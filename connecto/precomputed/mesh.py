"""Precomputed meshes.

Three formats live behind ``info["mesh"]``, and connecto meets all of them:

``neuroglancer_multilod_draco``
    The modern one, and what every flat volume connecto reads uses. An object is an
    octree of draco-encoded fragments; a manifest says where each fragment sits and
    how to un-quantize its vertices. Sharded or, rarely, one file per object.
``neuroglancer_legacy_mesh``
    A JSON manifest naming raw triangle-soup fragments. Old, still out there.

The vertex maths is the fiddly part and is spelled out in
:func:`_to_model_space`: draco stores each fragment's positions as small integers
inside its own box, so a fragment is meaningless without its manifest.

Spec: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..exceptions import MissingDependencyError
from .limits import DEFAULT_MESH_PARALLEL, decode_workers
from .sharding import ShardingSpec, ShardReader
from .store import Store

__all__ = ["mesh_source", "MultiResMeshSource", "LegacyMeshSource", "MultiResManifest"]


def _decode_draco(binary: bytes):
    """``(vertices, faces)`` out of one draco fragment."""
    try:
        import DracoPy
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "Reading meshes requires the `DracoPy` package, which is not "
            "installed. Install it with `pip install DracoPy`."
        ) from exc

    mesh = DracoPy.decode(binary)
    verts = np.asarray(mesh.points, dtype="float64").reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype="int64").reshape(-1, 3)
    return verts, faces


def _apply_transform(vertices: np.ndarray, transform) -> np.ndarray:
    """Model space -> physical (nm), via the mesh info's 3x4 matrix."""
    if transform is None:
        return vertices
    matrix = np.asarray(transform, dtype="float64").reshape(3, 4)
    return vertices @ matrix[:, :3].T + matrix[:, 3]


def _build(vertices, faces):
    import trimesh

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _join(pieces) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``(vertices, faces)`` fragments, re-basing each face index."""
    vertices, faces, placed = [], [], 0
    for v, f in pieces:
        vertices.append(v)
        faces.append(f + placed)
        placed += len(v)
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype="int64")
    return np.concatenate(vertices), np.concatenate(faces)


# --------------------------------------------------------------------- manifest


@dataclass
class MultiResManifest:
    """Where each fragment of one object lives, and how to place its vertices."""

    chunk_shape: np.ndarray
    grid_origin: np.ndarray
    lod_scales: np.ndarray
    vertex_offsets: np.ndarray
    fragment_positions: list
    fragment_offsets: list

    @property
    def num_lods(self) -> int:
        return len(self.fragment_offsets)

    def lod_byte_sizes(self) -> list:
        return [int(np.sum(sizes.astype(np.int64))) for sizes in self.fragment_offsets]

    @classmethod
    def from_binary(cls, binary: bytes) -> MultiResManifest:
        # num_lods is the 7th word; the header's own size depends on it.
        num_lods = int(np.frombuffer(binary[24:28], dtype="<u4")[0])
        header_dt = np.dtype(
            [
                ("chunk_shape", "<f4", (3,)),
                ("grid_origin", "<f4", (3,)),
                ("num_lods", "<u4"),
                ("lod_scales", "<f4", (num_lods,)),
                ("vertex_offsets", "<f4", (num_lods, 3)),
                ("num_fragments_per_lod", "<u4", (num_lods,)),
            ]
        )
        header = np.frombuffer(binary[: header_dt.itemsize], dtype=header_dt)[0]

        offset = header_dt.itemsize
        positions, sizes = [], []
        for lod in range(num_lods):
            n = int(header["num_fragments_per_lod"][lod])
            # Stored as three contiguous runs of x, y and z - hence order="F".
            positions.append(
                np.frombuffer(binary[offset : offset + 12 * n], dtype="<u4").reshape(
                    (n, 3), order="F"
                )
            )
            offset += 12 * n
            sizes.append(np.frombuffer(binary[offset : offset + 4 * n], dtype="<u4"))
            offset += 4 * n

        if offset != len(binary):
            raise ValueError(
                f"Mesh manifest is {len(binary)} bytes but describes {offset}; "
                f"it is truncated or not a multi-resolution manifest."
            )

        return cls(
            chunk_shape=np.asarray(header["chunk_shape"], dtype="float64"),
            grid_origin=np.asarray(header["grid_origin"], dtype="float64"),
            lod_scales=np.asarray(header["lod_scales"], dtype="float64"),
            vertex_offsets=np.asarray(header["vertex_offsets"], dtype="float64"),
            fragment_positions=positions,
            fragment_offsets=sizes,
        )


def _to_model_space(vertices, manifest, lod, frag, quantization_bits):
    """Un-quantize a fragment's vertices into the object's own coordinate frame.

    Each draco position component is an integer in ``[0, 2**bits)`` naming a point
    inside that fragment's box; the box is found from the fragment's octree
    position, the chunk shape and the level of detail.
    """
    return (
        manifest.grid_origin
        + manifest.vertex_offsets[lod]
        + manifest.chunk_shape
        * (2**lod)
        * (
            manifest.fragment_positions[lod][frag, :]
            + vertices / (2.0**quantization_bits - 1)
        )
    )


# ---------------------------------------------------------------- mesh sources


class MultiResMeshSource:
    """``neuroglancer_multilod_draco``, sharded or one file per object."""

    # Whether `lod` means anything here. Callers ask the source rather than sniffing
    # the URL, so a format that grows levels only has to say so in one place.
    has_lods = True

    def __init__(self, store: Store, info: dict, parallel: int | None = None):
        self.store = store
        self.info = info
        # Decode workers, not download workers - see `get`. `None` defers to
        # `limits.decode_workers()`, which cannot be settled at import time because
        # it depends on which DracoPy is installed.
        self.parallel = parallel
        self.transform = info.get("transform")
        self.quantization_bits = int(info.get("vertex_quantization_bits", 16))
        spec = info.get("sharding")
        self.spec = ShardingSpec.from_dict(spec) if spec else None
        self.reader = ShardReader(store, self.spec) if self.spec else None

    # `manifest_at` returns the manifest bytes plus where the fragment data that
    # precedes them starts, because the two are only locatable together.
    def _manifest_and_data_origin(self, segid: int):
        if self.reader is not None:
            hit = self.reader.locate(int(segid))
            if hit is None:
                return None, None, None
            filename, offset, size = hit
            from .store import decompress

            raw = self.store.get_range(filename, offset, offset + size)
            if raw is None:
                return None, None, None
            manifest = MultiResManifest.from_binary(
                decompress(raw, self.spec.data_encoding)
            )
            # Fragment data sits immediately before the manifest in the same shard.
            origin = offset - sum(manifest.lod_byte_sizes())
            return manifest, filename, origin

        raw = self.store.get(f"{int(segid)}.index")
        if raw is None:
            return None, None, None
        return MultiResManifest.from_binary(raw), f"{int(segid)}", 0

    def get(
        self, segid: int, lod: int = 0, parallel: int | None = None,
        clamp: bool = False,
    ):
        """One object's mesh, at one level of detail.

        How deep the octree goes is a property of the *object*, not of the bucket:
        the Janelia FlyEM volumes are uniformly four deep, but a small FlyWire v783
        neuron has exactly one level where a large one has four. ``clamp`` pins
        ``lod`` to the coarsest level the object has instead of raising.

        ``parallel`` buys something different here than on the other two sources. A
        whole LOD is one contiguous byte range, so there is exactly one request to
        make and no downloading to spread out - but that range holds dozens of
        independently draco-encoded fragments, and those can decode in parallel.
        Whether that is worth doing depends on the installed decoder, so the default
        comes from :func:`~connecto.precomputed.limits.decode_workers`; at one
        worker this runs the plain serial loop rather than a pool of one.
        """
        segid = int(segid)
        manifest, filename, origin = self._manifest_and_data_origin(segid)
        if manifest is None:
            raise KeyError(f"No mesh for segment {segid} in {self.store.url}.")

        if lod is None:
            lod = 0
        if lod < 0:
            lod += manifest.num_lods
        if clamp:
            lod = min(max(lod, 0), manifest.num_lods - 1)
        elif not 0 <= lod < manifest.num_lods:
            raise ValueError(
                f"Level of detail {lod} out of range for segment {segid}; "
                f"{manifest.num_lods} available (0 is the finest). How many levels "
                f"an object has depends on its size, so a level that works for one "
                f"segment need not exist for another."
            )

        per_lod = manifest.lod_byte_sizes()
        start = origin + sum(per_lod[:lod])
        blob = self.store.get_range(filename, start, start + per_lod[lod])
        if blob is None:
            raise KeyError(f"Mesh fragments for segment {segid} are missing.")

        sizes = manifest.fragment_offsets[lod]
        fragments, cursor = [], 0
        for frag in range(len(sizes)):
            size = int(sizes[frag])
            piece = blob[cursor : cursor + size]
            cursor += size
            # An empty fragment is legal: a child exists at a finer level but this
            # level has nothing there, which marching cubes run per-level produces.
            if size:
                fragments.append((frag, piece))

        def decode(item):
            frag, piece = item
            v, f = _decode_draco(piece)
            v = _to_model_space(v, manifest, lod, frag, self.quantization_bits)
            return _apply_transform(v, self.transform), f

        budget = self.parallel if parallel is None else int(parallel)
        if budget is None:
            budget = decode_workers()
        workers = max(1, min(budget, len(fragments) or 1))

        if workers == 1:
            # Not a one-worker pool: on a GIL-holding DracoPy that is the common
            # case, and it should cost exactly what the plain loop costs.
            pieces = [decode(item) for item in fragments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pieces = list(pool.map(decode, fragments))

        return _build(*_join(pieces))


class LegacyMeshSource:
    """``neuroglancer_legacy_mesh``: a JSON manifest naming raw fragments."""

    has_lods = False

    def __init__(
        self, store: Store, info: dict | None = None,
        parallel: int = DEFAULT_MESH_PARALLEL,
    ):
        self.store = store
        self.info = info or {}
        self.transform = (self.info or {}).get("transform")
        self.parallel = parallel

    def _fragments(self, segid: int) -> list:
        raw = self.store.get(f"{int(segid)}:0")
        if raw is None:
            return []
        return json.loads(raw).get("fragments", [])

    def get(self, segid: int, lod: int = 0, parallel: int | None = None):
        """One object's mesh. ``lod`` is accepted and ignored - this format has none.

        The fragments are separate objects, so they are fetched concurrently - see
        :data:`~connecto.precomputed.limits.DEFAULT_MESH_PARALLEL`. Decoding stays on
        this thread, which loses nothing: it is numpy on bytes already in hand, and
        it holds the GIL either way.
        """
        segid = int(segid)
        fragments = self._fragments(segid)
        if not fragments:
            raise KeyError(f"No mesh for segment {segid} in {self.store.url}.")

        budget = self.parallel if parallel is None else int(parallel)
        workers = max(1, min(budget, len(fragments)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blobs = pool.map(self.store.get, fragments)

            pieces = []
            for blob in blobs:
                if blob is None:
                    continue
                v, f = decode_legacy_fragment(blob)
                pieces.append((_apply_transform(v, self.transform), f))

        return _build(*_join(pieces))


def decode_legacy_fragment(binary: bytes):
    """``uint32 num_vertices``, then float32 xyz, then uint32 triangle indices."""
    n = int(np.frombuffer(binary[:4], dtype="<u4")[0])
    vertices = np.frombuffer(binary, dtype="<f4", count=3 * n, offset=4).reshape(-1, 3)
    faces = np.frombuffer(binary, dtype="<u4", offset=4 + 12 * n).reshape(-1, 3)
    return vertices.astype("float64"), faces.astype("int64")


def mesh_source(store: Store, mesh_dir: str):
    """Pick the reader that matches whatever is under `mesh_dir`."""
    sub = store.at(mesh_dir)
    info = sub.get_json("info", missing_ok=True)
    kind = (info or {}).get("@type", "neuroglancer_legacy_mesh")

    if kind == "neuroglancer_multilod_draco":
        return MultiResMeshSource(sub, info)
    if kind == "neuroglancer_legacy_mesh":
        return LegacyMeshSource(sub, info)
    raise ValueError(f"Unsupported mesh format {kind!r} at {sub.url}.")
