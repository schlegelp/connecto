"""Graphene meshes: static shards plus whatever proofreading has changed since.

A chunkedgraph's meshes come in two halves. The *initial* half was meshed in bulk
and packed into shards. The *dynamic* half is everything re-meshed since, one file
per fragment, because an edit that merges two neurons cannot rewrite a shard. A
neuron is assembled from both, and which fragments it needs is a question only the
graph can answer - so the meshing service answers it.

That service does more than name fragments. Asked with ``verify=True`` it returns
each *initial* fragment as ``~<layer>/<shard>.shard:<offset>:<size>`` - it has
already looked the fragment up in the shard index for us, so a fragment costs one
byte-range GET and no index reads at all. Where a server does not do that, we fall
back to reading the shard indices ourselves.

Unlike a flat volume's multi-resolution meshes, these fragments are draco-encoded
in *absolute nanometres*, so there is no manifest, no octree and no un-quantizing:
decode and concatenate.
"""

from __future__ import annotations

import functools
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..exceptions import MissingDependencyError
from .graphene import GrapheneMeta
from .mesh import _build, _decode_draco, _join, decode_legacy_fragment
from .sharding import ShardingSpec, ShardReader
from .store import Store

__all__ = ["GrapheneMeshSource"]


def _deduplicate_vertices(vertices: np.ndarray, faces: np.ndarray, is_chunk_aligned):
    """Merge vertices that appear exactly twice and lie on a chunk boundary.

    Exactly twice, because a seam vertex is shared by the two fragments either
    side of it. A vertex repeated more often than that is a genuine feature of the
    surface, not a seam, and merging it would weld unrelated geometry together.

    The merge is expressed as an integer key per vertex - shared where two vertices
    are to be fused, unique otherwise - so the deduplication is a 1-D ``np.unique``.
    Doing it on the coordinates themselves means lexsorting a ``(3 x faces, 4)``
    float array, which for a million-vertex neuron is a 150 MB temporary and several
    seconds.
    """
    _, inverse, counts = np.unique(
        vertices, return_inverse=True, return_counts=True, axis=0
    )
    inverse = np.asarray(inverse).reshape(-1)
    doubled = np.isin(inverse, np.flatnonzero(counts == 2))
    merge = doubled & np.asarray(is_chunk_aligned, dtype=bool)

    # Fusing pairs share their coordinate's id; everyone else gets an id of their
    # own, offset past the coordinate ids so the two ranges cannot collide.
    key = np.where(merge, inverse, np.arange(len(vertices), dtype=np.int64) + len(counts))

    corners = faces.reshape(-1)
    _, first, new_faces = np.unique(
        key[corners], return_index=True, return_inverse=True
    )
    return vertices[corners[first]], np.asarray(new_faces).reshape(-1, 3)


# e.g. ~2/344239114-0.shard:224659:442
_INITIAL = re.compile(r"^~(\d+)/([\d\-]+\.shard):(\d+):(\d+)$")


class _GrapheneShardReader(ShardReader):
    """Same container, different filenames: graphene names shards per chunk."""

    def __init__(self, store: Store, spec: ShardingSpec, meta: GrapheneMeta):
        super().__init__(store, spec)
        self.meta = meta

    def _filename(self, key: int, shard: int) -> str:
        return f"{self.meta.decode_chunk_position_number(key)}-{shard}.shard"


class GrapheneMeshSource:
    """Meshes for a chunkedgraph datastack."""

    def __init__(self, meta: GrapheneMeta, session=None, parallel: int = 8):
        self.meta = meta
        # `session` talks to the CAVE server, and only to it. The bucket is public,
        # and handing Google Storage a CAVE bearer token makes it try, and fail, to
        # authenticate with it - a 401 on an object anyone can read anonymously. No
        # session is threaded through to the store, so that stays true.
        self.session = session
        self.parallel = parallel
        self.store = Store(meta.data_dir).at(meta.mesh_path)
        self._info = self.store.get_json("info", missing_ok=True)
        self.mesh_mip = int((self._info or {}).get("mip", 0))
        self._readers: dict[int, _GrapheneShardReader] = {}

    @property
    def sharded(self) -> bool:
        return bool((self._info or {}).get("sharding"))

    def _reader(self, layer: int) -> _GrapheneShardReader:
        layer = int(layer)
        if layer not in self._readers:
            spec = ShardingSpec.from_dict(self._info["sharding"][str(layer)])
            self._readers[layer] = _GrapheneShardReader(
                self.store.at(self.meta.sharded_mesh_dir, str(layer)), spec, self.meta
            )
        return self._readers[layer]

    def _dynamic_store(self) -> Store:
        # An unsharded deployment keeps its fragments in the mesh directory itself;
        # a sharded one puts the re-meshed ones in a subdirectory beside `initial`.
        return self.store.at(self.meta.unsharded_mesh_dir) if self.sharded else self.store

    # ------------------------------------------------------------------ manifest

    def fetch_manifest(self, segid: int, lod: int = 0) -> dict:
        """Ask the meshing service which fragments make up this object."""
        if self.session is None:
            raise ValueError(
                "Graphene meshes need an authenticated session; pass the CAVE "
                "client's session."
            )
        segid = int(segid)
        level = min(self.meta.decode_layer_id(segid), self.meta.max_meshed_layer)

        url = f"{self.meta.manifest_endpoint}/{segid}:{int(lod)}"
        response = self.session.get(
            url,
            params={"verify": True, "return_seg_ids": 1},
            data=json.dumps({"start_layer": level}),
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _classify(manifest: dict):
        """Split the manifest into byte-ranged shard reads and whole-file reads."""
        fragments = manifest.get("fragments") or []
        segids = manifest.get("seg_ids") or [None] * len(fragments)

        initial, dynamic, unresolved = [], [], []
        for name, segid in zip(fragments, segids):
            if not name:
                continue
            if name[0] != "~":
                # A dynamic fragment is named "<label>:0:<bbox>"; the label is what
                # says which graph layer it came from, and so how big its chunk is.
                label = segid
                if label is None:
                    try:
                        label = int(name.split(":")[0])
                    except ValueError:
                        label = None
                dynamic.append((name, label))
                continue
            match = _INITIAL.match(name)
            if match:
                layer, filename, start, size = match.groups()
                initial.append((int(layer), filename, int(start), int(size), segid))
            else:
                # A server that named the fragment but did not resolve it for us.
                unresolved.append(int(segid) if segid is not None else None)
        return initial, dynamic, unresolved

    # -------------------------------------------------------------------- reading

    def _decode(self, blob: bytes | None):
        if not blob:
            return None
        try:
            return _decode_draco(blob)
        except MissingDependencyError:
            raise  # a missing DracoPy is not a corrupt fragment
        except Exception:
            # Older fragments predate draco and are plain precomputed triangle soup.
            try:
                return decode_legacy_fragment(blob)
            except Exception:
                return None

    def get(self, segid: int, lod: int = 0):
        """One neuron's mesh, in nanometres."""
        segid = int(segid)
        initial, dynamic, unresolved = self._classify(self.fetch_manifest(segid, lod))

        dyn = self._dynamic_store()
        reads = (
            [(dyn.get, (name,), label) for name, label in dynamic]
            + [
                (self._reader_store(layer).get_range, (filename, start, start + size), label)
                for layer, filename, start, size, label in initial
            ]
            + [(self._read_from_shard, (label,), label) for label in unresolved if label]
        )
        if not reads:
            raise KeyError(
                f"The meshing service lists no fragments for segment {segid}. "
                f"It may not be meshed yet."
            )

        workers = max(1, min(self.parallel, len(reads)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blobs = pool.map(lambda read: read[0](*read[1]), reads)

            pieces, masks = [], []
            for blob, (_, _, label) in zip(blobs, reads):
                decoded = self._decode(blob)
                if decoded is None:
                    continue
                pieces.append(decoded)
                if self.sharded:
                    masks.append(self._chunk_aligned(decoded[0], label))

        if not pieces:
            raise KeyError(f"No decodable mesh fragments for segment {segid}.")

        vertices, faces = _join(pieces)
        if masks:
            # Fragments are meshed independently, so the two sides of a chunk
            # boundary each carry their own copy of the vertices along the seam.
            # Merging those is what turns a pile of fragments into one surface.
            vertices, faces = _deduplicate_vertices(
                vertices, faces, np.concatenate(masks)
            )
        return _build(vertices, faces)

    # ------------------------------------------------------------------ stitching

    @functools.cached_property
    def _chunk_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """``(origin, level-2 chunk size)`` in nanometres.

        Fixed for the whole datastack, so it is derived once rather than per
        fragment - a large neuron has thousands of them.
        """
        meta = self.meta
        offset = (
            meta.voxel_offset(self.mesh_mip).astype("float64")
            * meta.resolution(self.mesh_mip).astype("float64")
            if meta.chunks_start_at_voxel_offset
            else np.zeros(3)
        )
        size = meta.chunk_size(self.mesh_mip).astype("float64") * meta.resolution(0)
        return offset, size

    def _chunk_aligned(self, vertices: np.ndarray, label) -> np.ndarray:
        """Which of a fragment's vertices sit on one of its chunk's boundaries."""
        if label is None:
            return np.zeros(len(vertices), dtype=bool)
        level = self.meta.decode_layer_id(int(label))
        grid = self.meta.get_draco_grid_size(min(level, self.meta.max_meshed_layer))

        offset, lvl2_size_nm = self._chunk_geometry
        chunk_size = lvl2_size_nm * (2.0 ** (level - 2))

        behind = np.mod(vertices - offset, chunk_size)
        ahead = chunk_size - behind
        # Draco rounds up, so "on the boundary" is within half a grid step of it.
        return np.any(behind < (grid / 2), axis=1) | np.any(ahead <= (grid / 2), axis=1)

    def _reader_store(self, layer) -> Store:
        """The store holding one layer's static shards."""
        return self._reader(int(layer)).store

    def _read_from_shard(self, label: int) -> bytes | None:
        """Fallback for a server that named a fragment without resolving it."""
        try:
            return self._reader(self.meta.decode_layer_id(int(label))).get(int(label))
        except KeyError:
            return None
