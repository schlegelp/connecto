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
import math
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..exceptions import MissingDependencyError
from .graphene import GrapheneMeta
from .limits import DEFAULT_MESH_PARALLEL
from .mesh import _build, _decode_draco, _join, decode_legacy_fragment
from .sharding import ShardingSpec, ShardReader
from .store import Store

__all__ = ["GrapheneMeshSource"]


def _packed_row_key(vertices: np.ndarray):
    """One integer per row that sorts exactly as the row does, or None.

    Mesh coordinates arrive on a lattice - draco quantises, and these fragments
    decode to whole nanometres - so the three columns can be folded into a single
    index of the box they span, x-major. That makes an ordinary integer sort
    reproduce the order a ``lexsort`` of the three columns gives, which is why this
    can be swapped in without moving a single vertex in the output.

    It is a *packing*, not a hash: the map is injective, so distinct coordinates
    cannot collide and no seam can be missed. A hash of comparable cost would leave
    a ~1e-8 chance per mesh of a silently unwelded seam, which is not a trade worth
    taking for a hairline crack nobody would ever trace back to here.

    ``None`` where the assumption fails - no rows, coordinates off the lattice, or a
    box too large to index - and the caller falls back to ``np.unique(axis=0)``.
    Nothing measures how often that happens, so the fallback has to stay correct
    rather than merely rare.
    """
    if not len(vertices):
        return None

    # Per column, not `min(axis=0)`. Reducing along the length-3 axis of an (N, 3)
    # array leaves numpy an inner loop of three, which it does not vectorise: 23 ms
    # against 2 ms for the six scalar reductions, on a 1.4M-vertex neuron.
    lo = np.array([vertices[:, c].min() for c in range(3)])
    hi = np.array([vertices[:, c].max() for c in range(3)])

    # Guard the cast before making it: `astype(int64)` of something past the integer
    # range is undefined, and numpy warns rather than answering. NaN and inf fail
    # this comparison too, so it doubles as the finiteness check.
    if not np.all(np.maximum(np.abs(lo), np.abs(hi)) < 2.0**62):
        return None

    packed = vertices.astype(np.int64)
    if not np.array_equal(packed, vertices):
        return None  # fractional coordinates: the cast would fuse distinct vertices

    packed -= lo.astype(np.int64)
    dims = [int(d) for d in (hi - lo).astype(np.int64) + 1]
    if math.prod(dims) > np.iinfo(np.intp).max:
        return None  # `ravel_multi_index` would refuse; ask it nothing it can't do

    return np.ravel_multi_index(tuple(packed.T), tuple(dims))


def _deduplicate_vertices(vertices: np.ndarray, faces: np.ndarray, is_chunk_aligned):
    """Merge vertices that appear exactly twice and lie on a chunk boundary.

    Exactly twice, because a seam vertex is shared by the two fragments either
    side of it. A vertex repeated more often than that is a genuine feature of the
    surface, not a seam, and merging it would weld unrelated geometry together.

    The merge is expressed as an integer key per vertex - shared where two vertices
    are to be fused, unique otherwise - so the fusing itself is integer bookkeeping.
    Both halves avoid a sort that numpy would otherwise do the slow way; on a
    1.4M-vertex mosquito neuron the two together are ~4.5x faster than the
    ``np.unique`` pair they replace, and return the identical arrays.

    Grouping identical coordinates is ``np.unique`` over :func:`_packed_row_key`
    rather than over the rows. ``np.unique(axis=0)`` gets its answer by viewing each
    row as one structured scalar and sorting *those* - a generic element-by-element
    comparator, called a few tens of millions of times. Given one integer per row it
    is an ordinary numeric sort instead: 0.90 s -> 0.06 s, of which a ``lexsort`` of
    the three float columns would still have cost 0.18 s.

    Renumbering is a lookup table, not a second ``np.unique``, because the keys are
    already integers with a known bound - one slot per coordinate group plus one per
    vertex - so "which distinct keys are used, and in what order" is a scatter and a
    ``flatnonzero`` rather than a sort of three million face corners: 0.4 s -> 0.06 s.
    """
    n = len(vertices)
    if n == 0:
        return vertices, faces

    # --- group identical coordinates -> `inverse` (group per vertex) and `counts`
    key = _packed_row_key(vertices)
    if key is None:
        _, inverse, counts = np.unique(
            vertices, axis=0, return_inverse=True, return_counts=True
        )
    else:
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)

    # `counts[inverse]`, not `np.isin(inverse, flatnonzero(counts == 2))`: the group
    # size is one gather away, and asking `isin` for it re-sorts to answer a question
    # already indexed by group id.
    merge = (counts[inverse] == 2) & np.asarray(is_chunk_aligned, dtype=bool)

    # Fusing pairs share their coordinate's id; everyone else gets an id of their
    # own, offset past the coordinate ids so the two ranges cannot collide.
    n_groups = len(counts)
    key = np.where(merge, inverse, np.arange(n, dtype=np.intp) + n_groups)

    # --- renumber: keep one vertex per distinct key, drop any the faces never name
    corners = faces.reshape(-1)
    corner_keys = key[corners]

    # One table, used for two things in turn. First it holds, per key, some vertex
    # bearing it - and *which* vertex does not matter, because two vertices share a
    # key only if they were fused, and they were fused only for having identical
    # coordinates. So an unordered scatter is enough; there is no need to hunt for a
    # first occurrence. `-1` marks the keys no face names, which are then dropped.
    slot = np.full(n_groups + n, -1, dtype=np.intp)
    slot[corner_keys] = corners
    kept = np.flatnonzero(slot >= 0)  # ascending, so: the distinct keys, in order
    out_vertices = vertices[slot[kept]]

    # Now the same slots become the key -> output-index map, which is the only thing
    # still wanted from them.
    slot[kept] = np.arange(len(kept), dtype=np.intp)
    return out_vertices, slot[corner_keys].reshape(-1, 3)


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
    """Meshes for a chunkedgraph datastack.

    ``parallel`` - fragment reads in flight - is the single biggest lever on how
    long a mesh takes here; see
    :data:`~connecto.precomputed.limits.DEFAULT_MESH_PARALLEL` for the measurements
    behind its default.
    """

    # The graph layer sets the resolution; `lod` only ends up in the manifest URL.
    has_lods = False

    def __init__(
        self, meta: GrapheneMeta, session=None,
        parallel: int = DEFAULT_MESH_PARALLEL,
    ):
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

    def _read_fragment(self, read):
        """One fragment, start to finish, on a worker thread.

        Decoding runs here rather than back on the calling thread because draco
        decoding is the one part of this that a thread can genuinely overlap:
        DracoPy 2.1 releases the GIL for it (seung-lab/DracoPy#67), and a large Aedes
        neuron's 71 fragments decode in 46 ms across the pool against 159 ms serially.
        Before 2.1 the decode holds the GIL, and this arrangement is still no worse:
        the work has to happen on some thread, and doing it here at least overlaps
        it with the reads still in flight. Unlike ``MultiResMeshSource``, then, there
        is no :func:`~connecto.precomputed.limits.decode_workers` gate here - the
        pool exists for the reads either way.

        Chunk-boundary marking comes along for the ride: it is numpy, so it holds the
        GIL and parallelises poorly, but it needs the fragment's own label and doing
        it here saves carrying labels back out.
        """
        fn, args, label = read
        decoded = self._decode(fn(*args))
        if decoded is None:
            return None
        return decoded, self._chunk_aligned(decoded[0], label) if self.sharded else None

    def get(self, segid: int, lod: int = 0, parallel: int | None = None):
        """One neuron's mesh, in nanometres.

        ``parallel`` overrides the source's own setting for this call - useful
        because volumes are cached per source, so the constructor's value is fixed
        once the first read has happened.
        """
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

        budget = self.parallel if parallel is None else int(parallel)
        workers = max(1, min(budget, len(reads)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = pool.map(self._read_fragment, reads)
            done = [piece for piece in fetched if piece is not None]

        pieces = [decoded for decoded, _ in done]
        masks = [mask for _, mask in done if mask is not None]

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
        half = grid / 2
        return np.any((behind < half) | (ahead <= half), axis=1)

    def _reader_store(self, layer) -> Store:
        """The store holding one layer's static shards."""
        return self._reader(int(layer)).store

    def _read_from_shard(self, label: int) -> bytes | None:
        """Fallback for a server that named a fragment without resolving it."""
        try:
            return self._reader(self.meta.decode_layer_id(int(label))).get(int(label))
        except KeyError:
            return None
