"""Sparse volumes from a chunkedgraph, the expensive way.

Ported from the ``pcg_sparse`` prototype, whose findings are the reason this
module is shaped the way it is.

A PyChunkedGraph stores the *graph*. The voxels underneath it are a static
precomputed watershed volume with **no per-body spatial index**: nothing maps a
root ID to the blocks it occupies. So the question DVID answers with one indexed
lookup degrades here to *read dense blocks, mask to one root, sparsify*, touching
100-1000x more voxels than it keeps. That cost is inherent to the absence of an
index, not a defect in this code, and :class:`FetchStats` reports it rather than
hiding it.

Three things that are not obvious, all measured:

*Coarser is not cheaper by itself.* The stored block is the same shape at every
scale and fly pyramids downsample XY only, so a coarse request still pulls whole
blocks and discards more of each. Fetching one box per graph chunk costs the same
bytes at scale 6 as at scale 0. What makes coarse reads cheap is planning requests
onto *distinct storage blocks* - which took a whole-neuron scale-6 fetch from about
ten minutes to ten seconds.

*Masking beats agglomerating.* Asking the volume to agglomerate a cutout re-fetches
the supervoxel manifest for every chunk - hundreds of chunkedgraph round-trips for
one neuron. The manifest is identical for all of them, so it is fetched once and
applied locally.

*Overlap is expected.* The chunk grid is anchored at the volume's voxel offset,
which is not divisible by coarse-scale downsample ratios, so chunk boxes round
outward onto the scale grid and overlap their neighbours. Coordinates are
deduplicated globally rather than trusted to be disjoint.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .rle import encode_runs

__all__ = ["FetchStats", "estimate", "fetch_sparsevol", "DEFAULT_MAX_VOXELS"]

# Guard on the dense read. A whole neuron at scale 0 is roughly 1000x over this;
# the ceiling exists so that request fails with an explanation instead of looking
# like a hang.
DEFAULT_MAX_VOXELS = 2_000_000_000

@dataclass
class FetchStats:
    """What the dense read actually cost.

    ``voxels_downloaded`` counts the requested boxes; ``voxels_transferred`` counts
    whole storage blocks, which is what object storage actually serves - a sub-block
    request costs a full block. The two are equal only when requests are
    block-aligned, so ``waste_ratio`` is derived from the transferred figure.
    """

    n_l2_nodes: int = 0
    n_chunks: int = 0        # chunks the neuron occupies
    n_chunks_read: int = 0   # chunks actually read (< n_chunks when truncated)
    n_requests: int = 0
    n_requests_empty: int = 0
    voxels_downloaded: int = 0
    voxels_transferred: int = 0
    voxels_kept: int = 0
    duplicate_voxels: int = 0
    truncated: bool = False
    seconds: float = 0.0

    @property
    def waste_ratio(self) -> float:
        if self.voxels_kept == 0:
            return float("inf")
        return self.voxels_transferred / self.voxels_kept

    def summary(self) -> str:
        head = (
            f"{self.n_requests} requests for {self.n_chunks} chunks "
            f"({self.n_requests_empty} empty) in {self.seconds:.1f}s | "
            f"transferred {self.voxels_transferred:,} voxels, "
            f"kept {self.voxels_kept:,} ({self.waste_ratio:,.0f}x waste)"
        )
        return head + (" | TRUNCATED (max_chunks)" if self.truncated else "")

    def as_dict(self) -> dict:
        waste = self.waste_ratio
        return {
            "l2_nodes": self.n_l2_nodes,
            "chunks": self.n_chunks,
            "chunks_read": self.n_chunks_read,
            "requests": self.n_requests,
            "requests_empty": self.n_requests_empty,
            "voxels_downloaded": self.voxels_downloaded,
            "voxels_transferred": self.voxels_transferred,
            "voxels_kept": self.voxels_kept,
            "duplicate_voxels": self.duplicate_voxels,
            # None rather than `inf`: an empty result has no meaningful ratio, and
            # `inf` is not JSON-serialisable, so it would break the moment anyone
            # tried to log or cache these stats.
            "waste_ratio": None if self.voxels_kept == 0 else round(waste, 1),
            "truncated": self.truncated,
            "seconds": round(self.seconds, 2),
        }


def get_volume(ds):
    """The volume onto the *chunkedgraph*, as opposed to the display volume.

    The distinction is the whole point: FlyWire's spec points ``_segmentation_source``
    at the flat v783 bucket, which is right for meshes and neuroglancer and useless
    here, because a flat volume has no supervoxels to mask by. So this asks for
    ``_graph_source`` by name.

    Everything else - credentials, ``fill_missing``, the shared cache - is
    :func:`connecto.core.volume.get_volume`'s business. Boxes are clipped against
    per-scale bounds by the caller, so the volume is left unbounded, which is the
    default.
    """
    from ..core.volume import get_volume as _get_volume

    return _get_volume(ds, ds._graph_source())


# ------------------------------------------------------------------- geometry


def _grid_origin(meta) -> np.ndarray:
    if meta.chunks_start_at_voxel_offset:
        return np.asarray(meta.voxel_offset(meta.watershed_mip), dtype=np.int64)
    return np.zeros(3, dtype=np.int64)


def chunk_bbox(meta, position, scale: int = 0, clip: bool = True):
    """Voxel bounding box of one graph chunk, in ``scale``-level coordinates.

    The inverse of ``meta.point_to_chunk_position``. Built at ``watershed_mip``,
    where the chunk grid is defined, and then converted - so an anisotropic pyramid
    is handled by the metadata rather than by assuming an isotropic ratio.
    """
    from ..precomputed import Bbox

    position = np.asarray(position, dtype=np.int64)
    size = np.asarray(meta.graph_chunk_size, dtype=np.int64)
    minpt = position * size + _grid_origin(meta)
    box = Bbox(minpt, minpt + size)

    if scale != meta.watershed_mip:
        box = meta.bbox_to_mip(box, mip=meta.watershed_mip, to_mip=scale)
    if clip:
        box = Bbox.clamp(box, meta.bounds(scale))
    return box.astype(np.int64)


def decode_chunk_positions(meta, labels) -> np.ndarray:
    """Unique ``(M, 3)`` chunk positions of a set of graphene labels.

    A graphene label packs its layer and chunk position into the high bits, so the
    chunks a neuron occupies are derivable from its L2 IDs alone - no volume access
    at all. Several L2 nodes routinely share a chunk (a neuron passing through in
    two disconnected branches gets one node per component), so deduplicating here
    removes downloads rather than merely tidying.
    """
    labels = np.asarray(labels, dtype=np.uint64).ravel()
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pos = np.array(
        [meta.decode_chunk_position(int(label)) for label in labels], dtype=np.int64
    )
    return np.unique(pos, axis=0)


def occupied_chunks(client, meta, root_id: int, scale: int = 0):
    """The chunks a root occupies, as ``scale``-level boxes. One graph call."""
    l2_ids = client.chunkedgraph.get_leaves(int(root_id), stop_layer=2)
    boxes = []
    for position in np.atleast_2d(decode_chunk_positions(meta, l2_ids)):
        box = chunk_bbox(meta, position, scale=scale)
        # Chunks on the volume's edge can clip away to nothing; an empty box would
        # raise on download.
        if not box.subvoxel():
            boxes.append(box)
    return boxes, l2_ids


def _block_index_range(meta, box, scale: int):
    """Inclusive grid index range of the storage blocks a box touches."""
    block = np.asarray(meta.chunk_size(scale), dtype=np.int64)
    origin = np.asarray(meta.voxel_offset(scale), dtype=np.int64)
    lo = (np.asarray(box.minpt, dtype=np.int64) - origin) // block
    # Exclusive maxpt: the last covered block contains maxpt - 1.
    hi = (np.asarray(box.maxpt, dtype=np.int64) - 1 - origin) // block
    return lo, hi


def blocks_touched(meta, box, scale: int = 0) -> int:
    """How many storage blocks one request actually reads.

    Uses grid position, not just extent: a 256-wide box straddling a boundary
    touches two 256-wide blocks, which ``ceil(size / block)`` would miss.
    """
    lo, hi = _block_index_range(meta, box, scale)
    return int(np.prod(hi - lo + 1))


def merge_to_storage_blocks(meta, boxes, scale: int = 0):
    """Collapse chunk boxes onto the distinct storage blocks covering them.

    At scale 0 a graph chunk is exactly a whole number of blocks and this changes
    nothing. It matters at coarse scales: the pyramid downsamples XY only, so a
    chunk's XY footprint shrinks below one block while its Z extent stays put. Many
    chunks then land in the same block, and fetching per chunk re-downloads that
    block once per chunk.

    Never loses voxels - a block covers at least what the chunks did, and the
    supervoxel mask decides membership regardless of which box was read.
    """
    from ..precomputed import Bbox

    if len(boxes) == 0:
        return []

    block = np.asarray(meta.chunk_size(scale), dtype=np.int64)
    origin = np.asarray(meta.voxel_offset(scale), dtype=np.int64)

    indices = set()
    for box in boxes:
        lo, hi = _block_index_range(meta, box, scale)
        for x in range(int(lo[0]), int(hi[0]) + 1):
            for y in range(int(lo[1]), int(hi[1]) + 1):
                for z in range(int(lo[2]), int(hi[2]) + 1):
                    indices.add((x, y, z))

    merged = []
    for index in sorted(indices):
        minpt = origin + np.asarray(index, dtype=np.int64) * block
        candidate = Bbox.clamp(Bbox(minpt, minpt + block), meta.bounds(scale))
        if not candidate.subvoxel():
            merged.append(candidate.astype(np.int64))
    return merged


# ---------------------------------------------------------------------- fetch


def estimate(ds, root_id: int, scale: int = 0) -> dict:
    """What a fetch would cost, from the graph alone - no volume reads."""
    vol = get_volume(ds)
    meta = vol.meta
    boxes, l2_ids = occupied_chunks(ds.client, meta, root_id, scale=scale)
    blocks = merge_to_storage_blocks(meta, boxes, scale=scale)
    block_voxels = int(np.prod(meta.chunk_size(scale)))

    return {
        "root_id": int(root_id),
        "scale": scale,
        "l2_nodes": int(len(l2_ids)),
        "chunks": len(boxes),
        "requests": len(blocks),
        "voxels_transferred": block_voxels
        * sum(blocks_touched(meta, b, scale) for b in blocks),
        "resolution": tuple(float(v) for v in meta.resolution(scale)),
        "block_shape": [int(v) for v in meta.chunk_size(scale)],
    }


def _mask_to_root(array: np.ndarray, supervoxels: np.ndarray) -> np.ndarray:
    """Boolean mask of voxels belonging to the root's supervoxel set."""
    try:
        import fastremap

        return fastremap.mask_except(array, list(supervoxels), in_place=False, value=0) != 0
    except ImportError:
        # ImportError, not ModuleNotFoundError: fastremap is a compiled extension, so
        # an ABI mismatch against the installed numpy raises the base class. Catching
        # only the narrower one would mean a *broken* install kills the fetch instead
        # of falling back - the exact case the fallback is for.
        #
        # np.isin against a 27,000-element manifest is markedly slower, but correct.
        return np.isin(array, supervoxels)


def fetch_sparsevol(
    ds,
    root_id: int,
    scale: int = 0,
    *,
    max_chunks: int | None = None,
    max_voxels: int | None = DEFAULT_MAX_VOXELS,
    parallel: int = 8,
) -> tuple[np.ndarray, tuple[float, float, float], dict]:
    """One root's sparse volume, as ``(M, 4)`` runs plus resolution and stats."""
    vol = get_volume(ds)
    meta = vol.meta
    started = time.time()
    stats = FetchStats()

    boxes, l2_ids = occupied_chunks(ds.client, meta, root_id, scale=scale)
    stats.n_l2_nodes = len(l2_ids)
    stats.n_chunks = len(boxes)

    if max_chunks is not None and len(boxes) > max_chunks:
        boxes = boxes[:max_chunks]
        stats.truncated = True
    # Kept separately from `n_chunks` so a truncated run still reports how much of
    # the neuron it skipped, rather than quietly renaming the smaller number.
    stats.n_chunks_read = len(boxes)

    requests = merge_to_storage_blocks(meta, boxes, scale=scale)

    block_voxels = int(np.prod(meta.chunk_size(scale)))
    transferred = block_voxels * sum(blocks_touched(meta, b, scale) for b in requests)
    if max_voxels is not None and transferred > max_voxels:
        raise ValueError(
            f"Reading root {root_id} at scale {scale} would transfer "
            f"{transferred:,} voxels ({len(requests)} block reads for "
            f"{len(boxes)} chunks), over the {max_voxels:,} ceiling.\n"
            f"A chunkedgraph has no per-body index, so this is a dense read - see "
            f"`voxels.estimate()`. Use a coarser `scale=`, pass `max_chunks=` to "
            f"sample, or raise `max_voxels=`."
        )

    # One manifest fetch for the whole neuron, reused by every block.
    supervoxels = np.asarray(
        ds.client.chunkedgraph.get_leaves(int(root_id)), dtype=np.uint64
    )

    def fetch(box):
        array = np.asarray(vol.download(box, mip=scale, agglomerate=False))[..., 0]
        keep = _mask_to_root(array, supervoxels)
        return int(keep.size), np.argwhere(keep) + np.asarray(box.minpt, dtype=np.int64)

    pieces = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for downloaded, coords in pool.map(fetch, requests):
            stats.voxels_downloaded += downloaded
            stats.n_requests += 1
            if coords.shape[0] == 0:
                stats.n_requests_empty += 1
            else:
                pieces.append(coords)
    stats.voxels_transferred = transferred

    if pieces:
        coords = np.concatenate(pieces, axis=0)
        # Requests can overlap - boxes round outward onto the scale grid, and a
        # block may be reached from several chunks - so repeats are expected.
        before = coords.shape[0]
        coords = np.unique(coords, axis=0)
        stats.duplicate_voxels = before - coords.shape[0]
    else:
        coords = np.zeros((0, 3), dtype=np.int64)

    stats.voxels_kept = int(coords.shape[0])
    stats.seconds = time.time() - started

    resolution = tuple(float(v) for v in meta.resolution(scale))
    return encode_runs(coords), resolution, stats.as_dict()
