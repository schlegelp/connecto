"""Assembling a cutout from stored chunks.

A precomputed volume is a grid of independently-encoded chunks. Reading a box means
working out which chunks it touches, fetching those, and pasting the overlapping
part of each into the output. Chunks are addressed either by filename (unsharded)
or by *compressed morton code* into a shard (sharded).

Fetches run on a thread pool: each one is a single HTTPS GET with the GIL released,
so threads are both the right tool and the cheap one. Shard indices are cached by
:class:`~connecto.precomputed.sharding.ShardReader`, so a cutout landing many
chunks in one shard pays for that shard's indices once.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .codecs import decode_chunk
from .lib import Bbox
from .meta import PrecomputedMeta
from .sharding import ShardReader
from .store import Store

__all__ = ["ImageSource", "compressed_morton_code"]


def compressed_morton_code(position, grid_size) -> int:
    """Interleave the bits of a grid position, skipping exhausted axes.

    "Compressed" is the skipping: once an axis has run out of significant bits
    (its grid is shorter than the others) it contributes nothing further, so the
    code stays dense instead of leaving gaps.
    """
    bits = [max(0, math.ceil(math.log2(int(g)))) if int(g) > 1 else 0 for g in grid_size]
    code = 0
    out_bit = 0
    for i in range(max(bits) if bits else 0):
        for dim in range(3):
            if i < bits[dim]:
                code |= ((int(position[dim]) >> i) & 1) << out_bit
                out_bit += 1
    return code


class ImageSource:
    """Reads boxes out of one precomputed volume."""

    def __init__(
        self,
        store: Store,
        meta: PrecomputedMeta,
        *,
        fill_missing: bool = True,
        bounded: bool = False,
        parallel: int = 8,
    ):
        self.store = store
        self.meta = meta
        self.fill_missing = fill_missing
        self.bounded = bounded
        self.parallel = parallel
        self._shards: dict[int, ShardReader] = {}

    def _shard_reader(self, mip: int) -> ShardReader | None:
        spec = self.meta.sharding(mip)
        if spec is None:
            return None
        if mip not in self._shards:
            self._shards[mip] = ShardReader(self.store.at(self.meta.key(mip)), spec)
        return self._shards[mip]

    # ------------------------------------------------------------------ chunks

    def _chunk_boxes(self, bbox: Bbox, mip: int):
        """Every stored chunk overlapping `bbox`, as (chunk box, grid position)."""
        chunk = self.meta.chunk_size(mip)
        offset = self.meta.voxel_offset(mip)
        bounds = self.meta.bounds(mip)

        lo = np.floor((bbox.minpt - offset) / chunk).astype(np.int64)
        hi = np.ceil((bbox.maxpt - offset) / chunk).astype(np.int64)

        out = []
        for index in itertools.product(*(range(int(a), int(b)) for a, b in zip(lo, hi))):
            pos = np.array(index, dtype=np.int64)
            minpt = offset + pos * chunk
            # A chunk at the volume's edge is *stored* clipped, so its decoded shape
            # is the clipped one, not `chunk_size`.
            box = Bbox.clamp(Bbox(minpt, minpt + chunk), bounds)
            if not box.subvoxel():
                out.append((box, pos))
        return out

    def _fetch_chunk(self, box: Bbox, pos: np.ndarray, mip: int, source) -> np.ndarray:
        """Read and decode one chunk. `source` is a ShardReader or a plain Store."""
        if isinstance(source, ShardReader):
            data = source.get(compressed_morton_code(pos, self.meta.grid_size(mip)))
        else:
            name = "_".join(f"{int(a)}-{int(b)}" for a, b in zip(box.minpt, box.maxpt))
            data = source.get(name)

        if data is None and not self.fill_missing:
            raise FileNotFoundError(
                f"Chunk {tuple(int(v) for v in box.minpt)} of "
                f"{self.meta.path} is not stored. Pass fill_missing=True to read "
                f"it as empty."
            )

        return decode_chunk(
            data,
            self.meta.encoding(mip),
            box.size3,
            self.meta.dtype,
            block_size=self.meta.compressed_segmentation_block_size(mip),
            num_channels=self.meta.num_channels,
        )

    # ---------------------------------------------------------------- download

    def download(self, bbox: Bbox, mip=0, parallel: int | None = None) -> np.ndarray:
        """The segmentation inside `bbox`, as ``(x, y, z, channels)``."""
        mip = self.meta.to_mip(mip)
        bbox = Bbox(
            np.asarray(bbox.minpt, dtype=np.int64), np.asarray(bbox.maxpt, dtype=np.int64)
        )

        bounds = self.meta.bounds(mip)
        if self.bounded and not bounds.contains_bbox(bbox):
            raise ValueError(
                f"{bbox} is outside the volume's bounds at scale {mip} "
                f"({bounds}). Pass bounded=False to read it clipped."
            )

        # The caller gets the shape it asked for, even where that runs off the end
        # of the volume - the part outside is empty by definition, not an error.
        # Only the *chunk search* is clipped, so we do not go looking for chunks
        # that cannot exist.
        shape = np.maximum(bbox.size3, 0)
        out = np.zeros(
            (*(int(v) for v in shape), self.meta.num_channels),
            dtype=self.meta.dtype,
            order="F",
        )
        if bbox.subvoxel():
            return out

        inside = Bbox.clamp(bbox, bounds)
        if inside.subvoxel():
            return out
        chunks = self._chunk_boxes(inside, mip)
        if not chunks:
            return out

        # Resolved once, before any worker starts: every thread would otherwise race
        # to build its own reader on the first chunk, and each loser throws away a
        # freshly-fetched shard index.
        source = self._shard_reader(mip) or self.store.at(self.meta.key(mip))

        def work(item):
            box, pos = item
            return box, self._fetch_chunk(box, pos, mip, source)

        workers = max(1, min(parallel if parallel is not None else self.parallel, len(chunks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for box, data in pool.map(work, chunks):
                self._paste(out, bbox, box, data)

        return out

    @staticmethod
    def _paste(out: np.ndarray, bbox: Bbox, box: Bbox, data: np.ndarray) -> None:
        """Copy the part of one chunk that falls inside the requested box."""
        overlap = Bbox.clamp(box, bbox)
        if overlap.subvoxel():
            return
        dst = (overlap.minpt - bbox.minpt).astype(np.int64)
        src = (overlap.minpt - box.minpt).astype(np.int64)
        n = overlap.size3.astype(np.int64)
        out[
            dst[0] : dst[0] + n[0], dst[1] : dst[1] + n[1], dst[2] : dst[2] + n[2]
        ] = data[src[0] : src[0] + n[0], src[1] : src[1] + n[1], src[2] : src[2] + n[2]]
