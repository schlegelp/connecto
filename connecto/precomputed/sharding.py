"""Reading the ``neuroglancer_uint64_sharded_v1`` container format.

A shard packs many chunks (or meshes) into one object so that storage sees a few
big files instead of millions of small ones. Finding one value inside it costs two
index reads and one data read - unless you cache the indices, which is the whole
game: a cutout touching 500 chunks in the same shard should pay for that shard's
indices *once*, not 500 times.

Layout of a ``.shard`` file::

    [ shard index      ]  2^minishard_bits entries x 16 bytes: (start, end) of
                          each minishard index, relative to the end of this block
    [ minishard indices]  3 x N uint64, delta-encoded: ids, offsets, sizes
    [ data             ]  the values themselves

Spec: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from . import mmh3
from .store import Store, decompress

__all__ = ["ShardingSpec", "ShardReader"]


@dataclass(frozen=True)
class ShardingSpec:
    preshift_bits: int
    hash: str
    minishard_bits: int
    shard_bits: int
    minishard_index_encoding: str = "raw"
    data_encoding: str = "raw"

    @classmethod
    def from_dict(cls, d: dict) -> ShardingSpec:
        kind = d.get("@type", "neuroglancer_uint64_sharded_v1")
        if kind != "neuroglancer_uint64_sharded_v1":
            raise ValueError(f"Unsupported sharding format {kind!r}.")
        if d["hash"] not in ("identity", "murmurhash3_x86_128"):
            raise ValueError(f"Unsupported shard hash {d['hash']!r}.")
        return cls(
            preshift_bits=int(d["preshift_bits"]),
            hash=str(d["hash"]),
            minishard_bits=int(d["minishard_bits"]),
            shard_bits=int(d["shard_bits"]),
            minishard_index_encoding=str(d.get("minishard_index_encoding", "raw")),
            data_encoding=str(d.get("data_encoding", "raw")),
        )

    @property
    def index_length(self) -> int:
        """Bytes of shard index at the head of every shard file."""
        return (1 << self.minishard_bits) * 16

    def locate(self, key: int) -> tuple[int, int]:
        """``(shard number, minishard number)`` holding `key`."""
        chunkid = int(key) >> self.preshift_bits
        if self.hash == "identity":
            hashed = chunkid & 0xFFFFFFFFFFFFFFFF
        else:
            hashed = mmh3.hash64_low(chunkid)
        minishard = hashed & ((1 << self.minishard_bits) - 1)
        shard = (hashed >> self.minishard_bits) & ((1 << self.shard_bits) - 1)
        return shard, minishard

    def shard_filename(self, shard: int) -> str:
        return format(int(shard), "x").zfill(math.ceil(self.shard_bits / 4)) + ".shard"


class ShardReader:
    """Cached random access into the shards under one prefix."""

    def __init__(self, store: Store, spec: ShardingSpec):
        self.store = store
        self.spec = spec
        self._shard_index: dict[str, np.ndarray | None] = {}
        self._minishard: dict[tuple[str, int], tuple | None] = {}
        self._lock = threading.Lock()

    def _filename(self, key: int, shard: int) -> str:
        """Which file holds `key`. Graphene names shards per chunk; see its reader."""
        return self.spec.shard_filename(shard)

    # ------------------------------------------------------------------ indices

    def _get_shard_index(self, filename: str):
        """``(2^minishard_bits, 2)`` of (start, end), or None if no such shard."""
        with self._lock:
            if filename in self._shard_index:
                return self._shard_index[filename]

        raw = self.store.get_range(filename, 0, self.spec.index_length)
        index = None
        if raw is not None and len(raw) >= self.spec.index_length:
            index = np.frombuffer(raw, dtype="<u8").reshape(-1, 2)

        with self._lock:
            self._shard_index[filename] = index
        return index

    def _get_minishard(self, filename: str, minishard: int):
        """``(keys, offsets, sizes)`` for one minishard, sorted by key."""
        cache_key = (filename, minishard)
        with self._lock:
            if cache_key in self._minishard:
                return self._minishard[cache_key]

        table = None
        index = self._get_shard_index(filename)
        if index is not None:
            start, end = (int(v) for v in index[minishard])
            if end > start:
                raw = self.store.get_range(
                    filename,
                    self.spec.index_length + start,
                    self.spec.index_length + end,
                )
                if raw is not None:
                    table = self._decode_minishard(
                        decompress(raw, self.spec.minishard_index_encoding)
                    )

        with self._lock:
            self._minishard[cache_key] = table
        return table

    def _decode_minishard(self, raw: bytes):
        """Delta-decode the ``3 x N`` table into sorted ``(keys, offsets, sizes)``.

        Kept as three numpy arrays rather than a dict: these are cached for the life
        of the process, and a whole-neuron read can touch hundreds of minishards of
        thousands of entries each - which is hundreds of megabytes of boxed Python
        ints for something ``searchsorted`` answers just as fast.
        """
        flat = np.frombuffer(raw, dtype="<u8")
        table = np.array(flat.reshape(3, len(flat) // 3), dtype=np.uint64)

        # Ids wrap on purpose: the deltas are defined in uint64 arithmetic.
        with np.errstate(over="ignore"):
            keys = np.cumsum(table[0], dtype=np.uint64)
        sizes = table[2]
        # An offset is a *gap* from the end of the previous value, so the running
        # total of preceding sizes has to be added back in.
        offsets = np.cumsum(table[1].astype(np.int64))
        offsets[1:] += np.cumsum(sizes[:-1].astype(np.int64))
        offsets += self.spec.index_length

        order = np.argsort(keys)
        return keys[order], offsets[order], sizes[order].astype(np.int64)

    # ------------------------------------------------------------------ reading

    def locate(self, key: int):
        """``(filename, absolute offset, size)`` of `key`, or None if absent."""
        key = int(key)
        shard, minishard = self.spec.locate(key)
        filename = self._filename(key, shard)
        table = self._get_minishard(filename, minishard)
        if table is None:
            return None

        keys, offsets, sizes = table
        i = int(np.searchsorted(keys, np.uint64(key)))
        if i >= len(keys) or int(keys[i]) != key:
            return None
        return filename, int(offsets[i]), int(sizes[i])

    def get(self, key: int) -> bytes | None:
        """The stored value for `key`, decoded, or None if it is not there."""
        hit = self.locate(key)
        if hit is None:
            return None
        filename, offset, size = hit
        raw = self.store.get_range(filename, offset, offset + size)
        if raw is None:
            return None
        return decompress(raw, self.spec.data_encoding)
