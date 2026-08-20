"""MurmurHash3 x86_128, the hash the neuroglancer sharded format specifies.

Only the low 64 bits of the 128-bit digest are ever used (that is what the spec
asks for), and the input is always an 8-byte little-endian uint64, so this is a
few dozen lines of integer arithmetic rather than a compiled dependency.

Reference: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
"""

from __future__ import annotations

import struct

__all__ = ["hash128", "hash64_low"]

_M32 = 0xFFFFFFFF

_C1 = 0x239B961B
_C2 = 0xAB0E9789
_C3 = 0x38B34AE5
_C4 = 0xA1E38B93


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M32


def _fmix32(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _M32
    h ^= h >> 16
    return h


def hash128(data: bytes, seed: int = 0) -> int:
    """The full 128-bit digest, as one unsigned Python int."""
    length = len(data)
    nblocks = length // 16
    h1 = h2 = h3 = h4 = seed & _M32

    for i in range(nblocks):
        k1, k2, k3, k4 = struct.unpack_from("<IIII", data, i * 16)

        k1 = (_rotl32((k1 * _C1) & _M32, 15) * _C2) & _M32
        h1 ^= k1
        h1 = _rotl32(h1, 19)
        h1 = (h1 + h2) & _M32
        h1 = (h1 * 5 + 0x561CCD1B) & _M32

        k2 = (_rotl32((k2 * _C2) & _M32, 16) * _C3) & _M32
        h2 ^= k2
        h2 = _rotl32(h2, 17)
        h2 = (h2 + h3) & _M32
        h2 = (h2 * 5 + 0x0BCAA747) & _M32

        k3 = (_rotl32((k3 * _C3) & _M32, 17) * _C4) & _M32
        h3 ^= k3
        h3 = _rotl32(h3, 15)
        h3 = (h3 + h4) & _M32
        h3 = (h3 * 5 + 0x96CD1C35) & _M32

        k4 = (_rotl32((k4 * _C4) & _M32, 18) * _C1) & _M32
        h4 ^= k4
        h4 = _rotl32(h4, 13)
        h4 = (h4 + h1) & _M32
        h4 = (h4 * 5 + 0x32AC3B17) & _M32

    tail = data[nblocks * 16 :]
    k1 = k2 = k3 = k4 = 0
    n = len(tail)
    if n >= 15:
        k4 ^= tail[14] << 16
    if n >= 14:
        k4 ^= tail[13] << 8
    if n >= 13:
        k4 ^= tail[12]
        k4 = (_rotl32((k4 * _C4) & _M32, 18) * _C1) & _M32
        h4 ^= k4
    if n >= 12:
        k3 ^= tail[11] << 24
    if n >= 11:
        k3 ^= tail[10] << 16
    if n >= 10:
        k3 ^= tail[9] << 8
    if n >= 9:
        k3 ^= tail[8]
        k3 = (_rotl32((k3 * _C3) & _M32, 17) * _C4) & _M32
        h3 ^= k3
    if n >= 8:
        k2 ^= tail[7] << 24
    if n >= 7:
        k2 ^= tail[6] << 16
    if n >= 6:
        k2 ^= tail[5] << 8
    if n >= 5:
        k2 ^= tail[4]
        k2 = (_rotl32((k2 * _C2) & _M32, 16) * _C3) & _M32
        h2 ^= k2
    if n >= 4:
        k1 ^= tail[3] << 24
    if n >= 3:
        k1 ^= tail[2] << 16
    if n >= 2:
        k1 ^= tail[1] << 8
    if n >= 1:
        k1 ^= tail[0]
        k1 = (_rotl32((k1 * _C1) & _M32, 15) * _C2) & _M32
        h1 ^= k1

    h1 ^= length
    h2 ^= length
    h3 ^= length
    h4 ^= length

    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    h3 = (h3 + h1) & _M32
    h4 = (h4 + h1) & _M32

    h1 = _fmix32(h1)
    h2 = _fmix32(h2)
    h3 = _fmix32(h3)
    h4 = _fmix32(h4)

    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    h3 = (h3 + h1) & _M32
    h4 = (h4 + h1) & _M32

    return h1 | (h2 << 32) | (h3 << 64) | (h4 << 96)


def hash64_low(value: int, seed: int = 0) -> int:
    """Low 64 bits of the digest of `value` as a little-endian uint64."""
    return hash128(struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF), seed) & 0xFFFFFFFFFFFFFFFF
