"""Decoding the chunk encodings connecto reads.

Only segmentation encodings are here. connecto never reads an EM image volume -
``_image_source`` exists solely to build a neuroglancer URL - so there is no jpeg,
png or jxl path, and none of the image codecs they would drag in.

``compressed_segmentation`` has two implementations: the C extension of the same
name if it is installed, and a pure-numpy fallback otherwise. The fallback exists
so that a plain ``pip install connecto`` can read every volume connecto knows
about with no compiled dependency at all; the C one is perhaps 10x quicker on a
dense read, so it is used when present.

Format (neuroglancer's ``compressed_segmentation``)::

    [ uint32 x num_channels ]   word offset of each channel's block header table
    per channel, per block (x fastest over the block grid):
      word0: bits 0-23 lookup table offset, bits 24-31 bits-per-encoded-value
      word1: bits 0-23 offset of the bit-packed value indices
    All offsets are in 32-bit words, relative to that channel's base offset.

Because ``bits`` is always one of 0, 1, 2, 4, 8, 16 or 32, an encoded value never
straddles a word boundary, which is what makes the numpy fallback simple.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decode_chunk", "decompress_segmentation"]


def decode_chunk(
    data: bytes | None,
    encoding: str,
    shape,
    dtype,
    *,
    block_size=None,
    num_channels: int = 1,
) -> np.ndarray:
    """One stored chunk as a ``(x, y, z, channels)`` array.

    `data` of ``None`` means the chunk is not stored, which for a segmentation is
    not an error - it is empty space. That is ``fill_missing``, and it is the
    default because a chunkedgraph's own chunk list routinely names chunks that
    were never written.
    """
    shape = tuple(int(v) for v in shape)
    dtype = np.dtype(dtype)
    full = (*shape, num_channels)

    if data is None or len(data) == 0:
        return np.zeros(full, dtype=dtype, order="F")

    if encoding == "raw":
        arr = np.frombuffer(data, dtype=dtype)
        return arr.reshape(full, order="F")

    if encoding == "compressed_segmentation":
        if block_size is None:
            raise ValueError("compressed_segmentation needs a block size.")
        return decompress_segmentation(
            data, shape, dtype, block_size, num_channels=num_channels
        )

    raise ValueError(
        f"Unsupported chunk encoding {encoding!r}. connecto reads segmentation "
        f"volumes only ('raw' and 'compressed_segmentation')."
    )


def decompress_segmentation(data, shape, dtype, block_size, *, num_channels: int = 1):
    shape = tuple(int(v) for v in shape)
    dtype = np.dtype(dtype)
    try:
        import compressed_segmentation as cseg
    except ImportError:
        return _decompress_segmentation_numpy(
            data, shape, dtype, block_size, num_channels=num_channels
        )

    # Deliberately outside the `try`: an ImportError raised from *inside* the
    # decoder means a broken install, and silently rerouting to the slow path would
    # hide it rather than report it.
    out = np.asarray(
        cseg.decompress(
            bytes(data),
            shape,
            dtype=dtype,
            block_size=tuple(int(v) for v in block_size),
            order="F",
        )
    )
    return out if out.ndim == 4 else out.reshape((*shape, num_channels), order="F")


def _block_values(words, base: int, header, block_size, dtype) -> np.ndarray:
    """One block's voxels, as a ``block_size``-shaped array in Fortran order.

    A block stores a lookup table of the distinct labels it contains plus an index
    per voxel, bit-packed at whatever width those indices need. Because that width
    is always 0, 1, 2, 4, 8, 16 or 32, an index never straddles a 32-bit word - which
    is what lets the unpacking be a shift-and-mask over the whole block at once.
    """
    word0, word1 = int(header[0]), int(header[1])
    bits = word0 >> 24
    table = words[base + (word0 & 0xFFFFFF) :].view(dtype)

    if bits == 0:
        # A uniform block stores no indices at all, only the one value.
        return np.full(tuple(block_size), table[0], dtype=dtype, order="F")

    n_values = int(np.prod(block_size))
    per_word = 32 // bits
    values_at = base + (word1 & 0xFFFFFF)
    packed = words[values_at : values_at + -(-n_values // per_word)]

    shifts = (np.arange(per_word, dtype=np.uint32) * bits).astype(np.uint32)
    index = (packed[:, None] >> shifts[None, :]) & np.uint32((1 << bits) - 1)
    return table[index.reshape(-1)[:n_values]].reshape(tuple(block_size), order="F")


def _decompress_segmentation_numpy(data, shape, dtype, block_size, *, num_channels=1):
    """The fallback. Correct, and about an order of magnitude slower."""
    shape = np.asarray(shape, dtype=np.int64)
    block_size = np.asarray(block_size, dtype=np.int64)
    dtype = np.dtype(dtype)
    words = np.frombuffer(data, dtype="<u4")

    out = np.zeros((*shape, num_channels), dtype=dtype, order="F")
    grid = -(-shape // block_size)  # ceil
    n_blocks = int(np.prod(grid))

    for channel in range(num_channels):
        base = int(words[channel])
        headers = words[base : base + 2 * n_blocks].reshape(-1, 2)

        for i in range(n_blocks):
            # Blocks run x fastest over the grid.
            bz, rem = divmod(i, int(grid[0] * grid[1]))
            by, bx = divmod(rem, int(grid[0]))

            start = np.array([bx, by, bz], dtype=np.int64) * block_size
            stop = np.minimum(start + block_size, shape)
            extent = stop - start
            if np.any(extent <= 0):
                continue

            # A block is always encoded at its full nominal size, even where the
            # volume cuts it short, so the clipping happens on the way out.
            block = _block_values(words, base, headers[i], block_size, dtype)
            out[
                start[0] : stop[0], start[1] : stop[1], start[2] : stop[2], channel
            ] = block[: extent[0], : extent[1], : extent[2]]

    return out
