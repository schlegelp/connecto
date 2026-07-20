"""Run-length encoded voxels, and the one decoder every source shares.

A run is ``(x, y, z, length)``: ``length`` voxels starting at ``(x, y, z)`` and
extending along +X. That is DVID's ``sparsevol`` wire format, and it is also - not
by coincidence - what the aedes lookup service emits and what the CAVE path encodes
into. So all three of connecto's voxel sources meet here, and only this module ever
has to know how a run is laid out.

Runs are the cheap form and connecto keeps them as long as it can. A hemibrain
neuron at scale 0 is 1.17 billion voxels - 14 GB as int32 triples - but only 16
million runs, a factor of 73. ``voxels.get(..., output="rle")`` hands them over
undecoded for exactly that reason.
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = [
    "decode_runs",
    "encode_runs",
    "run_voxel_count",
    "decode_sparsevol",
    "decode_service_response",
]

# DVID's RLE payload: a 12-byte header, then 16 bytes per run.
#
#   byte 0    payload descriptor
#   byte 1    number of dimensions (3)
#   byte 2    dimension along which runs extend (0 = x)
#   byte 3    reserved
#   bytes 4-7 number of blocks (unused for this format)
#   bytes 8-11 number of runs
#
# Then `n_runs` * (int32 x, int32 y, int32 z, int32 run_length).
_HEADER = struct.Struct("<bbbbii")
_HEADER_FIELDS = ("descriptor", "n_dims", "run_dim", "reserved", "n_blocks", "n_runs")
RUN_DTYPE = np.dtype("<i4")


def decode_sparsevol(payload: bytes) -> np.ndarray:
    """Decode a DVID ``sparsevol`` response to an ``(M, 4)`` array of runs.

    Vectorised on purpose. The obvious implementation - a Python loop calling
    ``struct.unpack`` once per run - is what dvid-tools does, and at 16 million runs
    for one hemibrain neuron it takes minutes and builds 16 million temporary
    arrays. ``frombuffer`` does the same work as a single memcpy-shaped read.
    """
    if len(payload) < _HEADER.size:
        raise ValueError(
            f"Truncated sparsevol response: got {len(payload)} bytes, "
            f"need at least {_HEADER.size} for the header."
        )

    header = dict(zip(_HEADER_FIELDS, _HEADER.unpack(payload[: _HEADER.size])))

    # Assert the layout rather than assume it. DVID has a second ("blocks") encoding
    # and a run dimension that is nominally configurable; if either ever shows up we
    # would otherwise reinterpret the bytes as x-runs and return plausible nonsense.
    if header["n_dims"] != 3 or header["run_dim"] != 0:
        raise ValueError(
            f"Unexpected sparsevol encoding: n_dims={header['n_dims']}, "
            f"run_dim={header['run_dim']}. connecto can only decode 3D runs "
            f"along X."
        )

    n_runs = header["n_runs"]
    body = payload[_HEADER.size : _HEADER.size + n_runs * 16]
    if len(body) != n_runs * 16:
        raise ValueError(
            f"Truncated sparsevol response: header declares {n_runs:,} runs "
            f"({n_runs * 16:,} bytes) but only {len(body):,} bytes followed."
        )

    return np.frombuffer(body, dtype=RUN_DTYPE).reshape(-1, 4)


def decode_service_response(payload: bytes) -> np.ndarray:
    """Decode a sparsevol *service* response to an ``(M, 4)`` array of runs.

    Same runs as :func:`decode_sparsevol`, minus DVID's header - the service writes
    the array out bare.
    """
    if len(payload) % 16:
        raise ValueError(
            f"Malformed sparsevol response: {len(payload):,} bytes is not a whole "
            f"number of 16-byte runs."
        )
    return np.frombuffer(payload, dtype=RUN_DTYPE).reshape(-1, 4)


def run_voxel_count(runs: np.ndarray) -> int:
    """How many voxels a run array expands to, without expanding it."""
    if len(runs) == 0:
        return 0
    # int64 before summing: 16M runs of int32 lengths overflow a 32-bit accumulator
    # around 2.1 billion voxels, and a hemibrain neuron passes that at scale 0.
    return int(runs[:, 3].astype(np.int64).sum())


def decode_runs(runs: np.ndarray, dtype=np.int32, *, sort: bool = True) -> np.ndarray:
    """Expand ``(M, 4)`` runs to ``(N, 3)`` voxel coordinates.

    Output is sorted by ``(z, y, x)`` so that ``encode_runs(decode_runs(r))``
    reproduces ``r`` exactly rather than merely as a set. DVID does *not* emit its
    runs in that order - measured on a hemibrain neuron, not one of its 225,030 runs
    was out of place in the voxel sense, but the run list itself was unsorted - so
    the ordering has to be imposed here rather than assumed.

    The sort is applied to the *runs*, not to the expanded voxels, which is the
    whole reason it is affordable: at scale 0 a hemibrain neuron is 16 million runs
    and 1.17 billion voxels, so sorting runs is ~73x less work than sorting the
    result would be.
    """
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError(f"Expected an (M, 4) array of runs, got {runs.shape}.")
    if len(runs) == 0:
        return np.zeros((0, 3), dtype=dtype)

    if sort:
        runs = runs[np.lexsort((runs[:, 0], runs[:, 1], runs[:, 2]))]

    lengths = runs[:, 3].astype(np.int64)
    # A zero- or negative-length run is not a run. Left unchecked it simply
    # contributes nothing, so a malformed payload decodes to a short array that
    # agrees with its own voxel count and looks entirely healthy - which is the one
    # failure mode this module exists to make impossible.
    if (lengths < 1).any():
        bad = int((lengths < 1).sum())
        raise ValueError(
            f"{bad} of {len(runs):,} runs have a length < 1. Run lengths must be "
            f"at least 1; this payload is malformed."
        )
    total = int(lengths.sum())

    out = np.empty((total, 3), dtype=dtype)
    out[:, 1] = np.repeat(runs[:, 1], lengths)
    out[:, 2] = np.repeat(runs[:, 2], lengths)
    # Walk +X within each run: repeat the start, then add 0, 1, 2, ... per run.
    run_start = np.repeat(np.cumsum(lengths) - lengths, lengths)
    out[:, 0] = np.repeat(runs[:, 0], lengths) + (np.arange(total) - run_start)
    return out


def encode_runs(
    coords: np.ndarray, *, assume_sorted: bool = False, assume_unique: bool = False
) -> np.ndarray:
    """Compress ``(N, 3)`` voxel coordinates to ``(M, 4)`` runs.

    Duplicates are removed unless ``assume_unique``. They have to be: a repeated x
    gives a step of 0, which is neither a continuation of the run nor a clean break
    from it, so a duplicated voxel silently splits one run into two and inflates the
    decoded voxel count. Nothing downstream would notice - the count agrees with the
    array length either way.
    """
    coords = np.asarray(coords)
    if coords.size == 0:
        return np.zeros((0, 4), dtype=RUN_DTYPE)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) array of voxels, got {coords.shape}.")

    if not assume_unique:
        # np.unique sorts as a side effect, which is the order we want anyway.
        coords = np.unique(coords, axis=0)
        coords = coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]
    elif not assume_sorted:
        # Sort by (z, y, x) so runs along X are contiguous within a scanline.
        coords = coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]

    step = np.diff(coords, axis=0)
    # A run continues only along the same scanline, with x advancing by exactly one.
    breaks = (step[:, 2] != 0) | (step[:, 1] != 0) | (step[:, 0] != 1)
    cut = np.nonzero(breaks)[0] + 1

    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(coords)]])

    runs = np.empty((len(starts), 4), dtype=RUN_DTYPE)
    runs[:, :3] = coords[starts]
    runs[:, 3] = ends - starts
    return runs
