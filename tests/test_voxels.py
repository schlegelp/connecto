"""Sparse volumes - the parts that need no network.

The RLE codec is the load-bearing piece here: all three routes (DVID, the lookup
service, the CAVE dense read) meet in it, so a bug in the codec is a bug in every
dataset at once. It is tested against the real wire layout rather than a paraphrase
of it.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import connecto as co
from connecto.core.registry import REGISTRY
from connecto.core.spec import Cap, SparseVolSource
from connecto.voxels import _DEFAULT_SCALE, default_scale, rle, route, service
from connecto.voxels import _check_opts as _fetch_opts_check

# ------------------------------------------------------------------ the codec

def _roundtrip(coords):
    """encode -> decode must reproduce the input exactly, not merely as a set."""
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]
    return np.array_equal(rle.decode_runs(rle.encode_runs(coords)), coords)


@pytest.mark.parametrize("seed", range(5))
def test_rle_round_trips_exactly(seed):
    rng = np.random.default_rng(seed)
    coords = np.unique(rng.integers(0, 15, size=(3000, 3)).astype(np.int32), axis=0)
    assert _roundtrip(coords)


def test_rle_handles_the_degenerate_shapes():
    assert rle.encode_runs(np.zeros((0, 3), dtype=np.int32)).shape == (0, 4)
    assert rle.decode_runs(np.zeros((0, 4), dtype=np.int32)).shape == (0, 3)
    assert rle.run_voxel_count(np.zeros((0, 4), dtype=np.int32)) == 0

    single = np.array([[5, 6, 7]], dtype=np.int32)
    runs = rle.encode_runs(single)
    assert runs.tolist() == [[5, 6, 7, 1]]
    assert _roundtrip(single)


def test_a_contiguous_x_run_collapses_to_one_run():
    coords = np.array([[10, 2, 3], [11, 2, 3], [12, 2, 3]], dtype=np.int32)
    assert rle.encode_runs(coords).tolist() == [[10, 2, 3, 3]]

    # ... but a break in x, y or z starts a new run.
    for gap in ([14, 2, 3], [10, 3, 3], [10, 2, 4]):
        broken = np.vstack([coords, np.array([gap], dtype=np.int32)])
        assert len(rle.encode_runs(broken)) == 2


def test_duplicate_voxels_do_not_silently_split_a_run():
    """A repeated x steps by 0 - neither a continuation nor a clean break.

    Left alone it splits one run in two and inflates the decoded voxel count, and
    nothing downstream notices because the count agrees with the array length.
    """
    coords = np.array([[10, 2, 3], [10, 2, 3], [11, 2, 3]], dtype=np.int32)
    runs = rle.encode_runs(coords)
    assert runs.tolist() == [[10, 2, 3, 2]]
    assert rle.run_voxel_count(runs) == 2


def test_malformed_run_lengths_raise_instead_of_vanishing():
    """A zero-length run decodes to nothing and looks perfectly healthy."""
    with pytest.raises(ValueError, match="length < 1"):
        rle.decode_runs(np.array([[1, 1, 1, 0], [5, 5, 5, 2]], dtype="<i4"))
    with pytest.raises(ValueError, match=r"\(M, 4\)"):
        rle.decode_runs(np.zeros((3, 5), dtype="<i4"))


def test_decode_runs_sorts_even_when_the_source_does_not():
    """DVID does not emit runs in (z, y, x) order - measured, not assumed."""
    runs = np.array([[0, 5, 9, 2], [0, 0, 0, 2], [0, 3, 4, 1]], dtype="<i4")
    out = rle.decode_runs(runs)
    assert np.array_equal(out, out[np.lexsort((out[:, 0], out[:, 1], out[:, 2]))])
    assert len(out) == rle.run_voxel_count(runs) == 5


def test_run_voxel_count_does_not_overflow_int32():
    """A hemibrain neuron at scale 0 is 1.17 billion voxels - past a 32-bit sum."""
    runs = np.zeros((3, 4), dtype="<i4")
    runs[:, 3] = 2_000_000_000 // 2
    assert rle.run_voxel_count(runs) == 3_000_000_000


# ------------------------------------------------------------- the DVID header

def _dvid_payload(runs, *, n_dims=3, run_dim=0):
    header = struct.pack("<bbbbii", 0, n_dims, run_dim, 0, 0, len(runs))
    return header + np.asarray(runs, dtype="<i4").tobytes()


def test_decode_sparsevol_reads_the_real_layout():
    runs = [[1, 2, 3, 4], [10, 20, 30, 1]]
    out = rle.decode_sparsevol(_dvid_payload(runs))
    assert out.tolist() == runs
    assert rle.run_voxel_count(out) == 5


def test_decode_sparsevol_rejects_truncation_rather_than_guessing():
    payload = _dvid_payload([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="[Tt]runcated"):
        rle.decode_sparsevol(payload[:8])          # header cut short
    with pytest.raises(ValueError, match="[Tt]runcated"):
        rle.decode_sparsevol(payload[:-4])         # body cut short


def test_decode_sparsevol_rejects_an_encoding_it_cannot_read():
    """Silently reinterpreting a different encoding would return plausible nonsense."""
    with pytest.raises(ValueError, match="[Uu]nexpected sparsevol encoding"):
        rle.decode_sparsevol(_dvid_payload([[1, 2, 3, 4]], run_dim=1))
    with pytest.raises(ValueError, match="[Uu]nexpected sparsevol encoding"):
        rle.decode_sparsevol(_dvid_payload([[1, 2, 3, 4]], n_dims=2))


def test_service_response_is_the_same_runs_without_the_header():
    runs = np.array([[1, 2, 3, 4], [9, 9, 9, 2]], dtype="<i4")
    assert rle.decode_service_response(runs.tobytes()).tolist() == runs.tolist()
    with pytest.raises(ValueError, match="whole number of 16-byte runs"):
        rle.decode_service_response(b"\x00" * 20)


# ------------------------------------------------------------------ resolution

def test_anisotropic_downsample_is_honoured():
    """aedes halves X and Y only: scale 1 is 32x32x45 nm, not 32x32x90."""
    src = SparseVolSource(url="{id}/{scale}", scales=(1,), downsample=(2, 2, 1))
    assert src.resolution(0, (16, 16, 45)) == (16.0, 16.0, 45.0)
    assert src.resolution(1, (16, 16, 45)) == (32.0, 32.0, 45.0)

    isotropic = SparseVolSource(url="{id}/{scale}", scales=(0,))
    assert isotropic.resolution(3, (8, 8, 8)) == (64.0, 64.0, 64.0)


def test_a_service_refuses_a_scale_it_does_not_serve_before_any_request():
    """The aedes service answers a bare HTTP 500 for scale 2; we say what exists."""
    src = SparseVolSource(url="http://nope.invalid/{id}/{scale}", scales=(1,))
    with pytest.raises(ValueError, match="serves scale 1 only"):
        service.fetch_sparsevol(src, 123, 2)


# --------------------------------------------------------------------- routing

class _Stub:
    """Just enough dataset to exercise routing, with no server behind it."""

    def __init__(self, backend_kind, sparsevol_source=None):
        self.backend_kind = backend_kind
        self.spec = type("S", (), {"sparsevol_source": sparsevol_source})()


def test_route_picks_the_source_the_dataset_actually_has():
    src = SparseVolSource(url="x", scales=(1,))
    assert route(_Stub("cave", src)) == "service"      # aedes
    assert route(_Stub("neuprint")) == "dvid"          # the Janelia datasets
    assert route(_Stub("cave")) == "pcg"               # FlyWire, BANC, FANC, MICrONS


def test_default_scale_is_never_zero():
    """Scale 0 on a whole neuron is billions of voxels; it must be opt-in."""
    assert all(s > 0 for s in _DEFAULT_SCALE.values())
    # A service gets the finest scale it actually serves - no volume to consult.
    assert default_scale(_Stub("cave", SparseVolSource(url="x", scales=(1, 4)))) == 1


def test_the_dense_route_defaults_coarser_than_the_indexed_one():
    """Regression: the default must stay inside the dense route's own ceiling.

    At scale 3 a mid-sized FlyWire neuron transfers ~1.2 billion voxels and a MICrONS
    cell exceeds `DEFAULT_MAX_VOXELS` outright - `voxels.get(x)` with no scale raised
    instead of returning anything. Caught by the conformance suite, not by review.
    """
    assert _DEFAULT_SCALE["pcg"] > _DEFAULT_SCALE["dvid"]


def test_an_option_for_another_route_is_refused_by_name():
    """The dense route's own ceiling message suggests `max_chunks=`.

    Carrying that call to a DVID dataset used to raise a bare TypeError naming a
    private function.
    """
    with pytest.raises(TypeError, match="max_chunks.*only on the 'pcg' route"):
        _fetch_opts_check("dvid", {"max_chunks": 5})
    with pytest.raises(TypeError, match="timeout"):
        _fetch_opts_check("pcg", {"timeout": 5})
    # ... and a legitimate option for the route passes straight through.
    _fetch_opts_check("pcg", {"max_chunks": 5, "parallel": 2})
    _fetch_opts_check("dvid", {"timeout": 30})


# ------------------------------------------------------- what the specs promise

def test_voxels_is_declared_only_where_there_is_a_route_to_it():
    """A dataset claiming VOXELS must have somewhere to get them from.

    The neuPrint *mirrors* of FlyWire and BANC are the trap this guards: they are
    imports, not DVID deployments, so the `dvid` route would resolve to nothing.
    """
    for spec in REGISTRY.values():
        for backend in spec.backends:
            if Cap.VOXELS not in spec.capabilities_for(backend.kind):
                continue
            stub = _Stub(backend.kind, spec.sparsevol_source)
            kind = route(stub)
            if kind == "pcg":
                assert Cap.CHUNKEDGRAPH in spec.capabilities_for(backend.kind), (
                    f"{spec.name}/{backend.kind} claims VOXELS via the dense read, "
                    f"which needs a chunkedgraph to mask against."
                )


def test_the_neuprint_mirrors_do_not_claim_voxels():
    """Regression: FlyWire/BANC via neuPrint must point at the CAVE door."""
    for name in ("flywire", "banc"):
        spec = REGISTRY[name]
        assert Cap.VOXELS not in spec.capabilities_for("neuprint"), (
            f"{name}'s neuPrint mirror is an import, not a DVID server."
        )
        assert Cap.VOXELS in spec.capabilities_for("cave")


def test_the_janelia_datasets_all_claim_voxels():
    for name in ("hemibrain", "malecns", "manc", "fish2"):
        assert Cap.VOXELS in REGISTRY[name].capabilities_for("neuprint")


def test_voxels_namespace_is_absent_without_the_capability():
    """A missing namespace must be absent, not broken."""
    ds = co.FlyWire(backend="neuprint")
    assert not hasattr(ds, "voxels")
    with pytest.raises(co.CapabilityError, match="voxels"):
        _ = ds.voxels
