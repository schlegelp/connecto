"""The conformance suite.

One file that runs the *same* assertions against *every* registered dataset. This
is the highest-leverage test in the project: it is what keeps eleven datasets and two
backends honest, and `test_capabilities_are_honest` is the executable form of the
"nothing degrades silently" promise.

Needs network and credentials::

    pytest tests/test_conformance.py -m network
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import connecto as co
from connecto.core.registry import REGISTRY
from connecto.core.spec import Cap

pytestmark = pytest.mark.network

# Every dataset that has declared example neurons to test against.
TESTABLE = [s for s in REGISTRY.values() if s.example_ids]


def _build(spec, backend=None):
    from connecto.backends import build

    return build(spec, backend=backend)


@pytest.fixture(scope="module", params=TESTABLE, ids=lambda s: s.name)
def ds(request):
    return _build(request.param)


# ------------------------------------------------------------------- the schemas

def test_edge_contract(ds):
    """Every dataset returns the same edge frame. This is the whole pitch."""
    e = ds.connectivity.edges(ds.spec.example_ids)

    assert list(e.columns) == ["pre", "post", "weight"]
    assert e.dtypes["pre"] == np.int64
    assert e.dtypes["post"] == np.int64
    assert e.dtypes["weight"] == np.int32
    assert len(e), f"{ds.label} returned no edges for its own example_ids"

    prov = e.attrs["connecto"]
    assert prov["dataset"] == ds.name
    assert prov["backend"] == ds.backend_kind
    assert prov["version"] == str(ds.version)


def test_annotation_contract(ds):
    if not ds.supports(Cap.ANNOTATIONS):
        pytest.skip("no annotations")

    ann = ds.annotations.get()
    assert "id" in ann.columns
    assert ann["id"].dtype == np.int64

    # Every ID is a real segment. Root ID 0 means "this annotation point doesn't
    # land on anything" - CAVE tables are full of them (MICrONS especially) and
    # they must never reach the user, or ds.ids() hands back a 0 to query with.
    assert (ann["id"] > 0).all(), f"{ds.label} leaked unassigned (id <= 0) rows"

    # Note we deliberately do NOT assert ann["id"].is_unique. A MICrONS root can
    # legitimately contain two nuclei (a merge error) and so carry two annotation
    # rows. Promising uniqueness would mean inventing a fact about the data.

    # Side, where present, is always the same three words.
    if "side" in ann.columns:
        vocab = set(ann["side"].dropna().unique())
        assert vocab <= {"left", "right", "center"}, f"{ds.label} side vocab: {vocab}"


def test_ids_roundtrip(ds):
    ids = ds.ids(list(ds.spec.example_ids))
    assert set(ids) == set(ds.spec.example_ids)
    assert ids.dtype == np.int64


def test_skeletons_are_navis_neurons_in_nm(ds):
    if not ds.supports(Cap.SKELETONS):
        pytest.skip("no skeletons")

    import navis

    sk = ds.skeletons.get(ds.spec.example_ids[:1])
    assert isinstance(sk, navis.NeuronList) and len(sk) == 1

    n = sk[0]
    assert isinstance(n, navis.TreeNeuron)
    assert "nanometer" in str(n.units)
    assert n.n_nodes > 1, f"{ds.label} returned a degenerate skeleton"


# --------------------------------------------------------------- the capabilities

def test_capabilities_are_honest(ds):
    """Declared capabilities work; undeclared ones raise rather than shrug.

    The executable form of the promise. fafbseg quietly sets `min_score = None`
    for datasets that have no scores, so a user believes they filtered when they
    did not. Here that is a test failure.
    """
    x = ds.spec.example_ids

    # An argument the dataset cannot honour must raise - never be ignored.
    if not ds.supports(Cap.SYNAPSE_SCORES):
        with pytest.raises(co.CapabilityError, match="min_score"):
            ds.connectivity.synapses(x, min_score=100)

    if not ds.supports(Cap.NT_PER_SYNAPSE):
        with pytest.raises(co.CapabilityError, match="transmitters"):
            ds.connectivity.synapses(x, transmitters=True)

    # A whole namespace the dataset lacks must be *absent*, not broken.
    for ns, cap in (
        ("segmentation", Cap.SEGMENTATION),
        ("proofreading", Cap.PROOFREADING),
        ("l2", Cap.L2CACHE),
    ):
        if ds.supports(cap):
            assert hasattr(ds, ns)
        else:
            assert not hasattr(ds, ns), f"{ds.label} exposes .{ns} without {cap}"
            with pytest.raises(co.CapabilityError):
                getattr(ds, ns)


def test_the_l2_cache_actually_works_where_it_is_claimed(ds):
    """`Cap.L2CACHE` used to gate nothing user-facing: it was an internal hint to the
    skeleton fallback, so five datasets could claim it with no test able to hold them
    to it. Now it carries the `.l2` namespace, so the claim is checkable - which is the
    whole reason the namespace exists rather than a `method=` flag."""
    if not ds.supports(Cap.L2CACHE):
        pytest.skip("no L2 cache")

    import navis

    x = ds.spec.example_ids[:1]

    info = ds.l2.info(x)
    assert len(info), f"{ds.label} claims an L2 cache but it returned no chunks"
    assert {"id", "l2_id", "x", "y", "z"}.issubset(info.columns)

    sk = ds.l2.skeleton(x)
    assert isinstance(sk[0], navis.TreeNeuron)
    assert sk[0].n_nodes > 1, f"{ds.label}: degenerate L2 skeleton"
    assert "nanometer" in str(sk[0].units)

    dp = ds.l2.dotprops(x)
    assert isinstance(dp[0], navis.Dotprops)
    assert len(dp[0].points) > 1
    # A zero-length vector is not a direction. NBLAST would treat it as one.
    assert (np.linalg.norm(dp[0].vect, axis=1) > 0).all()


def test_dotprops_can_be_asked_for_in_microns(ds):
    """NBLAST is calibrated in microns; nanometre dotprops score at the floor and look
    like "nothing matches" rather than like an error. Both routes must offer the unit."""
    if not ds.supports(Cap.SKELETONS):
        pytest.skip("no skeletons")

    x = ds.spec.example_ids[:1]

    nm = ds.skeletons.dotprops(x, units="nm")
    um = ds.skeletons.dotprops(x, units="um")
    # pint spells it "micrometer".
    assert "nanometer" in str(nm[0].units)
    assert "micrometer" in str(um[0].units)
    # Same neuron, 1000x apart.
    assert np.allclose(
        nm[0].points.max(axis=0) / 1000, um[0].points.max(axis=0), rtol=1e-3
    )

    with pytest.raises(ValueError, match="units"):
        ds.skeletons.dotprops(x, units="parsecs")


def test_a_scene_is_in_the_dialect_its_own_viewer_speaks(ds):
    """Neuroglancer forked and the halves cannot read each other's scenes - but a scene
    in the wrong schema does not error, it opens on an empty viewer. So the dataset
    declares which viewer it belongs to, and the scene has to match it. This test is the
    only thing standing between "we emit JSON" and "we emit JSON that viewer can read".
    """
    if not ds.supports(Cap.NEUROGLANCER):
        pytest.skip("no neuroglancer")

    from connecto.viz import decode_url

    x = list(ds.spec.example_ids)
    scene = ds.viz.scene(x, seg_colors="red")

    seg = next(lyr for lyr in scene["layers"] if lyr["type"].startswith("segmentation"))
    assert seg["source"], f"{ds.label} put no source on its segmentation layer"
    assert seg["segments"] == [str(i) for i in x]
    assert set(seg["segmentColors"].values()) == {"#ff0000"}

    if ds.spec.viewer_dialect == "seunglab":
        # The old fork keeps the voxel size inside the camera pose and has never heard
        # of `dimensions`.
        assert scene["navigation"]["pose"]["position"]["voxelSize"]
        assert "dimensions" not in scene
    else:
        assert scene["dimensions"], f"{ds.label} emitted a modern scene with no dimensions"
        assert "navigation" not in scene
        # Without `middleauth+` a modern viewer never runs CAVE's login flow, and a
        # protected datastack silently fails to load.
        if seg["source"].startswith("graphene://"):
            assert "middleauth+" in seg["source"], f"{ds.label}: graphene without middleauth"

    if ds.backend_kind == "cave":
        assert scene["layers"][0]["type"] == "image", f"{ds.label} has no EM layer"

    assert decode_url(ds.viz.neuroglancer_url(x, seg_colors="red")) == scene


def test_scene_coordinates_are_nanometres(ds):
    """Every coordinate connecto hands you is nanometres; neuroglancer wants voxels. If
    the conversion is skipped the camera lands 4-45x away from the thing you asked to
    look at - and the scene still opens, just somewhere else entirely."""
    if not ds.supports(Cap.NEUROGLANCER):
        pytest.skip("no neuroglancer")

    voxel = np.asarray(ds.spec.voxel_size, dtype=float)
    nm = voxel * 1000  # a point exactly 1000 voxels out along each axis

    scene = ds.viz.scene(ds.spec.example_ids[:1], position=nm)
    if ds.spec.viewer_dialect == "seunglab":
        pos = scene["navigation"]["pose"]["position"]["voxelCoordinates"]
    else:
        pos = scene["position"]

    assert np.allclose(pos, [1000, 1000, 1000]), f"{ds.label} did not convert nm -> voxels"


def test_live_queries_rejected_on_frozen_releases(ds):
    if ds.supports(Cap.LIVE):
        pytest.skip("dataset is live")
    with pytest.raises((co.CapabilityError, co.NoSuchVersionError)):
        ds.connectivity.edges(ds.spec.example_ids, version="live")


# ------------------------------------------------------- the cross-backend claim

def test_banc_backends_agree():
    """The normalisation is real, not aspirational.

    BANC is served by CAVE (materialization 888) and by neuPrint (banc:v888) -
    the *same snapshot*, and its neuPrint body IDs are valid CAVE root IDs. So the
    two backends must return the same edges. If they ever stop doing so, one of
    them is lying and this test says which.

    This is what caught the autapse divergence: CAVE's synapse table contains
    self-connections and neuPrint's simply does not, so the same neuron had
    different connectivity depending on which backend you asked.
    """
    spec = co.get_spec("banc")
    ids = spec.example_ids

    cave = _build(spec, backend="cave").connectivity.edges(ids)
    neuprint = _build(spec, backend="neuprint").connectivity.edges(ids)

    cave = cave.sort_values(["pre", "post"]).reset_index(drop=True)
    neuprint = neuprint.sort_values(["pre", "post"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(cave, neuprint, check_like=True)


def test_autapses_are_opt_in_not_a_backend_accident():
    """Autapses appear only when asked for - and then from the backend that has them."""
    spec = co.get_spec("banc")
    ids = spec.example_ids
    cave = _build(spec, backend="cave")

    assert not (cave.connectivity.edges(ids).eval("pre == post")).any()

    with_autapses = cave.connectivity.edges(ids, autapses=True)
    assert (with_autapses["pre"] == with_autapses["post"]).any()
