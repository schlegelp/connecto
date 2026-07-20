"""The conformance suite.

One file that runs the *same* assertions against *every* registered dataset. This
is the highest-leverage test in the project: it is what keeps ten datasets and two
backends honest, and `test_capabilities_are_honest` is the executable form of the
"nothing degrades silently" promise.

It runs per **door**, not per dataset: `flywire` and `banc` are each served by both
backends, and the two are not equally capable - neuPrint adds ROIs and takes away the
chunkedgraph - so a suite that only exercised the default backend would leave half of
each dataset's promises unchecked. That is not hypothetical: the neuPrint-backed FlyWire
claimed per-synapse transmitters it does not have, and returned a frame without them
rather than raising. Nothing tested it, because nothing built it.

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

# Every (dataset, backend) pair that has declared example neurons to test against.
TESTABLE = [
    (s, b.kind) for s in REGISTRY.values() if s.example_ids for b in s.backends
]


def _build(spec, backend=None):
    from connecto.backends import build

    return build(spec, backend=backend)


@pytest.fixture(scope="module", params=TESTABLE, ids=lambda p: f"{p[0].name}-{p[1]}")
def ds(request):
    spec, backend = request.param
    return _build(spec, backend=backend)


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


def test_voxels_are_sparse_volumes_of_the_right_neuron(ds):
    """Every declared VOXELS route returns real voxels, in a real place.

    The containment check is the one that matters. All three routes could plausibly
    return a well-formed array of the wrong thing - the DVID route by addressing the
    wrong node, the dense route by mis-converting the chunk grid at a coarse scale -
    and neither would look wrong. Landing inside the neuron's own mesh is the
    independent evidence that it is the right neuron.
    """
    if not ds.supports(Cap.VOXELS):
        pytest.skip("no voxels")

    import navis

    from connecto import voxels as _voxels

    body = ds.spec.example_ids[0]
    # Whatever the route picks for itself. Hard-coding a coarse scale here is what
    # the first version did, and it sent the dense route to a scale that exceeded its
    # own transfer ceiling on MICrONS - so the default is the thing worth testing.
    scale = _voxels.default_scale(ds)

    # One fetch, three renderings of it: re-fetching per output form is four dense
    # reads on the CAVE route, which is minutes of wall clock for no extra coverage.
    try:
        runs, resolution, _ = _voxels.fetch(ds, int(body), scale)
    except co.ConnectoError as e:
        # The aedes service caps segments at 256 chunks and serves one scale only,
        # so its largest neurons genuinely cannot be fetched. That connecto says so
        # clearly *is* the correct behaviour - skip loudly rather than pretend.
        if "too large for it" in str(e):
            pytest.skip(f"{ds.label}: {e}")
        raise
    n_voxels = _voxels.rle.run_voxel_count(runs)
    assert runs.shape[1] == 4
    assert n_voxels > 0, f"{ds.label} returned an empty sparse volume"

    vn = ds.voxels.get(body, scale=scale, progress=False)
    assert isinstance(vn, navis.NeuronList) and len(vn) == 1
    n = vn[0]
    assert isinstance(n, navis.VoxelNeuron)
    assert "nanometer" in str(n.units)
    assert len(n.voxels) == n_voxels

    if not ds.supports(Cap.SKELETONS):
        return

    # Independent check: the voxels belong to *this* neuron, in *this* place.
    #
    # Checked against the skeleton, not the mesh. neuPrint-backed meshes come back in
    # voxel units while labelled nanometres (hemibrain/maleCNS/MANC are 8x small,
    # FlyWire's mirror 4/4/40x) - a real, separate bug - so the mesh is not a sound
    # reference frame. Skeletons are correct in nm on both backends.
    sk = ds.skeletons.get(body, progress=False)[0]
    nodes = sk.nodes[["x", "y", "z"]].values
    nm = _voxels.to_nm(_voxels.rle.decode_runs(runs), resolution)

    # The two bounding boxes must essentially coincide.
    #
    # Not strict enclosure: at these scales a neuron's thinnest neurites downsample
    # away entirely, so the skeleton legitimately reaches a little past the voxels -
    # measured at 0.3-1.5% of extent on FlyWire. The tolerance is therefore relative,
    # which still leaves it enormously tighter than either failure it guards against.
    # A wrong node puts the neuron somewhere else entirely, and a bad grid conversion
    # is off by a whole voxel-size factor (the neuPrint mesh bug is 8x = 700%).
    extent = nodes.max(0) - nodes.min(0)
    tol = np.maximum(0.05 * extent, 4 * np.asarray(resolution))
    off = np.maximum(np.abs(nm.min(0) - nodes.min(0)), np.abs(nm.max(0) - nodes.max(0)))
    assert (off <= tol).all(), (
        f"{ds.label}: the sparse volume and the neuron's own skeleton do not agree.\n"
        f"  voxels    {nm.min(0)} .. {nm.max(0)}\n"
        f"  skeleton  {nodes.min(0)} .. {nodes.max(0)}\n"
        f"  worst corner offset {off} vs tolerance {tol} "
        f"({np.round(100 * off / extent, 1)}% of extent)\n"
        f"Wrong node, or a bad voxel-grid conversion."
    )


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
        ("voxels", Cap.VOXELS),
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


def test_the_segmentation_volume_holds_the_ids_the_dataset_talks_about(ds):
    """`Cap.SEGMENTATION` claims two things at once: that there is a volume, and that
    its values are *this dataset's* IDs. The second is the one that fails quietly.

    A bucket can open perfectly and still be the wrong release, or be registered at the
    wrong voxel size, and either way `locs_to_segments` returns plausible 64-bit
    integers that are simply not the neuron you asked about. So the check is a
    round-trip through the data: take a neuron's own T-bars - which sit *inside* it, by
    definition - and require the volume to name that neuron.

    (Skeleton nodes would not do. They are centrelines, and in fine neurites they cut
    corners into the neighbouring body often enough - ~45% of the time on hemibrain -
    that they cannot tell a coordinate bug from ordinary skeleton coarseness.)
    """
    if not (ds.supports(Cap.SEGMENTATION) and ds.supports(Cap.SYNAPSES)):
        pytest.skip("no segmentation volume, or no synapses to probe it with")

    x = int(ds.spec.example_ids[0])
    syn = ds.connectivity.synapses(x)
    pre = syn.loc[syn["pre"] == x, ["pre_x", "pre_y", "pre_z"]].to_numpy()
    if len(pre) < 5:
        pytest.skip(f"{ds.label}: example neuron has too few T-bars")

    # Nanometres - which is what every connecto call returns, and what this one takes.
    got = ds.segmentation.locs_to_segments(pre[:10], progress=False)

    hits = int((got == x).sum())
    assert hits >= 8, (
        f"{ds.label}: only {hits}/10 of neuron {x}'s own T-bars land inside it "
        f"(got {sorted(set(got.tolist()))[:4]}...). The segmentation volume and the "
        f"database disagree - wrong release, or wrong voxel size."
    )


def test_a_flat_segmentation_does_not_pretend_to_be_a_chunkedgraph(ds):
    """The whole point of splitting the capability. A frozen dataset must not offer
    `update_ids`, and must not accept a `version=` it cannot honour."""
    if not ds.supports(Cap.SEGMENTATION) or ds.supports(Cap.CHUNKEDGRAPH):
        pytest.skip("not a flat-volume dataset")

    x = list(ds.spec.example_ids[:1])
    with pytest.raises(co.CapabilityError, match="chunkedgraph"):
        ds.segmentation.update_ids(x)
    with pytest.raises(co.CapabilityError, match="chunkedgraph"):
        ds.segmentation.locs_to_supervoxels([[0, 0, 0]])


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


@pytest.mark.parametrize("name", ["banc", "flywire"])
def test_the_neuprint_doors_synapses_land_in_the_cave_doors_volume(name):
    """One door's coordinates, the other door's segmentation. They must agree.

    Edges agreeing is not enough, because *coordinates* can be wrong on their own -
    and they were. The neuPrint mirrors of FlyWire and BANC were imported from CAVE
    and kept its nanometres, unlike every Janelia FIB-SEM dataset on neuPrint, which
    reports 8 nm voxels. connecto took the backend's usual word for it and multiplied
    by `voxel_size` again, so every synapse position came back 4-45x out - while the
    edges, the IDs and the weights all stayed perfectly correct. That is the shape of
    bug that reaches publication.

    Comparing the two doors' coordinates directly is not the test to write: they return
    genuinely different *sets* of synapses (different tables, different thresholds), so
    any extremum is dominated by whichever outlier one of them happens to include. This
    asks the physical question instead - is the T-bar inside the neuron? - and it is
    only askable because backend and dataset are orthogonal: BANC's neuPrint door has
    no segmentation volume, but BANC does, and it is one `backend="cave"` away.

    A pre-synapse is inside its own body by definition. Get the units wrong and the
    points land in another neuron, in empty space, or outside the volume entirely.
    """
    spec = co.get_spec(name)
    x = int(spec.example_ids[0])

    syn = _build(spec, backend="neuprint").connectivity.synapses(x)
    tbars = syn.loc[syn["pre"] == x, ["pre_x", "pre_y", "pre_z"]].to_numpy()[:10]
    assert len(tbars) >= 5, "not enough T-bars to be worth testing"

    segs = _build(spec, backend="cave").segmentation.locs_to_segments(
        tbars, progress=False
    )
    hits = int((segs == x).sum())
    assert hits >= 8, (
        f"{name}: only {hits}/{len(tbars)} of the neuPrint door's T-bars land inside "
        f"the neuron they belong to, per the CAVE door's segmentation. The positions "
        f"are in the wrong units.\n  T-bars (nm): {tbars[:3].tolist()}\n  got: {segs[:3]}"
    )


def test_autapses_are_opt_in_not_a_backend_accident():
    """Autapses appear only when asked for - and then from the backend that has them."""
    spec = co.get_spec("banc")
    ids = spec.example_ids
    cave = _build(spec, backend="cave")

    assert not (cave.connectivity.edges(ids).eval("pre == post")).any()

    with_autapses = cave.connectivity.edges(ids, autapses=True)
    assert (with_autapses["pre"] == with_autapses["post"]).any()
