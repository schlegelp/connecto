"""Offline tests. No network, no credentials, fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import connecto as co
from connecto.core.criteria import to_criteria
from connecto.core.schemas import normalize_annotations, normalize_edges
from connecto.core.spec import Cap
from connecto.core.version import semantic_key, sort_versions

# --------------------------------------------------------------------- versions

def test_semantic_version_sort_beats_string_sort():
    # The bug this exists to prevent: a plain string sort puts v0.9 after v0.13.
    assert sorted(["v0.9", "v0.13"]) == ["v0.13", "v0.9"]
    assert sort_versions(["v0.9", "v0.13"]) == ["v0.9", "v0.13"]


def test_semantic_sort_handles_unversioned_datasets():
    # fish2, cns and mushroombody carry no version at all.
    assert sort_versions(["fish2"]) == ["fish2"]
    assert sort_versions(["male-cns:v0.9", "male-cns:v1.0"])[-1] == "male-cns:v1.0"
    assert semantic_key("fish2")[0] == 0  # non-numeric sorts first, doesn't explode


def test_version_compares_equal_to_bare_value():
    v = co.Version(783, "cave")
    assert v == 783
    assert str(v) == "783"


# --------------------------------------------------------------------- criteria

def test_mini_language_desugars_to_criteria():
    assert to_criteria(720575940604407468).id.tolist() == [720575940604407468]
    assert to_criteria("DA1_lPN").type == "DA1_lPN"

    rx = to_criteria("/AOTU00.*")
    assert rx.type == "AOTU00.*" and rx.regex is True

    col = to_criteria("cell_class:ALPN")
    assert col.extra == {"cell_class": "ALPN"}

    assert to_criteria("DA1_lPN", side="left").side == "left"


def test_id_list_stays_ids():
    c = to_criteria([1, 2, 3])
    assert c.id.tolist() == [1, 2, 3]
    assert c.type is None


# ---------------------------------------------------------------------- schemas

class _FakeSpec:
    fields = {"type": ("cell_type", "hemibrain_type"), "side": ("side",)}
    derive: dict = {}
    side_map = {"L": "left", "R": "right"}
    voxel_size = (4, 4, 40)


class _FakeDS:
    spec = _FakeSpec()
    name = "fake"
    label = "Fake"
    backend_kind = "cave"
    source = "fake_stack"
    _raw_position_units = "nm"


def test_edges_are_a_closed_schema():
    # The FlyWire edge view carries transmitter columns; hemibrain's does not. If
    # those leaked through, "same code, two backends" would be a lie.
    raw = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "n_syn": [10, 20],
            "ach": [0.9, 0.1],  # FlyWire-only noise
            "gaba": [0.1, 0.9],
        }
    )
    colmap = {"pre": "pre_pt_root_id", "post": "post_pt_root_id", "weight": "n_syn"}
    out = normalize_edges(raw, _FakeDS(), colmap=colmap)

    assert list(out.columns) == ["pre", "post", "weight"]
    assert out.dtypes["pre"] == np.int64
    assert out.dtypes["weight"] == np.int32
    # Dropped, but said so.
    assert set(out.attrs["connecto"]["dropped_columns"]) == {"ach", "gaba"}

    kept = normalize_edges(raw, _FakeDS(), colmap=colmap, extra=True)
    assert "ach" in kept.columns


def test_type_is_coalesced_by_priority():
    raw = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "cell_type": ["DA1_lPN", None, ""],
            "hemibrain_type": ["X", "FALLBACK", "ALSO_FALLBACK"],
            "side": ["L", "R", None],
        }
    )
    out = normalize_annotations(raw, _FakeDS(), id_column="root_id")

    # First non-null wins; empty string counts as null.
    assert out["type"].tolist() == ["DA1_lPN", "FALLBACK", "ALSO_FALLBACK"]
    # Side vocabulary normalised...
    assert out["side"].tolist()[:2] == ["left", "right"]
    # ...and the raw columns are still there.
    assert "cell_type" in out.columns and "hemibrain_type" in out.columns


def test_absent_fields_do_not_become_all_null_columns():
    # hemibrain v1.2.1 has no `class`. A column of NaNs would read as "we have this
    # field and it's empty", which is a different claim from "we don't have it".
    raw = pd.DataFrame({"root_id": [1], "cell_type": ["X"]})
    out = normalize_annotations(
        raw, _FakeDS(), id_column="root_id", fields={"class": ("nonexistent",)}
    )
    assert "class" not in out.columns


def test_derive_extracts_a_value_from_another_column():
    # hemibrain encodes side as an instance suffix: "DA1_lPN_R".
    class Spec(_FakeSpec):
        fields = {"side": ("side_from_instance",)}
        derive = {"side_from_instance": ("instance", r"_([LR])$")}

    class DS(_FakeDS):
        spec = Spec()

    raw = pd.DataFrame({"bodyId": [1, 2, 3], "instance": ["DA1_lPN_R", "DA1_lPN_L", "x"]})
    out = normalize_annotations(raw, DS(), id_column="bodyId")
    assert out["side"].tolist()[:2] == ["right", "left"]
    assert pd.isna(out["side"].iloc[2])  # no suffix -> no side, not a guess


# ------------------------------------------------------------------ capabilities

def _row(m, name, backend):
    """One (dataset, backend) row out of the matrix."""
    return m[m["backend"] == backend].loc[name]


def test_capability_matrix_covers_every_dataset():
    m = co.capability_matrix()
    assert "flywire" in m.index and "microns" in m.index and "fish2" in m.index

    # The promises the specs make. Note the matrix has a row per *door*, not per
    # dataset - see test_a_capability_belongs_to_a_backend_not_a_dataset.
    fw = _row(m, "flywire", "cave")
    assert fw["nt_per_synapse"]
    assert not _row(m, "microns", "cave")["nt_per_synapse"]
    assert not fw["live"]  # frozen public release
    assert _row(m, "flywire-production", "cave")["live"]

    # hemibrain has a segmentation volume (a flat precomputed bucket) but no
    # chunkedgraph under it - its body IDs are frozen. Two capabilities, because
    # collapsing them would force us to lie about one or the other.
    hb = _row(m, "hemibrain", "neuprint")
    assert hb["segmentation"]
    assert not hb["chunkedgraph"]
    assert fw["segmentation"] and fw["chunkedgraph"]

    # fish2 publishes no volume we could verify, so it claims none.
    assert not _row(m, "fish2", "neuprint")["segmentation"]


def test_a_capability_belongs_to_a_backend_not_a_dataset():
    """FlyWire *has* a chunkedgraph; you cannot reach it through neuPrint.

    Before this, capabilities were declared per dataset, so a neuPrint-backed FlyWire
    reported `supports(CHUNKEDGRAPH) == True` while every chunkedgraph call raised -
    and, worse, `synapses(transmitters=True)` returned a frame with no `nt` column and
    no error. That is the exact fafbseg failure this library exists to prevent, so the
    capability now lives on the (dataset, backend) pair.
    """
    spec = co.get_spec("flywire")

    cave = spec.capabilities_for("cave")
    neuprint = spec.capabilities_for("neuprint")

    # The narrow door: no supervoxels, no edit history, and - the one that would
    # otherwise bite silently - no per-synapse transmitters in this copy.
    for cap in (Cap.CHUNKEDGRAPH, Cap.PROOFREADING, Cap.NT_PER_SYNAPSE):
        assert cap in cave
        assert cap not in neuprint

    # ...but a *wider* one in the other direction: neuPrint ships an ROI hierarchy
    # that the CAVE datastack has no equivalent of.
    assert Cap.ROIS in neuprint
    assert Cap.ROIS not in cave

    assert spec.backends_with(Cap.CHUNKEDGRAPH) == ("cave",)
    assert spec.backends_with(Cap.CONNECTIVITY) == ("neuprint", "cave")


def test_a_backend_may_not_add_what_it_cannot_serve():
    """The contradiction is unregisterable, rather than discovered by a user."""
    with pytest.raises(ValueError, match="cannot serve"):
        co.BackendSpec("neuprint", "x/y:v1", extra_capabilities={Cap.CHUNKEDGRAPH})


def test_a_refusal_names_the_backend_that_can_do_it():
    """An error that only says "no" makes you go and read the source."""
    fw = co.get_dataset("flywire")  # neuPrint by default
    with pytest.raises(co.CapabilityError) as exc:
        _ = fw.proofreading
    assert 'backend="cave"' in str(exc.value)


def test_backend_and_dataset_are_orthogonal():
    # BANC is served by both, at the same snapshot. This is the fact the whole
    # spec/registry design exists to express. neuPrint is listed first, so it is the
    # default: same snapshot, faster, and it has ROIs.
    banc = co.get_spec("banc")
    assert banc.backend_kinds == ("neuprint", "cave")
    assert banc.backend() is banc.backend("neuprint")  # first == default
    assert banc.backend("cave").default_version == 888
    assert banc.backend("neuprint").source.endswith("banc:v888")


def test_asking_for_a_backend_a_dataset_lacks_says_so():
    with pytest.raises(ValueError, match="not served by"):
        co.get_spec("hemibrain").backend("cave")


def test_unknown_dataset_lists_the_known_ones():
    with pytest.raises(co.NoSuchDatasetError, match="flywire"):
        co.get_spec("nope")


def test_auto_prefers_a_public_source_but_falls_back_to_a_private_one():
    """`annotations="auto"` skips non-public sources so an ordinary user gets one they
    can actually read. But a gated dataset (aedes) may have *only* a private source -
    and there, returning None would make `.annotations.get()` claim "no source", which
    is false. So auto falls back to the first source rather than giving up."""
    from connecto.core.spec import AnnotationSource, BackendSpec, DatasetSpec

    be = (BackendSpec("cave", "x"),)
    pub = AnnotationSource("public", "github_tsv", "u")
    priv = AnnotationSource("flytable", "seatable", "b.t", public=False)

    both = DatasetSpec(name="both", backends=be, annotation_sources=(pub, priv))
    assert both.annotation_source("auto") is pub  # public wins when present

    private_only = DatasetSpec(name="priv", backends=be, annotation_sources=(priv,))
    assert private_only.annotation_source("auto") is priv  # ...else fall back, not None

    none_at_all = DatasetSpec(name="bare", backends=be, annotation_sources=())
    assert none_at_all.annotation_source("auto") is None  # genuinely nothing -> None


# ----------------------------------------------------------------- transmitters

def test_nt_is_an_argmax_over_the_declared_classes():
    from connecto.core.schemas import add_transmitters

    raw = pd.DataFrame(
        {
            "ach": [0.9, 0.1, np.nan],
            "gaba": [0.05, 0.8, np.nan],
            "glut": [0.05, 0.1, np.nan],
        }
    )
    out = add_transmitters(
        raw, {"ach": "acetylcholine", "gaba": "gaba", "glut": "glutamate"}, label="X"
    )

    assert out["nt"].tolist()[:2] == ["acetylcholine", "gaba"]
    assert out["nt_confidence"].tolist()[:2] == pytest.approx([0.9, 0.8])
    # A synapse nobody predicted is null, not a guess. BANC's neuPrint copy has
    # 121 of these on one neuron, and `idxmax` raises on an all-NA row rather
    # than returning NA - so this is the case that breaks first if it regresses.
    assert pd.isna(out["nt"].iloc[2])
    assert pd.isna(out["nt_confidence"].iloc[2])
    # Renamed onto the canonical vocabulary, raw columns kept alongside.
    assert out["nt_acetylcholine"].tolist()[:2] == pytest.approx([0.9, 0.1])
    assert "ach" in out.columns


def test_a_missing_transmitter_class_raises_rather_than_losing_quietly():
    """`nt` is an argmax, so a class the server didn't return cannot show up as a
    gap - it shows up as the runner-up, confidently. The only safe answer is to stop."""
    from connecto.core.schemas import add_transmitters

    raw = pd.DataFrame({"ach": [0.9], "gaba": [0.1]})
    with pytest.raises(co.CapabilityError, match="ntGlutamateProb|glut"):
        add_transmitters(
            raw, {"ach": "acetylcholine", "gaba": "gaba", "glut": "glutamate"}, label="X"
        )


def test_transmitter_columns_must_use_the_canonical_vocabulary():
    from connecto.core.spec import BackendSpec

    with pytest.raises(ValueError, match="TRANSMITTERS"):
        BackendSpec("cave", "x", nt_columns={"ach": "ACh"})


def test_claiming_transmitters_without_wiring_is_unregisterable():
    """The check that would have caught the bug this library was written about: a
    capability is a promise, and `Cap.NT_PER_SYNAPSE` used to be one with nothing
    behind it, so `transmitters=True` returned a frame with no `nt` and no error."""
    from connecto.core.spec import BackendSpec, DatasetSpec

    with pytest.raises(ValueError, match="nt_columns"):
        DatasetSpec(
            name="liar",
            backends=(BackendSpec("cave", "x"),),
            capabilities=frozenset({Cap.NT_PER_SYNAPSE}),
        )


def test_neuprint_no_longer_denies_transmitters_wholesale():
    """It used to, and that stated a gap in connecto's code as a fact about the
    server. banc, manc and male-cns all carry `ntGabaProb` & co. on their Synapse
    nodes; hemibrain and fish2 genuinely do not, and FlyWire's mirror carries them
    on Neuron nodes only - so the denial belongs per dataset, not per backend."""
    from connecto.core.spec import BACKEND_LIMITS

    assert Cap.NT_PER_SYNAPSE not in BACKEND_LIMITS["neuprint"]

    m = co.capability_matrix()
    for name in ("banc", "manc", "malecns"):
        assert _row(m, name, "neuprint")["nt_per_synapse"], name
    for name in ("hemibrain", "fish2"):
        assert not _row(m, name, "neuprint")["nt_per_synapse"], name
    # BANC is the one dataset that has them through *both* doors.
    assert co.get_spec("banc").backends_with(Cap.NT_PER_SYNAPSE) == ("neuprint", "cave")


def test_nt_records_which_column_it_came_from():
    """`known_nt` is somebody's immunostaining; `top_nt` is a CNN's argmax. Coalescing
    them into one column without saying which would let "this neuron is GABAergic"
    mean either "we measured it" or "a model thinks so"."""
    raw = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "known_nt": ["acetylcholine", "", None],
            "top_nt": ["gaba", "glutamate", None],
        }
    )

    class _DS(_FakeDS):
        class spec(_FakeSpec):
            fields = {"nt": ("known_nt", "top_nt")}

    out = normalize_annotations(raw, _DS(), id_column="root_id")

    assert out["nt"].tolist()[:2] == ["acetylcholine", "glutamate"]
    assert pd.isna(out["nt"].iloc[2])  # blank in both -> no call
    assert out["nt_source"].tolist()[:2] == ["known_nt", "top_nt"]
    assert pd.isna(out["nt_source"].iloc[2])  # no value -> no source
    # It sits next to the column it explains.
    assert list(out.columns).index("nt_source") == list(out.columns).index("nt") + 1


def test_a_raw_nt_source_column_does_not_get_clobbered():
    """FlyWire and BANC both carry `known_nt_source` - a *citation*. Nothing carries
    `nt_source` today, but if one ever does it means something else, so it steps aside
    rather than being overwritten by ours."""
    raw = pd.DataFrame(
        {"root_id": [1], "top_nt": ["gaba"], "nt_source": ["Davis et al., 2020"]}
    )

    class _DS(_FakeDS):
        class spec(_FakeSpec):
            fields = {"nt": ("top_nt",)}

    out = normalize_annotations(raw, _DS(), id_column="root_id")
    assert out["nt_source"].tolist() == ["top_nt"]
    assert out["nt_source_raw"].tolist() == ["Davis et al., 2020"]
