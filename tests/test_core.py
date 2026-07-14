"""Offline tests. No network, no credentials, fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import connecto as co
from connecto.core.criteria import to_criteria
from connecto.core.schemas import normalize_annotations, normalize_edges
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

def test_capability_matrix_covers_every_dataset():
    m = co.capability_matrix()
    assert "flywire" in m.index and "microns" in m.index and "fish2" in m.index
    # The promises the specs make.
    assert m.loc["flywire", "nt_per_synapse"]
    assert not m.loc["microns", "nt_per_synapse"]
    assert not m.loc["flywire", "live"]  # frozen public release
    assert m.loc["flywire-production", "live"]

    # hemibrain has a segmentation volume (a flat precomputed bucket) but no
    # chunkedgraph under it - its body IDs are frozen. Two capabilities, because
    # collapsing them would force us to lie about one or the other.
    assert m.loc["hemibrain", "segmentation"]
    assert not m.loc["hemibrain", "chunkedgraph"]
    assert m.loc["flywire", "segmentation"] and m.loc["flywire", "chunkedgraph"]

    # fish2 publishes no volume we could verify, so it claims none.
    assert not m.loc["fish2", "segmentation"]


def test_backend_and_dataset_are_orthogonal():
    # BANC is served by both, at the same snapshot. This is the fact the whole
    # spec/registry design exists to express.
    banc = co.get_spec("banc")
    assert banc.backend_kinds == ("cave", "neuprint")
    assert banc.backend("cave").default_version == 888
    assert banc.backend("neuprint").source.endswith("banc:v888")


def test_asking_for_a_backend_a_dataset_lacks_says_so():
    with pytest.raises(ValueError, match="not served by"):
        co.get_spec("hemibrain").backend("cave")


def test_unknown_dataset_lists_the_known_ones():
    with pytest.raises(co.NoSuchDatasetError, match="flywire"):
        co.get_spec("nope")
