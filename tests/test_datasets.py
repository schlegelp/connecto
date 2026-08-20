"""Dataset wiring, and the three bugs that adding FANC/aedes/MANC exposed.

These are offline: they test the spec layer and the machinery that reads it, not the
servers. Each of the three sections below is a regression test for something that was
silently wrong - in every case the library returned a *plausible* answer (an empty ID
list, a generic AttributeError) rather than saying it could not do the thing, which is
precisely the failure mode connecto exists to prevent.
"""

from __future__ import annotations

import pandas as pd
import pytest

import connecto as co
from connecto.core.criteria import _match
from connecto.core.registry import REGISTRY
from connecto.core.spec import MULTI_SEP, Cap

# --------------------------------------------------------------- the datasets exist


def test_the_shipped_datasets_are_all_registered():
    for name in ("flywire", "banc", "fanc", "microns", "aedes",
                 "hemibrain", "malecns", "manc", "fish2"):
        assert name in REGISTRY, f"{name} is not registered"


def test_optic_lobe_is_gone():
    """It was the optic lobes of the same specimen `malecns` now covers whole - two
    names for one dataset, and the narrower one could only give you fewer neurons
    under different body IDs. Registering both invited a silent mismatch."""
    assert "optic-lobe" not in REGISTRY
    assert not hasattr(co, "OpticLobe")


@pytest.mark.parametrize("name", ["Aedes", "BANC", "FANC", "MANC", "FlyWire",
                                  "Hemibrain", "MaleCNS", "MICrONS", "Fish2"])
def test_every_dataset_has_a_constructor(name):
    """MANC was registered but had no reachable constructor: the spec constant `MANC`
    shadowed it, so the factory was renamed `MANC_` and then never exported. It showed
    up in the README's generated capability matrix while `co.MANC` did not exist."""
    assert callable(getattr(co, name))


def test_manc_spec_and_constructor_do_not_collide():
    assert co.datasets.MANC_SPEC.name == "manc"
    assert co.MANC is not co.datasets.MANC_SPEC


# ------------------------------------------------- a set in a cell stays searchable


def test_a_pivoted_tag_set_matches_each_of_its_members():
    """FANC tags one MDN with "MDN", "MDN3" *and* "moonwalker descending neuron", so
    the pivot stores all three in one cell. Matching only the whole cell made every
    member unfindable - and silently: ds.ids("MDN") returned an empty array."""
    s = pd.Series(["MDN, MDN3, moonwalker descending neuron", "DNa02", None])

    assert _match(s, "MDN", False).tolist() == [True, False, False]
    assert _match(s, "MDN3", False).tolist() == [True, False, False]
    assert _match(s, "moonwalker descending neuron", False).tolist() == [True, False, False]
    assert _match(s, "DNa02", False).tolist() == [False, True, False]


def test_a_tag_set_does_not_match_a_mere_substring():
    """Set membership, not `in`. "MDN" must not match "MDN3" on its own."""
    s = pd.Series(["MDN3", "MDN3, MDN4"])
    assert _match(s, "MDN", False).tolist() == [False, False]


def test_single_valued_columns_are_unaffected():
    """The overwhelmingly common case: no cell holds a set, so nothing changes."""
    s = pd.Series(["DA1_lPN", "DA2_lPN", None])
    assert _match(s, "DA1_lPN", False).tolist() == [True, False, False]
    assert _match(s, "DA2_lPN", False).tolist() == [False, True, False]
    assert _match(s, "DA.*", True).tolist() == [True, True, False]
    assert _match(s, "nope", False).tolist() == [False, False, False]


def test_regex_matches_members_of_a_set_too():
    s = pd.Series(["MDN, MDN3, moonwalker descending neuron"])
    assert _match(s, "moonwalker.*", True).tolist() == [True]


def test_the_join_and_the_split_share_one_constant():
    """The pivot writes the set and the matcher reads it. Two copies of ", " would be
    a silent divergence, so both import MULTI_SEP."""
    from connecto.sources import pivot_long

    src = co.AnnotationSource("x", "cave_table", "t", id_column="root_id",
                              pivot=("key", "value"))
    long = pd.DataFrame(
        {
            "root_id": [1, 1, 2],
            "key": ["type", "type", "type"],
            "value": ["MDN", "MDN3", "DNa02"],
        }
    )
    wide = pivot_long(long, src)
    assert wide.loc[wide.root_id == 1, "type"].iloc[0] == MULTI_SEP.join(["MDN", "MDN3"])
    assert _match(wide["type"], "MDN3", False).sum() == 1


# ------------------------------------------ a missing namespace says which dataset


def test_a_missing_namespace_says_which_dataset():
    """A capability a dataset lacks must raise a CapabilityError that *names the
    dataset*, not an AttributeError about the class. aedes has no ROIs; accessing the
    namespace used to fall through to Dataset.__getattr__ - because CapabilityError
    subclasses AttributeError, the descriptor's precise error read as "lookup failed" -
    and came back as "'CAVEDataset' object has no attribute 'rois'"."""
    ds = co.get_dataset("aedes")

    with pytest.raises(co.CapabilityError) as exc:
        _ = ds.rois

    msg = str(exc.value)
    assert "Aedes" in msg  # the dataset, not the class
    assert "rois" in msg
    assert "CAVEDataset" not in msg


def test_feature_detection_still_works():
    ds = co.get_dataset("aedes")
    assert not hasattr(ds, "rois")  # aedes has no ROI hierarchy
    assert hasattr(ds, "annotations")  # ...but it does have annotations, via FlyTable
    assert hasattr(ds, "connectivity")


def test_a_real_typo_is_still_a_plain_attribute_error():
    ds = co.get_dataset("aedes")
    with pytest.raises(AttributeError) as exc:
        _ = ds.nonsense
    assert not isinstance(exc.value, co.CapabilityError)


# ----------------------------------------------------- the score column is a name


def test_the_synapse_score_column_comes_off_the_spec():
    """FlyWire calls it `cleft_score`, FANC calls it `score`. Same concept - so it is
    a column name on the BackendSpec, not a capability."""
    assert co.get_spec("flywire").backend("cave").score_column == "cleft_score"
    assert co.get_spec("fanc").backend("cave").score_column == "score"

    fanc = co.get_dataset("fanc")
    assert fanc._synapse_colmap["score"] == "score"
    assert co.get_dataset("flywire", backend="cave")._synapse_colmap["score"] == "cleft_score"
    # `score_column` is a CAVE table's column name. neuPrint has its own notion of a
    # confidence score, and reads it from its own schema.
    assert co.get_dataset("flywire")._synapse_colmap["score"] == "confidence_pre"


# ------------------------------------------------------------- specs match reality


def test_aedes_gets_annotations_from_flytable_but_still_no_scores():
    """The CAVE datastack has no annotation table; the cell typing lives in FlyTable
    (the `aedes_main` table of the `aedes` base). So aedes *does* claim annotations -
    from a lab-internal SeaTable source - but still no per-synapse score (the synapse
    table has `size`, not a confidence)."""
    spec = co.get_spec("aedes")
    assert Cap.ANNOTATIONS in spec.capabilities
    assert Cap.SYNAPSE_SCORES not in spec.capabilities

    (src,) = spec.annotation_sources
    assert (src.name, src.kind, src.location) == ("flytable", "seatable", "aedes.aedes_main")
    assert src.id_column == "root_id"
    assert src.public is False  # lab-internal; needs a SEATABLE_TOKEN

    # Its only source is non-public, so `auto` must still resolve to it rather than
    # giving up - otherwise `.annotations.get()` would wrongly say "no source".
    assert spec.annotation_source("auto") is src
    # And it can find neurons by the fields we wired up.
    assert "class" in spec.fields and "type" in spec.fields


@pytest.mark.network
def test_flywire_flytable_concatenates_both_bases_into_clean_int_ids():
    """Live: the two tables really do concatenate, and the `root_783` key gives clean
    int64 ids (not float - the very bug the caveclient pin guards against), with the
    known DA1_lPN examples present. Skipped without a SEATABLE_TOKEN."""
    import os

    if not os.environ.get("SEATABLE_TOKEN"):
        pytest.skip("needs SEATABLE_TOKEN")

    fw = co.FlyWire()
    ann = fw.annotations.get(source="flytable")
    # 58k central-brain + 89k optic-lobe neurons; far more than `info` alone (~58k).
    assert len(ann) > 100_000
    assert ann["id"].dtype == "int64"
    assert (ann["id"] > 0).all()
    assert set(fw.spec.example_ids) <= set(ann["id"])  # frozen 783 ids line up


@pytest.mark.network
def test_banc_flytable_resolves_on_the_cloud_instance():
    """Live: BANC's flytable really is reachable on cloud.seatable.io (the old
    `banc.main` on the lab instance 404'd), with clean int64 ids on `root_888` and
    the known example neurons present. Skipped without a SEATABLE_TOKEN."""
    import os

    if not os.environ.get("SEATABLE_TOKEN"):
        pytest.skip("needs SEATABLE_TOKEN")

    b = co.BANC()
    ann = b.annotations.get(source="flytable")
    assert len(ann) > 100_000
    assert ann["id"].dtype == "int64"
    assert (ann["id"] > 0).all()
    assert set(b.spec.example_ids) <= set(ann["id"])  # frozen 888 ids line up


def test_flywire_flytable_is_two_bases_keyed_on_the_frozen_root():
    """FlyWire's live annotations are two SeaTable tables in two different bases -
    the central brain in `main.info`, the optic lobes in `optic_lobes.optic` -
    concatenated. Reading only `info` silently drops ~89k optic-lobe neurons. And the
    public release is frozen at mat 783, so its id column is `root_783`, not the live
    `root_id` that tracks edits - otherwise every edited neuron mis-joins in silence."""
    from connecto.sources.seatable import _table_specs

    pub = co.get_spec("flywire").annotation_source("flytable")
    assert pub.location == "main.info,optic_lobes.optic"
    assert _table_specs(pub) == [("main", "info"), ("optic_lobes", "optic")]
    assert pub.id_column == "root_783"  # frozen release
    assert pub.public is False  # lab-internal; needs a SEATABLE_TOKEN

    # Production is the *same* two tables but live, so it keys on `root_id`. A shared
    # source with one static id column cannot be right for both, so they diverge.
    prod = co.get_spec("flywire-production").annotation_source("flytable")
    assert prod.location == pub.location
    assert prod.id_column == "root_id"  # live release


def test_seatable_sources_declare_which_deployment_they_live_on():
    """SeaTable is not one server. FlyWire and aedes are on the lab's own instance
    ('flytable'); BANC's `banc_meta` is on the official cloud ('seatable'). A base
    named on one does not exist on the other, so the instance must be explicit."""
    from connecto.sources.seatable import _INSTANCES, _server

    fw = co.get_spec("flywire").annotation_source("flytable")
    aedes = co.get_spec("aedes").annotation_source("flytable")
    banc = co.get_spec("banc").annotation_source("flytable")

    assert fw.instance == "flytable" and aedes.instance == "flytable"
    assert banc.instance == "seatable"

    # flytable -> the SEATABLE_SERVER env default (None); seatable -> the cloud URL.
    assert _server(fw) is None
    assert _server(banc) == "https://cloud.seatable.io/"
    assert set(_INSTANCES) >= {"flytable", "seatable"}


def test_banc_flytable_points_at_the_cloud_banc_meta_keyed_on_the_frozen_root():
    """BANC's live annotations are `banc_meta` on cloud.seatable.io - not the lab's
    flytable instance, where the old `banc.main` did not even exist. And BANC is
    frozen at CAVE mat 888, so the id column is `root_888`, not the live `root_id`."""
    banc = co.get_spec("banc").annotation_source("flytable")
    assert (banc.kind, banc.location, banc.instance) == ("seatable", "banc_meta.banc_meta", "seatable")
    assert banc.id_column == "root_888"  # frozen release, like FlyWire's root_783
    assert banc.public is False


def test_an_unknown_seatable_instance_is_a_loud_error_not_a_silent_default():
    """Fat-fingering the instance must fail with the known names, not quietly fall
    back to one deployment and 404 on a base that lives on the other."""
    import pytest

    from connecto.core.spec import AnnotationSource
    from connecto.sources.seatable import _server

    bad = AnnotationSource("flytable", "seatable", "x.y", instance="nope")
    with pytest.raises(ValueError, match="Unknown SeaTable instance"):
        _server(bad)


def test_table_specs_parses_base_qualified_and_bare_locations():
    """The `base.table` convention is what keeps SeaTable fast: named, resolution is
    ~1s; bare, seaserpent searches every base (~20s). A bare location still parses -
    to `(None, table)` - but every shipped source names its base."""
    from connecto.core.spec import AnnotationSource
    from connecto.sources.seatable import _table_specs

    one = AnnotationSource("flytable", "seatable", "aedes.aedes_main")
    two = AnnotationSource("flytable", "seatable", "main.info,optic_lobes.optic")
    bare = AnnotationSource("flytable", "seatable", "info")
    assert _table_specs(one) == [("aedes", "aedes_main")]
    assert _table_specs(two) == [("main", "info"), ("optic_lobes", "optic")]
    assert _table_specs(bare) == [(None, "info")]


def test_aedes_uses_the_synapse_table_whose_coordinates_are_right():
    """`synapses_v2` is newer and 23% larger, but CAVE has it registered at the
    datastack's 16x16x45 voxel size while its coordinates are already in nm. connecto
    asks CAVE for nm, so v2 comes back scaled up again - a 9.9 mm mosquito brain."""
    assert co.get_spec("aedes").backend("cave").synapse_table == "synapses"


@pytest.mark.parametrize("name", ["fanc", "aedes", "manc", "malecns"])
def test_the_new_datasets_are_conformance_testable(name):
    """example_ids are what put a dataset in the conformance suite. Without them it is
    silently skipped - which is how manc went untested."""
    assert co.get_spec(name).example_ids


# ------------------------------------------------------------------ the L2 namespace


def test_l2_is_present_exactly_where_the_cache_is():
    """Cap.L2CACHE used to gate nothing user-facing - it was an internal hint to the
    skeleton fallback, so nothing could hold a dataset to the claim. Now it carries the
    `.l2` namespace, and the conformance suite checks it."""
    assert hasattr(co.get_dataset("flywire-production"), "l2")   # has an L2 cache
    assert hasattr(co.get_dataset("banc", backend="cave"), "l2")
    # The public FlyWire stack has no L2 cache, on either backend.
    assert not hasattr(co.get_dataset("flywire", backend="cave"), "l2")
    assert not hasattr(co.get_dataset("hemibrain"), "l2")        # neuPrint: no chunkedgraph
    # ...and neuPrint never has one, even for a dataset whose CAVE copy does.
    assert not hasattr(co.get_dataset("banc"), "l2")             # neuPrint is the default


def test_asking_for_l2_without_a_cache_names_the_dataset():
    with pytest.raises(co.CapabilityError) as exc:
        _ = co.get_dataset("hemibrain").l2
    assert "hemibrain" in str(exc.value)
    assert "l2cache" in str(exc.value)


# ------------------------------------------------------------------- provenance


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_dataset_says_what_it_is_and_where_it_lives(name):
    """A dataset is somebody's years of work. The least we can do is describe it and
    say where to find it."""
    spec = co.get_spec(name)
    assert spec.description, f"{name} has no description"
    assert spec.links, f"{name} has no links"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_a_dataset_cites_its_papers_or_admits_it_has_none(name):
    """Every dataset carries citable publications - except the two that genuinely have
    none, and those say so rather than borrowing a neighbour's paper.

    This is the point of the test. It would be easy to give `aedes` the Lee lab's
    mosquito preprint and `fish2` one of the two published zebrafish connectomes; both
    would look right and both would be wrong, and the error would surface in somebody's
    methods section. So the honest empty set is asserted, not tolerated."""
    spec = co.get_spec(name)
    if name in ("aedes", "fish2"):
        assert spec.publications == ()
        assert "no publication" in spec.description or "no paper" in spec.description
        return
    assert spec.publications, f"{name} cites nothing"
    for pub in spec.publications:
        assert pub.doi and pub.year and pub.authors and pub.title
        assert pub.url == f"https://doi.org/{pub.doi}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_a_dataset_you_cannot_have_says_how_to_ask(name):
    """`public=False` is not a secret - it is a warning that a fresh token is not
    enough. Saying so without saying what *would* be is worse than not saying it."""
    spec = co.get_spec(name)
    assert spec.public == (not spec.access)
    if not spec.public:
        assert len(spec.access) > 40, f"{name}'s access note says too little"


def test_the_non_public_datasets_are_the_ones_we_think():
    got = {s.name for s in REGISTRY.values() if not s.public}
    assert got == {"flywire-production", "fanc", "aedes", "fish2"}


def test_a_spec_cannot_claim_privacy_without_saying_how_to_ask():
    with pytest.raises(ValueError, match="access"):
        co.DatasetSpec(
            name="x", backends=(co.BackendSpec("cave", "y"),), public=False
        )


def test_cite_names_the_papers_and_the_access_terms():
    text = co.get_dataset("fanc").cite()
    assert "Azevedo" in text and "10.1038/s41586-024-07389-x" in text
    assert "FANC_edit" in text  # non-public: the terms are part of the citation block


@pytest.mark.parametrize("ns,call", [("skeletons", "dotprops"), ("l2", "dotprops")])
def test_dotprops_reject_a_unit_they_cannot_honour(ns, call):
    """NBLAST is calibrated in microns and connecto returns nm by default, so `units` is
    the one knob that matters here. A typo in it must raise, not be ignored."""
    ds = co.get_dataset("flywire-production")
    fn = getattr(getattr(ds, ns), call)
    with pytest.raises(ValueError, match="units"):
        fn([720575940604407468], units="parsecs")


# ------------------------------------------------------- the documented matrices

CAP_COLUMNS = {
    "annot": Cap.ANNOTATIONS, "conn": Cap.CONNECTIVITY, "syn": Cap.SYNAPSES,
    "scores": Cap.SYNAPSE_SCORES, "NT/syn": Cap.NT_PER_SYNAPSE,
    "roi-conn": Cap.ROI_CONN, "rois": Cap.ROIS, "skel": Cap.SKELETONS,
    "mesh": Cap.MESHES, "seg": Cap.SEGMENTATION, "cgraph": Cap.CHUNKEDGRAPH,
    "vox": Cap.VOXELS, "proof": Cap.PROOFREADING, "soma": Cap.SOMAS,
    "live": Cap.LIVE,
}


def _doc_matrix(path):
    """Parse the `| dataset | ... |` table out of a markdown file."""
    from pathlib import Path

    lines = Path(path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| dataset |"))
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    rows, dataset = {}, None
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        dataset = cells[0].strip("`") or dataset
        row = dict(zip(header, cells))
        rows[(dataset, row["backend"].strip("*"))] = row
    return rows


@pytest.mark.parametrize("path", ["README.md", "docs/index.md"])
def test_the_documented_capability_matrix_matches_the_code(path):
    """The README and the landing page both hand-maintain the capability matrix, and
    both say they are generated from it. They drifted anyway: when BANC, maleCNS and
    MANC gained per-synapse transmitters the tables still showed FlyWire as the only
    dataset with any, and neither had ever grown the `vox` column. A table that claims
    to be generated has to be checkable, or it is just a comment that ages.
    """
    doc = _doc_matrix(path)
    assert doc, f"no capability table found in {path}"

    for (name, backend), row in doc.items():
        spec = REGISTRY[name]
        have = spec.capabilities_for(backend)
        for label, cap in CAP_COLUMNS.items():
            assert label in row, f"{path}: {name}/{backend} has no {label!r} column"
            claimed = row[label] == "✅"
            assert claimed == (cap in have), (
                f"{path}: {name}/{backend} shows {label}={row[label]!r}, "
                f"but capabilities_for({backend!r}) says {cap in have}"
            )

    # Every door in the registry must appear - a dataset silently missing from the
    # table is the other way for it to be wrong.
    doors = {(s.name, b.kind) for s in REGISTRY.values() for b in s.backends}
    assert doors == set(doc), f"{path}: table covers {set(doc) ^ doors} differently"


@pytest.mark.parametrize("path", ["README.md", "docs/index.md"])
def test_the_documented_neuron_level_nt_column_matches_the_specs(path):
    """`NT/neuron` is not a capability - it is whether the default annotation source
    has an `nt` field - so nothing in `capability_matrix()` can hold it honest.

    Three marks, not two. A dataset where a classifier covers the volume and one where
    a handful of neurons carry a transmitter somebody read off a paper are not the same
    claim, and averaging over the second as though it were the first turns annotation
    effort into a result. `aedes` is the only ◐ today: 40% coverage, all of it curated.

    Dense is read off `Cap.NT_PER_SYNAPSE` rather than declared separately, because the
    per-neuron call *is* the per-synapse model rolled up - a dataset has a dense one
    exactly when somebody ran the classifier. Should a dataset ever import per-neuron
    predictions without the synapses behind them, this is the assumption to revisit.
    """
    for (name, backend), row in _doc_matrix(path).items():
        spec = REGISTRY[name]
        src = spec.annotation_source("auto")
        fields = dict(spec.fields) | dict(src.fields if src else {})
        if not fields.get("nt"):
            expected = "·"
        else:
            expected = "✅" if spec.backends_with(Cap.NT_PER_SYNAPSE) else "◐"
        assert row["NT/neuron"] == expected, (
            f"{path}: {name}/{backend} shows NT/neuron={row['NT/neuron']!r}, "
            f"expected {expected!r}"
        )
