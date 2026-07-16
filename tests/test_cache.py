"""The on-disk cache, and its Feather -> pickle fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from connecto import cache as _cache


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "CACHE_ROOT", tmp_path)
    return tmp_path


def _entry():
    return _cache.CacheEntry("ds", "v1", "annotations_flytable", "base.table")


def test_a_feather_friendly_frame_is_stored_as_feather(cache_root):
    df = pd.DataFrame({"root_id": [1, 2, 3], "type": ["a", "b", "c"]})
    e = _entry()
    e.write(df)

    assert e.path.exists() and not e._pickle.exists()
    assert e.exists()
    pd.testing.assert_frame_equal(e.read(), df)


def test_a_frame_feather_cannot_hold_falls_back_to_pickle(cache_root):
    """FlyTable's `aedes_main` has a `serial_id` column that mixes `int` and `str` in
    one object column - Arrow refuses it. Before the fallback the write raised, the
    entry was deleted, and every `.annotations.get()` re-downloaded all 17k rows."""
    df = pd.DataFrame({
        "root_id": [1, 2, 3],
        "serial_id": [10, "20", 30],  # mixed int/str object column - feather-hostile
    })
    with pytest.raises(Exception):  # noqa: B017 - confirm it really is hostile
        df.to_feather(cache_root / "probe.feather")

    e = _entry()
    e.write(df)

    assert e._pickle.exists() and not e.path.exists()  # pickle, not feather
    assert e.exists()
    # A hit is exactly what a miss would have returned - mixed types and all.
    got = e.read()
    pd.testing.assert_frame_equal(got, df)
    assert sorted({type(v).__name__ for v in got["serial_id"]}) == ["int", "str"]


def test_feather_is_preferred_and_leaves_no_stale_pickle(cache_root):
    e = _entry()
    e.write(pd.DataFrame({"x": [1, "2"]}))  # hostile -> pickle
    assert e._pickle.exists()

    e.write(pd.DataFrame({"x": [1, 2]}))  # clean -> feather, and the pickle is cleared
    assert e.path.exists() and not e._pickle.exists()


def test_disabling_the_cache_serves_nothing(cache_root, monkeypatch):
    e = _entry()
    e.write(pd.DataFrame({"x": [1, 2]}))
    assert e.exists()

    monkeypatch.setenv("CONNECTO_NO_CACHE", "1")
    assert not e.exists()  # present on disk, but never served when disabled


# --- verbose narration --------------------------------------------------------


def test_source_phrase_names_where_the_data_lives():
    """The one-line feedback has to say *which* source - and for SeaTable/CAVE the
    table is worth showing; a `github_tsv` URL is not."""
    from connecto.core.namespaces import _source_phrase
    from connecto.core.spec import AnnotationSource

    ft = AnnotationSource("flytable", "seatable", "main.info,optic_lobes.optic")
    assert _source_phrase(ft) == "FlyTable main.info,optic_lobes.optic"
    # A cloud-instance SeaTable source names the host, not "FlyTable" - BANC does not
    # live on the lab's instance, and the feedback must not pretend it does.
    cloud = AnnotationSource("flytable", "seatable", "banc_meta.banc_meta", instance="seatable")
    assert _source_phrase(cloud) == "cloud.seatable.io banc_meta.banc_meta"
    assert _source_phrase(AnnotationSource("public", "github_tsv", "http://x/y.tsv")) == "public TSV"
    assert _source_phrase(AnnotationSource("cave", "cave_table", "codex")) == "CAVE table codex"
    assert _source_phrase(AnnotationSource("np", "neuprint", "")) == "neuPrint"


def test_annotations_get_narrates_the_cache_decision(cache_root, monkeypatch, capsys):
    """`verbose=True` says which source, and whether it came from the remote server,
    the local cache, the session memo, or a refresh - the feedback a confused user
    reaches for. The states are driven entirely by the freshness token and what is on
    disk, so they can be exercised with a stub dataset and no network."""
    import connecto.sources as _sources
    from connecto.core.namespaces import Annotations
    from connecto.core.spec import AnnotationSource

    src = AnnotationSource("flytable", "seatable", "aedes.aedes_main", public=False)
    monkeypatch.setattr(_sources, "freshness", lambda s, ds, v: "TOKEN-1")  # live

    class FakeDS:
        name = "aedes"
        label = "Aedes (mosquito brain)"

        def _fetch_annotations(self, source, version):
            return pd.DataFrame({"root_id": [1, 2, 3]})

    # 1. cold cache -> download
    Annotations(FakeDS())._table(src, "v1", verbose=True)
    out = capsys.readouterr().out
    assert "downloading from remote" in out and "FlyTable aedes.aedes_main" in out

    # 2. fresh handle, warm cache, live token -> re-validated local cache
    Annotations(FakeDS())._table(src, "v1", verbose=True)
    out = capsys.readouterr().out
    assert "local cache" in out and "re-validated" in out

    # 3. same handle again -> session memo, no re-read
    ann = Annotations(FakeDS())
    ann._table(src, "v1", verbose=True)
    capsys.readouterr()
    ann._table(src, "v1", verbose=True)
    assert "already loaded this session" in capsys.readouterr().out

    # 4. verbose=False -> silent
    Annotations(FakeDS())._table(src, "v1", verbose=False)
    assert capsys.readouterr().out == ""

    # 5. the source moved on (token turns over) -> refresh, not a first fetch
    monkeypatch.setattr(_sources, "freshness", lambda s, ds, v: "TOKEN-2")
    Annotations(FakeDS())._table(src, "v1", verbose=True)
    assert "refreshing the local cache" in capsys.readouterr().out


def test_a_frozen_source_reads_from_local_cache_without_claiming_validation(cache_root, capsys):
    """A `github_tsv` is frozen at its release (freshness None), so the feedback must
    say "reading from local cache" - not "re-validated", which would imply a live
    check that never happened."""
    import connecto.sources as _sources  # noqa: F401 - real freshness returns None here
    from connecto.core.namespaces import Annotations
    from connecto.core.spec import AnnotationSource

    src = AnnotationSource("public", "github_tsv", "http://x/y.tsv")  # freshness -> None

    class FakeDS:
        name = "flywire"
        label = "FlyWire (FAFB) public release"

        def _fetch_annotations(self, source, version):
            return pd.DataFrame({"root_id": [1, 2, 3]})

    Annotations(FakeDS())._table(src, "783", verbose=True)  # cold -> download
    capsys.readouterr()
    Annotations(FakeDS())._table(src, "783", verbose=True)  # warm -> local cache
    out = capsys.readouterr().out
    assert "reading from local cache" in out and "re-validated" not in out


# --- freshness / invalidation -------------------------------------------------


def test_freshness_is_none_for_materialization_scoped_sources():
    """A CAVE table is frozen at its materialization and FlyWire's public TSV is a
    per-release artefact - both are already pinned by the version in the key, so they
    need no token. (SeaTable and neuPrint are live and *do* get one - network tests.)"""
    from connecto.core.spec import AnnotationSource
    from connecto.sources import freshness

    cave = AnnotationSource("cave", "cave_table", "t", id_column="pt_root_id")
    gh = AnnotationSource("public", "github_tsv", "http://x/y.tsv")
    assert freshness(cave, object(), "v1") is None
    assert freshness(gh, object(), "v1") is None


def test_supersede_drops_its_own_family_but_no_other_kind(cache_root):
    """When a live source's freshness token turns over, the new entry supersedes the
    old - but it must not touch a *different* kind. The connectivity cache keeps many
    `edges` entries under one kind, and sweeping those would be a data-loss bug."""
    old = _cache.CacheEntry("ds", "v1", "annotations_flytable", "loc", "OLD")
    old.write(pd.DataFrame({"x": [1]}))
    edges = _cache.CacheEntry("ds", "v1", "edges", b"query-A")
    edges.write(pd.DataFrame({"pre": [1]}))

    new = _cache.CacheEntry("ds", "v1", "annotations_flytable", "loc", "NEW")
    new.write(pd.DataFrame({"x": [2]}))
    new.supersede()

    assert not old.exists()  # the previous freshness token's entry is gone
    assert new.exists()      # the current one stays
    assert edges.exists()    # a different kind is untouched


@pytest.mark.network
def test_a_live_seatable_source_gets_a_stable_freshness_token():
    """aedes' FlyTable is a live curation DB with no version of its own; its token is
    `COUNT(*):MAX(_mtime)`, stable while the table is, and part of the cache key."""
    import connecto as cn
    from connecto.sources import freshness

    a = cn.get_dataset("aedes")
    src = a._annotation_source
    tok1 = freshness(src, a, a.version)
    tok2 = freshness(src, a, a.version)
    assert tok1 and tok1 == tok2

    keyed = _cache.CacheEntry(a.name, a.version, "annotations_flytable", src.location, tok1)
    plain = _cache.CacheEntry(a.name, a.version, "annotations_flytable", src.location, None)
    assert keyed.path != plain.path  # the token genuinely turns the key over


@pytest.mark.network
def test_a_live_neuprint_source_gets_a_freshness_token():
    """maleCNS is re-curated under the same version tag; `lastDatabaseEdit` catches it."""
    import connecto as cn
    from connecto.sources import freshness

    m = cn.get_dataset("malecns")
    assert freshness(m.spec.annotation_source("neuprint"), m, m.version)
