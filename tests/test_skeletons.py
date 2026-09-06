"""Fetching skeletons, and the two things threading them put at risk.

Offline. Every route is one request per neuron, so the fan-out is the whole of the
speed-up and there is nothing to test about it that needs a live server - what needs
testing is that going wide did not change *what* comes back, or the order it comes
back in, and that the shared state the CAVE route mutates survives being touched by
several threads at once.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from connecto.backends.cave import skeletons as cave_skeletons

# ------------------------------------------------------------------ the CAVE patch


class _Info:
    """Stands in for `client.info`, whose method the patch swaps out."""

    def segmentation_cloudvolume(self, *a, **kw):
        return "the real volume"


def _ds_with_info():
    return SimpleNamespace(client=SimpleNamespace(info=_Info()))


def test_cloudvolume_patch_restores_the_original():
    """Restored to the arrangement we were handed, not merely to something equivalent.

    `info.segmentation_cloudvolume` is a class method here, so assigning the bound
    method back would leave an instance attribute shadowing it - working, but a
    change to an object connecto does not own. `__dict__` is the assertion.
    """
    ds = _ds_with_info()

    with cave_skeletons._without_cloudvolume_root_check(ds):
        assert ds.client.info.segmentation_cloudvolume() is None

    assert ds.client.info.segmentation_cloudvolume() == "the real volume"
    assert ds.client.info.__dict__ == {}


def test_cloudvolume_patch_restores_after_an_error():
    ds = _ds_with_info()

    with pytest.raises(RuntimeError), cave_skeletons._without_cloudvolume_root_check(ds):
        raise RuntimeError("the skeleton service fell over")

    assert ds.client.info.segmentation_cloudvolume() == "the real volume"
    assert ds.client.info.__dict__ == {}


def test_cloudvolume_patch_survives_concurrent_use():
    """The failure mode threading introduced - see `_without_cloudvolume_root_check`.

    Overlap is forced rather than hoped for: every thread waits on the same barrier
    while inside the patch, so all of them are holding it at once.
    """
    ds = _ds_with_info()
    barrier = threading.Barrier(8)
    seen = []

    def use_it(_):
        with cave_skeletons._without_cloudvolume_root_check(ds):
            barrier.wait(timeout=10)
            seen.append(ds.client.info.segmentation_cloudvolume())

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(use_it, range(8)))

    # Every thread saw the stub while inside...
    assert seen == [None] * 8
    # ...and the client is left exactly as it was found, with no bookkeeping behind.
    assert ds.client.info.segmentation_cloudvolume() == "the real volume"
    assert ds.client.info.__dict__ == {}


# --------------------------------------------------------------- the CAVE fan-out


def _nodes(root: int) -> pd.DataFrame:
    return pd.DataFrame({"node_id": [0], "parent_id": [-1], "x": [float(root)],
                         "y": [0.0], "z": [0.0], "radius": [1.0]})


def test_cave_skeletons_keep_the_order_they_were_asked_for(monkeypatch):
    """Threads finish out of order; the results must not.

    `Skeletons.get` pairs each node table with the id it yielded alongside, so a
    generator that let results overtake each other would hand back node tables
    labelled with the wrong neuron - the sort of wrong that plots fine.
    """
    delays = {}

    def fake_skeleton(source, root):
        # Later ids finish first, so anything relying on completion order is caught.
        time.sleep(delays.get(root, 0))
        return _nodes(root)

    monkeypatch.setattr(cave_skeletons, "precomputed_skeleton", fake_skeleton)
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket/skeletons")

    ids = [10, 20, 30, 40, 50, 60]
    # Descending, so the ids asked for first are the ones that finish last.
    delays.update({10: 0.02, 20: 0.015, 30: 0.01})

    got = list(cave_skeletons.fetch_skeletons(ds, ids, None, progress=False))

    assert [root for root, _ in got] == ids
    for root, nodes in got:
        assert nodes["x"].iloc[0] == float(root)


def test_cave_skeletons_actually_run_concurrently(monkeypatch):
    """Eight neurons are genuinely in flight at once.

    A barrier rather than a stopwatch. "This finished faster than serial would have"
    needs a margin, and a margin on a loaded machine is a test that fails for
    reasons that have nothing to do with connecto - which this one did, roughly one
    run in five, before it was written this way. Here the eight calls can only get
    past the barrier if all eight are inside it simultaneously; anything less
    deadlocks and `BrokenBarrierError` says so. Nothing to tune, and no way for it
    to pass by luck.
    """
    barrier = threading.Barrier(8, timeout=10)

    def fake_skeleton(source, root):
        barrier.wait()
        return _nodes(root)

    monkeypatch.setattr(cave_skeletons, "precomputed_skeleton", fake_skeleton)
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket/skeletons")

    got = list(
        cave_skeletons.fetch_skeletons(ds, range(8), None, progress=False, max_workers=8)
    )

    assert [root for root, _ in got] == list(range(8))
    assert not barrier.broken


def test_cave_skeletons_max_workers_is_honoured(monkeypatch):
    """`max_workers=1` is the escape hatch, so it has to really mean one.

    No sleep: with one worker there is exactly one thread by construction, so the
    set of thread ids says everything and waiting would only slow the suite.
    """
    threads = set()

    def fake_skeleton(source, root):
        threads.add(threading.current_thread().ident)
        return _nodes(root)

    monkeypatch.setattr(cave_skeletons, "precomputed_skeleton", fake_skeleton)
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket/skeletons")

    list(cave_skeletons.fetch_skeletons(ds, range(6), None, progress=False, max_workers=1))
    assert len(threads) == 1


def test_cave_skeletons_refuse_an_unknown_keyword():
    """A misspelled option must not run silently at the default.

    `**opts` used to swallow it: `skeletons.get(x, wrokers=2)` fetched eight at a
    time and said nothing.
    """
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket/skeletons")

    with pytest.raises(TypeError, match="wrokers"):
        list(cave_skeletons.fetch_skeletons(ds, [1], None, wrokers=2))


# ------------------------------------------------------- the neuPrint fan-out


def _neuprint_ds(source, client):
    """The three methods under test, given the two attributes they need.

    Borrowing the real methods onto a bare class rather than building a
    `NeuPrintDataset` keeps this offline - a real one wants a live server - while
    still exercising the actual pooling and copying, not a paraphrase of it.
    """
    from connecto.backends.neuprint.dataset import NeuPrintDataset

    class _DS:
        _skeleton_clients = NeuPrintDataset._skeleton_clients
        _borrowed_client = NeuPrintDataset._borrowed_client
        _fetch_skeletons = NeuPrintDataset._fetch_skeletons

        def _skeleton_source(self, version):
            return source

    ds = _DS()
    ds.client = client
    return ds


def test_neuprint_skeletons_give_each_thread_its_own_client():
    """neuprint-python keeps per-thread copies of its own client; so must we.

    connecto passes its client explicitly, which bypasses neuprint's
    `DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES` machinery entirely - so if this door did
    not copy, several threads would share one `Client` and one `requests.Session`,
    which is the arrangement neuprint itself goes out of its way to avoid.

    All eight are held at a barrier so eight threads really are live at once;
    otherwise a fast worker can drain the queue before its colleagues start, one
    copy serves everything, and this passes or fails on scheduling luck.

    The fake is defined in the test body on purpose: `deepcopy` treats a class as
    atomic, so every copy shares these closure cells and reports into the same list.
    """
    used = []
    barrier = threading.Barrier(8, timeout=10)

    class FakeClient:
        def fetch_skeleton(self, body, heal=True, format="pandas"):
            used.append(id(self))
            barrier.wait()
            return _nodes(body)

    client = FakeClient()
    ds = _neuprint_ds(None, client)

    bodies = list(range(8))
    got = list(ds._fetch_skeletons(bodies, None, progress=False, max_workers=8))

    assert [body for body, _ in got] == bodies
    # The client connecto holds must never be the one that serves a query...
    assert id(client) not in used, "the shared client served a query"
    # ...and with eight threads provably live, there must be eight separate copies.
    assert len(set(used)) == 8


def test_neuprint_skeleton_clients_are_reused_across_calls():
    """Copies are pooled, not remade - a fresh one has no warmed connections.

    `requests` drops its pool manager on pickle, so a copied `Client` reconnects
    from scratch: 645 ms against 114 ms once warm, measured against hemibrain. Made
    per call, that handshake was most of what a small fetch cost.
    """
    used = []

    class FakeClient:
        def fetch_skeleton(self, body, heal=True, format="pandas"):
            used.append(id(self))
            return _nodes(body)

    ds = _neuprint_ds(None, FakeClient())

    list(ds._fetch_skeletons(range(4), None, progress=False, max_workers=2))
    first = set(used)
    list(ds._fetch_skeletons(range(4), None, progress=False, max_workers=2))

    assert set(used) == first, "the second call built fresh clients"
    assert ds._skeleton_clients.qsize() == len(first)


def test_neuprint_skeletons_prefer_a_published_bucket(monkeypatch):
    """A precomputed bucket wins, and then no neuPrint client is touched at all."""
    from connecto.core import volume as core_volume

    used = []

    class FakeClient:
        def fetch_skeleton(self, body, heal=True, format="pandas"):
            used.append(id(self))
            return _nodes(body)

    monkeypatch.setattr(core_volume, "precomputed_skeleton", lambda src, body: _nodes(body))
    ds = _neuprint_ds("https://bucket", FakeClient())

    got = list(ds._fetch_skeletons([1, 2, 3], None, progress=False))

    assert [body for body, _ in got] == [1, 2, 3]
    assert used == []


def test_neuprint_skeletons_refuse_an_unknown_keyword():
    """Same as the CAVE door: a misspelled option raises rather than being dropped."""
    ds = _neuprint_ds(None, object())

    with pytest.raises(TypeError, match="wrokers"):
        list(ds._fetch_skeletons([1], None, wrokers=2))


# ---------------------------------------------------------------- the L2 namespace


def test_l2_split_preserves_missing_attributes_as_null():
    """A missing L2 attribute must stay missing, not become uninitialised memory.

    caveclient's own `split_columns=True` fills an absent attribute with `np.empty`
    rather than NaN, so a chunk the cache has no axis or no position for arrives as
    whatever bytes were lying around - usually zeros, which is why it looked
    harmless, sometimes 3e-30, and never the same twice. connecto asks for the
    unsplit frame and does the flattening itself so that null stays null; downstream
    then reads missingness off `.notna()` instead of guessing from the values.
    """
    from connecto.backends.cave.skeletons import _split_vector_columns

    raw = pd.DataFrame(
        {
            "size_nm3": [10.0, 20.0, 30.0],
            "rep_coord_nm": [[1, 2, 3], [4, 5, 6], np.nan],
            "pca": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.nan, np.nan],
        },
        index=pd.Index([11, 22, 33], name="l2_id"),
    )

    out = _split_vector_columns(raw)

    # Scalars pass through; vectors become one column per component.
    assert list(out["size_nm3"]) == [10.0, 20.0, 30.0]
    assert list(out["rep_coord_nm_x"][:2]) == [1.0, 4.0]
    assert out["pca_0_x"].iloc[0] == 1.0

    # And the rows the cache had nothing for are null - not zero, not garbage.
    assert out["rep_coord_nm_x"].isna().tolist() == [False, False, True]
    assert out["pca_0_x"].isna().tolist() == [False, True, True]
    # The unsplit columns are gone, so nothing can read the raw lists by accident.
    assert "pca" not in out.columns and "rep_coord_nm" not in out.columns


def test_l2_info_asks_for_the_unsplit_frame():
    """It must be connecto that flattens, not caveclient.

    The fake below honours `split_columns` the way caveclient does: asked to split,
    it fills a missing attribute with a number (uninitialised memory in the real
    thing, zeros here); asked not to, it hands back nulls. So a NaN in the result is
    proof `l2_info` took the second route. Splitting the already-split frame would
    be a no-op and the garbage would flow straight through, which is exactly the
    edit this guards against.
    """
    from connecto.backends.cave.skeletons import l2_info

    index = pd.Index([11, 22], name="l2_id")

    def get_l2data_table(l2_ids, attributes=None, split_columns=True):
        if split_columns:  # caveclient's lossy branch
            return pd.DataFrame(
                {"pca_0_x": [1.0, 0.0], "rep_coord_nm_x": [1.0, 0.0]}, index=index
            )
        return pd.DataFrame(
            {"pca": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.nan],
             "rep_coord_nm": [[1, 2, 3], np.nan]},
            index=index,
        )

    ds = SimpleNamespace(
        client=SimpleNamespace(
            chunkedgraph=SimpleNamespace(get_leaves=lambda root, stop_layer: np.array([11, 22])),
            l2cache=SimpleNamespace(get_l2data_table=get_l2data_table),
        )
    )

    got = l2_info(ds, 1, attributes=("rep_coord_nm", "pca"))
    assert got["pca_0_x"].isna().tolist() == [False, True]
    assert got["rep_coord_nm_x"].isna().tolist() == [False, True]

    # And `dropna` then actually removes the positionless chunk, which it could not
    # do while a missing position arrived as a number.
    assert len(l2_info(ds, 1, attributes=("rep_coord_nm", "pca"), dropna=True)) == 1


def test_l2_info_honours_max_workers(monkeypatch):
    """The knob has to reach the pool, not just appear in the signature.

    A signature check would pass on a method that accepted `max_workers` and dropped
    it; this counts the threads that actually ran.
    """
    from connecto.backends.cave import l2 as l2_mod
    from connecto.core.spec import Cap

    threads = set()

    def fake_info(_ds, root, **kw):
        threads.add(threading.current_thread().ident)
        time.sleep(0.01)  # long enough that a real pool overlaps
        return pd.DataFrame(
            {"rep_coord_nm_x": [1.0], "rep_coord_nm_y": [2.0], "rep_coord_nm_z": [3.0]},
            index=pd.Index([root], name="l2_id"),
        )

    monkeypatch.setattr(l2_mod, "l2_info", fake_info)
    # Enough of a dataset for the capability gate in `requires` to let the call
    # through; the fan-out is what is under test.
    ds = SimpleNamespace(
        capabilities=frozenset({Cap.L2CACHE}),
        backend_kind="cave",
        _auth_server=None,
        source="fake",
        label="fake",
        ids=lambda x, version=None: list(x),
    )

    l2_mod.L2(ds).info(list(range(6)), progress=False, max_workers=1)
    assert len(threads) == 1

    threads.clear()
    l2_mod.L2(ds).info(list(range(6)), progress=False, max_workers=4)
    assert len(threads) > 1


# ------------------------------------------------------- the CAVE session pool


def test_cave_session_pool_is_raised_without_clobbering_other_defaults(monkeypatch):
    """Enough retained connections for the fan-out, and nothing else disturbed.

    caveclient gives each sub-client its own session capped at 10-20 connections,
    which a per-neuron fan-out overruns immediately: urllib3 then discards the
    connection it cannot keep and re-handshakes on the next read. Measured on aedes,
    `l2.skeleton(max_workers=32)` logged 28 "Connection pool is full" warnings at 20
    and none at 128.

    The second half of the assertion is the fiddly bit: `set_session_defaults`
    assigns *every* field from its arguments, so passing only `pool_maxsize` would
    silently reset a caller's retry policy and backoff to caveclient's own defaults.
    """
    import inspect

    import caveclient

    from connecto.backends.cave import dataset as cave_ds
    from connecto.core.parallel import SESSION_POOL_MAXSIZE

    state = {
        "max_retries": 5, "pool_block": False, "pool_maxsize": 20,
        "backoff_factor": 0.2, "backoff_max": 120, "status_forcelist": (502, 503),
    }

    # The stand-in is built from the real signature rather than hand-written, so it
    # reproduces the part that matters: an argument left out is not left alone, it is
    # reset to caveclient's default. A forgiving fake here would pass a naive
    # `set_session_defaults(pool_maxsize=...)`, which is the whole thing under test.
    signature = inspect.signature(caveclient.set_session_defaults)

    def set_defaults(**kwargs):
        bound = signature.bind(**kwargs)
        bound.apply_defaults()
        state.clear()
        state.update(bound.arguments)

    monkeypatch.setattr(caveclient, "get_session_defaults", lambda: dict(state))
    monkeypatch.setattr(caveclient, "set_session_defaults", set_defaults)

    cave_ds._widen_session_pool()

    assert state["pool_maxsize"] == SESSION_POOL_MAXSIZE
    assert state["max_retries"] == 5, "clobbered the caller's retry policy"
    assert state["backoff_factor"] == 0.2, "clobbered the caller's backoff"
    assert state["status_forcelist"] == (502, 503)

    # A caller who already asked for more keeps it - this raises, never lowers.
    state["pool_maxsize"] = SESSION_POOL_MAXSIZE * 2
    cave_ds._widen_session_pool()
    assert state["pool_maxsize"] == SESSION_POOL_MAXSIZE * 2
