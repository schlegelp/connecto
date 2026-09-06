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
