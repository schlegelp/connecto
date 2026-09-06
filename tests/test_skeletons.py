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
    """The failure mode that threading `fetch_skeletons` introduced.

    The patch swaps a method on a client every worker shares. Save-patch-restore per
    thread loses that race in the worst possible way: the second thread in saves the
    *first* thread's stub as "the original" and restores that on the way out, so the
    stub stays installed for the life of the client and every later caller gets
    `None` for a volume it may genuinely need - a wrong answer that raises nothing.

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
    delays.update({10: 0.20, 20: 0.15, 30: 0.10})

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
        cave_skeletons.fetch_skeletons(ds, range(8), None, progress=False, workers=8)
    )

    assert [root for root, _ in got] == list(range(8))
    assert not barrier.broken


def test_cave_skeletons_workers_is_honoured(monkeypatch):
    """`workers=1` is the escape hatch, so it has to really mean one."""
    threads = set()

    def fake_skeleton(source, root):
        threads.add(threading.current_thread().ident)
        time.sleep(0.02)
        return _nodes(root)

    monkeypatch.setattr(cave_skeletons, "precomputed_skeleton", fake_skeleton)
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket/skeletons")

    list(cave_skeletons.fetch_skeletons(ds, range(6), None, progress=False, workers=1))
    assert len(threads) == 1


# ------------------------------------------------------- the neuPrint fan-out

_USED_CLIENTS: list[int] = []

#: Module-level so the deep copies handed to each thread all reach the *same* one -
#: an attribute on the client would be copied along with it, one barrier per thread,
#: and nothing would ever wait.
_BARRIER: threading.Barrier | None = None


class _FakeNeuPrintClient:
    """Records which client instance served each call, across deep copies."""

    def fetch_skeleton(self, body, heal=True, format="pandas"):
        _USED_CLIENTS.append(id(self))
        if _BARRIER is not None:
            _BARRIER.wait()
        return _nodes(body)


def test_neuprint_skeletons_give_each_thread_its_own_client():
    """neuprint-python keeps per-thread copies of its own client; so must we.

    connecto passes its client explicitly, which bypasses neuprint's
    `DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES` machinery entirely - so if this door did
    not copy, several threads would share one `Client` and one `requests.Session`,
    which is the arrangement neuprint itself goes out of its way to avoid.

    All eight are held at a barrier so that eight threads really are live at once -
    otherwise a fast worker can drain the whole queue before its colleagues start,
    one copy serves everything, and "each thread got its own" passes or fails on
    scheduling luck.
    """
    global _BARRIER
    from connecto.backends.neuprint.dataset import NeuPrintDataset

    _USED_CLIENTS.clear()
    _BARRIER = threading.Barrier(8, timeout=10)
    client = _FakeNeuPrintClient()
    ds = SimpleNamespace(_skeleton_source=lambda version: None, client=client)

    bodies = list(range(8))
    try:
        got = list(
            NeuPrintDataset._fetch_skeletons(ds, bodies, None, progress=False, workers=8)
        )
    finally:
        _BARRIER = None

    assert [body for body, _ in got] == bodies
    # The client connecto holds must never be the one that serves a query...
    assert id(client) not in _USED_CLIENTS, "the shared client served a query"
    # ...and with eight threads provably live, there must be eight separate copies.
    assert len(set(_USED_CLIENTS)) == 8


def test_neuprint_skeletons_prefer_a_published_bucket(monkeypatch):
    """A precomputed bucket wins, and then no neuPrint client is touched at all."""
    from connecto.backends.neuprint import dataset as np_dataset
    from connecto.core import volume as core_volume

    _USED_CLIENTS.clear()
    monkeypatch.setattr(core_volume, "precomputed_skeleton", lambda src, body: _nodes(body))

    client = _FakeNeuPrintClient()
    ds = SimpleNamespace(_skeleton_source=lambda version: "https://bucket", client=client)

    got = list(np_dataset.NeuPrintDataset._fetch_skeletons(ds, [1, 2, 3], None, progress=False))

    assert [body for body, _ in got] == [1, 2, 3]
    assert _USED_CLIENTS == []
