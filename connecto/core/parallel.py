"""Fanning one call out over threads, in order.

Three places fetch N things that are each one request and none of which depend on
each other - meshes in :func:`connecto.core.volume.fetch_meshes`, and skeletons in
each backend's ``_fetch_skeletons``. They had grown the same six lines three times:
clamp the worker count to the work, map over a pool, and hand results back paired
with what produced them.

The part worth having in one place is the *ordering guarantee*. Threads finish out
of order; callers here pair each result with the id yielded beside it, so a
generator that let results overtake each other would label a skeleton with the wrong
neuron - wrong in a way that plots perfectly well. ``pool.map`` gives back input
order, and stating that once is better than three call sites each being separately
right about it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

__all__ = ["DEFAULT_SKELETON_WORKERS", "map_ordered"]

#: Neurons in flight in each backend's ``_fetch_skeletons``.
#:
#: A skeleton is one request per neuron - a read from a published bucket, a call to
#: CAVE's skeleton service, or the three CAVE calls an L2 skeleton needs - so this is
#: the whole of the parallelism available, and it matters more here than anywhere
#: else in connecto. Measured over 16 neurons:
#:
#: * precomputed bucket (FlyWire): 2.2 s serial, 0.22 s at 8 workers.
#: * CAVE skeleton service (MICrONS): 52.6 s serial, 5.3 s at 8.
#: * neuPrint store (hemibrain): 5.6 s serial, 1.3 s at 8.
#:
#: 8 rather than more because all three curves stop there and two of them turn back
#: up at 16 (0.27 s and 1.5 s) - and unlike mesh fragments, which come off a public
#: bucket, two of these three routes are somebody's query service.
#:
#: Not in :mod:`connecto.precomputed.limits` with the mesh budgets, because it is not
#: coupled to them: those nest inside one another and their product has to stay under
#: a connection pool, while this one is a flat fan-out over three different services,
#: only one of which is a precomputed bucket at all.
DEFAULT_SKELETON_WORKERS = 8


def map_ordered(
    items: Iterable,
    fn: Callable[[Any], Any],
    *,
    workers: int,
    desc: str,
    progress: bool = True,
):
    """Yield ``(item, fn(item))`` for every item, in the order they were given.

    ``workers`` is an upper bound, not a promise: a pool is never made wider than
    the work there is to do. The progress bar stays hidden for a single item, where
    it would be noise.
    """
    from tqdm.auto import tqdm

    items = list(items)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items) or 1))) as pool:
        yield from tqdm(
            zip(items, pool.map(fn, items)),
            desc=desc,
            total=len(items),
            disable=not progress or len(items) < 2,
            leave=False,
        )
