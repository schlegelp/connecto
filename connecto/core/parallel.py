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

__all__ = ["DEFAULT_NEURON_WORKERS", "SESSION_POOL_MAXSIZE", "map_ordered"]

#: Neurons in flight when a call is one request per neuron.
#:
#: Skeletons in each backend's ``_fetch_skeletons``, and every method of the
#: :class:`~connecto.backends.cave.l2.L2` namespace. In all of them a neuron costs a
#: round trip and nothing else - a read from a published bucket, a call to CAVE's
#: skeleton service, or the chunkedgraph and l2cache calls behind an L2 query - so
#: this is the whole of the parallelism available, and it matters more here than
#: anywhere else in connecto. Measured over 16 neurons:
#:
#: * precomputed bucket (FlyWire): 2.2 s serial, 0.22 s at 8 workers.
#: * CAVE skeleton service (MICrONS): 52.6 s serial, 5.3 s at 8.
#: * neuPrint store (hemibrain): 5.6 s serial, 1.3 s at 8.
#:
#: and over 8 MICrONS neurons through the L2 cache: 36.8 s serial, 8.2 s at 8.
#:
#: 8 rather than more because every one of those curves stops there - L2 gains 3%
#: going to 16, and two of the skeleton routes turn back up - and unlike mesh
#: fragments, which come off a public bucket, most of these routes are somebody's
#: query service.
#:
#: Not quite a flat fan-out, and the one route that isn't should be read off here
#: rather than discovered: ``l2.skeleton`` and the L2 fallback in ``fetch_skeletons``
#: put two requests per neuron on the wire, because a chunk graph and its chunk
#: attributes are independent and go out together. So that route runs at twice this
#: number - 16 CAVE calls at the default, which is measured and fine (16 wide costs
#: the same as 8 wide, and the overlap is worth 1.4x on top).
#:
#: Not in :mod:`connecto.precomputed.limits` with the mesh budgets, because it is not
#: coupled to them: those nest to keep a *connection pool* from overflowing, and
#: `POOL_MAXSIZE` is derived from their product. These routes go to query services
#: over their own clients' sessions, and want a number chosen from what the service
#: will answer quickly, not from how many sockets are retained.
DEFAULT_NEURON_WORKERS = 8


#: Connections a client's session may retain per host, where connecto gets to say.
#:
#: A ceiling on *retained idle* connections, not on concurrency: urllib3 opens them
#: lazily, so headroom costs nothing until it is used. Running out is what costs -
#: the pool discards the connection it cannot keep, logs "Connection pool is full",
#: and the next read pays a fresh TLS handshake, so the fan-out is charged for and
#: then spent on setup.
#:
#: 128 because the routes above put up to two requests per neuron on one host, so
#: the shipped default needs 16 and a caller tuning ``max_workers=64`` needs all of
#: it. See ``connecto.precomputed.limits.POOL_MAXSIZE`` for the same argument on
#: connecto's own session, which is sized larger because mesh fragments fan out
#: wider still.
SESSION_POOL_MAXSIZE = 128


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
