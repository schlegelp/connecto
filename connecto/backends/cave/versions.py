"""Materialization versions.

CAVE root IDs change every time a neuron is edited, so a set of IDs is only
*jointly* valid in some materializations. ``find_version`` scans them newest-first
and returns the first in which all the given IDs exist.

This is the best idea in fafbseg (``find_mat_version``); caveclient has no
equivalent. Here it is promoted to a first-class concept so that
``version="auto"`` is available - and correct - on every CAVE dataset.
"""

from __future__ import annotations

import logging

import numpy as np

from ...core.version import Version
from ...exceptions import AmbiguousVersionError, NoSuchVersionError

logger = logging.getLogger("connecto")

__all__ = ["find_version", "resolve"]


def _versions_metadata(client) -> dict[int, dict]:
    return {m["version"]: m for m in client.materialize.get_versions_metadata()}


def resolve(client, request, *, backend="cave") -> Version:
    """Turn a version request into a concrete :class:`Version`."""
    from datetime import datetime

    if isinstance(request, Version):
        return request

    if request == "live":
        return Version("live", backend, timestamp=None)

    meta = _versions_metadata(client)

    if request in (None, "latest"):
        value = max(meta)
    elif isinstance(request, datetime):
        # Nearest materialization at or before the timestamp.
        candidates = [
            v for v, m in meta.items()
            if m.get("time_stamp") and m["time_stamp"] <= request
        ]
        if not candidates:
            raise NoSuchVersionError(f"No materialization at or before {request}.")
        value = max(candidates)
    elif isinstance(request, (int, np.integer)):
        value = int(request)
        if value not in meta:
            raise NoSuchVersionError(
                f"Materialization {value} does not exist. "
                f"Available: {sorted(meta)}."
            )
    else:
        raise NoSuchVersionError(
            f"CAVE versions are integers, 'latest', 'live' or 'auto' - got {request!r}."
        )

    m = meta[value]
    return Version(
        value,
        backend,
        timestamp=m.get("time_stamp"),
        expires=m.get("expires_on"),
    )


def find_version(client, ids: np.ndarray, *, raise_missing: bool = True, backend="cave") -> Version:
    """The newest materialization in which *all* ``ids`` are valid root IDs."""
    ids = np.unique(np.asarray(ids, dtype="int64"))
    if not len(ids):
        return resolve(client, "latest", backend=backend)

    meta = _versions_metadata(client)
    for value in sorted(meta, reverse=True):
        ts = client.materialize.get_timestamp(value)
        if np.all(client.chunkedgraph.is_latest_roots(ids, timestamp=ts)):
            logger.info("Using materialization version %s.", value)
            m = meta[value]
            return Version(value, backend, timestamp=m.get("time_stamp"),
                           expires=m.get("expires_on"))

    # Nothing worked. Is it that they're current-but-unmaterialized, or bogus?
    if np.all(client.chunkedgraph.is_latest_roots(ids, timestamp=None)):
        logger.info("IDs are current but not yet materialized; using live query.")
        return Version("live", backend, timestamp=None)

    if not raise_missing:
        return resolve(client, "latest", backend=backend)

    stale = ids[~client.chunkedgraph.is_latest_roots(ids, timestamp=None)]
    raise AmbiguousVersionError(
        f"No materialization contains all {len(ids)} IDs, and "
        f"{len(stale)} of them are not current either "
        f"(e.g. {stale[:3].tolist()}). They may never have co-existed. "
        f"Update them with `ds.segmentation.update_ids(...)`, or pin an explicit "
        f"version."
    )
