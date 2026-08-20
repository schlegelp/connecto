"""Annotations from a CAVE table.

Note the client here comes from the *spec's* CAVE backend, not from whichever
backend is currently answering queries. That is not a technicality - it is the
orthogonality claim made good.

BANC is served by CAVE (materialization 888) and by neuPrint (``banc:v888``), and
those are the same snapshot: its neuPrint body IDs are valid CAVE root IDs. But
the rich annotations (32 classification systems) live only in CAVE's
``codex_annotations``. So ``BANC(backend="neuprint")`` reads its edges from
neuPrint and its annotations from CAVE, and neither knows about the other.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("connecto")

__all__ = ["fetch_cave_table", "cave_context"]

CHUNK_SIZE = 100_000


def cave_context(ds):
    """(CAVEclient, version-kwargs) for this dataset's CAVE backend.

    Note this borrows a whole CAVE *handle*, not just a client, so the version it
    reads at is resolved by the same rules a first-class CAVE handle uses - rather
    than the "newest materialization" this used to guess at, which was a third
    spelling of a decision `CAVEDataset._resolve_version` already owns.
    """
    from . import borrow

    cave = borrow(ds, "cave")
    return cave.client, cave._mat_kwargs(cave.version)


def fetch_cave_table(source, ds, version) -> pd.DataFrame:
    client, kwargs = cave_context(ds)

    if not source.chunked:
        return client.materialize.query_table(source.location, **kwargs)

    # Some tables (BANC's codex_annotations) reliably fail to come back in one go.
    n = client.materialize.get_annotation_count(
        source.location,
        version=kwargs.get("materialization_version"),
    )
    logger.info("Fetching %s (%s rows) in chunks of %s...", source.location, n, CHUNK_SIZE)

    frames = []
    for offset in range(0, n, CHUNK_SIZE):
        chunk = client.materialize.query_table(
            source.location, offset=offset, limit=CHUNK_SIZE, **kwargs
        )
        frames.append(chunk)
        if len(chunk) < CHUNK_SIZE:
            break

    return pd.concat(frames, ignore_index=True)
