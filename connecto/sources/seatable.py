"""Annotations from SeaTable ("flytable").

Optional and lab-internal: ``pip install connecto[flytable]``, plus
``SEATABLE_SERVER`` / ``SEATABLE_TOKEN``.

**Instances.** SeaTable is not one server. The lab runs its own deployment
("flytable", the default, at ``SEATABLE_SERVER``), and there is the official
``cloud.seatable.io`` ("seatable") - and a base named on one does not exist on the
other. A source picks its deployment with ``AnnotationSource(..., instance=...)``,
which :data:`_INSTANCES` maps to a server URL. FlyWire and aedes are on flytable;
BANC's ``banc_meta`` is on the cloud.

**Location grammar.** A comma-separated list of ``base.table``. Naming the base is
not optional decoration: without it, seaserpent has to search *every* base the
token can see to find the table (~20s for FlyWire's ``info``); with it, resolution
is ~1s. So ``aedes.aedes_main`` and ``banc_meta.banc_meta`` name one table each, and
FlyWire names *two* -

    main.info,optic_lobes.optic

- because FlyWire's live annotations are genuinely two SeaTable tables in two
different bases (the central brain in ``main.info``, the optic lobes in
``optic_lobes.optic``), concatenated. This mirrors exactly how cocoa assembles
them; reading only ``info`` silently drops ~89k optic-lobe neurons.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

from ..auth import missing_token
from ..exceptions import MissingDependencyError

__all__ = ["fetch_seatable", "seatable_freshness"]


# SeaTable deployment name -> server URL. ``None`` means "the SEATABLE_SERVER env
# default", which is how the lab's own flytable instance has always been reached, so
# nothing about it changes. A new deployment is one more line here.
_INSTANCES: dict[str, str | None] = {
    "flytable": None,                          # the lab's own instance (SEATABLE_SERVER)
    "seatable": "https://cloud.seatable.io/",   # the official SeaTable cloud
}


# Authenticated base handles, memoised by (server, base) for the life of the process.
# Resolving a base (find_base + auth) is the ~1s cost; the tables within it are then
# cheap, and both the freshness probe and the fetch want the same base. The server is
# part of the key because the same base name can exist on two deployments.
_BASES: dict = {}


def _server(source) -> str | None:
    """The server URL for a source's SeaTable instance, or None for the env default."""
    instance = getattr(source, "instance", "flytable")
    if instance not in _INSTANCES:
        raise ValueError(
            f"Unknown SeaTable instance {instance!r} on source {source.name!r}. "
            f"Known: {', '.join(_INSTANCES)}."
        )
    return _INSTANCES[instance]


def _table_specs(source) -> list[tuple[str | None, str]]:
    """``[(base, table), ...]`` from a ``base.table[,base.table...]`` location.

    A bare ``table`` (no base) yields ``(None, table)`` - seaserpent then has to
    search for it, which is slow; every connecto source names its base.
    """
    specs = []
    for part in source.location.split(","):
        base, _, table = part.strip().partition(".")
        specs.append((base, table) if table else (None, base))
    return specs


def _require_seaserpent():
    try:
        import seaserpent as ss
    except ModuleNotFoundError as e:
        raise MissingDependencyError.for_extra(
            "sea-serpent", "flytable", "Fetching flytable annotations"
        ) from e
    return ss


def _base(server: str | None, base_name: str | None, table: str):
    """An authenticated base handle, *without* the table-metadata pull.

    ``ss.Table(...)`` resolves the base *and* downloads the table's full column
    metadata on construction; a freshness probe needs neither - only
    ``base.query`` - so this skips it. The handle is memoised by (server, base)
    because resolving it is the expensive part, and a bare (base-less) location is
    never memoised, since ``None`` is not a base anyone can look up.
    """
    memo_key = (server, base_name)
    if base_name and memo_key in _BASES:
        return _BASES[memo_key]
    ss = _require_seaserpent()
    find_kw = {"base": base_name, "required_table": table}
    if server:
        find_kw["server"] = server
    ws, resolved, auth_token, base_token, srv = ss.find_base(**find_kw)
    if base_token:
        b = ss.SeaTableAPI(base_token, srv)
        b.auth()
    else:
        account = ss.Account(None, None, srv)
        account.token = auth_token
        b = account.get_base(ws, resolved)
    if base_name:
        _BASES[memo_key] = b
    return b


def fetch_seatable(source, ds, version) -> pd.DataFrame:
    ss = _require_seaserpent()

    if not os.environ.get("SEATABLE_TOKEN"):
        # One credential exception in the library, not a bespoke one per source.
        raise missing_token(
            "seatable", resource=f"{ds.label} {source.name!r} annotations"
        )

    server = _server(source)
    # `ss.Table(...).to_frame()` paginates the download (SeaTable's SQL API caps a
    # single query at ~10k rows); with the base named its construction is cheap. The
    # server= keyword is what sends BANC to the cloud and FlyWire to the lab instance.
    frames = [
        pd.DataFrame(ss.Table(table, base=base, **({"server": server} if server else {})).to_frame())
        for base, table in _table_specs(source)
    ]
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    # concat unions columns - the two FlyWire tables differ by a couple - filling
    # the gaps with NaN. The FutureWarning is about the dtype of those all-NA gap
    # columns; they are raw passthrough columns core never coerces, so the change
    # is immaterial and the warning would only alarm users. Suppress it locally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df = pd.concat(frames, axis=0, ignore_index=True)
    return df.reset_index(drop=True)


def seatable_freshness(source, ds) -> str | None:
    """A token that changes whenever the SeaTable table(s) do, or None if unknowable.

    FlyTable is a *live* curation database - edited daily - and it is independent of
    any CAVE materialization, so the ``(dataset, version)`` key never turns over on
    its own: FlyWire is pinned at 783 forever. Without this the first snapshot would
    be served until someone ran ``cn.cache.clear()`` by hand.

    ``MAX(_mtime)`` catches any edit or insert, ``COUNT(*)`` catches a delete - one
    SQL round-trip per table, so validating a cache hit stays far cheaper than
    re-downloading. When a source is several tables, the token folds in each of
    them, so an edit to *either* FlyWire table turns the cache over. Returns None
    when we simply can't ask (no seaserpent, no token, a SQL error): the caller then
    falls back to the version-scoped key - i.e. today's behaviour - rather than
    failing the query.
    """
    if not os.environ.get("SEATABLE_TOKEN"):
        return None
    try:
        server = _server(source)
        parts = []
        for base_name, table in _table_specs(source):
            b = _base(server, base_name, table)
            rows = b.query(f"SELECT COUNT(*) AS n, MAX(`_mtime`) AS m FROM `{table}`")
            r = rows[0] if rows else {}
            parts.append(f"{table}={r.get('n')}:{r.get('m')}")
    except Exception:
        return None
    return ";".join(parts) if parts else None
