"""Annotations from SeaTable ("flytable").

Optional and lab-internal: ``pip install connecto[flytable]``, plus
``SEATABLE_SERVER`` / ``SEATABLE_TOKEN``.
"""

from __future__ import annotations

import os

import pandas as pd

from ..auth import missing_token
from ..exceptions import MissingDependencyError

__all__ = ["fetch_seatable"]


def fetch_seatable(source, ds, version) -> pd.DataFrame:
    try:
        import seaserpent as ss
    except ModuleNotFoundError as e:
        raise MissingDependencyError.for_extra(
            "sea-serpent", "flytable", f"Fetching {source.name} annotations"
        ) from e

    if not os.environ.get("SEATABLE_TOKEN"):
        # One credential exception in the library, not a bespoke one per source.
        raise missing_token(
            "seatable", resource=f"{ds.label} {source.name!r} annotations"
        )

    base, _, table = source.location.partition(".")
    tbl = ss.Table(table or base, base if table else None)
    return pd.DataFrame(tbl.to_frame()).reset_index(drop=True)
