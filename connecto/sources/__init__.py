"""Annotation sources.

The annotation source is *not* the query backend, and that distinction is the
thing cocoa most needed:

    maleCNS  = neuPrint backend x (neuprint | clio)  annotations
    FlyWire  = CAVE     backend x (GitHub TSV | SeaTable) annotations
    BANC     = CAVE     backend x (CAVE table | SeaTable) annotations

Modelling it as a cross-product rather than a class hierarchy is what lets
``annotations="clio"`` be a parameter instead of a subclass.
"""

from __future__ import annotations

import pandas as pd

from ..core.spec import MULTI_SEP, AnnotationSource

__all__ = ["fetch"]


def fetch(source: AnnotationSource, ds, version) -> pd.DataFrame:
    """Fetch a whole annotation table. Returns a *raw* frame; core normalises it."""
    if source.kind == "cave_table":
        from .cave_table import fetch_cave_table

        df = fetch_cave_table(source, ds, version)
    elif source.kind == "github_tsv":
        from .github_tsv import fetch_github_tsv

        df = fetch_github_tsv(source, ds, version)
    elif source.kind == "neuprint":
        from .neuprint_source import fetch_neuprint

        df = fetch_neuprint(source, ds, version)
    elif source.kind == "clio":
        from .clio import fetch_clio

        df = fetch_clio(source, ds, version)
    elif source.kind == "seatable":
        from .seatable import fetch_seatable

        df = fetch_seatable(source, ds, version)
    else:
        raise ValueError(f"Unknown annotation source kind: {source.kind!r}")

    if source.pivot:
        df = pivot_long(df, source)
    return df


def pivot_long(df: pd.DataFrame, source: AnnotationSource) -> pd.DataFrame:
    """Turn a long (id, key, value) table into one row per neuron.

    A neuron can legitimately carry several values for one key - a FANC MDN is tagged
    "MDN", "MDN3" *and* "moonwalker descending neuron" - so collisions are joined with
    ``MULTI_SEP`` rather than silently dropped. ``criteria._match`` splits on the same
    constant, so each member of the set stays individually findable.
    """
    key, value = source.pivot
    idc = source.id_column

    missing = [c for c in (idc, key, value) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot pivot annotation table: missing {missing} "
            f"(got {list(df.columns)})."
        )

    wide = df.pivot_table(
        index=idc,
        columns=key,
        values=value,
        aggfunc=lambda s: MULTI_SEP.join(sorted({str(v) for v in s if pd.notna(v)})) or pd.NA,
    )
    wide.columns.name = None
    return wide.reset_index()
