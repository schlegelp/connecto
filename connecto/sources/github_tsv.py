"""Annotations from a file in a GitHub repo.

FlyWire's authoritative annotations live in `flyconnectome/flywire_annotations`,
not in CAVE. This is exactly why criteria like ``type="DA1_lPN"`` cannot compile
to a CAVE filter and must be resolved client-side - see
:func:`connecto.core.criteria.resolve_criteria`.
"""

from __future__ import annotations

import pandas as pd

from .. import cache

__all__ = ["fetch_github_tsv"]


def fetch_github_tsv(source, ds, version) -> pd.DataFrame:
    url = source.location.format(version=version)
    path = cache.download(url)

    sep = "\t" if path.suffix in (".tsv", ".dat") or ".tsv" in url else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    if source.id_column in df.columns:
        df[source.id_column] = pd.to_numeric(
            df[source.id_column], errors="coerce"
        ).astype("Int64")
        df = df[df[source.id_column].notna()]

    return df.reset_index(drop=True)
