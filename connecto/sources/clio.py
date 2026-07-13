"""Annotations from clio, Janelia's curation database.

Optional: ``pip install connecto[clio]``. clio is the *live* annotation store for
maleCNS and MANC; the neuPrint copy is a snapshot of it.
"""

from __future__ import annotations

import pandas as pd

from ..exceptions import MissingDependencyError

__all__ = ["fetch_clio"]


def fetch_clio(source, ds, version) -> pd.DataFrame:
    try:
        import clio
    except ModuleNotFoundError as e:
        raise MissingDependencyError.for_extra(
            "clio-py", "clio", f"Fetching clio annotations for {ds.label}"
        ) from e

    dataset = source.location or str(ds.version).split(":")[0]
    ann = clio.fetch_annotations(dataset=dataset)
    df = pd.DataFrame(ann) if not isinstance(ann, pd.DataFrame) else ann

    # clio speaks snake_case where neuPrint speaks camelCase. connecto does not
    # guess between them (cocoa's `_find_column` did): the dataset's `fields`
    # priority lists name the exact columns, so the two sources just declare
    # different ones.
    return df.reset_index(drop=True)
