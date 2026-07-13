"""Proofreading status and edit history. CAVE-only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.namespaces import _Namespace
from ...exceptions import CapabilityError

__all__ = ["Proofreading"]


class Proofreading(_Namespace):
    def is_proofread(self, x, *, version=None) -> np.ndarray:
        """Boolean mask: has each neuron been marked as proofread?"""
        ds = self._ds
        table = ds._backend.proofreading_table
        if not table:
            raise CapabilityError(f"{ds.label} has no proofreading table.")

        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version)

        df = ds.client.materialize.query_table(
            table,
            filter_in_dict={"pt_root_id": ids.tolist()},
            **ds._mat_kwargs(v),
        )
        proofread = set(df["pt_root_id"].astype("int64"))
        return np.isin(ids, list(proofread))

    def edit_history(self, x) -> pd.DataFrame:
        """Every edit that touched these neurons."""
        ds = self._ds
        ids = ds.ids(x)
        logs = ds.client.chunkedgraph.get_tabular_change_log(ids.tolist())
        frames = []
        for root, log in logs.items():
            log = pd.DataFrame(log)
            log.insert(0, "root_id", int(root))
            frames.append(log)
        if not frames:
            return pd.DataFrame(columns=["root_id"])
        return pd.concat(frames, ignore_index=True)

    def lineage_graph(self, x, *, as_nx: bool = True):
        """The merge/split tree that produced this neuron."""
        return self._ds.client.chunkedgraph.get_lineage_graph(
            int(np.atleast_1d(x)[0]), as_nx_graph=as_nx
        )
