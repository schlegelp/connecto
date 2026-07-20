"""The CAVE backend.

Implements the ten ``_fetch_*`` hooks against ``caveclient``. Everything the user
sees - column names, dtypes, units, capability errors - is applied above this, in
:mod:`connecto.core`. Nothing in this file constructs a user-facing frame.
"""

from __future__ import annotations

import logging
from datetime import UTC

import numpy as np
import pandas as pd

from ...core.dataset import Dataset, namespace
from ...core.namespaces import (
    Annotations,
    Connectivity,
    Meshes,
    ROIs,
    Skeletons,
    Somas,
    Viz,
    Voxels,
)
from ...core.segmentation import Segmentation
from ...core.spec import Cap
from ...core.version import Version
from ...exceptions import CapabilityError
from . import versions as _versions
from .l2 import L2
from .proofreading import Proofreading

logger = logging.getLogger("connecto")

__all__ = ["CAVEDataset"]

# FlyWire-style per-synapse transmitter probability columns.
NT_COLUMNS = {
    "ach": "acetylcholine",
    "gaba": "gaba",
    "glut": "glutamate",
    "oct": "octopamine",
    "ser": "serotonin",
    "da": "dopamine",
}

_CLIENTS: dict = {}


class CAVEDataset(Dataset):
    """A dataset served by a CAVE datastack."""

    annotations = namespace(Annotations, Cap.ANNOTATIONS)
    connectivity = namespace(Connectivity, Cap.CONNECTIVITY)
    skeletons = namespace(Skeletons, Cap.SKELETONS)
    meshes = namespace(Meshes, Cap.MESHES)
    voxels = namespace(Voxels, Cap.VOXELS)
    rois = namespace(ROIs, Cap.ROIS)
    somas = namespace(Somas, Cap.SOMAS)
    viz = namespace(Viz, Cap.NEUROGLANCER)
    segmentation = namespace(Segmentation, Cap.SEGMENTATION)
    proofreading = namespace(Proofreading, Cap.PROOFREADING)
    l2 = namespace(L2, Cap.L2CACHE)

    # CAVE hands back nanometres because we ask for them (desired_resolution).
    _default_position_units = "nm"

    # ------------------------------------------------------------------ volumes

    def _segmentation_source(self, *, format_for: str = "raw") -> str | None:
        """The volume to read and display."""
        # The spec may override this, and FlyWire does: with the *flat* v783 bucket,
        # which needs no CAVE login and is exactly what the public release is. That
        # is the right volume for meshes and for neuroglancer. It is the wrong one
        # for supervoxels - see `_graph_source`.
        if self.spec.segmentation_source is not None:
            return self.spec.segmentation_source
        # `middleauth+` is not decoration: it tells a modern viewer to run CAVE's
        # login flow, and a protected datastack will not load without it. caveclient
        # knows which format wants it, so ask rather than splice the prefix on here.
        return self.client.info.segmentation_source(format_for=format_for)

    def _graph_source(self) -> str:
        """The chunkedgraph, always - never the spec's display volume."""
        return self.client.info.segmentation_source()

    def _image_source(self) -> str | None:
        # `image_source(format_for=...)` returns None for the viewer formats -
        # caveclient only maps the image for "raw"/"cloudvolume".
        return self.client.info.image_source(format_for="raw")

    _edge_colmap = {"pre": "pre_pt_root_id", "post": "post_pt_root_id", "weight": "n_syn", "roi": "neuropil"}
    _SYNAPSE_COLMAP = {
        "id": "id",
        "pre": "pre_pt_root_id",
        "post": "post_pt_root_id",
        "pre_x": "pre_pt_position_x",
        "pre_y": "pre_pt_position_y",
        "pre_z": "pre_pt_position_z",
        "post_x": "post_pt_position_x",
        "post_y": "post_pt_position_y",
        "post_z": "post_pt_position_z",
        "roi": "neuropil",
    }

    @property
    def _synapse_colmap(self) -> dict:
        # The score column is the one name that varies between datastacks
        # (FlyWire: `cleft_score`; FANC: `score`), so it comes off the spec.
        return {**self._SYNAPSE_COLMAP, "score": self._backend.score_column}
    _skeleton_colmap = {
        "node_id": "node_id",
        "parent_id": "parent_id",
        "x": "x", "y": "y", "z": "z", "radius": "radius",
    }

    # ------------------------------------------------------------------- client

    @property
    def client(self):
        """The underlying CAVEclient (cached per datastack).

        Rebuilt when the pinned materialization expires - CAVEclient caches
        version metadata at construction, which is why fafbseg needs a hand-rolled
        staleness hack. Here ``Version.expires`` says exactly when to let go.
        """
        from caveclient import CAVEclient

        from ...auth import get_token
        from ...servers import upstream_errors

        key = self.source
        client = _CLIENTS.get(key)
        if client is None:
            token = get_token("cave").token
            # CAVEclient hits the info service during construction, so this is where
            # both a bad token and a dead deployment first bite - translate them here
            # rather than let a raw AuthException/401/503 escape.
            with upstream_errors("cave", resource=self.source, dataset=self):
                client = CAVEclient(self.source, auth_token=token)
            _CLIENTS[key] = client
        return client

    # ------------------------------------------------------------------ versions

    def _resolve_version(self, request) -> Version:
        if request == "auto":
            # "auto" with no IDs to pin against just means "latest".
            return _versions.resolve(self.client, "latest")
        if request == "live" and not self.supports(Cap.LIVE):
            raise CapabilityError(
                f"{self.label} does not support live queries "
                f"(it is a frozen public release). Use a materialization version."
            )
        return _versions.resolve(self.client, request)

    def _list_versions(self) -> list:
        return sorted(self.client.materialize.get_versions())

    def _find_version(self, ids, *, raise_missing=True) -> Version:
        return _versions.find_version(self.client, ids, raise_missing=raise_missing)

    def _ids_exist(self, ids, version) -> np.ndarray:
        ts = None if version.is_live else self.client.materialize.get_timestamp(version.value)
        return np.asarray(self.client.chunkedgraph.is_latest_roots(ids, timestamp=ts))

    def _mat_kwargs(self, version) -> dict:
        """Version selector for materialize queries."""
        if version.is_live:
            from datetime import datetime

            return {"timestamp": datetime.now(UTC)}
        return {"materialization_version": int(version.value)}

    # --------------------------------------------------------------- annotations

    def _fetch_annotations(self, source, version) -> pd.DataFrame:
        from ...sources import fetch as fetch_source

        return fetch_source(source, self, version)

    # -------------------------------------------------------------- connectivity

    @property
    def _synapse_source(self) -> tuple[str, bool]:
        """(name, is_view) of the table/view holding synapses."""
        b = self._backend
        if b.synapse_table:
            # Curated views (FlyWire's valid_synapses_nt_np_v6) are already
            # score-filtered and neuropil-annotated; prefer them.
            if b.synapse_table in _list_views(self.client):
                return b.synapse_table, True
            return b.synapse_table, False
        table = self.client.materialize.synapse_table
        if not table:
            raise CapabilityError(
                f"{self.label}: no synapse table configured and the datastack "
                f"does not declare one."
            )
        return table, False

    def _query(self, name, *, is_view, **kwargs):
        mat = self.client.materialize
        if is_view:
            kwargs.pop("timestamp", None)  # views are materialization-only
            return mat.query_view(name, **kwargs)
        return mat.query_table(name, **kwargs)

    def _fetch_edges(self, pre, post, version, *, by_roi, min_weight, rois) -> pd.DataFrame:
        b = self._backend
        filters = {}
        if pre is not None:
            filters["pre_pt_root_id"] = np.asarray(pre, dtype="int64").tolist()
        if post is not None:
            filters["post_pt_root_id"] = np.asarray(post, dtype="int64").tolist()

        # Fast path: a pre-aggregated edge view (FlyWire's valid_connection_v2).
        # Orders of magnitude cheaper than pulling every synapse and grouping.
        if b.edge_view and not by_roi:
            df = self.client.materialize.query_view(
                b.edge_view,
                filter_in_dict=filters,
                materialization_version=int(version.value) if not version.is_live else None,
            )
            if min_weight > 1:
                df = df[df["n_syn"] >= min_weight]
            return df.reset_index(drop=True)

        # General path: aggregate synapses.
        syn = self._fetch_synapses(
            pre, post, version, min_score=None, transmitters=False, rois=rois,
            columns=["pre_pt_root_id", "post_pt_root_id"] + (["neuropil"] if by_roi else []),
        )
        group = ["pre_pt_root_id", "post_pt_root_id"] + (["neuropil"] if by_roi else [])
        group = [g for g in group if g in syn.columns]
        edges = syn.groupby(group, as_index=False, observed=True).size()
        edges = edges.rename(columns={"size": "n_syn"})
        if min_weight > 1:
            edges = edges[edges["n_syn"] >= min_weight]
        return edges.reset_index(drop=True)

    def _fetch_synapses(
        self, pre, post, version, *, min_score=None, transmitters=False, rois=None, columns=None
    ) -> pd.DataFrame:
        name, is_view = self._synapse_source

        filters = {}
        if pre is not None:
            filters["pre_pt_root_id"] = np.asarray(pre, dtype="int64").tolist()
        if post is not None:
            filters["post_pt_root_id"] = np.asarray(post, dtype="int64").tolist()

        kwargs = dict(
            filter_in_dict=filters,
            split_positions=True,
            desired_resolution=[1, 1, 1],  # nanometres, straight from the server
            **self._mat_kwargs(version),
        )
        if columns:
            kwargs["select_columns"] = columns
            kwargs.pop("split_positions")
            kwargs.pop("desired_resolution")

        df = self._query(name, is_view=is_view, **kwargs)

        if min_score is not None:
            score = self._backend.score_column
            if score not in df.columns:
                raise CapabilityError(
                    f"{self.label}: synapse table {name!r} has no {score!r} column."
                )
            df = df[df[score] >= min_score]

        if rois is not None and "neuropil" in df.columns:
            df = df[df["neuropil"].isin(np.atleast_1d(rois))]

        if transmitters:
            df = _add_nt(df, self.label)

        return df.reset_index(drop=True)

    # ---------------------------------------------------------------- morphology

    def _fetch_skeletons(self, ids, version, **opts):
        """Skeletons, preferring CAVE's skeleton service over L2.

        The skeleton service post-dates fafbseg, which is why fafbseg rolls its own
        L2 skeletonisation. Try the service first; fall back to L2 only if it isn't
        there.
        """
        from .skeletons import fetch_skeletons

        yield from fetch_skeletons(self, ids, version, **opts)

    def _fetch_meshes(self, ids, version, *, lod=None, **opts):
        from .meshes import fetch_meshes

        yield from fetch_meshes(self, ids, version, lod=lod, **opts)

    # --------------------------------------------------------------------- somas

    def _fetch_somas(self, ids, version) -> pd.DataFrame:
        table = self._backend.nucleus_table
        if not table:
            raise CapabilityError(f"{self.label} has no nucleus/soma table.")

        filters = {}
        if ids is not None:
            filters["pt_root_id"] = np.asarray(ids, dtype="int64").tolist()

        df = self.client.materialize.query_table(
            table,
            filter_in_dict=filters or None,
            split_positions=True,
            desired_resolution=[1, 1, 1],
            **self._mat_kwargs(version),
        )
        out = pd.DataFrame(
            {
                "id": df["pt_root_id"].astype("int64"),
                "x": df["pt_position_x"].astype("float32"),
                "y": df["pt_position_y"].astype("float32"),
                "z": df["pt_position_z"].astype("float32"),
            }
        )
        if "volume" in df.columns:
            out["volume"] = df["volume"].astype("float32")
        return out.reset_index(drop=True)


def _list_views(client) -> set:
    try:
        return set(client.materialize.get_views())
    except Exception:
        return set()


def _add_nt(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Derive a single `nt` call from per-synapse transmitter probabilities."""
    present = [c for c in NT_COLUMNS if c in df.columns]
    if not present:
        raise CapabilityError(
            f"{label}: synapse table has no transmitter probability columns."
        )
    probs = df[present].astype("float32")
    df = df.copy()
    df["nt"] = probs.idxmax(axis=1).map(NT_COLUMNS)
    df["nt_confidence"] = probs.max(axis=1)
    for short, full in NT_COLUMNS.items():
        if short in df.columns:
            df[f"nt_{full}"] = df[short].astype("float32")
    return df
