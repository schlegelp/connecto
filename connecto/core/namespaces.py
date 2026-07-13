"""The public API surface.

One generic implementation per namespace, shared by every backend. These are the
methods users call; they resolve versions, enforce capabilities, hit the backend's
``_fetch_*`` hook, and hand the raw result to :mod:`connecto.core.schemas`.

Method names inside a namespace carry information - ``ds.connectivity.edges()``,
never ``ds.connectivity.get()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import cache as _cache
from ..exceptions import CapabilityError
from . import schemas
from .criteria import to_criteria
from .dataset import UNSET, requires
from .spec import Cap

__all__ = ["Annotations", "Connectivity", "Skeletons", "Meshes", "ROIs", "Viz"]


class _Namespace:
    def __init__(self, ds):
        self._ds = ds

    def __repr__(self):
        methods = [
            m for m in dir(type(self)) if not m.startswith("_") and callable(getattr(type(self), m, None))
        ]
        return f"<{type(self).__name__} of {self._ds.label}: {', '.join(methods)}>"


class Annotations(_Namespace):
    """Neuron metadata: types, sides, classes, somas."""

    @requires(Cap.ANNOTATIONS)
    def get(
        self,
        x=None,
        *,
        fields: dict | None = None,
        source: str | None = None,
        raw: bool = False,
        version=None,
        units: str = "nm",
        **filters,
    ) -> pd.DataFrame:
        """Annotations, with canonical columns added and every raw column kept.

        Canonical: ``id, type, side, class, nt, status, soma_x/y/z``. ``type`` is
        coalesced from ``spec.fields["type"]`` in priority order; ``side`` is
        mapped to ``left``/``right``/``center``. Pass ``raw=True`` for the
        untouched backend frame.
        """
        ds = self._ds
        v = ds._resolve_version_arg(version)
        src = (
            ds.spec.annotation_source(source)
            if source is not None
            else ds._annotation_source
        )
        if src is None:
            raise CapabilityError(
                f"{ds.label} has no annotation source configured."
            )

        table = self._table(src, v)

        if raw:
            return table.copy()

        ann = schemas.normalize_annotations(
            table, ds, id_column=src.id_column, fields=fields, version=v, units=units
        )

        if x is not None or filters:
            crit = to_criteria(x) if x is not None else None
            ids = ds.ids(crit, version=version) if crit is not None else None
            if filters:
                from .criteria import NeuronCriteria, resolve_criteria

                extra = resolve_criteria(NeuronCriteria(**filters), ds, version=version)
                ids = extra if ids is None else np.intersect1d(ids, extra)
            ann = ann[ann["id"].isin(ids)].reset_index(drop=True)

        return ann

    def _table(self, src, version) -> pd.DataFrame:
        """Fetch (and cache) the whole annotation table for a source."""
        ds = self._ds
        key = _cache.CacheEntry(ds.name, version, f"annotations_{src.name}", src.location)
        if key.exists():
            return key.read()
        table = ds._fetch_annotations(src, version)
        return key.write(table)

    @requires(Cap.ANNOTATIONS)
    def search(self, term: str, *, version=None, regex: bool = True) -> pd.DataFrame:
        """Rows whose `type`, `class` or `instance` matches ``term``."""
        ann = self.get(version=version)
        cols = [c for c in ("type", "class", "instance") if c in ann.columns]
        mask = pd.Series(False, index=ann.index)
        for col in cols:
            s = ann[col].astype("string")
            mask |= (
                s.str.contains(term, case=False, regex=regex, na=False)
                if regex
                else s.str.lower().eq(term.lower())
            )
        return ann[mask].reset_index(drop=True)

    @requires(Cap.ANNOTATIONS)
    def ids(self, *, version=None) -> np.ndarray:
        """Every annotated neuron in the dataset."""
        return np.unique(self.get(version=version)["id"].to_numpy(dtype="int64"))

    @property
    def sources(self) -> list[str]:
        return [s.name for s in self._ds.spec.annotation_sources]

    @property
    def fields(self) -> dict:
        return dict(self._ds.spec.fields)


class Connectivity(_Namespace):
    """Edges, adjacency and synapses."""

    @requires(Cap.CONNECTIVITY)
    def edges(
        self,
        x,
        *,
        upstream: bool = True,
        downstream: bool = True,
        by_roi: bool = False,
        rois=UNSET,
        min_weight: int = 1,
        version=None,
        cache: bool = False,
        extra: bool = False,
        autapses: bool = False,
    ) -> pd.DataFrame:
        """Edges to/from the given neurons as ``pre, post, weight``.

        A closed schema: identical columns and dtypes on every backend, so
        ``bodyId_pre`` and ``pre_pt_root_id`` both come back as ``pre``, and
        FlyWire's edge view does not smuggle in seventeen extra columns that
        hemibrain has never heard of. ``extra=True`` keeps whatever else the
        backend happened to return.

        ``autapses`` (pre == post) default to **off**. They are nearly always
        segmentation errors, neuPrint omits them entirely, and caveclient's own
        ``synapse_query`` drops them by default - so leaving them in would make the
        same neuron have different connectivity depending on which backend you
        asked. Pass ``autapses=True`` if you want them; you will get them from
        every backend that has them.
        """
        ds = self._ds
        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version)
        if not len(ids):
            return schemas.normalize_edges(
                pd.DataFrame(columns=["pre", "post", "weight"]), ds,
                colmap={}, version=v, extra=extra,
            )

        by_roi = ds._resolve_opt("by_roi", by_roi if by_roi else UNSET,
                                 Cap.ROI_CONN, default=False, neutral=False)
        rois = ds._resolve_opt("rois", rois, Cap.ROI_CONN, default=None)

        entry = None
        if cache:
            entry = _cache.CacheEntry(
                ds.name, v, "edges",
                ids.tobytes(), upstream, downstream, by_roi, rois, min_weight,
            )
            if entry.exists():
                return schemas.normalize_edges(
                    entry.read(), ds, colmap=ds._edge_colmap, version=v, extra=extra
                )

        frames = []
        if downstream:
            frames.append(
                ds._fetch_edges(pre=ids, post=None, version=v,
                                by_roi=by_roi, min_weight=min_weight, rois=rois)
            )
        if upstream:
            frames.append(
                ds._fetch_edges(pre=None, post=ids, version=v,
                                by_roi=by_roi, min_weight=min_weight, rois=rois)
            )
        if not frames:
            raise ValueError("One of `upstream` or `downstream` must be True.")

        raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if len(frames) > 1:
            # A query neuron connecting to another query neuron appears in both
            # the upstream and the downstream fetch.
            keys = [c for c in raw.columns if c in set(ds._edge_colmap.values())]
            dedup_on = [
                ds._edge_colmap[k] for k in ("pre", "post") if ds._edge_colmap.get(k) in keys
            ]
            if by_roi and ds._edge_colmap.get("roi") in raw.columns:
                dedup_on.append(ds._edge_colmap["roi"])
            if dedup_on:
                raw = raw.drop_duplicates(subset=dedup_on, ignore_index=True)

        if entry is not None:
            entry.write(raw)

        out = schemas.normalize_edges(
            raw, ds, colmap=ds._edge_colmap, version=v, extra=extra
        )
        if not autapses:
            prov = out.attrs["connecto"]
            out = out[out["pre"] != out["post"]].reset_index(drop=True)
            out.attrs["connecto"] = prov  # boolean masking doesn't reliably keep attrs
        return out

    @requires(Cap.CONNECTIVITY)
    def adjacency(
        self,
        sources,
        targets=None,
        *,
        version=None,
        min_weight: int = 1,
    ) -> pd.DataFrame:
        """Weight matrix, rows = sources, columns = targets."""
        ds = self._ds
        src = ds.ids(sources, version=version)
        tgt = src if targets is None else ds.ids(targets, version=version)

        edges = self.edges(
            src, upstream=False, downstream=True, min_weight=min_weight, version=version
        )
        edges = edges[edges["post"].isin(tgt)]

        adj = (
            edges.pivot_table(
                index="pre", columns="post", values="weight",
                aggfunc="sum", fill_value=0,
            )
            .reindex(index=src, columns=tgt, fill_value=0)
            .astype("int32")
        )
        adj.index.name, adj.columns.name = "pre", "post"
        return schemas.stamp(adj, ds, query="adjacency", version=ds._resolve_version_arg(version))

    @requires(Cap.SYNAPSES)
    def synapses(
        self,
        x,
        *,
        pre: bool = True,
        post: bool = True,
        min_score=UNSET,
        transmitters=UNSET,
        rois=UNSET,
        version=None,
        units: str = "nm",
    ) -> pd.DataFrame:
        """Individual synapses, positions in nanometres.

        ``min_score`` and ``transmitters`` raise on datasets that don't have them,
        rather than being quietly ignored.
        """
        ds = self._ds
        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version)

        min_score = ds._resolve_opt("min_score", min_score, Cap.SYNAPSE_SCORES, default=None)
        transmitters = ds._resolve_opt(
            "transmitters", transmitters, Cap.NT_PER_SYNAPSE, default=False, neutral=False
        )
        rois = ds._resolve_opt("rois", rois, Cap.ROI_CONN, default=None)

        frames = []
        if pre:
            frames.append(ds._fetch_synapses(
                pre=ids, post=None, version=v,
                min_score=min_score, transmitters=transmitters, rois=rois))
        if post:
            frames.append(ds._fetch_synapses(
                pre=None, post=ids, version=v,
                min_score=min_score, transmitters=transmitters, rois=rois))
        if not frames:
            raise ValueError("One of `pre` or `post` must be True.")

        raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        idcol = ds._synapse_colmap.get("id")
        if len(frames) > 1 and idcol in raw.columns:
            raw = raw.drop_duplicates(subset=[idcol], ignore_index=True)

        return schemas.normalize_synapses(
            raw, ds, colmap=ds._synapse_colmap, version=v, units=units
        )

    @requires(Cap.CONNECTIVITY)
    def synapse_counts(self, x, *, by_roi: bool = False, version=None) -> pd.DataFrame:
        """Pre- and post-synapse counts per neuron."""
        ds = self._ds
        ids = ds.ids(x, version=version)

        out = self.edges(ids, version=version, by_roi=by_roi)
        group = ["roi"] if by_roi and "roi" in out.columns else []

        up = (
            out[out["post"].isin(ids)]
            .groupby(["post"] + group, as_index=False)["weight"].sum()
            .rename(columns={"post": "id", "weight": "post"})
        )
        down = (
            out[out["pre"].isin(ids)]
            .groupby(["pre"] + group, as_index=False)["weight"].sum()
            .rename(columns={"pre": "id", "weight": "pre"})
        )
        counts = pd.merge(down, up, on=["id"] + group, how="outer").fillna(0)
        for col in ("pre", "post"):
            counts[col] = counts[col].astype("int32")
        return counts[["id"] + group + ["pre", "post"]]

    @requires(Cap.NT_PER_SYNAPSE)
    def transmitters(self, x, *, version=None) -> pd.DataFrame:
        """Predicted transmitter per neuron, from its presynapses."""
        syn = self.synapses(x, pre=True, post=False, transmitters=True, version=version)
        if "nt" not in syn.columns:
            raise CapabilityError(f"{self._ds.label} returned no transmitter column.")
        counts = (
            syn.groupby(["pre", "nt"], observed=True).size().rename("n").reset_index()
        )
        top = counts.sort_values("n", ascending=False).drop_duplicates("pre")
        total = counts.groupby("pre", observed=True)["n"].sum().rename("total")
        top = top.merge(total, on="pre")
        top["confidence"] = top["n"] / top["total"]
        return (
            top.rename(columns={"pre": "id"})[["id", "nt", "confidence"]]
            .sort_values("id")
            .reset_index(drop=True)
        )


class Skeletons(_Namespace):
    """Skeletons, as navis TreeNeurons."""

    @requires(Cap.SKELETONS)
    def get(self, x, *, output: str = "navis", version=None, **opts):
        """Skeletons for the given neurons -> ``navis.NeuronList``."""
        ds = self._ds
        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version)

        out = []
        for nid, raw in ds._fetch_skeletons(ids, v, **opts):
            nodes = schemas.normalize_skeleton(raw, ds, colmap=ds._skeleton_colmap)
            out.append((nid, nodes))

        if output == "raw":
            return {nid: nodes for nid, nodes in out}

        import navis

        neurons = []
        for nid, nodes in out:
            n = navis.TreeNeuron(nodes, id=nid, units="1 nm")
            n.name = str(nid)
            neurons.append(n)
        return navis.NeuronList(neurons)

    @requires(Cap.SKELETONS)
    def dotprops(self, x, *, k: int = 5, units: str = "nm", version=None):
        """Dotprops, for NBLAST. Built from the skeleton.

        ``units`` defaults to ``"nm"``, because everything connecto returns is in
        nanometres. **NBLAST is calibrated in microns.** Score it on nanometre dotprops
        and every pair collapses to the floor - two copies of the same cell type come
        back at -0.88 instead of +0.70 - which reads as "nothing is similar" rather than
        as an error. navis notices and logs a warning, but a log line is easy to miss.

        So when the next step is NBLAST, ask for microns::

            dp = ds.skeletons.dotprops(x, units="um")
            navis.nblast(dp, dp)

        If the dataset has an L2 cache, :meth:`~connecto.backends.cave.l2.L2.dotprops`
        gets there without building a skeleton first, and is much cheaper.
        """
        import navis

        if units not in ("nm", "um"):
            raise ValueError(f"`units` must be 'nm' or 'um', got {units!r}.")

        skels = self.get(x, version=version)
        if units == "um":
            skels = skels / 1000
        return navis.make_dotprops(skels, k=k)


class Meshes(_Namespace):
    """Meshes, as navis MeshNeurons."""

    @requires(Cap.MESHES)
    def get(self, x, *, output: str = "navis", lod=None, version=None, **opts):
        ds = self._ds
        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version)

        out = list(ds._fetch_meshes(ids, v, lod=lod, **opts))

        if output == "raw":
            return {nid: m for nid, m in out}

        import navis

        neurons = []
        for nid, mesh in out:
            n = navis.MeshNeuron(mesh, id=nid, units="1 nm")
            n.name = str(nid)
            neurons.append(n)
        return navis.NeuronList(neurons)


class ROIs(_Namespace):
    """Neuropils / brain regions."""

    @requires(Cap.ROIS)
    def list(self) -> pd.DataFrame:
        return self._ds._fetch_rois()

    @requires(Cap.ROIS)
    def hierarchy(self):
        """ROI containment tree as a ``networkx.DiGraph``."""
        return self._ds._fetch_roi_hierarchy()

    @requires(Cap.ROIS)
    def mesh(self, roi: str):
        return self._ds._fetch_roi_mesh(roi)


class Somas(_Namespace):
    @requires(Cap.SOMAS)
    def get(self, x=None, *, version=None) -> pd.DataFrame:
        ds = self._ds
        v = ds._resolve_version_arg(version)
        ids = ds.ids(x, version=version) if x is not None else None
        return ds._fetch_somas(ids, v)


class Viz(_Namespace):
    """Neuroglancer.

    ``scene()`` hands back the raw dict; ``neuroglancer_url()`` hands back a link. Both
    take the same arguments - see :func:`connecto.viz.construct_scene` - so::

        ds.viz.neuroglancer_url(ids, color_by="type")

    is a link with every cell type in its own colour, and::

        scene = ds.viz.scene(ids)
        scene = add_skeleton_layer(scene, ds.skeletons.get(ids))
        encode_url(scene, viewer="https://ngl.flywire.ai")

    is the escape hatch when the canned scene is not enough.
    """

    @requires(Cap.NEUROGLANCER)
    def scene(self, x=None, *, version=None, **kwargs) -> dict:
        from ..viz.neuroglancer import construct_scene

        ds = self._ds
        return construct_scene(ds, _viz_ids(ds, x, version), version=version, **kwargs)

    @requires(Cap.NEUROGLANCER)
    def neuroglancer_url(self, x=None, *, version=None, shorten: bool = False, **kwargs) -> str:
        from ..viz.neuroglancer import build_url

        ds = self._ds
        ids = _viz_ids(ds, x, version)
        return build_url(ds, ids, shorten=shorten, version=version, **kwargs)


def _viz_ids(ds, x, version) -> np.ndarray:
    """Resolve `x`, but keep the order the caller wrote it in.

    `ds.ids()` ends in `np.unique`, so it sorts. Everywhere else that is harmless -
    an edge list does not care. Here it is not: `seg_colors=["red", "blue"]` zips onto
    the IDs positionally, and a silently re-sorted list paints the second neuron with
    the first neuron's colour. The user sees a scene that is wrong but not obviously
    wrong, which is the worst kind. fafbseg never sorts, so this is also the difference
    between matching it and being quietly worse than it.
    """
    if x is None:
        return np.array([], dtype="int64")

    ids = ds.ids(x, version=version)

    written = _as_written(x)
    if written is None:  # a criteria query - the caller never chose an order
        return ids

    resolved = set(ids.tolist())
    kept = dict.fromkeys(i for i in written if i in resolved)
    return np.asarray(list(kept), dtype="int64")


def _as_written(x) -> list[int] | None:
    """`x` as the plain list of IDs the caller typed, or None if it was a query."""
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        values = np.asarray(x).ravel()
        if values.dtype.kind in "iu":
            return [int(v) for v in values]
    return None
