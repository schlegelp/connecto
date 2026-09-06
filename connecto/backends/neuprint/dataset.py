"""The neuPrint backend.

Implements the same ten ``_fetch_*`` hooks as the CAVE backend, against
``neuprint-python``. Everything the user sees is normalised above this.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import logging
import queue

import numpy as np
import pandas as pd

from ...core import schemas
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
from ...core.parallel import DEFAULT_SKELETON_WORKERS, map_ordered
from ...core.segmentation import Segmentation
from ...core.spec import Cap
from ...core.version import Version
from ...exceptions import CapabilityError
from . import versions as _versions

logger = logging.getLogger("connecto")

__all__ = ["NeuPrintDataset"]

_CLIENTS: dict = {}


class NeuPrintDataset(Dataset):
    """A dataset served by a neuPrint server."""

    annotations = namespace(Annotations, Cap.ANNOTATIONS)
    connectivity = namespace(Connectivity, Cap.CONNECTIVITY)
    skeletons = namespace(Skeletons, Cap.SKELETONS)
    meshes = namespace(Meshes, Cap.MESHES)
    voxels = namespace(Voxels, Cap.VOXELS)
    rois = namespace(ROIs, Cap.ROIS)
    somas = namespace(Somas, Cap.SOMAS)
    viz = namespace(Viz, Cap.NEUROGLANCER)
    segmentation = namespace(Segmentation, Cap.SEGMENTATION)

    # neuPrint usually returns voxel coordinates; spec.voxel_size takes them to nm.
    # "Usually" is load-bearing: the mirrors of FlyWire and BANC were imported from
    # CAVE and kept its nanometres, and they say so via `BackendSpec.position_units`.
    _default_position_units = "voxel"

    # ------------------------------------------------------------------ volumes

    def _ngl_layer(self, data_type: str) -> str | None:
        """The source URL of the first `neuroglancerMeta` layer of this type.

        neuPrint advertises its volumes in `Client.meta["neuroglancerMeta"]` - a
        *list* of layers, image and segmentation together and in no promised order.
        So it has to be searched by `dataType`, not indexed: on hemibrain, entry 0
        is the grayscale.

        Only some servers fill this in at all (hemibrain does; MANC, maleCNS and
        fish2 all return None), which is why the specs carry a verified
        `segmentation_source` and this is only the fallback.
        """
        for layer in self.client.meta.get("neuroglancerMeta") or []:
            if layer.get("dataType") == data_type and layer.get("source"):
                return layer["source"]
        return None

    def _segmentation_source(self, *, format_for: str = "raw") -> str | None:
        return self.spec.segmentation_source or self._ngl_layer("segmentation")

    def _image_source(self) -> str | None:
        return self._ngl_layer("image")

    _edge_colmap = {"pre": "bodyId_pre", "post": "bodyId_post", "weight": "weight", "roi": "roi"}
    _synapse_colmap = {
        "pre": "bodyId_pre",
        "post": "bodyId_post",
        "pre_x": "x_pre", "pre_y": "y_pre", "pre_z": "z_pre",
        "post_x": "x_post", "post_y": "y_post", "post_z": "z_post",
        "score": "confidence_pre",
        "roi": "roi_pre",
    }
    _skeleton_colmap = {
        "node_id": "rowId",
        "parent_id": "link",
        "x": "x", "y": "y", "z": "z", "radius": "radius",
    }

    # ------------------------------------------------------------------- client

    @property
    def _server_and_name(self) -> tuple[str, str]:
        server, name, _ = _versions.parse_source(self.source)
        return server, name

    @property
    def _auth_server(self) -> str:
        """neuPrint tokens are per-server, and the servers are separate deployments."""
        return self._server_and_name[0]

    @property
    def _token(self):
        from ...auth import get_token

        server, _ = self._server_and_name
        return get_token("neuprint", server=server).token

    @property
    def client(self):
        """A client pinned to this dataset *and version*."""
        from neuprint import Client

        from ...auth import get_token
        from ...servers import upstream_errors

        server, _ = self._server_and_name
        dataset = str(self.version)  # e.g. "hemibrain:v1.2.1" or, for fish2, "fish2"
        key = (server, dataset)
        if key not in _CLIENTS:
            token = get_token("neuprint", server=server).token
            with upstream_errors(
                "neuprint", server=server, resource=dataset, dataset=self
            ):
                _CLIENTS[key] = Client(server, dataset=dataset, token=token)
        return _CLIENTS[key]

    # ------------------------------------------------------------------ versions

    def _criteria(self, ids=None, *, label="Segment", **kwargs):
        """A neuPrint criteria object.

        Note ``label="Segment"``, not neuprint's own default of ``"Neuron"``.
        neuPrint splits its bodies into ``:Neuron`` (above a synapse threshold) and
        ``:Segment`` (everything, including small fragments), and its default
        criteria quietly matches only the former. CAVE draws no such distinction,
        so leaving the default in place would mean the same query returned *fewer
        partners* on neuPrint than on CAVE - a silent divergence of exactly the
        kind connecto exists to prevent. Matching ``:Segment`` includes the
        fragments, which is what CAVE does.

        Filter them out afterwards if you don't want them; that is a choice, and
        it should be the caller's.
        """
        from neuprint import NeuronCriteria as NC

        if ids is not None:
            kwargs["bodyId"] = np.asarray(ids, dtype="int64").tolist()
        return NC(client=self.client, label=label, **kwargs)

    def _resolve_version(self, request) -> Version:
        server, name, pinned = _versions.parse_source(self.source)
        # The spec's own version wins over "latest": a spec that says
        # hemibrain:v1.2.1 means v1.2.1, not "whatever landed this morning".
        if request in (None, "latest") and pinned:
            request = pinned
        return _versions.resolve(server, name, request, token=self._token)

    def _list_versions(self) -> list:
        server, name = self._server_and_name
        return _versions.available(server, name, self._token)

    def _find_version(self, ids, *, raise_missing=True) -> Version:
        # Body IDs are immutable, so there is nothing to search for. This is the
        # point of version="auto": the same call is correct on both backends.
        return self.version

    def _ids_exist(self, ids, version) -> np.ndarray:
        from neuprint import fetch_neurons

        found, _ = fetch_neurons(self._criteria(ids), client=self.client)
        return np.isin(ids, found["bodyId"].to_numpy())

    # --------------------------------------------------------------- annotations

    def _fetch_annotations(self, source, version) -> pd.DataFrame:
        from ...sources import fetch as fetch_source

        return fetch_source(source, self, version)

    # -------------------------------------------------------------- connectivity

    def _fetch_edges(self, pre, post, version, *, by_roi, min_weight, rois) -> pd.DataFrame:
        from neuprint import fetch_adjacencies, fetch_simple_connections

        # An explicit Segment-matching criteria on the *unconstrained* side, rather
        # than None - passing None lets neuprint fall back to its :Neuron-only
        # default and quietly drop fragment partners. See _criteria().
        src = self._criteria(pre)
        tgt = self._criteria(post)

        if by_roi:
            _, conn = fetch_adjacencies(
                sources=src, targets=tgt, rois=rois,
                min_total_weight=min_weight, client=self.client,
            )
            return conn.reset_index(drop=True)

        conn = fetch_simple_connections(
            upstream_criteria=src,
            downstream_criteria=tgt,
            rois=rois,
            min_weight=min_weight,
            properties=[],
            client=self.client,
        )
        return conn.reset_index(drop=True)

    def _fetch_synapses(
        self, pre, post, version, *, min_score=None, transmitters=False, rois=None, **kw
    ) -> pd.DataFrame:
        if transmitters:
            # neuprint-python cannot ask for the transmitter properties, so this
            # path is ours. Same match pattern, same frame, extra columns.
            from .synapses import fetch_synapses_with_transmitters

            syn = fetch_synapses_with_transmitters(
                self, pre, post, nt_columns=self._backend.nt_columns, rois=rois
            )
        else:
            # An explicit Segment-matching criteria on the *unconstrained* side,
            # rather than None - exactly as `_fetch_edges` does, and for the same
            # reason: `fetch_synapse_connections` turns a None into neuprint's
            # default criteria, which matches `:Neuron` only and so drops every
            # partner that is a mere `:Segment` fragment.
            #
            # This was silently costing 81% of BANC's synapses - one body returned
            # 187 rows here and 994 through `edges()`, which counts fragments
            # because that call already passes explicit criteria. So `synapses(x)`
            # and `synapse_counts(x)` disagreed by 5x on the same neuron, and
            # neither said why.
            from neuprint import SynapseCriteria as SC
            from neuprint import fetch_synapse_connections

            src = self._criteria(pre)
            tgt = self._criteria(post)

            sc = SC(rois=rois, client=self.client) if rois is not None else None
            syn = fetch_synapse_connections(
                source_criteria=src, target_criteria=tgt,
                synapse_criteria=sc, client=self.client,
            )

        if min_score is not None:
            syn = syn[syn["confidence_pre"] >= min_score]

        if transmitters:
            syn = schemas.add_transmitters(
                syn, self._backend.nt_columns, label=self.label
            )

        return syn.reset_index(drop=True)

    # ---------------------------------------------------------------- morphology

    @functools.cached_property
    def _skeleton_clients(self) -> queue.LifoQueue:
        """Warm neuPrint clients waiting to be borrowed. See `_borrowed_client`."""
        return queue.LifoQueue()

    @contextlib.contextmanager
    def _borrowed_client(self, pool: queue.LifoQueue):
        """Borrow a neuPrint client for the calling thread, and give it back.

        neuprint-python keeps per-thread deep copies of its own default client
        (``DEFAULT_NEUPRINT_CLIENT_THREAD_COPIES``, keyed by thread and pid) rather
        than share one, which is as clear a statement as the library makes that a
        ``Client`` is not to be shared across threads. That machinery only runs for
        callers who let neuprint pick the client; connecto passes one explicitly, so
        it never fires for us and this door has to do the same thing itself. (The
        CAVE door shares its client instead - see ``backends.cave.skeletons``, where
        caveclient's own use of threads settles the question the other way.)

        Pooled on the dataset rather than made fresh per call, because a copied
        ``Client`` copies no connections: ``requests`` drops the pool manager on
        pickle, so the copy's first request pays a fresh TCP and TLS handshake -
        measured at 645 ms against 114 ms once warm. Made per call, eight copies
        meant eight handshakes every time, which was most of what a small fetch
        cost. The queue holds at most one client per concurrent worker, since a
        borrower always returns it.
        """
        try:
            client = pool.get_nowait()
        except queue.Empty:
            client = copy.deepcopy(self.client)
        try:
            yield client
        finally:
            pool.put(client)

    def _fetch_skeletons(
        self, ids, version, *, heal: bool = True, progress: bool = True,
        max_workers: int = DEFAULT_SKELETON_WORKERS,
    ):
        """Yield ``(body_id, node_table)`` for each neuron, in the order asked for.

        Threaded: a skeleton is one request per neuron either way, so serially it is
        a round trip of dead time each. See
        :data:`~connecto.core.parallel.DEFAULT_SKELETON_WORKERS`.

        No ``**opts``: an unrecognised keyword should be a ``TypeError`` naming
        itself rather than a silently ignored request.
        """
        from ...core.volume import precomputed_skeleton

        # A published precomputed bucket wins over neuPrint's own skeleton store.
        # Not every neuPrint dataset *has* a store - `flywire-fafb:v783b` answers
        # HTTP 400, "no store found supporting the datatype and dataset" - and where
        # a dataset publishes skeletons of its own, those are the authoritative ones
        # anyway. Backend-independent by design: it is a plain HTTPS bucket.
        source = self._skeleton_source(version)

        # Resolved here, on the calling thread, so the workers never race to build it.
        clients = self._skeleton_clients

        def one(body: int):
            if source is not None:
                return precomputed_skeleton(source, body)
            with self._borrowed_client(clients) as client:
                return client.fetch_skeleton(body, heal=heal, format="pandas")

        yield from map_ordered(
            (int(i) for i in ids),
            one,
            workers=max_workers,
            desc="Skeletons",
            progress=progress,
        )

    def _fetch_meshes(self, ids, version, *, lod=None, progress: bool = True, **opts):
        """Yield ``(body_id, trimesh.Trimesh)``.

        Read straight from the published precomputed bucket where the dataset
        names one - every neuPrint dataset connecto ships does. That bucket is
        plain HTTPS and needs no neuPrint login, and reading it here rather than
        through navis keeps the mesh path on connecto's own reader.

        ``lod=1`` by default, not 0. These are deep octrees and level 0 is the
        full-resolution surface: one hemibrain neuron at level 0 is 36 million
        vertices, which is not what somebody asking for "the mesh" wants.

        ``**opts`` reaches :func:`~connecto.core.volume.fetch_meshes`, so
        ``max_workers=`` and ``parallel=`` tune this door exactly as they tune the
        CAVE one. They used to be swallowed here, which made the same keyword work
        on FlyWire-via-CAVE and do nothing at all on FlyWire-via-neuPrint.
        """
        from ...core.volume import fetch_meshes

        source = self._segmentation_source()
        if source is None:
            # No published volume: fall back to whatever navis can find. navis has
            # no notion of connecto's fetch options, so passing them on would be a
            # TypeError from inside somebody else's library; say so here instead,
            # and say where the keyword *does* work - `parallel=` is real on every
            # other mesh route, so "not supported" alone would read as a connecto
            # limitation rather than as a property of this one fallback.
            if opts:
                raise TypeError(
                    f"{self.label} publishes no mesh volume, so its meshes come from "
                    f"navis, which takes none of {sorted(opts)} - only `lod` and "
                    f"`progress` mean anything here. A dataset with a precomputed "
                    f"mesh bucket, or the same dataset through a CAVE door, takes "
                    f"`max_workers` and `parallel`."
                )
            import navis.interfaces.neuprint as neu

            neurons = neu.fetch_mesh_neuron(
                np.asarray(ids, dtype="int64").tolist(),
                lod=1 if lod is None else lod,
                client=self.client,
                progress=progress,
            )
            for n in navis_list(neurons):
                yield int(n.id), n.trimesh
            return

        yield from fetch_meshes(
            self, ids, source=source, lod=1 if lod is None else lod,
            progress=progress, **opts,
        )

    # --------------------------------------------------------------------- somas

    def _fetch_somas(self, ids, version) -> pd.DataFrame:
        from neuprint import fetch_neurons

        crit = self._criteria(ids) if ids is not None else self._criteria(soma=True)
        neurons, _ = fetch_neurons(crit, client=self.client)

        if "somaLocation" not in neurons.columns:
            raise CapabilityError(f"{self.label} has no soma locations.")

        loc = neurons["somaLocation"].dropna()
        xyz = np.array([list(v) for v in loc], dtype="float32").reshape(-1, 3)
        vox = np.asarray(self.spec.voxel_size or (1, 1, 1), dtype="float32")
        xyz = xyz * vox  # -> nm

        out = pd.DataFrame(
            {
                "id": neurons.loc[loc.index, "bodyId"].astype("int64").to_numpy(),
                "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            }
        )
        if "somaRadius" in neurons.columns:
            out["radius"] = neurons.loc[loc.index, "somaRadius"].to_numpy()
        return out.reset_index(drop=True)

    # ---------------------------------------------------------------------- ROIs

    def _fetch_rois(self) -> pd.DataFrame:
        client = self.client
        primary = set(client.primary_rois)
        return pd.DataFrame(
            {
                "roi": sorted(client.all_rois),
                "primary": [r in primary for r in sorted(client.all_rois)],
            }
        )

    def _fetch_roi_hierarchy(self):
        import networkx as nx

        tree = self.client.meta.get("roiHierarchy")
        if not tree:
            raise CapabilityError(f"{self.label} has no ROI hierarchy.")

        g = nx.DiGraph()

        def walk(node):
            name = node["name"]
            g.add_node(name)
            for child in node.get("children", []) or []:
                g.add_edge(name, child["name"])
                walk(child)

        walk(tree)
        return g

    def _fetch_roi_mesh(self, roi: str):
        import io

        import trimesh

        obj = self.client.fetch_roi_mesh(roi)
        return trimesh.load(io.BytesIO(obj), file_type="obj")

    def expand_rois(self, rois) -> list[str]:
        """Expand a super-ROI ("Brain", "VNC") into its primary ROIs."""
        import networkx as nx

        g = self._fetch_roi_hierarchy()
        primary = set(self.client.primary_rois)

        out = []
        for roi in np.atleast_1d(rois):
            if roi in primary:
                out.append(roi)
            elif roi in g:
                out += [n for n in nx.descendants(g, roi) if n in primary]
            else:
                raise ValueError(f"{self.label} has no ROI {roi!r}.")
        return sorted(set(out))


def navis_list(x):
    import navis

    return x if isinstance(x, navis.NeuronList) else navis.NeuronList(x)
