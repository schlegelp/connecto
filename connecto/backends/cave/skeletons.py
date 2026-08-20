"""CAVE skeletons.

Three possible sources, in order of preference:

1. **Precomputed neuroglancer skeletons**, if the dataset publishes them
   (``DatasetSpec.skeleton_source``). FlyWire does, one bucket per materialization.
2. CAVE's **skeleton service** (``client.skeleton``). Post-dates fafbseg, which is
   why fafbseg doesn't use it. Note it *requires an L2 cache* - so it is not
   available for every datastack even when ``get_versions()`` cheerfully answers.
3. The **L2 cache** directly - build a skeleton from the level-2 chunk graph,
   using ``rep_coord_nm`` for node positions and ``size_nm3``/``area_nm2`` for radii.

If none of the three is available, a clear :class:`CapabilityError`. Never a silent
empty result.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import pandas as pd

from ...core.spec import Cap
from ...core.volume import precomputed_skeleton
from ...exceptions import CapabilityError, ConnectoError

logger = logging.getLogger("connecto")

__all__ = ["fetch_skeletons", "l2_skeleton", "l2_info"]


def fetch_skeletons(ds, ids, version, *, progress: bool = True, **opts):
    """Yield ``(root_id, node_table)`` for each neuron."""
    from tqdm.auto import tqdm

    source = ds._skeleton_source(version)
    use_service = source is None and _service_available(ds)

    if source is None and not use_service and not ds.supports(Cap.L2CACHE):
        raise CapabilityError(
            f"{ds.label}: no precomputed skeletons, no working skeleton service, "
            f"and no L2 cache - skeletons cannot be built for this dataset."
        )

    for root in tqdm(ids, desc="Skeletons", disable=not progress or len(ids) < 2, leave=False):
        root = int(root)
        if source is not None:
            yield root, precomputed_skeleton(source, root)
            continue

        if use_service:
            nodes = _service_skeleton(ds, root)
            # The service happily returns a *one-vertex* skeleton for neurons it
            # hasn't gotten around to generating yet. Handing that back as a
            # navis.TreeNeuron would be a 0 um "neuron" - silent degradation of
            # exactly the kind this library exists to prevent.
            if len(nodes) > 1:
                yield root, nodes
                continue
            if not ds.supports(Cap.L2CACHE):
                raise CapabilityError(
                    f"{ds.label}: the skeleton service has not generated a skeleton "
                    f"for {root} yet, and there is no L2 cache to build one from. "
                    f"Queue generation with "
                    f"`ds.client.skeleton.generate_bulk_skeletons_async([{root}])` "
                    f"and try again later."
                )
            logger.info(
                "No precomputed skeleton for %s yet; building it from the L2 cache.", root
            )

        yield root, l2_skeleton(ds, root)


def _service_available(ds) -> bool:
    """Is the skeleton service actually usable here?

    ``get_versions()`` answers even for datastacks the service cannot serve, so we
    also need to know whether there is an L2 cache - the service requires one.

    That answer comes from the *spec*, not from ``l2cache.has_cache()``. Asking the
    l2cache service would couple skeletons to l2cache uptime, and the failure is
    nasty: a transient 503 from the l2cache reads as "this datastack has no L2
    cache", which silently downgrades us to the L2 skeleton path - the one route
    that is guaranteed to be broken, because the l2cache is exactly what just went
    down. FANC's l2cache does this. The spec knows the answer for curated datasets,
    and ``probe_cave`` fills it in for the rest.
    """
    try:
        if not ds.client.skeleton.get_versions():
            return False
        return bool(ds.supports(Cap.L2CACHE))
    except Exception:  # noqa: BLE001 - no service is a fact, not an error
        return False


@contextlib.contextmanager
def _without_cloudvolume_root_check(ds):
    """Let the skeleton service work without cloud-volume installed.

    caveclient's ``skeleton.get_skeleton`` calls ``info.segmentation_cloudvolume()``
    purely to check that the id it was handed is a root and not, say, a supervoxel -
    and that call hard-raises ImportError when cloud-volume is absent. The check
    itself is bit arithmetic on the label, which :func:`_check_is_root` above has
    already done, so we hand caveclient a ``None`` volume, which it already handles
    (``if cv and ...``).

    Unconditionally, not only when the import fails: gating on whether some other
    package happens to be importable would make connecto behave differently in two
    environments for no reason anybody could see, and quietly route back through the
    dependency this reader exists to replace.

    Remove this once caveclient no longer reaches for cloud-volume to answer a
    question about a 64-bit integer.
    """
    info = ds.client.info
    original = info.segmentation_cloudvolume
    info.segmentation_cloudvolume = lambda *a, **kw: None
    try:
        yield
    finally:
        info.segmentation_cloudvolume = original


def _check_is_root(ds, root: int) -> None:
    """Refuse a non-root id here, rather than let the service answer nonsense."""
    from ...core.volume import get_volume

    try:
        meta = get_volume(ds, ds._graph_source()).meta
    except ConnectoError:
        return  # cannot ask right now; the request itself will fail loudly enough
    layer = meta.decode_layer_id(int(root))
    if layer != meta.n_layers:
        raise ValueError(
            f"{root} is a layer-{layer} id, not a root id (roots are at layer "
            f"{meta.n_layers}). Skeletons are built for whole neurons."
        )


def _service_skeleton(ds, root: int) -> pd.DataFrame:
    _check_is_root(ds, root)
    with _without_cloudvolume_root_check(ds):
        sk = ds.client.skeleton.get_skeleton(root, output_format="dict")
    verts = np.asarray(sk["vertices"], dtype="float32")
    edges = np.asarray(sk["edges"], dtype="int64")
    radius = np.asarray(sk.get("radius", np.full(len(verts), -1)), dtype="float32")

    # The service gives an undirected edge list; navis wants a parent per node.
    parent = _edges_to_parents(len(verts), edges, root_node=int(sk.get("root", 0)))

    return pd.DataFrame(
        {
            "node_id": np.arange(len(verts), dtype="int32"),
            "parent_id": parent,
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "radius": radius,
        }
    )


def _edges_to_parents(n_nodes: int, edges: np.ndarray, root_node: int = 0) -> np.ndarray:
    """Root an undirected tree and return a parent array (-1 at the root)."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges.tolist())

    parent = np.full(n_nodes, -1, dtype="int32")
    # A skeleton can be disconnected; root each component separately rather than
    # dropping the fragments on the floor.
    for component in nx.connected_components(g):
        src = root_node if root_node in component else next(iter(component))
        for child, par in nx.bfs_predecessors(g, src):
            parent[child] = par
    return parent


def l2_info(ds, root: int) -> pd.DataFrame:
    """Level-2 chunk attributes for one neuron.

    Note ``split_columns=True`` (caveclient's default): ``rep_coord_nm`` comes back
    as ``rep_coord_nm_x/_y/_z``, not as a single column of triples.
    """
    l2_ids = ds.client.chunkedgraph.get_leaves(root, stop_layer=2)
    return ds.client.l2cache.get_l2data_table(
        l2_ids.tolist(),
        attributes=["rep_coord_nm", "size_nm3", "area_nm2"],
        split_columns=True,
    )


def l2_skeleton(ds, root: int) -> pd.DataFrame:
    """Build a skeleton from the L2 chunk graph. Positions are already in nm."""
    import networkx as nx

    edges = ds.client.chunkedgraph.level2_chunk_graph(root)
    info = l2_info(ds, root)

    xyz_cols = ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]
    missing = [c for c in xyz_cols if c not in info.columns]
    if len(info) == 0 or missing:
        raise CapabilityError(
            f"{ds.label}: the L2 cache returned no usable coordinates for {root} "
            f"(missing {missing or 'all rows'}), so no skeleton can be built."
        )

    info = info.dropna(subset=xyz_cols)
    if not len(info):
        raise CapabilityError(f"{ds.label}: no L2 coordinates for {root}.")

    order = [int(i) for i in info.index]
    index = {l2: i for i, l2 in enumerate(order)}
    xyz = info[xyz_cols].to_numpy(dtype="float32")

    # Radius of the sphere with the same volume-to-surface ratio as the chunk.
    if {"size_nm3", "area_nm2"}.issubset(info.columns):
        size = info["size_nm3"].to_numpy(dtype="float64")
        area = info["area_nm2"].to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(area > 0, 3 * size / area, -1.0).astype("float32")
    else:
        radius = np.full(len(order), -1.0, dtype="float32")

    g = nx.Graph()
    g.add_nodes_from(range(len(order)))
    g.add_edges_from(
        [
            (index[int(a)], index[int(b)])
            for a, b in edges
            if int(a) in index and int(b) in index
        ]
    )

    parent = _edges_to_parents(
        len(order),
        np.array(list(g.edges), dtype="int64").reshape(-1, 2),
    )

    return pd.DataFrame(
        {
            "node_id": np.arange(len(order), dtype="int32"),
            "parent_id": parent,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "radius": radius,
            "l2_id": np.array(order, dtype="int64"),
        }
    )
