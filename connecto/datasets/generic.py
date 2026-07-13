"""Datasets connecto has never heard of.

Adding a datastack must not require writing a class, or opening a PR:

    ca3  = connecto.CAVE("zheng_ca3", fields={"type": ("cell_type",)})
    wasp = connecto.NeuPrint("wasp3:v0.8", server="neuprint-pre.janelia.org")

The spec is *probed* from the server. A probed spec claims fewer capabilities and
has no `fields`, so `ds.ids("DA1_lPN")` raises a clear error rather than guessing
which column means "type" - curated specs for the datasets you care about, working
defaults for everything else.
"""

from __future__ import annotations

import logging
import re

from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

logger = logging.getLogger("connecto")

__all__ = ["CAVE", "NeuPrint", "probe_cave", "probe_neuprint"]

# What a table has to look like for us to guess what it is. We guess, but loudly.
_NUCLEUS_RE = re.compile(r"nucle|soma", re.I)
_PROOFREAD_RE = re.compile(r"proofread", re.I)
_ANNOTATION_RE = re.compile(r"annotation|cell_info|cell_type|celltype", re.I)


def probe_cave(datastack: str, **overrides) -> DatasetSpec:
    """Build a DatasetSpec by asking the CAVE server what it has."""
    from caveclient import CAVEclient

    from ..auth import get_token

    client = CAVEclient(datastack, auth_token=get_token("cave").token)
    info = client.info.get_datastack_info()

    voxel = (
        float(info.get("viewer_resolution_x") or 1),
        float(info.get("viewer_resolution_y") or 1),
        float(info.get("viewer_resolution_z") or 1),
    )

    tables = set(client.materialize.get_tables())

    synapse_table = info.get("synapse_table")
    nucleus = next((t for t in sorted(tables) if _NUCLEUS_RE.search(t)), None)
    proofread = next((t for t in sorted(tables) if _PROOFREAD_RE.search(t)), None)
    ann_tables = [t for t in sorted(tables) if _ANNOTATION_RE.search(t)]

    caps = {Cap.SEGMENTATION, Cap.MESHES, Cap.NEUROGLANCER}
    if synapse_table:
        caps |= {Cap.CONNECTIVITY, Cap.SYNAPSES}
    if ann_tables:
        caps.add(Cap.ANNOTATIONS)
    if nucleus:
        caps.add(Cap.SOMAS)
    if proofread:
        caps.add(Cap.PROOFREADING)
    try:
        if client.l2cache.has_cache():
            caps |= {Cap.L2CACHE, Cap.SKELETONS}
    except Exception:  # noqa: BLE001 - absence of an L2 cache is not an error
        pass

    logger.info(
        "Probed %s: synapses=%s nucleus=%s proofreading=%s annotations=%s caps=%s",
        datastack, synapse_table, nucleus, proofread, ann_tables,
        sorted(str(c) for c in caps),
    )

    spec = DatasetSpec(
        name=datastack,
        label=info.get("description") or datastack,
        backends=(
            BackendSpec(
                "cave", datastack,
                synapse_table=synapse_table,
                nucleus_table=nucleus,
                proofreading_table=proofread,
            ),
        ),
        annotation_sources=tuple(
            AnnotationSource(t, "cave_table", t, id_column="pt_root_id")
            for t in ann_tables
        ),
        voxel_size=voxel,
        segmentation_source=info.get("segmentation_source"),
        capabilities=frozenset(caps),
    )
    return spec.evolve(**overrides) if overrides else spec


def probe_neuprint(dataset: str, server: str = "neuprint.janelia.org", **overrides) -> DatasetSpec:
    """Build a DatasetSpec by asking a neuPrint server what it has."""
    from neuprint import Client

    from ..auth import get_token
    from ..backends.neuprint.versions import available

    name = dataset.split(":")[0]
    token = get_token("neuprint", server=server).token

    versions = available(server, name, token)
    full = dataset if dataset in versions else versions[-1]

    client = Client(server, dataset=full, token=token)
    meta = client.meta
    voxel = tuple(meta.get("voxelSize") or (1, 1, 1))

    caps = {
        Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SYNAPSE_SCORES,
        Cap.ROI_CONN, Cap.ROIS, Cap.SKELETONS, Cap.SOMAS,
    }
    if meta.get("neuroglancerMeta"):
        caps |= {Cap.MESHES, Cap.NEUROGLANCER}

    logger.info("Probed %s/%s: voxel=%s rois=%s", server, full, voxel, len(client.primary_rois))

    spec = DatasetSpec(
        name=name,
        label=name,
        backends=(BackendSpec("neuprint", f"{server}/{full}"),),
        annotation_sources=(AnnotationSource("neuprint", "neuprint", id_column="bodyId"),),
        voxel_size=voxel,
        capabilities=frozenset(caps),
    )
    return spec.evolve(**overrides) if overrides else spec


def CAVE(datastack: str, *, version=None, annotations="auto", **overrides):
    """Any CAVE datastack, without writing a class.

    Spec fields (`fields`, `side_map`, `capabilities`, ...) can be passed as
    keyword arguments to refine what was probed.
    """
    from ..backends import build

    ds_kwargs = {"version": version, "annotations": annotations}
    spec = probe_cave(datastack, **overrides)
    return build(spec, **ds_kwargs)


def NeuPrint(dataset: str, *, server: str = "neuprint.janelia.org", version=None,
             annotations="auto", **overrides):
    """Any neuPrint dataset, without writing a class."""
    from ..backends import build

    spec = probe_neuprint(dataset, server=server, **overrides)
    return build(spec, version=version, annotations=annotations)
