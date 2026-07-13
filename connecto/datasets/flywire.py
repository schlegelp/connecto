"""FlyWire (FAFB)."""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

__all__ = ["FLYWIRE", "FLYWIRE_PRODUCTION", "FlyWire"]

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

_FIELDS = {
    "type": ("cell_type", "hemibrain_type"),
    "side": ("side",),
    "class": ("super_class", "cell_class"),
    "nt": ("known_nt", "top_nt"),
    "status": ("status",),
    "soma": ("soma_x", "soma_y", "soma_z"),
}

# FlyWire already speaks left/right/center. "na" is not a side.
_SIDES = {"left": "left", "right": "right", "center": "center"}

_ANNOTATIONS = (
    AnnotationSource("public", "github_tsv", ANNOTATIONS_URL, id_column="root_id"),
    AnnotationSource("flytable", "seatable", "info", id_column="root_id", public=False),
)

_CAPS = {
    Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SYNAPSE_SCORES,
    Cap.NT_PER_SYNAPSE, Cap.ROI_CONN, Cap.SKELETONS, Cap.MESHES,
    Cap.SEGMENTATION, Cap.PROOFREADING, Cap.SOMAS, Cap.NEUROGLANCER,
    # Note: no Cap.L2CACHE. The `flywire_fafb_public` datastack genuinely has no
    # L2 cache (only production does) - the server says so, and declaring it here
    # would mean the skeleton fallback fails with an HTTP 500 instead of a clear
    # message. Skeletons still work: the CAVE skeleton service serves this stack.
}

FLYWIRE = DatasetSpec(
    name="flywire",
    label="FlyWire (FAFB) public release",
    species="Drosophila melanogaster",
    backends=(
        BackendSpec(
            "cave",
            "flywire_fafb_public",
            default_version=783,
            # The *filtered* view, not the raw synapses_nt_v1 table: it is already
            # cleft-score filtered and neuropil-annotated.
            synapse_table="valid_synapses_nt_np_v6",
            # Pre-aggregated edges. Orders of magnitude cheaper than pulling every
            # synapse and grouping.
            edge_view="valid_connection_v2",
            nucleus_table="nuclei_v1",
            proofreading_table="proofread_neurons",
            # FlyWire publishes precomputed skeletons, one bucket per
            # materialization. Needed because `flywire_fafb_public` has no L2
            # cache, and the CAVE skeleton service requires one.
            skeleton_source=(
                "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_{version}"
            ),
        ),
        BackendSpec("neuprint", "neuprint-cns.janelia.org/flywire-fafb:v783b"),
    ),
    annotation_sources=_ANNOTATIONS,
    fields=_FIELDS,
    side_map=_SIDES,
    voxel_size=(4, 4, 40),
    template_space="FAFB14.1",
    # The *flat* segmentation of materialization 783 - which is exactly what the
    # public release is, so its root IDs are the ones in this volume. Preferred over
    # the graphene source because it loads without a CAVE login. Note this is only
    # true of a frozen release; see FLYWIRE_PRODUCTION below.
    segmentation_source="precomputed://gs://flywire_v141_m783",
    # FlyWire's own neuroglancer, which is a fork old enough that it speaks a
    # different state schema than every other dataset here.
    viewer="https://ngl.flywire.ai",
    viewer_dialect="seunglab",
    # No Cap.LIVE: the public release is a frozen snapshot.
    capabilities=frozenset(_CAPS),
    example_ids=(720575940604407468, 720575940623543881),  # two DA1_lPN @ mat 783
)

FLYWIRE_PRODUCTION = FLYWIRE.evolve(
    name="flywire-production",
    label="FlyWire (FAFB) production",
    backends=(
        BackendSpec(
            "cave",
            "flywire_fafb_production",
            default_version="latest",
            synapse_table="synapses_nt_v1",
            nucleus_table="nuclei_v1",
            proofreading_table="proofreading_status_public_v1",
        ),
    ),
    # NOT the flat v783 volume it would otherwise inherit from FLYWIRE. Production
    # root IDs are live: they change with every edit and do not exist in a frozen
    # snapshot, so a scene built on the flat segmentation selects *nothing* - it
    # opens, it just shows an empty brain. `None` means "ask the info service", which
    # hands back the graphene source that actually tracks the chunkedgraph.
    segmentation_source=None,
    # Production is live and editable, and unlike the public stack it does have an
    # L2 cache.
    capabilities=frozenset(_CAPS | {Cap.LIVE, Cap.L2CACHE}),
    example_ids=(),
)

register(FLYWIRE)
register(FLYWIRE_PRODUCTION)


def FlyWire(release: str = "public", **kwargs):
    """FlyWire, the whole-brain FAFB reconstruction.

        fw = connecto.FlyWire()                  # public release, mat 783
        fw = connecto.FlyWire("production")      # live, needs permissions
    """
    from ..backends import build

    spec = {"public": FLYWIRE, "production": FLYWIRE_PRODUCTION}.get(release)
    if spec is None:
        raise ValueError(f"`release` must be 'public' or 'production', got {release!r}.")
    return build(spec, **kwargs)
