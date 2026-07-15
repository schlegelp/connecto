"""BANC - the Brain And Nerve Cord dataset.

BANC is the reason connecto treats backend and dataset as orthogonal: it is served
by CAVE (``brain_and_nerve_cord_public``, materialization 888) *and* by neuPrint
(``banc:v888``) - the same snapshot, two backends. So::

    connecto.BANC()                 # neuPrint, the default
    connecto.BANC(backend="cave")   # CAVE - for the chunkedgraph and the L2 cache

which also gives us a real test that the normalisation is genuine rather than
aspirational: both should return the same edges.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec, Publication

__all__ = ["BANC_SPEC", "BANC"]

BANC_SPEC = DatasetSpec(
    name="banc",
    label="BANC (brain and nerve cord)",
    species="Drosophila melanogaster",
    description=(
        "The first synapse-resolution connectome to hold the brain *and* the ventral "
        "nerve cord of one animal - a female adult Drosophila, imaged by GridTape-TEM "
        "and proofread by the community. 158,262 neurons at v888. The dataset you need "
        "when the circuit does not stop at the neck."
    ),
    publications=(
        Publication(
            authors="Bates AS, Phelps JS, Kim M, Yang HH, Matsliah A, et al.",
            year=2026,
            title="Distributed control circuits across a brain-and-cord connectome",
            journal="Nature",
            doi="10.1038/s41586-026-10735-w",
        ),
    ),
    links={
        "wiki": "https://github.com/jasper-tms/the-BANC-fly-connectome/wiki",
        "codex": "https://codex.flywire.ai/banc",
        "neuroglancer": "https://ng.banc.community/view",
        "data": "gs://lee-lab_brain-and-nerve-cord-fly-connectome",
    },
    backends=(
        # neuPrint first, and therefore the default: same snapshot, far quicker, and
        # it ships an ROI hierarchy that the CAVE datastack has no equivalent of.
        #
        # It is, for now, a *connectivity* mirror. `banc:v888` hosts no skeleton store
        # (HTTP 400: "no store found supporting the datatype and dataset") and BANC
        # publishes no flat volume - its only segmentation is the CAVE graphene one -
        # so meshes, cutouts and neuroglancer scenes have nothing to read either.
        # Declaring that here is what turns four confusing upstream failures (a 400, an
        # *empty* NeuronList, and two `None` sources) into one CapabilityError that
        # names the CAVE door. Delete the lines as the server gains the stores.
        BackendSpec(
            "neuprint",
            "neuprint.janelia.org/banc:v888",
            extra_capabilities={Cap.ROIS},
            # Imported from CAVE, and it kept CAVE's nanometres - the two doors report
            # the *identical* T-bar bounding box, to the nanometre. Scaling it by the
            # voxel size again put every synapse 4-45x out. Nothing caught it: BANC's
            # neuPrint door has no segmentation, so the T-bar round-trip test cannot
            # run there. `test_banc_backends_agree` compares the positions instead.
            position_units="nm",
            missing_capabilities={
                Cap.SKELETONS,      # no skeleton store on this server (yet)
                Cap.MESHES,         # no volume to read them from
                Cap.SEGMENTATION,   # ditto: BANC's only volume is CAVE's graphene
                Cap.NEUROGLANCER,   # a scene needs a segmentation source
            },
        ),
        BackendSpec(
            "cave",
            "brain_and_nerve_cord_public",
            default_version=888,
            synapse_table="synapses_v2",
            nucleus_table="somas_v1a",
        ),
    ),
    annotation_sources=(
        # codex_annotations is long-format: one row per (root_id, key, value),
        # with 32 distinct keys where other datasets would have 32 columns. It also
        # reliably fails to download in one request, hence chunked=True.
        AnnotationSource(
            "cave",
            "cave_table",
            "codex_annotations",
            id_column="pt_root_id",
            chunked=True,
            pivot=("classification_system", "cell_type"),
        ),
        AnnotationSource("flytable", "seatable", "banc.main", id_column="root_id", public=False),
    ),
    fields={
        "type": ("cell_type", "fafb_783_cell_type", "malecns_09_cell_type", "manc_121_cell_type"),
        "side": ("side",),
        "class": ("super_class", "cell_class"),
        "nt": ("neurotransmitter_verified", "neurotransmitter_predicted"),
        "status": ("status",),
    },
    side_map={"left": "left", "right": "right", "center": "center"},
    voxel_size=(4, 4, 45),
    capabilities=frozenset(
        {
            Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.ROI_CONN,
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH,
            Cap.SOMAS, Cap.NEUROGLANCER,
        }
    ),
    # Central-brain neurons, valid in CAVE mat 888 *and* neuPrint banc:v888 - the
    # two backends share an ID space, which is what makes the cross-backend
    # equivalence test in tests/ possible.
    example_ids=(720575941350526512, 720575941350528560),
)

register(BANC_SPEC)


def BANC(**kwargs):
    """BANC. Defaults to neuPrint; pass ``backend="cave"`` for the CAVE copy."""
    from ..backends import build

    return build(BANC_SPEC, **kwargs)
