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
from ..core.spec import (
    AnnotationSource,
    BackendSpec,
    Cap,
    DatasetSpec,
    Publication,
    neuprint_nt_columns,
)

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
                # BANC's voxels are real, but they are CAVE's. This mirror is a
                # neuPrint *import*, not a DVID deployment, so there is no sparsevol
                # endpoint behind it - unlike every Janelia dataset, where the
                # neuPrint door does front a DVID server. Denying it here is what
                # makes the error name the CAVE door instead of failing later with
                # "cannot locate a DVID server".
                Cap.VOXELS,
            },
            # Eight transmitters, on the Synapse nodes - verified against the server
            # and against a real body: 720575941350526512's presynapses come back
            # serotonin-dominant, matching its own `predictedNt`. This is the wider
            # door of the two for transmitters: BANC's *CAVE* copy keeps the same
            # predictions in a reference table that only stores the winner, so the
            # full probability vector is here and nowhere else.
            nt_columns=neuprint_nt_columns(
                "acetylcholine",
                "gaba",
                "glutamate",
                "dopamine",
                "serotonin",
                "octopamine",
                "histamine",
                "tyramine",
            ),
        ),
        BackendSpec(
            "cave",
            "brain_and_nerve_cord_public",
            default_version=888,
            synapse_table="synapses_v2",
            nucleus_table="somas_v1a",
            # Not columns on the synapse table: BANC keeps its predictions in a
            # separate table, already argmaxed - one row per predicted synapse with
            # the winning transmitter and its probability. Same eight classes as the
            # neuPrint copy, but this door cannot give you the runner-up.
            #
            # The *view*, not the `synapses_v2_nt_prediction_5` reference table it
            # wraps, and that is not cosmetic: the reference table can only be
            # filtered by its own `target_id`, so using it would mean fetching the
            # synapses, collecting their ids and sending them back as a filter - two
            # round trips and a filter list the size of the neuron. The view exposes
            # `pre_pt_root_id`/`post_pt_root_id`, so it takes the *same* filter as
            # the synapse query, and it has already done the left join server-side.
            #
            # Only synapses of size >= 5 were predicted, so this covers a subset.
            # connecto joins it onto the synapse frame rather than querying it
            # instead, so `transmitters=True` returns the same synapses as
            # `transmitters=False` - the unpredicted ones just have a null `nt`.
            nt_table="synapses_v2_nt_prediction_5_human_readable",
        ),
    ),
    annotation_sources=(
        # neuPrint first, and therefore the default. Its `bodyId` *is* the CAVE root
        # ID at materialization 888, so it joins onto everything either backend
        # returns, and unlike `codex_annotations` it needs no CAVE login - which
        # matters most on the neuPrint door, where annotations were the one thing
        # still reaching across to CAVE.
        #
        # It is also much the better source for transmitters: 153,986 bodies carry
        # `neurotransmitterPredicted` here against codex's 82,286, which is the
        # FlyTable figure without the FlyTable token.
        #
        # The exception is `side`, and it is a big one - see `fields` below.
        AnnotationSource(
            "neuprint", "neuprint", id_column="bodyId",
            # 8,153 of 175,420 bodies carry a `side` here, against all 158,250 in
            # codex. Declared empty rather than left to look like a side column
            # that happens to be mostly blank - see `AnnotationSource.fields`.
            fields={"side": ()},
        ),
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
        # BANC's live annotations live on the *official* SeaTable cloud (base
        # `banc_meta`), not the lab's own flytable instance - so `instance="seatable"`.
        # Like FlyWire, this is a frozen release (CAVE mat 888), so it keys on the
        # `root_888` column, not the live `root_id` that tracks edits; a neuron edited
        # since 888 would otherwise mis-join in silence.
        AnnotationSource(
            "flytable", "seatable", "banc_meta.banc_meta",
            instance="seatable", id_column="root_888", public=False,
        ),
    ),
    # Every spelling any source uses, in priority order. The sources are
    # alternatives, never merged, so only one frame's columns are present at a time
    # and the extra names cost nothing. snake_case is codex's and FlyTable's,
    # camelCase neuPrint's.
    fields={
        "type": (
            "cell_type", "fafb_783_cell_type", "malecns_09_cell_type",
            "manc_121_cell_type", "type", "fafbCellType", "malecnsCellType",
        ),
        # Both sources spell it `side`, so one entry covers both - but they are not
        # equally populated, and the default one is not usable: see the `fields`
        # override on the neuPrint source above, which declares it absent so that
        # `ids(side=...)` refuses instead of answering from 5% of the dataset.
        "side": ("side",),
        "class": ("super_class", "cell_class", "superclass", "class"),
        "nt": (
            "neurotransmitter_verified", "neurotransmitter_predicted",
            "neurotransmitterVerified", "neurotransmitterPredicted",
        ),
        "status": ("status",),
        # Only the neuPrint source has these (codex carries no soma point at all);
        # naming them makes the column canonical - ordered, float32, and put through
        # the unit conversion - rather than raw passthrough that happens to be right.
        "soma": ("soma_x", "soma_y", "soma_z"),
    },
    side_map={"left": "left", "right": "right", "center": "center"},
    voxel_size=(4, 4, 45),
    capabilities=frozenset(
        {
            Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.ROI_CONN,
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH,
            Cap.SOMAS, Cap.NEUROGLANCER,
            Cap.VOXELS,
            # Both doors have per-synapse transmitters, in two different shapes -
            # see `nt_columns` on the neuPrint backend and `nt_table` on the CAVE
            # one. Declared on the dataset because the *data* has them; which shape
            # you get is the backend's business.
            Cap.NT_PER_SYNAPSE,
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
