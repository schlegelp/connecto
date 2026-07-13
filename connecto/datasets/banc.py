"""BANC - the Brain And Nerve Cord dataset.

BANC is the reason connecto treats backend and dataset as orthogonal: it is served
by CAVE (``brain_and_nerve_cord_public``, materialization 888) *and* by neuPrint
(``banc:v888``) - the same snapshot, two backends. So::

    connecto.BANC()                     # CAVE
    connecto.BANC(backend="neuprint")   # neuPrint

which also gives us a real test that the normalisation is genuine rather than
aspirational: both should return the same edges.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

__all__ = ["BANC_SPEC", "BANC"]

BANC_SPEC = DatasetSpec(
    name="banc",
    label="BANC (brain and nerve cord)",
    species="Drosophila melanogaster",
    backends=(
        BackendSpec(
            "cave",
            "brain_and_nerve_cord_public",
            default_version=888,
            synapse_table="synapses_v2",
            nucleus_table="somas_v1a",
        ),
        BackendSpec("neuprint", "neuprint.janelia.org/banc:v888"),
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
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION,
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
    """BANC. Pass ``backend="neuprint"`` for the neuPrint copy."""
    from ..backends import build

    return build(BANC_SPEC, **kwargs)
