"""MICrONS - mouse visual cortex.

MICrONS is the forcing function that keeps connecto from quietly becoming a fly
library. It has no `side`, no fly-style `super_class`, and its cell types are
spread across several tables produced by different methods (a transformer model,
an SVM on nuclei, a hand-drawn column) rather than living in one canonical column.

So it ships with `fields={}` beyond the ID: annotations come back raw, with every
native column present and no invented `type`. That is the honest default. If you
want a `type`, say which table's opinion you want::

    mic = connecto.MICrONS(annotations="mtypes")
    mic.annotations.get(fields={"type": ("cell_type",)})
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

__all__ = ["MICRONS_SPEC", "MICrONS"]

MICRONS_SPEC = DatasetSpec(
    name="microns",
    label="MICrONS (minnie65) public",
    species="Mus musculus",
    backends=(
        BackendSpec(
            "cave",
            "minnie65_public",
            default_version="latest",
            synapse_table="synapses_pni_2",
            nucleus_table="nucleus_detection_v0",
        ),
    ),
    annotation_sources=(
        # Several tables, several opinions. Named, so the user picks one.
        AnnotationSource(
            "celltypes", "cave_table",
            "aibs_metamodel_celltypes_v661", id_column="pt_root_id",
        ),
        AnnotationSource(
            "mtypes", "cave_table",
            "aibs_metamodel_mtypes_v661_v2", id_column="pt_root_id",
        ),
        AnnotationSource(
            "nucleus_svm", "cave_table",
            "nucleus_ref_neuron_svm", id_column="pt_root_id",
        ),
    ),
    # Deliberately empty: MICrONS has no single canonical type column, and
    # inventing one would be exactly the kind of opinion connecto must not have.
    fields={},
    voxel_size=(4, 4, 40),
    capabilities=frozenset(
        {
            Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES,
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION,
            Cap.SOMAS, Cap.NEUROGLANCER,
            # No SYNAPSE_SCORES, no NT_PER_SYNAPSE, no ROI_CONN, no side.
            # Asking for min_score= or transmitters= here raises, loudly.
        }
    ),
    example_ids=(864691136274724621, 864691135489403194),
)

register(MICRONS_SPEC)


def MICrONS(**kwargs):
    """MICrONS minnie65, mouse visual cortex."""
    from ..backends import build

    return build(MICRONS_SPEC, **kwargs)
