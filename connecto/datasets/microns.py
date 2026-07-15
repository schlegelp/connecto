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
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec, Publication

__all__ = ["MICRONS_SPEC", "MICrONS"]

MICRONS_SPEC = DatasetSpec(
    name="microns",
    label="MICrONS (minnie65) public",
    species="Mus musculus",
    description=(
        "A cubic millimetre of mouse visual cortex - ~1.4 x 0.87 x 0.84 mm of "
        "serial-section EM spanning all six layers of VISp and neighbouring higher "
        "visual areas, with >200,000 cells and ~0.5 billion synapses, co-registered "
        "with two-photon calcium imaging of ~75,000 neurons in the same awake animal. "
        "Structure and function in one volume."
    ),
    publications=(
        Publication(
            authors="The MICrONS Consortium (Bae JA, Baptiste M, Bishop CA, et al.)",
            year=2025,
            title=(
                "Functional connectomics spanning multiple areas of mouse visual cortex"
            ),
            journal="Nature",
            doi="10.1038/s41586-025-08790-w",
        ),
        Publication(
            authors="Schneider-Mizell CM, Bodor AL, Brittain D, Buchanan J, et al.",
            year=2025,
            title="Inhibitory specificity from a connectomic census of mouse visual cortex",
            journal="Nature",
            doi="10.1038/s41586-024-07780-8",
        ),
    ),
    links={
        "website": "https://www.microns-explorer.org/cortical-mm3",
        "neuroglancer": "https://ngl.microns-explorer.org",
        "data": "https://bossdb.org/project/microns-minnie",
    },
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
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH,
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
