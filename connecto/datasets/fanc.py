"""FANC - the Female Adult Nerve Cord.

The female counterpart to MANC, but a CAVE dataset rather than a neuPrint one: an
editable segmentation with a chunkedgraph, so root IDs move and `version="auto"`
earns its keep.

Two things about FANC are not like the other CAVE datasets, and both are handled in
the spec rather than in the backend:

1. Its synapse table calls the confidence score ``score``, not ``cleft_score``. Same
   concept, different name - hence ``score_column``.
2. Its annotations are *long*: `neuron_information` is one row per (neuron, tag),
   with the category in ``tag2`` and the value in ``tag``. So a motor neuron carries
   ``tag2="primary class", tag="motor neuron"``. Pivoting it wide turns each category
   into a column, which is what the rest of connecto expects.

FANC is not public. It needs the ``FANC_edit`` group, so an unprivileged token gets a
403 - which connecto reports as a *permission* problem, not a bad token.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

__all__ = ["FANC_SPEC", "FANC"]

FANC_SPEC = DatasetSpec(
    name="fanc",
    label="FANC (female VNC)",
    species="Drosophila melanogaster",
    backends=(
        BackendSpec(
            "cave",
            "fanc_production_mar2021",
            # Every FANC materialization except v840 expires within days, so there
            # is no stable snapshot to pin to. v840 (Jan 2024) *is* permanent, but
            # pinning it would cost most of the annotations: at v840 only 95 of
            # 22k neurons carry a cell type and `hemilineage` does not exist at
            # all, against 4,265 and 9,715 today. Current annotations win.
            default_version="latest",
            synapse_table="synapses_nov2022",
            nucleus_table="neuron_somas_dec2022",
            proofreading_table="proofreading_status_table_v0",
            score_column="score",
        ),
    ),
    annotation_sources=(
        # Long-format: (pt_root_id, tag2, tag) where tag2 is the *category*
        # ("primary class", "soma side", "hemilineage", ...) and tag the value.
        # Pivoting gives one column per category - see the module docstring.
        AnnotationSource(
            "cave",
            "cave_table",
            "neuron_information",
            id_column="pt_root_id",
            pivot=("tag2", "tag"),
        ),
    ),
    # Column names below are the *pivoted* ones, i.e. the distinct `tag2` values,
    # spaces and all.
    fields={
        "type": ("neuron identity",),
        "side": ("soma side",),
        "class": ("primary class",),
        # No `nt`: exactly one neuron in the whole table carries a
        # "fast neurotransmitter" tag, and a field backed by one row is a
        # promise connecto cannot keep.
    },
    side_map={"left soma": "left", "right soma": "right", "midline soma": "center"},
    voxel_size=(4.3, 4.3, 45),
    template_space="JRCVNC2018F",
    capabilities=frozenset(
        {
            Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SYNAPSE_SCORES,
            Cap.SKELETONS, Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH,
            Cap.PROOFREADING, Cap.SOMAS, Cap.NEUROGLANCER, Cap.LIVE,
        }
    ),
    # Two MDNs (moonwalker descending neurons). Safe to pin on a live, editable
    # stack because they are published and finished: their root IDs are unchanged
    # across every materialization from v840 (Jan 2024) to today.
    example_ids=(648518346474413506, 648518346475400628),
)

register(FANC_SPEC)


def FANC(**kwargs):
    """FANC, the female adult nerve cord. Needs the ``FANC_edit`` group."""
    from ..backends import build

    return build(FANC_SPEC, **kwargs)
