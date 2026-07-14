"""Aedes - the mosquito brain (Wei-Chung Lee lab).

Notable mostly for what it does *not* have: no annotation table of any kind. The
datastack ships a synapse table, a nucleus table, and nothing else - so there is no
column that means "type" or "side", and connecto says so rather than guessing:

    aedes.connectivity.edges(root_ids)   # fine
    aedes.ids("SomeType")                # raises: aedes has no annotations

That is the whole point of `fields` being data. An empty `fields` is not a stub to be
filled in later; it is the honest description of this datastack.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import BackendSpec, Cap, DatasetSpec

__all__ = ["AEDES_SPEC", "Aedes"]

AEDES_SPEC = DatasetSpec(
    name="aedes",
    label="Aedes (mosquito brain)",
    species="Aedes aegypti",
    backends=(
        BackendSpec(
            "cave",
            "wclee_aedes_brain",
            # Every aedes materialization expires within weeks; there is no stable
            # snapshot to pin to.
            default_version="latest",
            # `synapses`, deliberately - NOT the newer, larger `synapses_v2`.
            #
            # Both hold positions in nanometres, but CAVE has `synapses_v2`
            # registered at the datastack's 16x16x45 voxel size. connecto asks CAVE
            # for nanometres (desired_resolution=[1,1,1]), so the server dutifully
            # multiplies the already-nm coordinates *again* and hands back a
            # mosquito brain 9.9 mm deep. Edges would be fine; every synapse
            # position would be silently wrong by 16-45x.
            #
            # So: 90.5M correct synapses beat 111.3M corrupt ones. Switch to
            # `synapses_v2` once its resolution is registered as (1, 1, 1) upstream
            # - and check `ctr_pt_position` against this comment before you do.
            synapse_table="synapses",
            nucleus_table="nuclei_v1_aedes",
        ),
    ),
    # No annotation source: the datastack has no annotation table at all.
    annotation_sources=(),
    fields={},
    voxel_size=(16, 16, 45),
    capabilities=frozenset(
        {
            # No Cap.ANNOTATIONS - there is nothing to annotate with.
            # No Cap.SYNAPSE_SCORES - the synapse table has `size`, not a score.
            Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SKELETONS, Cap.MESHES,
            Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH, Cap.SOMAS,
            Cap.NEUROGLANCER, Cap.LIVE,
        }
    ),
    # Two heavily-connected neurons, unchanged across every live materialization.
    # Pinned IDs are a bet on a live stack - aedes root IDs churn ~2%/month - so if
    # the conformance suite starts reporting no edges for these, they were edited:
    # re-pick rather than assuming connecto broke.
    example_ids=(648518347529750614, 648518347481448779),
)

register(AEDES_SPEC)


def Aedes(**kwargs):
    """The Aedes aegypti mosquito brain, via CAVE."""
    from ..backends import build

    return build(AEDES_SPEC, **kwargs)
