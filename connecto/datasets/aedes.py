"""Aedes - the mosquito brain (Wei-Chung Lee lab).

The CAVE datastack is bare: a synapse table, a nucleus table, and no annotation table
at all - no column that means "type" or "side". The project keeps its cell typing
elsewhere, in FlyTable (the lab's SeaTable database): the `aedes_main` table of the
`aedes` base. So connecto reads annotations from there rather than from CAVE:

    aedes.connectivity.edges(root_ids)   # from CAVE
    aedes.ids("class:KC")                # from FlyTable

FlyTable is lab-internal - it needs a SEATABLE_TOKEN on top of CAVE access, and there
is no public alternative - so the source is `public=False`. Because it is the only
source, `annotations="auto"` falls back to it (see `DatasetSpec.annotation_source`);
a caller without the token gets a clear missing-token error, not a silently empty
frame.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import (
    AnnotationSource,
    BackendSpec,
    Cap,
    DatasetSpec,
    SparseVolSource,
)

# aedes_main records side as L / R / M (with ~530 blanks, which are not a side).
_SIDES = {"L": "left", "R": "right", "M": "center"}

__all__ = ["AEDES_SPEC", "Aedes"]

AEDES_SPEC = DatasetSpec(
    name="aedes",
    label="Aedes (mosquito brain)",
    species="Aedes aegypti",
    description=(
        "An EM volume of the brain of the yellow fever mosquito, from the Wei-Chung "
        "Lee lab - the substrate for an in-progress whole-brain connectome (a Wellcome "
        "Discovery Award, with Jefferis, Marin and Younger). There is no paper for this "
        "dataset yet, so connecto cites none."
    ),
    # Deliberately empty. The Lee lab *has* a published mosquito preprint (Bao et al.
    # 2025, on CO2 sensitivity), but its Data Availability statement describes a
    # different, partial volume - the posterior antennal lobes, served over CATMAID -
    # and never mentions this datastack. Citing it here would put the wrong paper in
    # somebody's methods section, so: nothing, until there is a paper for *this*.
    publications=(),
    links={
        "project": (
            "https://flyconnecto.me/2025/05/07/"
            "new-project-an-aedes-aegypti-brain-connectome/"
        ),
    },
    public=False,
    access=(
        "Pre-publication. The `wclee_aedes_brain` datastack is not documented publicly "
        "and appears to be restricted to project members - ask the Lee lab or the "
        "whole-brain connectome project for a CAVE account with access."
    ),
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
    annotation_sources=(
        # Not a CAVE table - the datastack has none - but FlyTable (SeaTable): the
        # `aedes_main` table in the `aedes` base. Lab-internal, so `public=False`
        # and it needs a SEATABLE_TOKEN. `root_id` is a CAVE root ID, so it joins
        # straight onto everything the CAVE backend returns.
        AnnotationSource(
            "flytable", "seatable", "aedes.aedes_main",
            id_column="root_id", public=False,
        ),
    ),
    fields={
        "type": ("type", "flywire_type"),
        "side": ("side",),
        # aedes_main has both a fine `class` (KC, CX, ALSN, LHN, ...) and a coarse
        # `superclass` (cb_intrinsic, cb_sensory, visual_projection, ...). The fine
        # one wins so `ds.ids("class:KC")` works; `superclass` falls in behind it for
        # the ~3.6k neurons with no fine class, and both survive as raw columns.
        "class": ("class", "superclass"),
        "nt": ("neurotransmitter_verified",),
        "status": ("status",),
        # No `soma`: the flytable's `soma_xyz` is one voxel-space "x,y,z" string, and
        # somas already come from the nucleus table in nm (`.somas`). Promoting it to
        # soma_x/y/z would quietly mix voxels into an nm frame, so it stays raw.
    },
    side_map=_SIDES,
    voxel_size=(16, 16, 45),
    # A lookup service that keeps the per-body index the chunkedgraph does not, so
    # aedes gets sparse volumes in one request instead of the several-hundred-block
    # dense read every other CAVE dataset needs (see `connecto.voxels.pcg`).
    #
    # `scales=(1,)` is the literal truth and not a placeholder: scale 0 is refused
    # ("would require reading 6,039,797,760 voxels"), and scales 2 and 3 answer a
    # bare `500 Internal Server Error`. Declaring it means asking for scale 2 gets a
    # sentence about which scales exist rather than the server's stack trace.
    #
    # `downsample=(2, 2, 1)` because this pyramid halves X and Y only - so scale 1 is
    # 32x32x45 nm, not 32x32x90. Assuming isotropy would stretch every neuron 2x in Z
    # and look entirely plausible while doing it.
    #
    # Known ceiling: the service refuses any segment spanning more than 256 chunks
    # ("Use a coarser scale" - which, with one scale, cannot be done). The largest
    # neurons here are therefore simply not available as sparse volumes; the first of
    # `example_ids` below is one of them, at 370 chunks. Skeletons and meshes for
    # those neurons are unaffected.
    sparsevol_source=SparseVolSource(
        url=(
            "https://flyem.mrc-lmb.cam.ac.uk/transform-service/sparsevol"
            "/dataset/wclee_aedes_brain/s/{scale}/root/{id}"
        ),
        scales=(1,),
        downsample=(2, 2, 1),
    ),
    capabilities=frozenset(
        {
            # Annotations come from FlyTable, not from the (annotation-less) datastack.
            # No Cap.SYNAPSE_SCORES - the synapse table has `size`, not a score.
            Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SKELETONS,
            Cap.MESHES, Cap.L2CACHE, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH, Cap.SOMAS,
            Cap.NEUROGLANCER, Cap.LIVE, Cap.VOXELS,
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
