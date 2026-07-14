"""The neuPrint datasets: hemibrain, maleCNS, MANC, optic-lobe, fish2."""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec

__all__ = [
    "HEMIBRAIN", "MALECNS", "MANC_SPEC", "OPTIC_LOBE", "FISH2",
    "Hemibrain", "MaleCNS", "MANC", "OpticLobe", "Fish2",
]

# What every neuPrint dataset can do. Body IDs are immutable: they are frozen at
# publication and nobody is proofreading them any more. So there is no chunkedgraph,
# no edit history and no live query.
#
# There *is* a segmentation, though - a flat `precomputed://` volume, which
# cloud-volume reads as happily as it reads a graphene one. What you cannot do is
# ask it for supervoxels or `update_ids`; what you can do is ask it what body sits
# at a point. Hence Cap.SEGMENTATION without Cap.CHUNKEDGRAPH - see `Cap` for why
# those are two capabilities and not one.
_NEUPRINT_CAPS = frozenset(
    {
        Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SYNAPSE_SCORES,
        Cap.ROI_CONN, Cap.ROIS, Cap.SKELETONS, Cap.MESHES, Cap.SOMAS,
    }
)

# Only add SEGMENTATION where we have actually verified a public volume: opened it
# with cloud-volume, and checked that a body's T-bars land back inside that body, so
# the volume's IDs really are the neuPrint body IDs. The alternative - assuming the
# obvious bucket name - is how you end up promising a segmentation that 404s, or one
# whose IDs belong to a different release.
_NEUPRINT_SEG_CAPS = _NEUPRINT_CAPS | {Cap.SEGMENTATION}

# Janelia's side vocabularies. hemibrain/maleCNS use L/R/M; MANC uses LHS/RHS.
_LRM = {"L": "left", "R": "right", "M": "center", "C": "center"}
_LHS = {"LHS": "left", "RHS": "right", "M": "center", "UNK": None}

_NP = AnnotationSource("neuprint", "neuprint", id_column="bodyId")


HEMIBRAIN = DatasetSpec(
    name="hemibrain",
    label="hemibrain",
    species="Drosophila melanogaster",
    backends=(BackendSpec("neuprint", "neuprint.janelia.org/hemibrain:v1.2.1"),),
    annotation_sources=(_NP,),
    fields={
        "type": ("type",),
        # hemibrain v1.2.1 has no somaSide column at all - side lives as a suffix
        # on the instance name ("DA1_lPN_R"), so we dig it out (see `derive`).
        "side": ("side_from_instance",),
        "status": ("status", "statusLabel"),
        "soma": ("soma_x", "soma_y", "soma_z"),
    },
    derive={"side_from_instance": ("instance", r"_([LRM])$")},
    side_map=_LRM,
    voxel_size=(8, 8, 8),
    template_space="JRCFIB2018Fraw",
    # Also the one dataset that advertises its volume in `neuroglancerMeta`, so
    # this is belt and braces - but pinning it means we do not depend on the
    # server keeping that field populated.
    segmentation_source="precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    # No `class` and no predicted transmitters in v1.2.1 - so they are simply
    # absent from the frame rather than present-and-empty.
    capabilities=_NEUPRINT_SEG_CAPS,
    example_ids=(1734350788, 1734350908),  # two DA1 lPNs
)

MALECNS = DatasetSpec(
    name="malecns",
    label="male CNS",
    species="Drosophila melanogaster",
    backends=(BackendSpec("neuprint", "neuprint-cns.janelia.org/male-cns:v1.0"),),
    annotation_sources=(
        _NP,
        # clio is the live curation DB; neuPrint is a snapshot of it.
        AnnotationSource("clio", "clio", "male-cns", id_column="bodyid", public=False),
    ),
    fields={
        # maleCNS carries several communities' opinions about type. Priority order
        # is data, and overridable per call:
        #   mcns.annotations.get(fields={"type": ("flywireType", "type")})
        "type": ("type", "flywireType", "hemibrainType", "mancType"),
        "side": ("somaSide", "rootSide"),
        "class": ("class", "subclass"),
        "nt": ("predictedNt",),
        "status": ("status",),
        "soma": ("soma_x", "soma_y", "soma_z"),
    },
    side_map=_LRM,
    voxel_size=(8, 8, 8),
    template_space="JRCFIB2022Mraw",
    segmentation_source="precomputed://gs://flyem-male-cns/v1.0/segmentation",
    capabilities=_NEUPRINT_SEG_CAPS,
    example_ids=(10001, 10002),
)

# Named MANC_SPEC, not MANC, so the *factory* below can be `MANC` - an acronym has
# no CamelCase form to fall back on, unlike Hemibrain/OpticLobe. Same convention as
# BANC_SPEC and MICRONS_SPEC.
MANC_SPEC = DatasetSpec(
    name="manc",
    label="MANC (male VNC)",
    species="Drosophila melanogaster",
    backends=(BackendSpec("neuprint", "neuprint.janelia.org/manc:v1.2.3"),),
    annotation_sources=(_NP,),
    fields={
        "type": ("type", "systematicType", "instance"),
        "side": ("somaSide", "rootSide"),
        "class": ("class", "subclass"),
        "nt": ("predictedNt",),
        "status": ("status",),
    },
    side_map=_LHS,
    voxel_size=(8, 8, 8),
    template_space="JRCVNC2018M",
    # The neuPrint DB is v1.2.3 and the volume is v1.2. That is not a version
    # mismatch - v1.2.3 is a database revision on the same segmentation - and the
    # T-bar check confirms it: MDN's synapses land in MDN.
    segmentation_source="precomputed://gs://manc-seg-v1p2/manc-seg-v1.2",
    capabilities=_NEUPRINT_SEG_CAPS,
    example_ids=(13438, 13809),  # two MDNs (moonwalker descending neurons)
)

OPTIC_LOBE = DatasetSpec(
    name="optic-lobe",
    label="optic lobe",
    species="Drosophila melanogaster",
    backends=(BackendSpec("neuprint", "neuprint.janelia.org/optic-lobe:v1.1"),),
    annotation_sources=(_NP,),
    fields={
        "type": ("type", "instance"),
        # optic-lobe:v1.1 has neither `somaSide` nor `class` - side lives as a
        # suffix on the instance ("Tm1_R"), exactly as in hemibrain, so we dig it
        # out the same way. Declaring `somaSide` here would not corrupt anything
        # (a missing source column drops the canonical column rather than
        # inventing an all-null one) - it would just silently mean *no side*.
        "side": ("side_from_instance",),
        "nt": ("consensusNt", "predictedNt"),
        "status": ("status", "statusLabel"),
    },
    derive={"side_from_instance": ("instance", r"_([LRM])$")},
    side_map=_LRM,
    voxel_size=(8, 8, 8),
    segmentation_source="precomputed://gs://flyem-optic-lobe/v1.1/segmentation",
    capabilities=_NEUPRINT_SEG_CAPS,
    example_ids=(41566, 43090),  # two Tm1
)

FISH2 = DatasetSpec(
    name="fish2",
    label="fish2 (larval zebrafish)",
    species="Danio rerio",
    # Note: no version suffix. Nothing in connecto may assume `name:vX.Y`.
    backends=(BackendSpec("neuprint", "neuprint-fish2.janelia.org/fish2"),),
    annotation_sources=(_NP,),
    fields={
        "type": ("type", "proposedType", "connectivityType"),
        "side": ("somaSide",),
        "class": ("class",),
        "status": ("status",),
    },
    side_map=_LRM,
    voxel_size=(16, 16, 15),
    # No SEGMENTATION: fish2 has no public segmentation volume that we could find.
    # The server does not populate `neuroglancerMeta`, its API is entirely behind a
    # login, no paper cites it, and none of the plausible buckets exist. So connecto
    # says it has no segmentation - which is the honest answer, and a better one than
    # a guessed URL that 404s at the first lookup.
    capabilities=_NEUPRINT_CAPS,
    example_ids=(100000001, 100000123),
)

for _spec in (HEMIBRAIN, MALECNS, MANC_SPEC, OPTIC_LOBE, FISH2):
    register(_spec)


def _factory(spec):
    def make(**kwargs):
        from ..backends import build

        return build(spec, **kwargs)

    make.__name__ = spec.name
    make.__doc__ = f"{spec.label} ({spec.species}), via neuPrint."
    return make


Hemibrain = _factory(HEMIBRAIN)
MaleCNS = _factory(MALECNS)
MANC = _factory(MANC_SPEC)
OpticLobe = _factory(OPTIC_LOBE)
Fish2 = _factory(FISH2)
