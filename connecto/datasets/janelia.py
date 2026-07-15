"""The neuPrint datasets: hemibrain, maleCNS, MANC, fish2.

No `optic-lobe`. It was a partial release - the optic lobes of the same specimen
that `malecns` now covers whole - so it was two names for one dataset, and the
narrower one could only ever give you fewer neurons and a different set of body
IDs for them. Use `malecns` and select the optic-lobe ROIs.
"""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec, Publication

__all__ = [
    "HEMIBRAIN", "MALECNS", "MANC_SPEC", "FISH2",
    "Hemibrain", "MaleCNS", "MANC", "Fish2",
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
    description=(
        "Dense FIB-SEM reconstruction of much of the central brain of a female "
        "Drosophila - one hemisphere, cropped, and no optic lobes. ~25,000 typed "
        "neurons and ~20M synapses. The dataset most published fly connectomics "
        "was built on, and still the reference against which new ones are matched."
    ),
    publications=(
        Publication(
            authors="Scheffer LK, Xu CS, Januszewski M, Lu Z, et al.",
            year=2020,
            title="A connectome and analysis of the adult Drosophila central brain",
            journal="eLife",
            doi="10.7554/eLife.57443",
        ),
        Publication(
            authors="Xu CS, Hayworth KJ, Lu Z, Grob P, et al.",
            year=2017,
            title="Enhanced FIB-SEM systems for large-volume 3D imaging",
            journal="eLife",
            doi="10.7554/eLife.25916",
        ),
    ),
    links={
        "website": "https://www.janelia.org/project-team/flyem/hemibrain",
        "neuprint": "https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1",
    },
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
    description=(
        "The complete central nervous system - brain and ventral nerve cord - of a "
        "male Drosophila: 166,691 neurons in 11,691 cell types, annotated with "
        "fruitless/doublesex expression. The counterpart that makes synapse-resolution "
        "male-vs-female comparison possible, and it subsumes the old `optic-lobe` "
        "release, which was these optic lobes on their own."
    ),
    publications=(
        Publication(
            authors="Berg S, Beckett IR, Costa M, Schlegel P, et al.",
            year=2025,
            title=(
                "Sexual dimorphism in the complete connectome of the Drosophila male "
                "central nervous system"
            ),
            journal="bioRxiv",  # still a preprint - there is no journal version yet
            doi="10.1101/2025.10.09.680999",
        ),
    ),
    links={
        "website": "https://male-cns.janelia.org/",
        "neuprint": "https://neuprint.janelia.org/?dataset=male-cns:v1.0",
        "clio": "https://clio.janelia.org/",
        "data": "https://male-cns.janelia.org/download/",
    },
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
# no CamelCase form to fall back on, unlike Hemibrain/MaleCNS. Same convention as
# BANC_SPEC and MICRONS_SPEC.
MANC_SPEC = DatasetSpec(
    name="manc",
    label="MANC (male VNC)",
    species="Drosophila melanogaster",
    description=(
        "Dense FIB-SEM connectome of the complete ventral nerve cord of an adult male "
        "Drosophila (~23,000 neurons), with motor neurons, descending and ascending "
        "neurons, hemilineages and predicted transmitters. The male counterpart to "
        "the CAVE-backed `fanc`."
    ),
    publications=(
        Publication(
            authors="Takemura S, Hayworth KJ, Huang GB, Januszewski M, et al.",
            year=2024,
            title="A connectome of the male Drosophila ventral nerve cord",
            journal="eLife",
            doi="10.7554/eLife.97769",
        ),
        Publication(
            authors="Marin EC, Morris BJ, Stürner T, Champion AS, et al.",
            year=2024,
            title=(
                "Systematic annotation of a complete adult male Drosophila nerve cord "
                "connectome reveals principles of functional organisation"
            ),
            journal="eLife",
            doi="10.7554/eLife.97766",
        ),
        Publication(
            authors="Cheong HS, Eichler K, Stürner T, Asinof SK, et al.",
            year=2025,
            title=(
                "Transforming descending input into motor output: an analysis of the "
                "Drosophila Male Adult Nerve Cord connectome"
            ),
            journal="eLife",
            doi="10.7554/eLife.96084",
        ),
    ),
    links={
        "website": "https://www.janelia.org/project-team/flyem/manc-connectome",
        "neuprint": "https://neuprint.janelia.org/?dataset=manc:v1.2.3",
    },
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

FISH2 = DatasetSpec(
    name="fish2",
    label="fish2 (larval zebrafish)",
    species="Danio rerio",
    description=(
        "A larval zebrafish connectome, served on its own Janelia neuPrint deployment. "
        "It is the one dataset here with no publication to point at: no paper, no "
        "preprint, no project page names it. So connecto describes it and cites "
        "nothing, rather than attaching someone else's fish paper to it."
    ),
    # Deliberately empty. There is a published larval zebrafish connectome (Petkova et
    # al. 2025, "fish1") and a forthcoming Janelia/Google one, and it would be easy to
    # cite either - but nothing links them to *this* server, and a citation that is
    # merely plausible is worse than none: it would be wrong in a methods section.
    publications=(),
    links={"neuprint": "https://neuprint-fish2.janelia.org/"},
    public=False,
    access=(
        "Its own neuPrint deployment (neuprint-fish2.janelia.org), which issues its "
        "own tokens - a token for neuprint.janelia.org is *not* valid here. Access "
        "appears to be granted per account; ask Janelia."
    ),
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

for _spec in (HEMIBRAIN, MALECNS, MANC_SPEC, FISH2):
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
Fish2 = _factory(FISH2)
