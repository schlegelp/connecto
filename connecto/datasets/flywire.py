"""FlyWire (FAFB)."""

from __future__ import annotations

from ..core.registry import register
from ..core.spec import AnnotationSource, BackendSpec, Cap, DatasetSpec, Publication

__all__ = ["FLYWIRE", "FLYWIRE_PRODUCTION", "FlyWire"]

_PUBS = (
    Publication(
        authors="Dorkenwald S, Matsliah A, Sterling AR, Schlegel P, et al.",
        year=2024,
        title="Neuronal wiring diagram of an adult brain",
        journal="Nature",
        doi="10.1038/s41586-024-07558-y",
    ),
    Publication(
        authors="Schlegel P, Yin Y, Bates AS, Dorkenwald S, et al.",
        year=2024,
        title=(
            "Whole-brain annotation and multi-connectome cell typing of Drosophila"
        ),
        journal="Nature",
        doi="10.1038/s41586-024-07686-5",
    ),
    # The EM volume the whole thing is built on. Cite it too: the segmentation is
    # FlyWire's, the electrons are FAFB's.
    Publication(
        authors="Zheng Z, Lauritzen JS, Perlman E, Robinson CG, et al.",
        year=2018,
        title=(
            "A complete electron microscopy volume of the brain of adult "
            "Drosophila melanogaster"
        ),
        journal="Cell",
        doi="10.1016/j.cell.2018.06.019",
    ),
)

_LINKS = {
    "website": "https://flywire.ai/",
    "codex": "https://codex.flywire.ai/",
    "annotations": "https://github.com/flyconnectome/flywire_annotations",
}

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

# The snake_case spellings, which are the GitHub TSV's and FlyTable's. neuPrint
# spells the same concepts in camelCase and says so on its own AnnotationSource
# rather than here - a column name is a property of the table, not of the dataset.
# `side`, `status` and the soma point are named the same by all three, so they need
# no per-source entry at all.
_FIELDS = {
    "type": ("cell_type", "hemibrain_type"),
    "side": ("side",),
    # Three levels, three columns, no coalescing between them: `super_class` is
    # the flow/modality tier (optic, central, sensory, ...), `cell_class` the
    # familiar one (Kenyon_Cell, ALPN, CX, ...) and `cell_sub_class` finer still.
    # Coarse-first priority would have hidden `cell_class` behind a `super_class`
    # that 139,248 of 139,255 neurons carry.
    "superclass": ("super_class",),
    "class": ("cell_class",),
    "subclass": ("cell_sub_class",),
    "nt": ("known_nt", "top_nt"),
    "status": ("status",),
    "soma": ("soma_x", "soma_y", "soma_z"),  # neuPrint's somaLocation is split to these
}

# FlyWire already speaks left/right/center. "na" is not a side.
_SIDES = {"left": "left", "right": "right", "center": "center"}

# Per-synapse transmitter probabilities, as FlyWire's synapse tables spell them.
# The same six columns on the public view (`valid_synapses_nt_np_v6`) and on the
# production table (`synapses_nt_v1`) it is built from, so both backends share them.
#
# Six, and there is no histamine - even though FlyWire has plenty of histaminergic
# photoreceptors. The classifier has six classes, so the honest thing is six columns
# and an `nt` that can never say "histamine", rather than a seventh column of zeros
# implying it was considered and ruled out. (BANC's model, trained later, has eight.)
_NT_COLUMNS = {
    "ach": "acetylcholine",
    "gaba": "gaba",
    "glut": "glutamate",
    "oct": "octopamine",
    "ser": "serotonin",
    "da": "dopamine",
}

# FlyWire's live FlyTable annotations are *two* SeaTable tables in two bases - the
# central brain in `main.info`, the optic lobes in `optic_lobes.optic` -
# concatenated, exactly as cocoa assembles them. Reading only `info` silently drops
# ~89k optic-lobe neurons. The id column is `root_783`, not the live `root_id`: this
# is the frozen public release (materialization 783), so an id that tracks live
# edits would mis-join every neuron that has been edited since. Production, below,
# is live and keys on `root_id` instead.
_FLYTABLE = "main.info,optic_lobes.optic"

# neuPrint first, and therefore the default. Its `bodyId` *is* the CAVE root ID at
# materialization 783, so it joins onto everything either backend returns, and it
# needs no CAVE login and no SeaTable token. Reached through the spec's neuPrint
# backend whichever door is answering queries, so `backend="cave"` keeps it.
#
# It is very nearly the same table under different spelling - the mirror carries the
# hemilineages, `nerve`, `synonyms` and `dimorphism` as camelCase, and adds `statusLabel`,
# connectivity counts and the optic-column coordinates. What it does *not* have is
# `known_nt` and its citation - the measured transmitter, as opposed to the predicted
# one - plus `fbbt_id`, `nucleus_id`, `matching_notes` and the annotation point
# `pos_x/y/z`. All one argument away: `FlyWire(annotations="public")`.
_ANNOTATIONS = (
    AnnotationSource(
        "neuprint", "neuprint", id_column="bodyId",
        # Same concepts, neuPrint's spelling. Declared here rather than appended to
        # `_FIELDS` so the priority order within each source is exactly the order
        # that source intends - a dataset-wide union would only rank correctly for
        # as long as no two sources share a column name.
        fields={
            "type": ("type", "hemibrainType"),
            # The same three levels, camelCased. Spelling the middle one `class`
            # is why the priority list that used to stand here did real damage
            # rather than only ranking oddly: the coalesced result is written back
            # to `class`, so `superclass` first did not shadow the cell class, it
            # overwrote it, and 107,504 neurons lost it with no `class_raw` left.
            "superclass": ("superclass",),
            "class": ("class",),
            "subclass": ("subclass",),
            "nt": ("predictedNt",),
        },
    ),
    AnnotationSource("public", "github_tsv", ANNOTATIONS_URL, id_column="root_id"),
    AnnotationSource("flytable", "seatable", _FLYTABLE, id_column="root_783", public=False),
)

_CAPS = {
    Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES, Cap.SYNAPSE_SCORES,
    Cap.NT_PER_SYNAPSE, Cap.ROI_CONN, Cap.SKELETONS, Cap.MESHES,
    Cap.SEGMENTATION, Cap.CHUNKEDGRAPH, Cap.PROOFREADING, Cap.SOMAS, Cap.NEUROGLANCER,
    # Sparse volumes via the dense-read path - a chunkedgraph keeps no per-body
    # index, so this is expensive and `voxels.get` says so. Verified at scales 4
    # and 6: every voxel lands inside the neuron's own mesh bounding box.
    Cap.VOXELS,
    # Note: no Cap.L2CACHE. The `flywire_fafb_public` datastack genuinely has no
    # L2 cache (only production does) - the server says so, and declaring it here
    # would mean the skeleton fallback fails with an HTTP 500 instead of a clear
    # message. Skeletons still work: the CAVE skeleton service serves this stack.
}

FLYWIRE = DatasetSpec(
    name="flywire",
    label="FlyWire (FAFB) public release",
    species="Drosophila melanogaster",
    description=(
        "The whole brain of an adult female Drosophila at synapse resolution, "
        "proofread by the FlyWire community from the FAFB serial-section TEM volume. "
        "The v783 public release has 139,255 neurons and ~50M synapses, with "
        "comprehensive hierarchical annotations."
    ),
    publications=_PUBS,
    links=_LINKS,
    backends=(
        # neuPrint first, and therefore the default. It is the same release - its
        # body IDs *are* CAVE root IDs at materialization 783 - and it answers
        # connectivity queries in a fraction of the time, with an ROI hierarchy the
        # CAVE datastack has no equivalent of.
        #
        # It is also a narrower door, and connecto says so rather than papering over
        # it: no chunkedgraph (so no `update_ids`, no supervoxels), no proofreading,
        # and - the one that would otherwise bite silently - no per-synapse
        # transmitters, because this copy's Synapse nodes simply do not carry them.
        # All three raise here and point you at `backend="cave"`. See BACKEND_LIMITS.
        BackendSpec(
            "neuprint",
            "neuprint-cns.janelia.org/flywire-fafb:v783b",
            extra_capabilities={Cap.ROIS},
            # This mirror *is* CAVE materialization 783 - its body IDs are the root
            # IDs at 783 - so it reads the 783 skeleton bucket. Said explicitly
            # because its version string ("flywire-fafb:v783b") is not the bucket key.
            skeleton_version=783,
            # It was imported from CAVE and kept its nanometres, unlike every Janelia
            # FIB-SEM dataset on neuPrint, which reports 8 nm voxels. Verified against
            # the CAVE door: the same T-bar is (698640, 177936, 123200) here and
            # (698916, 169808, 115280) nm there. Take neuPrint's usual word for it and
            # every synapse position comes back 4-40x too big - silently, because the
            # edges are still right.
            position_units="nm",
            # FlyWire's voxels are real, but they are CAVE's. This mirror is a
            # neuPrint *import*, not a DVID deployment, so there is no sparsevol
            # endpoint behind it - unlike every Janelia dataset, where the neuPrint
            # door does front a DVID server. Denying it here is what makes the error
            # name the CAVE door instead of failing later with "cannot locate a DVID
            # server for 'flywire-fafb:v783b'".
            #
            # NT_PER_SYNAPSE for the reason in the comment above: verified against
            # the server, this mirror's Synapse nodes carry exactly `bodyId`,
            # `type`, `confidence`, `location` and their ROI flags - no `ntGabaProb`
            # and no siblings. Its *Neuron* nodes do carry `predictedNt` and six
            # probabilities, but that is a per-neuron call, not a per-synapse one,
            # and answering `transmitters=True` out of it would be a different
            # question than the one asked. Denied on this BackendSpec rather than
            # backend-wide: banc and manc and male-cns really do have them.
            missing_capabilities={Cap.VOXELS, Cap.NT_PER_SYNAPSE},
        ),
        BackendSpec(
            "cave",
            "flywire_fafb_public",
            default_version=783,
            # The *filtered* view, not the raw synapses_nt_v1 table: it is already
            # cleft-score filtered and neuropil-annotated.
            synapse_table="valid_synapses_nt_np_v6",
            # Pre-aggregated edges. Orders of magnitude cheaper than pulling every
            # synapse and grouping.
            edge_view="valid_connection_v2",
            nucleus_table="nuclei_v1",
            proofreading_table="proofread_neurons",
            nt_columns=_NT_COLUMNS,
        ),
    ),
    annotation_sources=_ANNOTATIONS,
    fields=_FIELDS,
    side_map=_SIDES,
    voxel_size=(4, 4, 40),
    template_space="FAFB14.1",
    # The *flat* segmentation of materialization 783 - which is exactly what the
    # public release is, so its root IDs are the ones in this volume. Preferred over
    # the graphene source because it loads without a CAVE login. Note this is only
    # true of a frozen release; see FLYWIRE_PRODUCTION below.
    segmentation_source="precomputed://gs://flywire_v141_m783",
    # FlyWire publishes precomputed skeletons, one bucket per materialization. A
    # plain HTTPS bucket - no login, no CAVE client - so *both* doors read it, which
    # is why it lives on the dataset and not on the CAVE backend. The public stack
    # has no L2 cache (so CAVE's skeleton service cannot serve it) and the neuPrint
    # mirror has no skeleton store at all, so this bucket is the only thing that
    # gives FlyWire skeletons on either backend.
    skeleton_source=(
        "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_{version}"
    ),
    # FlyWire's own neuroglancer, which is a fork old enough that it speaks a
    # different state schema than every other dataset here.
    viewer="https://ngl.flywire.ai",
    viewer_dialect="seunglab",
    # No Cap.LIVE: the public release is a frozen snapshot.
    capabilities=frozenset(_CAPS),
    example_ids=(720575940604407468, 720575940623543881),  # two DA1_lPN @ mat 783
)

FLYWIRE_PRODUCTION = FLYWIRE.evolve(
    name="flywire-production",
    label="FlyWire (FAFB) production",
    description=(
        "The live, actively-proofread FlyWire stack. Same brain as the public "
        "release, but a moving target: root IDs change with every edit."
    ),
    public=False,
    access=(
        "Needs FlyWire group membership, not just a token - an unprivileged account "
        "gets a 403, and a fresh token will not fix it. Ask a FlyWire admin. The "
        "public release (`flywire`) needs no such thing."
    ),
    backends=(
        BackendSpec(
            "cave",
            "flywire_fafb_production",
            default_version="latest",
            synapse_table="synapses_nt_v1",
            nucleus_table="nuclei_v1",
            proofreading_table="proofreading_status_public_v1",
            nt_columns=_NT_COLUMNS,
        ),
    ),
    # NOT the flat v783 volume it would otherwise inherit from FLYWIRE. Production
    # root IDs are live: they change with every edit and do not exist in a frozen
    # snapshot, so a scene built on the flat segmentation selects *nothing* - it
    # opens, it just shows an empty brain. `None` means "ask the info service", which
    # hands back the graphene source that actually tracks the chunkedgraph.
    segmentation_source=None,
    # Nor the precomputed skeletons, for the same reason: the buckets are published
    # per *materialization* of the frozen release, and a live root ID is in none of
    # them. Production has an L2 cache, so the CAVE skeleton service can serve it.
    skeleton_source=None,
    # Production is live and editable, and unlike the public stack it does have an
    # L2 cache.
    capabilities=frozenset(_CAPS | {Cap.LIVE, Cap.L2CACHE}),
    # Same two FlyTable tables, but keyed on the *live* `root_id`: production root IDs
    # are the current ones, so `root_783` (a frozen-release column) would be wrong
    # here in exactly the way it is right for the public release. The public GitHub
    # TSV stays as the default source - it is a 783 artefact, so on a live handle it
    # only lines up for neurons untouched since 783, but that is a pre-existing limit
    # of a static annotation file, not something the FlyTable source should inherit.
    annotation_sources=(
        AnnotationSource("public", "github_tsv", ANNOTATIONS_URL, id_column="root_id"),
        AnnotationSource("flytable", "seatable", _FLYTABLE, id_column="root_id", public=False),
    ),
    example_ids=(),
)

register(FLYWIRE)
register(FLYWIRE_PRODUCTION)


def FlyWire(release: str = "public", **kwargs):
    """FlyWire, the whole-brain FAFB reconstruction.

        fw = connecto.FlyWire()                  # public release, mat 783
        fw = connecto.FlyWire("production")      # live, needs permissions
    """
    from ..backends import build

    spec = {"public": FLYWIRE, "production": FLYWIRE_PRODUCTION}.get(release)
    if spec is None:
        raise ValueError(f"`release` must be 'public' or 'production', got {release!r}.")
    return build(spec, **kwargs)
