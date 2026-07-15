"""Declarative description of a dataset.

A dataset is *data*, not a class hierarchy. Everything connecto needs to know in
order to query a connectome - which backends serve it, which tables hold what,
which columns mean "type" or "side", what it can and cannot do - lives in a
frozen :class:`DatasetSpec`.

This matters because backend and dataset are orthogonal: BANC is served by *both*
CAVE (``brain_and_nerve_cord_public``, materialization 888) and neuPrint
(``banc:v888``) - the same snapshot, two backends. A rigid ``CaveDataset`` /
``NeuPrintDataset`` class split cannot express that; a spec with a tuple of
backends can.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

__all__ = [
    "Cap",
    "BackendSpec",
    "AnnotationSource",
    "DatasetSpec",
    "Publication",
    "BACKEND_LIMITS",
    "MULTI_SEP",
    "DIALECTS",
]

# The neuroglancer state schemas we can emit. See `DatasetSpec.viewer_dialect`.
DIALECTS = ("modern", "seunglab")

# How a set of values is encoded in one cell (see `AnnotationSource.pivot`).
# `sources.pivot_long` joins on it; `criteria._match` splits on it. Defined once so
# the two can never disagree - when they did, a FANC neuron tagged "MDN" was stored
# as "MDN, MDN3, moonwalker descending neuron" and `ds.ids("MDN")` found nothing.
MULTI_SEP = ", "


class Cap(str, Enum):
    """A thing a dataset can do.

    Capabilities are declared per dataset and enforced in two places: whole
    namespaces (``.segmentation`` simply does not exist without ``SEGMENTATION``)
    and individual arguments (``min_score=`` raises without ``SYNAPSE_SCORES``
    rather than being silently ignored).
    """

    ANNOTATIONS = "annotations"
    CONNECTIVITY = "connectivity"  # edge lists
    SYNAPSES = "synapses"  # synapse-resolution data
    SYNAPSE_SCORES = "synapse_scores"  # per-synapse confidence/cleft score
    NT_PER_SYNAPSE = "nt_per_synapse"  # per-synapse transmitter predictions
    ROI_CONN = "roi_connectivity"  # edges broken down by ROI/neuropil
    ROIS = "rois"  # ROI hierarchy + meshes
    SKELETONS = "skeletons"
    MESHES = "meshes"
    L2CACHE = "l2cache"

    # Two capabilities, because there are two different things people mean by
    # "segmentation", and a dataset can have the first without the second.
    #
    # SEGMENTATION: there is a segmentation *volume* you can query - "what body is
    # at this point", "give me the voxels of this body". Every CAVE dataset has one
    # (graphene), and so does every published neuPrint dataset (a flat precomputed
    # volume, advertised in `Client.meta`). It just needs cloud-volume.
    #
    # CHUNKEDGRAPH: the volume is a *proofreadable graph* - supervoxels underneath
    # root IDs, root IDs that change when someone makes an edit, and therefore
    # `update_ids`, `is_latest_root`, edit history. CAVE only.
    #
    # Collapsing the two would mean either denying that hemibrain has a
    # segmentation (it does) or promising `update_ids` on a dataset whose IDs are
    # immutable and have no supervoxels beneath them (it isn't). Both are lies, so:
    # two capabilities.
    SEGMENTATION = "segmentation"  # a queryable segmentation volume
    CHUNKEDGRAPH = "chunkedgraph"  # supervoxels, root-ID history, update_ids (CAVE)

    PROOFREADING = "proofreading"  # edit history, proofreading status (CAVE)
    SOMAS = "somas"
    LIVE = "live"  # non-materialized "right now" queries
    NEUROGLANCER = "neuroglancer"

    def __str__(self):
        return self.value


# What a backend cannot do, whatever the dataset asks for.
#
# A capability is a property of a *dataset seen through a backend*, not of the
# dataset alone. BANC and FlyWire are each served by both backends, and the two
# doors are not equally wide: FlyWire has a chunkedgraph, but you cannot reach it
# through neuPrint. So the dataset declares what the *data* has, the backend
# subtracts what it cannot deliver, and `DatasetSpec.capabilities_for` is the only
# thing anyone asks.
#
# Getting this wrong is not cosmetic. Before this existed, `FlyWire(backend=
# "neuprint").supports(CHUNKEDGRAPH)` returned True and every chunkedgraph call
# raised anyway; `synapses(x, transmitters=True)` returned a frame with no `nt`
# column and no error - the exact fafbseg failure this library exists to prevent.
BACKEND_LIMITS: Mapping[str, frozenset[Cap]] = {
    "cave": frozenset(),
    "neuprint": frozenset(
        {
            # Structural. neuPrint serves frozen snapshots of a finished
            # segmentation: there are no supervoxels under a body ID, no edit
            # history, no L2 cache and nothing to query "right now".
            Cap.CHUNKEDGRAPH,
            Cap.PROOFREADING,
            Cap.L2CACHE,
            Cap.LIVE,
            # Not structural - a gap in *our* backend. BANC's neuPrint copy really
            # does carry per-synapse transmitter probabilities (`ntGabaProb`, ...)
            # on its Synapse nodes; FlyWire's copy does not carry them at all. Our
            # `_fetch_synapses` fetches neither. Until it does, claiming the
            # capability would mean `transmitters=True` was silently ignored, so it
            # is denied here and `transmitters=True` raises. Delete this line the
            # day the backend learns to read those properties.
            Cap.NT_PER_SYNAPSE,
        }
    ),
}


@dataclass(frozen=True)
class BackendSpec:
    """How one backend serves one dataset.

    A dataset may have several of these - see the module docstring.
    """

    kind: str  # "cave" | "neuprint"
    source: str  # CAVE datastack, or "server/dataset[:version]"
    default_version: object = "latest"

    # Capabilities this backend adds to the dataset's, because the door is *wider*
    # than the dataset's baseline. Both neuPrint copies of FlyWire and BANC ship a
    # full ROI hierarchy that the CAVE datastack has no equivalent of, so the same
    # dataset gains `ds.rois` when you come in through neuPrint.
    extra_capabilities: frozenset[Cap] = frozenset()

    # What units *this server* reports positions in. None means the backend's usual
    # convention (CAVE asks for nm outright; neuPrint hands back voxels).
    #
    # It has to be per-pairing, because it is not a property of either alone. The
    # Janelia FIB-SEM datasets on neuPrint (hemibrain, MANC, maleCNS) report 8 nm
    # voxels, as expected - but the neuPrint *mirrors* of FlyWire and BANC were
    # imported from CAVE and kept its nanometres. Taking the backend's word for it
    # multiplied every synapse position by the voxel size again: FlyWire T-bars came
    # back 4-40x out, BANC's 4-45x, and nothing said a word. Edges were fine; only
    # the coordinates were wrong, which is the kind of bug that survives to
    # publication. Verified per dataset against the CAVE door, which is the ruler.
    position_units: str | None = None

    # Which key to interpolate into `DatasetSpec.skeleton_source`. None means "the
    # version this handle resolved to", which is right for CAVE - ask for
    # materialization 630 and you want the 630 skeletons.
    #
    # neuPrint needs to say it explicitly, because its version *string* is not the
    # bucket key: the mirror calls itself `flywire-fafb:v783b` and the bucket is
    # `flywire_skeletons_783`. Interpolating the former gives a 404 for every neuron.
    skeleton_version: object = None

    # ...and capabilities this *particular* pairing cannot serve, though both the
    # dataset and the backend can in general. BACKEND_LIMITS is for what a backend
    # can never do; this is for what one server happens not to host.
    #
    # BANC's neuPrint mirror is the case in point: neuPrint serves skeletons and
    # meshes perfectly well for hemibrain, but `banc:v888` has no skeleton store
    # (HTTP 400, "no store found supporting the datatype and dataset") and no volume
    # of its own - BANC's only segmentation is the CAVE graphene one. So that door is
    # connectivity, annotations and ROIs, and it says so instead of 400ing at you.
    missing_capabilities: frozenset[Cap] = frozenset()

    # --- CAVE table wiring. None means "probe at runtime".
    synapse_table: str | None = None
    edge_view: str | None = None  # pre-aggregated edges; much faster than synapses
    nucleus_table: str | None = None
    proofreading_table: str | None = None

    # What the synapse table calls its confidence score. FlyWire says `cleft_score`;
    # FANC says plain `score`. Same concept, different column - so this is a name,
    # not a capability. Whether the dataset *has* scores is Cap.SYNAPSE_SCORES.
    score_column: str = "cleft_score"

    def __post_init__(self):
        if self.kind not in BACKEND_LIMITS:
            raise ValueError(f"Unknown backend kind: {self.kind!r}")
        object.__setattr__(
            self, "extra_capabilities", frozenset(self.extra_capabilities)
        )
        object.__setattr__(
            self, "missing_capabilities", frozenset(self.missing_capabilities)
        )
        if self.position_units not in (None, "nm", "voxel"):
            raise ValueError(
                f"`position_units` must be 'nm', 'voxel' or None, got "
                f"{self.position_units!r}."
            )
        # A backend cannot add what it is structurally unable to serve. Catching
        # this here means the contradiction is impossible to register, rather than
        # being discovered by a user whose `update_ids` call raises.
        impossible = self.extra_capabilities & BACKEND_LIMITS[self.kind]
        if impossible:
            raise ValueError(
                f"The {self.kind!r} backend cannot serve "
                f"{', '.join(sorted(str(c) for c in impossible))}, so "
                f"{self.source!r} may not add it."
            )


@dataclass(frozen=True)
class Publication:
    """A paper to cite when you use a dataset.

    Datasets are other people's years of work. connecto is in the unusual position
    of knowing exactly which dataset you just queried, so it can also tell you who
    to credit for it - and `ds.cite()` is a great deal harder to forget than a
    citation buried in a README.
    """

    authors: str  # "Dorkenwald S, Matsliah A, Sterling AR, et al."
    year: int
    title: str
    journal: str = ""
    doi: str = ""

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""

    def __str__(self) -> str:
        bits = [f"{self.authors} ({self.year}) {self.title}."]
        if self.journal:
            bits.append(f"{self.journal}.")
        if self.doi:
            bits.append(f"doi:{self.doi}")
        return " ".join(bits)


@dataclass(frozen=True)
class AnnotationSource:
    """Where a dataset's annotations come from.

    Orthogonal to the query backend: maleCNS is a neuPrint backend with either
    neuPrint *or* clio annotations; FlyWire is a CAVE backend with either a
    GitHub TSV *or* SeaTable annotations. Modelling this as a cross-product,
    rather than as subclasses, is what lets ``annotations="clio"`` be a parameter.
    """

    name: str  # "public" | "clio" | "flytable" | "cave" | ...
    kind: str  # "cave_table" | "github_tsv" | "neuprint" | "clio" | "seatable"
    location: str = ""  # table name / URL / SeaTable base
    id_column: str = "root_id"
    public: bool = True  # non-public sources are skipped by annotations="auto"
    chunked: bool = False  # fetch in chunks (BANC's codex_annotations needs this)

    # Some tables are long-format: one row per (id, key, value), rather than one
    # row per neuron. BANC's `codex_annotations` is - its 32 `classification_system`
    # values (super_class, cell_type, side, ...) are what other datasets keep in 32
    # columns. Set this to (key_column, value_column) and connecto pivots it wide.
    #
    # A neuron can carry several values for one key - FANC's MDNs are each tagged
    # "MDN", "MDN3" *and* "moonwalker descending neuron" - so pivoting has to encode
    # a set in a cell. It joins them with MULTI_SEP (below), and `criteria._match`
    # splits on the same constant, so a neuron tagged MDN is findable as `MDN`.
    pivot: tuple[str, str] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Everything connecto knows about a dataset."""

    name: str
    backends: tuple[BackendSpec, ...]
    label: str = ""
    species: str = ""

    # --- Who made this, and may you have it?
    #
    # A dataset is a scientific artefact with authors, a home page and an access
    # policy, and connecto is the last place that knows which one you are using
    # before the results turn into a figure. So it carries all three.

    description: str = ""  # a sentence or two: what was imaged, and how much of it
    publications: tuple[Publication, ...] = ()
    links: Mapping[str, str] = field(default_factory=dict)  # name -> landing page

    # `public=False` does not mean secret - every dataset here is one you may read
    # *about*. It means connecto cannot get you the data on a fresh token, and it is
    # much kinder to say so in `list_datasets()` than to let you discover it from a
    # 403 halfway through a script. `access` is the sentence that says what you need.
    public: bool = True
    access: str = ""

    annotation_sources: tuple[AnnotationSource, ...] = ()

    # Canonical field -> source columns, in priority order. First non-null wins.
    # This one mapping replaces cocoa's _type_cols, _side_cols, _align_columns,
    # _find_column and _backfill_types.
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    # New column -> (source column, regex with one capture group).
    #
    # Some datasets don't put a value in a column of its own. hemibrain v1.2.1 has
    # no `somaSide` at all - side lives in the *instance* name, as a suffix
    # ("DA1_lPN_R"). Rather than special-casing hemibrain in the normaliser, the
    # spec says how to dig the value out, the extracted column joins the frame like
    # any other, and `fields` can then point at it.
    derive: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    # Raw side value -> canonical "left"/"right"/"center".
    side_map: Mapping[str, str] = field(default_factory=dict)

    voxel_size: tuple[float, float, float] | None = None  # nm
    template_space: str | None = None
    segmentation_source: str | None = None

    # Precomputed neuroglancer skeletons, if the dataset publishes them. May contain
    # "{version}" - FlyWire ships one bucket per materialization.
    #
    # A *dataset* property, not a backend one, because that is what it is: a public
    # HTTPS bucket that needs no login and no CAVE client, and it is the same
    # skeleton whichever door you came in by. Keeping it on the CAVE BackendSpec
    # meant the neuPrint-backed FlyWire could not see it, so `skeletons.get()` went
    # to neuPrint's skeleton store - which for `flywire-fafb:v783b` does not exist,
    # and answers HTTP 400.
    #
    # Preferred over both backends' own skeleton services: CAVE's needs an L2 cache
    # that `flywire_fafb_public` does not have, and neuPrint's is not always there.
    skeleton_source: str | None = None

    # Which neuroglancer to send people to, and which *state dialect* it speaks.
    #
    # This is not cosmetic. `ngl.flywire.ai` is a fork of a 2021 neuroglancer and
    # cannot parse a modern scene at all - its bundle has `voxelCoordinates` and
    # `hiddenSegments` and not one occurrence of `crossSectionScale`. Spelunker and
    # base neuroglancer are the mirror image. Same scene, two incompatible JSON
    # schemas, and a scene in the wrong one does not error: it opens on an empty
    # viewer. So the dataset says which, and `viz` serialises accordingly.
    #
    # `viewer=None` means "ask the datastack" - CAVE's info service knows.
    viewer: str | None = None
    viewer_dialect: str = "modern"  # "modern" | "seunglab"

    capabilities: frozenset[Cap] = frozenset()

    # A couple of IDs known to exist. Exists purely so the conformance suite can
    # run the same assertions against every registered dataset.
    example_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if not self.backends:
            raise ValueError(f"Dataset {self.name!r} declares no backends.")
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if self.viewer_dialect not in DIALECTS:
            raise ValueError(
                f"Dataset {self.name!r}: `viewer_dialect` must be one of "
                f"{', '.join(DIALECTS)}, got {self.viewer_dialect!r}."
            )
        if not self.public and not self.access:
            raise ValueError(
                f"Dataset {self.name!r} is marked non-public but says nothing about "
                f"`access`. Saying 'you cannot have this' without saying how to ask "
                f"for it is worse than not saying it at all."
            )
        # Normalise to immutable mappings so the spec is genuinely frozen.
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "derive", dict(self.derive))
        object.__setattr__(self, "side_map", dict(self.side_map))
        object.__setattr__(self, "links", dict(self.links))
        object.__setattr__(self, "publications", tuple(self.publications))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "backends", tuple(self.backends))

    def backend(self, kind: str | None = None) -> BackendSpec:
        """Get the backend spec of the given kind, or the default (the first)."""
        if kind is None:
            return self.backends[0]
        for b in self.backends:
            if b.kind == kind:
                return b
        available = ", ".join(b.kind for b in self.backends)
        raise ValueError(
            f"{self.label} is not served by the {kind!r} backend. Available: {available}."
        )

    @property
    def backend_kinds(self) -> tuple[str, ...]:
        return tuple(b.kind for b in self.backends)

    def capabilities_for(self, kind: str | None = None) -> frozenset[Cap]:
        """What this dataset can do *through this backend*.

        `self.capabilities` is what the data has. This is what you can actually
        reach, and it is the only one anyone should ask: a dataset object is always
        bound to one backend, so the unqualified set is a claim nobody can use.
        """
        b = self.backend(kind)
        return (
            (self.capabilities | b.extra_capabilities)
            - BACKEND_LIMITS[b.kind]
            - b.missing_capabilities
        )

    def backends_with(self, cap: Cap) -> tuple[str, ...]:
        """Which of this dataset's backends can serve `cap`. Possibly none.

        Exists so a CapabilityError can end with "the cave backend does" instead of
        a flat no - the difference between a dead end and a next step.
        """
        return tuple(
            b.kind for b in self.backends if cap in self.capabilities_for(b.kind)
        )

    def annotation_source(self, name: str | None = "auto") -> AnnotationSource | None:
        """Pick an annotation source by name.

        ``"auto"`` picks the first *public* source; ``None`` disables annotations.
        """
        if name is None or not self.annotation_sources:
            return None
        if name == "auto":
            for s in self.annotation_sources:
                if s.public:
                    return s
            # No public source. That is a gated dataset whose only annotations are
            # lab-internal (aedes: FlyTable). Returning None would make
            # `.annotations.get()` claim "no source configured" - false, there is
            # one; it just needs credentials. Fall back to the first source and let
            # the fetch raise a clear missing-token error if the caller can't read it.
            return self.annotation_sources[0]
        for s in self.annotation_sources:
            if s.name == name:
                return s
        available = ", ".join(s.name for s in self.annotation_sources)
        raise ValueError(
            f"{self.label} has no annotation source {name!r}. Available: {available}."
        )

    def evolve(self, **changes) -> DatasetSpec:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)
