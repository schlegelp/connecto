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
    "SparseVolSource",
    "DatasetSpec",
    "Publication",
    "BACKEND_LIMITS",
    "MULTI_SEP",
    "DIALECTS",
    "TRANSMITTERS",
    "neuprint_nt_columns",
]

# The neuroglancer state schemas we can emit. See `DatasetSpec.viewer_dialect`.
DIALECTS = ("modern", "seunglab")

# How a set of values is encoded in one cell (see `AnnotationSource.pivot`).
# `sources.pivot_long` joins on it; `criteria._match` splits on it. Defined once so
# the two can never disagree - when they did, a FANC neuron tagged "MDN" was stored
# as "MDN, MDN3, moonwalker descending neuron" and `ds.ids("MDN")` found nothing.
MULTI_SEP = ", "

# The transmitter vocabulary connecto normalises to. Datasets predict different
# subsets of it - MANC three, male-cns seven, BANC eight - and name the columns
# differently for the same molecule (`ach`, `ntAcetylcholineProb`, `acetylcholine`),
# so every `BackendSpec.nt_columns` maps its raw names onto these and nothing else.
# Without one vocabulary, `nt == "acetylcholine"` would be a per-dataset spelling
# test and cross-dataset comparison - the point of the library - would not work.
#
# "unknown" is a member because MANC predicts it: its model has an explicit fourth
# class for "none of these three", and folding that into NA would turn a confident
# "not one of the fast transmitters" into a missing value, which is a different and
# weaker claim.
TRANSMITTERS = (
    "acetylcholine",
    "gaba",
    "glutamate",
    "dopamine",
    "serotonin",
    "octopamine",
    "histamine",
    "tyramine",
    "unknown",
)


def neuprint_nt_columns(*transmitters: str) -> dict[str, str]:
    """``{"ntGabaProb": "gaba", ...}`` - neuPrint's per-synapse probability keys.

    neuPrint spells them ``nt<Transmitter>Prob``, so the mapping is mechanical.
    *Which* transmitters a dataset predicts is not, and guessing is silent: `nt` is
    an argmax over the columns we ask for, so naming one the dataset does not have
    costs nothing, while omitting one it does have cannot raise - it can only
    quietly return the runner-up. Hence an explicit list per dataset, verified
    against the server's Synapse properties.
    """
    unknown = [t for t in transmitters if t not in TRANSMITTERS]
    if unknown:
        raise ValueError(
            f"Not canonical transmitters: {unknown}. One of {list(TRANSMITTERS)}."
        )
    return {f"nt{t.capitalize()}Prob": t for t in transmitters}


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
    # volume, advertised in `Client.meta`). It just needs a readable volume.
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

    # A third thing, for the same reason the two above are separate: "there is a
    # segmentation volume" and "you can get one neuron's voxels out of it without
    # reading the whole brain" are different claims.
    #
    # DVID maintains a live per-body index (body -> blocks -> runs), so a sparse
    # volume is one cheap request. A chunkedgraph maintains no such index: the
    # voxels are a static precomputed volume with nothing mapping a root ID to the
    # blocks it occupies, so the same question degrades to "read dense blocks, mask,
    # sparsify" - 100-1000x more voxels touched than kept. Both can answer; the cost
    # differs by three orders of magnitude, and `voxels.get` says so up front rather
    # than appearing to hang.
    #
    # Kept apart from SEGMENTATION because Aedes has this *without* connecto being
    # able to read its graphene volume directly - it is served by a lookup service -
    # and hemibrain has both by two unrelated routes.
    VOXELS = "voxels"  # per-neuron sparse volumes (N, 3)

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
            # Note: NT_PER_SYNAPSE is *not* here, and used to be. It was denied
            # backend-wide on the grounds that connecto could not read neuPrint's
            # per-synapse transmitter properties - true at the time, but it stated a
            # gap in our code as a fact about the server, and the two diverged as
            # soon as the code was written. It is a per-*dataset* fact: banc, manc
            # and male-cns carry `ntGabaProb` & co. on their Synapse nodes and now
            # declare it; hemibrain and fish2 have no such properties, and FlyWire's
            # mirror carries them on Neuron nodes but not Synapse ones - so that one
            # says so on its own BackendSpec. See `BackendSpec.nt_columns`.
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

    # --- Per-synapse transmitters. Whether the dataset *has* them is
    # Cap.NT_PER_SYNAPSE; these say where they live and what they are called, which
    # is a property of the (dataset, backend) pair and of nothing smaller.
    #
    # `nt_columns` maps a raw probability column onto a canonical transmitter:
    # FlyWire's CAVE view calls them `ach`/`gaba`/`glut`, neuPrint calls the same
    # things `ntAcetylcholineProb` & co. (see `neuprint_nt_columns`). One row per
    # synapse, one column per transmitter, and `nt` is the argmax across them.
    nt_columns: Mapping[str, str] = field(default_factory=dict)

    # ...or, when the predictions are not columns on the synapse table at all.
    # BANC's CAVE door keeps them in a separate table, one row per synapse carrying
    # the winning transmitter and its probability rather than eight probability
    # columns. So it needs a join, not a rename, and the frame it produces has
    # `nt`/`nt_confidence` but no per-transmitter columns - which is also why
    # `transmitter_source` reports no `classes` for such a door: the table records
    # the choice, not what it was chosen from. The columns connecto reads off it
    # are `_NT_TABLE_COLMAP` in the CAVE backend.
    #
    # It is also a *subset*: only synapses of size >= 5 were predicted. connecto
    # therefore left-joins it rather than querying it instead of the synapse table,
    # so `transmitters=True` returns the same synapses as `transmitters=False`,
    # some of them with no call. Swapping the source would silently drop the rest.
    nt_table: str | None = None

    @property
    def transmitter_source(self) -> dict | None:
        """Which model run made this door's per-synapse calls, and over what classes.

        ``None`` if this door has no per-synapse transmitters. Lives here because
        the answer differs by *field*, not just by value: a door with `nt_columns`
        names the server and knows its own classes, while a door with an `nt_table`
        names the table and cannot know them - the table stores the winner, not the
        vector it was chosen from. Deriving it from `nt_columns` alone reported no
        classes at all for BANC's CAVE door, which is the one it was written for.
        """
        if self.nt_table:
            return {"source": self.nt_table, "classes": None}
        if self.nt_columns:
            return {
                "source": self.source,
                "classes": sorted(set(self.nt_columns.values())),
            }
        return None

    def __post_init__(self):
        if self.kind not in BACKEND_LIMITS:
            raise ValueError(f"Unknown backend kind: {self.kind!r}")
        object.__setattr__(
            self, "extra_capabilities", frozenset(self.extra_capabilities)
        )
        object.__setattr__(
            self, "missing_capabilities", frozenset(self.missing_capabilities)
        )
        object.__setattr__(self, "nt_columns", dict(self.nt_columns))
        # A raw column may map only onto the canonical vocabulary. Caught here
        # because the alternative is an `nt` column whose values are a spelling
        # nobody else uses, discovered when a cross-dataset comparison finds no
        # overlap and looks like biology.
        rogue = sorted(set(self.nt_columns.values()) - set(TRANSMITTERS))
        if rogue:
            raise ValueError(
                f"{self.source!r}: `nt_columns` maps onto {rogue}, which "
                f"{'is' if len(rogue) == 1 else 'are'} not in TRANSMITTERS."
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

    # Which SeaTable *deployment* a `seatable` source lives on. SeaTable is not one
    # server: the lab runs its own ("flytable", the default), and there is the
    # official "seatable" cloud - and a base named on one does not exist on the other.
    # FlyWire and aedes are on flytable; BANC's `banc_meta` is on the cloud. Ignored
    # for non-seatable kinds; the name maps to a URL in `connecto.sources.seatable`.
    instance: str = "flytable"

    chunked: bool = False  # fetch in chunks (BANC's codex_annotations needs this)

    # Field priorities that apply only when *this* source is the one being read,
    # overlaid on `DatasetSpec.fields`.
    #
    # Needed because `fields` is a property of the dataset while the columns are a
    # property of the table, and for a dataset with several sources those are not
    # the same thing. Listing every source's spellings in one priority list covers
    # most of it - the sources are alternatives, so only one frame's columns are
    # ever present - but not the case where two sources have a column of the *same
    # name* and only one of them means it. BANC is exactly that: both its neuPrint
    # mirror and its codex table call the field `side`, and codex has one for all
    # 158,250 neurons where the mirror has 8,153.
    #
    # An empty tuple is the useful value: "this source does not have this field",
    # which makes the canonical column *absent* rather than present and 95% null.
    # `ds.ids("PFNd", side="left")` then says it cannot answer instead of answering
    # 0 - the difference between a refusal and a wrong number.
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

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

    def __post_init__(self):
        object.__setattr__(self, "fields", dict(self.fields))


@dataclass(frozen=True)
class SparseVolSource:
    """A service that hands out one neuron's voxels, run-length encoded.

    The escape hatch for datasets whose voxels connecto cannot reach cheaply on its
    own. A chunkedgraph has no per-body spatial index, so the generic CAVE path has
    to read dense blocks and mask them; where somebody has stood up a service that
    *does* keep an index, this points at it and the expensive path is skipped.

    ``url`` is a template taking ``{id}`` and ``{scale}``. The response is expected
    to be the same wire format DVID's ``sparsevol`` emits - little-endian int32
    ``(x, y, z, run_length)`` runs along +X - which is what makes one decoder serve
    every source connecto has (see :mod:`connecto.voxels.rle`).
    """

    url: str

    # Which scales the service actually serves. Not decoration: the aedes service
    # answers 400 for scale 0 ("would require reading 6,039,797,760 voxels") and a
    # bare 500 for scales 2 and 3. Declaring the truth means `voxels.get(x, scale=2)`
    # says which scales exist instead of relaying an Internal Server Error.
    scales: tuple[int, ...] = (0,)

    # Per-scale downsample factor, per axis. Not always (2, 2, 2): fly pyramids
    # routinely halve X and Y while leaving Z alone, so aedes is (2, 2, 1) and its
    # 16x16x45 nm voxels become 32x32x45 at scale 1 - not 32x32x90. Getting this
    # wrong scales a neuron wrongly along one axis, which looks plausible and is not.
    downsample: tuple[int, int, int] = (2, 2, 2)

    def resolution(self, scale: int, voxel_size) -> tuple[float, float, float]:
        """nm per voxel at ``scale``, given the dataset's scale-0 ``voxel_size``."""
        return tuple(
            float(v) * float(d) ** int(scale)
            for v, d in zip(voxel_size, self.downsample)
        )


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

    # A service that serves per-neuron sparse volumes, if one exists for this
    # dataset. When set, `voxels.get` asks it instead of reading dense blocks - the
    # difference between one request and several hundred. See `SparseVolSource`.
    #
    # DVID datasets need nothing here: their server and node are discovered at
    # runtime from neuPrint or clio metadata rather than written down (see
    # `connecto.voxels.dvid`), because those servers are not public knowledge and a
    # URL pinned in source would be both a leak and wrong by the next release.
    sparsevol_source: SparseVolSource | None = None

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

        # A door that claims per-synapse transmitters must say where they live.
        # This is the check that would have caught the bug this library was written
        # about: `Cap.NT_PER_SYNAPSE` used to be a claim with nothing behind it, so
        # `transmitters=True` returned a frame with no `nt` column and no error.
        # Now the claim cannot be registered without the wiring that honours it.
        for b in self.backends:
            if (
                Cap.NT_PER_SYNAPSE in self.capabilities_for(b.kind)
                and b.transmitter_source is None
            ):
                raise ValueError(
                    f"Dataset {self.name!r} claims {Cap.NT_PER_SYNAPSE} through the "
                    f"{b.kind!r} backend but that BackendSpec declares neither "
                    f"`nt_columns` nor `nt_table`, so nothing could serve it."
                )

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
