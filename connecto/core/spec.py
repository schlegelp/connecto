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

__all__ = ["Cap", "BackendSpec", "AnnotationSource", "DatasetSpec", "MULTI_SEP", "DIALECTS"]

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
    SEGMENTATION = "segmentation"  # chunkedgraph / ID operations (CAVE)
    PROOFREADING = "proofreading"  # edit history, proofreading status (CAVE)
    SOMAS = "somas"
    LIVE = "live"  # non-materialized "right now" queries
    NEUROGLANCER = "neuroglancer"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class BackendSpec:
    """How one backend serves one dataset.

    A dataset may have several of these - see the module docstring.
    """

    kind: str  # "cave" | "neuprint"
    source: str  # CAVE datastack, or "server/dataset[:version]"
    default_version: object = "latest"

    # --- CAVE table wiring. None means "probe at runtime".
    synapse_table: str | None = None
    edge_view: str | None = None  # pre-aggregated edges; much faster than synapses
    nucleus_table: str | None = None
    proofreading_table: str | None = None

    # What the synapse table calls its confidence score. FlyWire says `cleft_score`;
    # FANC says plain `score`. Same concept, different column - so this is a name,
    # not a capability. Whether the dataset *has* scores is Cap.SYNAPSE_SCORES.
    score_column: str = "cleft_score"

    # Precomputed neuroglancer skeletons, if the dataset publishes them.
    # May contain "{version}" - FlyWire ships one bucket per materialization.
    # Preferred over the CAVE skeleton service, which needs an L2 cache that not
    # every datastack has (`flywire_fafb_public`, notably, does not).
    skeleton_source: str | None = None

    def __post_init__(self):
        if self.kind not in ("cave", "neuprint"):
            raise ValueError(f"Unknown backend kind: {self.kind!r}")


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
        # Normalise to immutable mappings so the spec is genuinely frozen.
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "derive", dict(self.derive))
        object.__setattr__(self, "side_map", dict(self.side_map))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "backends", tuple(self.backends))

    def backend(self, kind: str | None = None) -> BackendSpec:
        """Get the backend spec of the given kind, or the first one."""
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
            return None
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
