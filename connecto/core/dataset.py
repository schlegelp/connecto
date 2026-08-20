"""The Dataset base class.

connecto is a *normalisation layer*, not a wrapper layer. Backends implement ten
``_fetch_*`` hooks that return raw frames; everything the user actually touches -
validation, version resolution, capability checking, caching, schema
normalisation - happens above them, once, in code that both backends share.

A backend never constructs a user-facing DataFrame. That is the only thing that
structurally guarantees FlyWire and hemibrain hand back the same frame.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..exceptions import CapabilityError
from .criteria import parse_ids
from .spec import Cap, DatasetSpec
from .version import Version

__all__ = ["Dataset", "UNSET", "requires", "namespace", "capability_error"]


class _Unset:
    """'The user did not ask for this', as distinct from 'the user asked for None'."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return "UNSET"


UNSET = _Unset()


def _elsewhere(ds, caps) -> str:
    """"The cave backend does" - when that is true, and nothing when it isn't.

    A capability lives on a (dataset, backend) pair, not on a dataset: FlyWire has
    a chunkedgraph and you cannot reach it through neuPrint. So "FlyWire does not
    support chunkedgraph" is, on its own, a lie of omission that sends the reader
    off to find another dataset when what they need is another *door*.
    """
    spec = getattr(ds, "spec", None)
    if spec is None:  # a stub, or a dataset built without a spec
        return ""
    caps = list(caps)
    others = [
        b.kind
        for b in spec.backends
        if b.kind != ds.backend_kind
        and all(c in spec.capabilities_for(b.kind) for c in caps)
    ]
    if not others:
        return ""
    kind = others[0]
    return (
        f" The {kind} backend does: "
        f'cn.get_dataset("{spec.name}", backend="{kind}").'
    )


def capability_error(ds, caps, *, what: str | None = None) -> CapabilityError:
    """The one place a "this dataset cannot do that" message is written.

    Every refusal says three things: what is missing, what the dataset *can* do
    instead, and - if the dataset is served by a backend that has it - where to go.
    An error that only says "no" makes you go and read the source.
    """
    missing = ", ".join(str(c) for c in caps)
    subject = f"has no `{what}` - it does not support {missing}" if what else (
        f"does not support {missing}"
    )
    available = ", ".join(sorted(str(c) for c in ds.capabilities)) or "nothing"
    return CapabilityError(
        f"{ds.label} ({ds.backend_kind}) {subject}. "
        f"Available: {available}.{_elsewhere(ds, caps)}"
    )


def requires(*caps: Cap):
    """The public-call boundary.

    Every user-facing namespace method carries this, which makes it the one place
    that sees every call in and every exception out. So it does two jobs:

    1. Refuse, up front, anything the dataset cannot do (the capability gate).
    2. Translate upstream failures on the way back out. A 401 - or a 503 from a CAVE
       datastack halfway through a materialization - can surface at any depth: client
       construction, version resolution, or a private table you aren't allowed to
       read. Wrapping the single choke point catches all of them without decorating
       twenty methods.
    """

    def deco(fn):
        @functools.wraps(fn)
        def inner(self, *args, **kwargs):
            from ..servers import upstream_errors

            ds = getattr(self, "_ds", self)
            missing = [c for c in caps if c not in ds.capabilities]
            if missing:
                raise capability_error(ds, missing)

            with upstream_errors(
                ds.backend_kind, server=ds._auth_server, resource=ds.source, dataset=ds
            ):
                return fn(self, *args, **kwargs)

        inner.__connecto_caps__ = frozenset(caps)
        return inner

    return deco


class namespace:
    """A lazily-built, capability-gated sub-namespace.

    The point of namespaces is that the capability boundary becomes *syntactic*:
    ``ds.segmentation`` either exists or it doesn't, rather than sixteen separate
    methods each raising individually. Because :class:`CapabilityError` subclasses
    ``AttributeError``, ``hasattr(hb, "segmentation")`` is simply ``False``.
    """

    def __init__(self, cls, *caps: Cap):
        self.cls = cls
        self.caps = caps

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        missing = [c for c in self.caps if c not in obj.capabilities]
        if missing:
            raise capability_error(obj, missing, what=self.name)
        if self.name not in obj._namespaces:
            obj._namespaces[self.name] = self.cls(obj)
        return obj._namespaces[self.name]


class Dataset(ABC):
    """A connectome you can query.

    Datasets are immutable. There is no ``self.neurons``, no ``edges_``, no
    ``use_types``: a dataset is a *handle*, and the selection lives in the caller.
    Re-pinning a version returns a new object (:meth:`at`) rather than mutating -
    a mutable version setter plus a memoised annotation frame is exactly how you
    end up joining annotations from one materialization to edges from another.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        *,
        backend: str | None = None,
        version=None,
        annotations: str | None = "auto",
        fields: dict | None = None,
        edges=None,
    ):
        if fields:
            spec = spec.evolve(fields=dict(spec.fields) | dict(fields))

        self.spec = spec
        self._backend = spec.backend(backend)
        self._annotation_source = spec.annotation_source(annotations)
        self._edge_source = edges
        self._namespaces: dict = {}
        self._version_request = (
            version if version is not None else self._backend.default_version
        )
        self._resolved_version: Version | None = None

    # ------------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def backend_kind(self) -> str:
        return self._backend.kind

    @property
    def source(self) -> str:
        return self._backend.source

    @property
    def _auth_server(self) -> str | None:
        """Which server's credentials this dataset needs.

        None means "the service's default" (CAVE's global server). neuPrint
        overrides it, because its tokens are per-server.
        """
        return None

    @property
    def capabilities(self) -> frozenset[Cap]:
        """What this dataset can do *through the backend it is bound to*.

        Not `spec.capabilities` - that is what the *data* has, which is a different
        and, for a live handle, useless claim. FlyWire has a chunkedgraph; a FlyWire
        handle on the neuPrint backend cannot reach it, so it does not claim it.
        """
        return self.spec.capabilities_for(self.backend_kind)

    def supports(self, cap: Cap) -> bool:
        return cap in self.capabilities

    # ---------------------------------------------------------------- provenance

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def publications(self) -> tuple:
        """The papers to cite for this dataset."""
        return self.spec.publications

    @property
    def links(self) -> dict:
        """Landing pages and data repositories, by name."""
        return dict(self.spec.links)

    @property
    def public(self) -> bool:
        """False if you need permission that a fresh token will not give you."""
        return self.spec.public

    @property
    def access(self) -> str:
        """What you need in order to read this dataset. Empty if it is public."""
        return self.spec.access

    def cite(self) -> str:
        """Who to credit for this dataset.

        connecto knows exactly which dataset produced the numbers in your figure,
        which puts it in an unusually good position to tell you whose work it was::

            print(cn.Hemibrain().cite())
        """
        lines = [f"{self.label} ({self.spec.species})".strip()]
        if self.description:
            lines += ["", self.description]
        if self.publications:
            lines += ["", "Please cite:"]
            lines += [f"  {p}" for p in self.publications]
        if self.links:
            lines += ["", "See also:"]
            lines += [f"  {k:<10} {v}" for k, v in self.links.items()]
        if not self.public:
            lines += ["", f"Access: {self.access}"]
        return "\n".join(lines)

    # Namespaces that only some backends define at all. Without this, asking a
    # neuPrint dataset for `.segmentation` would raise a bare AttributeError with
    # no explanation - the descriptor that carries the good message lives on the
    # CAVE class, so it never gets consulted.
    _BACKEND_NAMESPACES = {
        "segmentation": Cap.SEGMENTATION,
        "proofreading": Cap.PROOFREADING,
        "l2": Cap.L2CACHE,
    }

    def __getattr__(self, name):
        # Only called when normal lookup has already failed - and note *why* it can
        # fail: a `namespace` descriptor that raises CapabilityError counts as a
        # failed lookup, because CapabilityError subclasses AttributeError (which is
        # what makes `hasattr` work). Python then lands here and, left alone, we would
        # replace the descriptor's precise message with a generic one. So if the name
        # *is* a namespace, let the descriptor speak.
        #
        # aedes found this: it is the first dataset with no annotations at all, and
        # `ds.ids("SomeType")` reported "'CAVEDataset' object has no attribute
        # 'annotations'" instead of saying that aedes has no annotations.
        desc = getattr(type(self), name, None)
        if isinstance(desc, namespace):
            return desc.__get__(self, type(self))

        cap = type(self)._BACKEND_NAMESPACES.get(name)
        if cap is not None:
            raise capability_error(self, [cap], what=name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __repr__(self):
        return (
            f"<{type(self).__name__} {self.label!r} "
            f"backend={self.backend_kind} version={self.version}>"
        )

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return NotImplemented
        return (self.name, self.backend_kind, self.version) == (
            other.name, other.backend_kind, other.version,
        )

    def __hash__(self):
        return hash((self.name, self.backend_kind, str(self.version)))

    # ------------------------------------------------------------------ versions

    @property
    def version(self) -> Version:
        """The version this dataset is pinned to (resolved on first access)."""
        from ..servers import upstream_errors

        if self._resolved_version is None or self._resolved_version.expired:
            # Resolving a version is the *first* authenticated call for most queries,
            # and it happens outside any @requires-decorated method - so it needs the
            # translation too, or a bad token (or a materializing CAVE server, which
            # 503s exactly here: every version lookup goes through the materialize
            # service) surfaces raw before the public boundary ever sees it.
            with upstream_errors(
                self.backend_kind,
                server=self._auth_server,
                resource=self.source,
                dataset=self,
            ):
                self._resolved_version = self._resolve_version(self._version_request)
        return self._resolved_version

    def versions(self) -> list:
        """All versions available for this dataset."""
        return self._list_versions()

    def at(self, version) -> Dataset:
        """A new handle onto the same dataset, pinned to a different version."""
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._namespaces = {}
        clone._version_request = version
        clone._resolved_version = None
        return clone

    def find_version(self, x, *, raise_missing: bool = True) -> Version:
        """The newest version in which *all* the given IDs are valid.

        On CAVE this scans materializations newest-first, because root IDs change
        as neurons are edited and a set of IDs is only jointly valid in some of
        them. On neuPrint body IDs are immutable, so this is an identity - which
        is the point: ``version="auto"`` is correct code on both backends.
        """
        ids = np.asarray(x, dtype="int64").ravel()
        return self._find_version(ids, raise_missing=raise_missing)

    def _resolve_version_arg(self, version) -> Version:
        """Per-call version override, falling back to the pinned one."""
        if version is None:
            return self.version
        return self._resolve_version(version)

    # ------------------------------------------------------------------ querying

    def ids(self, x=None, *, side=None, regex="auto", version=None) -> np.ndarray:
        """Resolve any neuron query to an array of int64 IDs.

            ds.ids(720575940621039145)   ds.ids("DA1_lPN")
            ds.ids("/AOTU00.*")          ds.ids("cell_class:ALPN", side="left")
        """
        return parse_ids(x, self, side=side, regex=regex, version=version)

    def _resolve_opt(self, name, value, cap: Cap, *, default, neutral=None):
        """Resolve an optional, capability-dependent argument.

        This is the fafbseg bug we refuse to repeat. fafbseg quietly sets
        ``min_score = None; filtered = False; transmitters = False`` for
        non-FlyWire datasets - so a user believes they filtered and they did not.

        Here: a default never raises, but an *explicit* request for something the
        dataset cannot do always does.
        """
        supported = cap in self.capabilities
        if value is UNSET:
            return default if supported else neutral
        if not supported and value not in (None, False):
            raise CapabilityError(
                f"{self.label} ({self.backend_kind}) does not support `{name}` "
                f"(no {cap}). Drop the argument, or use a dataset that has it."
                f"{_elsewhere(self, [cap])}"
            )
        return value

    # -------------------------------------------------------------- backend hooks
    # Ten methods. This is the entire surface for adding a new backend.

    @abstractmethod
    def _resolve_version(self, version) -> Version: ...

    @abstractmethod
    def _list_versions(self) -> list: ...

    @abstractmethod
    def _find_version(self, ids: np.ndarray, *, raise_missing: bool) -> Version: ...

    @abstractmethod
    def _ids_exist(self, ids: np.ndarray, version: Version) -> np.ndarray: ...

    @abstractmethod
    def _fetch_annotations(self, source, version: Version) -> pd.DataFrame: ...

    @abstractmethod
    def _fetch_edges(self, pre, post, version, *, by_roi, min_weight, rois) -> pd.DataFrame: ...

    @abstractmethod
    def _fetch_synapses(self, pre, post, version, **opts) -> pd.DataFrame: ...

    @abstractmethod
    def _fetch_skeletons(self, ids, version, **opts): ...

    @abstractmethod
    def _fetch_meshes(self, ids, version, **opts): ...

    def _fetch_somas(self, ids, version) -> pd.DataFrame:
        raise CapabilityError(f"{self.label} has no soma source.")

    def _fetch_rois(self) -> pd.DataFrame:
        raise CapabilityError(f"{self.label} has no ROIs.")

    def _fetch_roi_hierarchy(self):
        raise CapabilityError(f"{self.label} has no ROI hierarchy.")

    def _fetch_roi_mesh(self, roi: str):
        raise CapabilityError(f"{self.label} has no ROI meshes.")

    # ------------------------------------------------------------------ volumes

    # Where the segmentation and the EM image actually live. Two callers need this
    # - the volume reader (meshes, point lookups, cutouts) and neuroglancer - and they
    # used to work it out separately. They drifted, as duplicated lookups do: the
    # viz copy read `neuroglancerMeta[0]["dataInstance"]`, and entry 0 of that list
    # is the *grayscale* layer, so it would have handed neuroglancer the string
    # "grayscalejpeg" as a segmentation source. One resolver, asked by everyone.
    #
    # `format_for` is caveclient's vocabulary ("raw" | "neuroglancer" |
    # "cave_explorer"); it decides whether the graphene URL carries the
    # `middleauth+` prefix. Backends that have no such notion ignore it.

    def _segmentation_source(self, *, format_for: str = "raw") -> str | None:
        """This dataset's segmentation volume, or None if it has none."""
        return self.spec.segmentation_source

    def _graph_source(self) -> str:
        """The chunkedgraph source. Only a chunkedgraph dataset has one."""
        raise CapabilityError(
            f"{self.label} ({self.backend_kind}) has no chunkedgraph - its "
            f"segmentation is a flat volume."
        )

    def _image_source(self) -> str | None:
        """This dataset's EM image volume, or None if it does not advertise one."""
        return None

    def _skeleton_source(self, version) -> str | None:
        """The published precomputed skeleton bucket for `version`, if there is one.

        One resolver, because the bucket is keyed by the *release* and the two
        backends spell a release differently: CAVE says `783`, the neuPrint mirror
        says `flywire-fafb:v783b`. The backend supplies the key it needs
        (`BackendSpec.skeleton_version`); the dataset supplies the bucket.
        """
        source = self.spec.skeleton_source
        if source is None:
            return None
        key = self._backend.skeleton_version
        return source.format(version=version if key is None else key)

    # Column maps: backend-declared, consumed by connecto.core.schemas.
    _edge_colmap: dict = {}
    _synapse_colmap: dict = {}
    _skeleton_colmap: dict = {}

    # The backend's usual convention. A *server* may differ - see
    # `BackendSpec.position_units` - so nobody reads this directly.
    _default_position_units: str = "voxel"

    @property
    def _raw_position_units(self) -> str:
        """What units this server's raw positions are in. schemas._rescale converts."""
        return self._backend.position_units or self._default_position_units
