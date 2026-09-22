"""Selecting neurons.

Two layers, and they are not alternatives - the mini-language desugars into the
criteria object:

    ds.ids(720575940621039145)      # a raw ID
    ds.ids("DA1_lPN")              # a type, matched across spec.fields["type"]
    ds.ids("/AOTU00.*")            # a regex
    ds.ids("class:ALPN")           # an explicit column filter
    ds.ids(NeuronCriteria(type="DA1_lPN", side="left"))

``parse_ids`` is a *pure function*: it takes a query and returns IDs. It does not
mutate the dataset. (cocoa's ``add_neurons`` mutated ``self.neurons``, which is
what made its datasets stateful and its caches unsound.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .spec import MULTI_SEP

__all__ = ["NeuronCriteria", "parse_ids", "is_id"]

_ID_RE = re.compile(r"^\d+$")


def is_id(x) -> bool:
    """Is this an ID rather than a name/type?"""
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, str):
        return bool(_ID_RE.match(x.strip()))
    return False


@dataclass(frozen=True)
class NeuronCriteria:
    """A backend-agnostic neuron query.

    Compiled per backend: to Cypher on neuPrint, and on CAVE to
    ``filter_*_dict`` where the column lives in a queryable table - otherwise
    resolved client-side against the cached annotation frame.

    That fallback is load-bearing, not an accident: FlyWire's authoritative
    annotations live in a GitHub TSV, not in CAVE, so ``type="DA1_lPN"`` genuinely
    cannot compile to a CAVE filter and *must* be resolved locally first.
    """

    id: Any = None
    type: Any = None
    side: Any = None
    class_: Any = None
    status: Any = None
    rois: Any = None
    regex: bool | str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(self, id=None, type=None, side=None, class_=None, status=None,
                 rois=None, regex="auto", **extra):
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "class_", class_)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "rois", rois)
        object.__setattr__(self, "regex", regex)
        object.__setattr__(self, "extra", dict(extra))

    def __repr__(self):
        parts = []
        for k in ("id", "type", "side", "class_", "status", "rois"):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={v!r}")
        parts += [f"{k}={v!r}" for k, v in self.extra.items()]
        return f"NeuronCriteria({', '.join(parts)})"

    @property
    def is_empty(self) -> bool:
        return not any(
            getattr(self, k) is not None
            for k in ("id", "type", "side", "class_", "status", "rois")
        ) and not self.extra

    def filters(self) -> dict[str, Any]:
        """Canonical field -> value, excluding `id` and `rois`."""
        out = {}
        for canon, attr in (("type", "type"), ("side", "side"),
                            ("class", "class_"), ("status", "status")):
            v = getattr(self, attr)
            if v is not None:
                out[canon] = v
        out.update(self.extra)
        return out


def _split_regex(value, regex):
    """Resolve the ``/`` regex convention. Returns (value, is_regex)."""
    if isinstance(value, str) and regex == "auto":
        if value.startswith("/"):
            return value[1:], True
        return value, False
    return value, bool(regex) and regex != "auto"


def _match(series: pd.Series, value, use_regex: bool) -> pd.Series:
    """Match a column against a scalar or a list of values.

    A pivoted cell may hold a *set* of values joined with ``MULTI_SEP`` - a FANC MDN's
    type is "MDN, MDN3, moonwalker descending neuron". Matching only the whole string
    would make every member of every such set unfindable, and silently: ``ids("MDN")``
    would return nothing at all rather than raise. So a value matches if it matches the
    cell *or* any member of it.
    """
    s = series.astype("string")
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        mask = pd.Series(False, index=series.index)
        for v in value:
            mask |= _match(series, v, use_regex)
        return mask

    if use_regex:
        whole = s.str.match(str(value), na=False)
    else:
        whole = s == str(value)
    whole = whole.fillna(False).astype(bool)

    multi = s.str.contains(MULTI_SEP, regex=False, na=False).fillna(False).astype(bool)
    if not multi.any():
        return whole

    if use_regex:
        rx = re.compile(str(value))
        hit = lambda cell: any(rx.match(p) for p in cell.split(MULTI_SEP))  # noqa: E731
    else:
        target = str(value)
        hit = lambda cell: target in cell.split(MULTI_SEP)  # noqa: E731

    part = pd.Series(False, index=series.index)
    part.loc[multi] = s.loc[multi].map(hit).astype(bool)
    return whole | part


def to_criteria(x, *, side=None, regex="auto") -> NeuronCriteria:
    """Desugar the mini-language into a :class:`NeuronCriteria`."""
    if isinstance(x, NeuronCriteria):
        if side is not None and x.side is None:
            return NeuronCriteria(
                id=x.id, type=x.type, side=side, class_=x.class_,
                status=x.status, rois=x.rois, regex=x.regex, **x.extra,
            )
        return x

    if isinstance(x, str) and ":" in x and not is_id(x):
        col, _, value = x.partition(":")
        value, rx = _split_regex(value, regex)
        return NeuronCriteria(side=side, regex=rx, **{col.strip(): value})

    if isinstance(x, str) and not is_id(x):
        value, rx = _split_regex(x, regex)
        return NeuronCriteria(type=value, side=side, regex=rx)

    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        items = list(x)
        if items and all(is_id(i) for i in items):
            return NeuronCriteria(id=np.asarray(items, dtype="int64"), side=side)
        # A mixed/str list: OR the individual criteria together at resolve time.
        return _MultiCriteria(
            tuple(to_criteria(i, side=side, regex=regex) for i in items)
        )

    if is_id(x):
        return NeuronCriteria(id=np.asarray([int(x)], dtype="int64"), side=side)

    raise TypeError(f"Don't know how to interpret {x!r} as a neuron query.")


@dataclass(frozen=True)
class _MultiCriteria:
    """The union of several criteria (a heterogeneous list)."""

    criteria: tuple[NeuronCriteria, ...]


def parse_ids(x, ds, *, side=None, regex="auto", version=None) -> np.ndarray:
    """Resolve any neuron query down to an array of int64 IDs.

    Pure function. Returns IDs. Mutates nothing.
    """
    if x is None:
        return ds.annotations.ids(version=version)

    crit = to_criteria(x, side=side, regex=regex)

    if isinstance(crit, _MultiCriteria):
        out = [parse_ids(c, ds, version=version) for c in crit.criteria]
        return np.unique(np.concatenate(out)) if out else np.array([], dtype="int64")

    return resolve_criteria(crit, ds, version=version)


def resolve_criteria(crit: NeuronCriteria, ds, *, version=None) -> np.ndarray:
    """Resolve criteria against the dataset's annotation table.

    Backends may override this with a native query (neuPrint compiles to Cypher);
    this is the client-side fallback and the CAVE default.
    """
    ids = None
    if crit.id is not None:
        ids = np.unique(np.asarray(crit.id, dtype="int64").ravel())

    filters = crit.filters()
    if not filters:
        if ids is None:
            raise ValueError("Empty neuron query.")
        return ids

    ann = ds.annotations.get(version=version, verbose=False)
    use_regex = bool(crit.regex) and crit.regex != "auto"

    mask = pd.Series(True, index=ann.index)
    for col, value in filters.items():
        if col == "type" and col not in ann.columns:
            raise ValueError(
                f"{ds.label} has no `type` column configured. Pass "
                f"`fields={{'type': (...)}}` to the dataset, or query a raw "
                f"column with 'column:value'."
            )
        if col not in ann.columns:
            hint = _other_source(ds, col)
            if not hint and (near := _suggest(col, ann.columns)):
                hint = f" Did you mean {near!r}?"
            raise ValueError(
                f"{ds.label} annotations have no column {col!r}.{hint}"
            )
        mask &= _match(ann[col], value, use_regex)

    found = np.unique(ann.loc[mask, "id"].to_numpy(dtype="int64"))
    if ids is not None:
        found = np.intersect1d(found, ids)
    return found


def _other_source(ds, col: str) -> str:
    """"...the 'cave' source has one" - when another annotation source does.

    The annotation-source counterpart of `dataset._elsewhere`, and it exists for the
    same reason: a field can be missing from the *table you are reading* while the
    dataset has it. BANC's neuPrint mirror declares no usable `side`; its codex
    table has one for every neuron. Saying only "no column 'side'" sends the reader
    off to find another dataset when what they need is another source.
    """
    current = getattr(ds, "_annotation_source", None)
    # Only speak up when *this* source declared the field absent. A column missing
    # for any other reason is a different problem, and guessing here would send
    # people to a source that has not got it either. That test also excludes the
    # current source from the list below, so it needs no separate identity check.
    if current is None or current.fields.get(col, True):
        return ""
    others = [
        s.name for s in ds.spec.annotation_sources if s.fields.get(col, True)
    ]
    if not others:
        return ""
    # Carry the backend through when it is not the default one. Annotation source
    # and backend are independent axes, and a suggestion that silently resets the
    # other one sends a `backend="cave"` handle to neuPrint to fix its `side`.
    args = f'annotations="{others[0]}"'
    if ds.backend_kind != ds.spec.backends[0].kind:
        args = f'backend="{ds.backend_kind}", {args}'
    return (
        f" The {others[0]!r} annotation source has one: "
        f'cn.get_dataset("{ds.spec.name}", {args}).'
    )


def _suggest(name: str, options) -> str | None:
    import difflib

    hits = difflib.get_close_matches(name, [str(o) for o in options], n=1, cutoff=0.6)
    return hits[0] if hits else None
