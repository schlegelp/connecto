"""The one place normalisation happens.

Backends return raw frames. Nothing else in connecto renames a column, maps a
vocabulary or converts a unit - it all happens here. That is the only thing that
structurally guarantees FlyWire and hemibrain hand back the same frame.

The rule that resolves "clean interface" against "not too opinionated":

    Canonical columns are produced by *consuming* raw columns (renaming them, or
    deriving from several). Any raw column that is not consumed is passed through
    untouched. Nothing is ever silently dropped, and ``raw=True`` skips this
    module entirely.

So the closed schemas (edges, synapses, skeleton nodes) come back tidy, while
annotations - the open-ended, dataset-specific frame - keeps every native column
alongside the canonical ones.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..exceptions import CapabilityError

__all__ = [
    "add_transmitters",
    "EDGE_DTYPES",
    "SYNAPSE_DTYPES",
    "ANNOTATION_DTYPES",
    "normalize_edges",
    "normalize_synapses",
    "normalize_annotations",
    "normalize_skeleton",
    "stamp",
]

EDGE_DTYPES = {"pre": "int64", "post": "int64", "weight": "int32", "roi": "category"}

SYNAPSE_DTYPES = {
    "id": "int64",
    "pre": "int64",
    "post": "int64",
    "pre_x": "float32",
    "pre_y": "float32",
    "pre_z": "float32",
    "post_x": "float32",
    "post_y": "float32",
    "post_z": "float32",
    "score": "float32",
    "roi": "category",
    "nt": "category",
    "nt_confidence": "float32",
}

ANNOTATION_DTYPES = {
    "id": "int64",
    "type": "string",
    "side": "category",
    "class": "string",
    "nt": "category",
    "nt_source": "category",
    "status": "string",
    "soma_x": "float32",
    "soma_y": "float32",
    "soma_z": "float32",
}

SIDES = ("left", "right", "center")


def stamp(df: pd.DataFrame, ds, *, query: str, version=None, raw_columns=None):
    """Attach provenance to a frame.

    Survives most pandas operations via ``df.attrs`` and means a frame can always
    tell you which dataset, backend and version it came from - which matters a
    lot once you have frames from six datasets in one notebook.
    """
    df.attrs["connecto"] = {
        "dataset": ds.name,
        "backend": ds.backend_kind,
        "source": ds.source,
        "version": str(version) if version is not None else None,
        "version_timestamp": getattr(version, "timestamp", None),
        "query": query,
        "fetched": datetime.now(UTC),
        "raw_columns": list(raw_columns) if raw_columns is not None else None,
    }
    return df


def _consume(df: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
    """Rename raw -> canonical, pass everything else through.

    ``colmap`` maps canonical name -> raw name. A raw column that is not consumed
    but collides with a canonical name gets suffixed ``_raw`` so the canonical
    column always means what the schema says it means.
    """
    consumed = {raw for raw in colmap.values() if raw in df.columns}
    canonical = set(colmap)

    collisions = {
        c: f"{c}_raw" for c in df.columns if c in canonical and c not in consumed
    }
    if collisions:
        df = df.rename(columns=collisions)

    rename = {raw: canon for canon, raw in colmap.items() if raw in df.columns}
    return df.rename(columns=rename)


def _coerce(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # A nullable column of ints comes back as float/object; don't fight it.
            if dtype.startswith(("int", "float")):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _order(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    lead = [c for c in first if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def _rescale(df, prefixes, ds, units: str, have: str | None = None):
    """Get positions into the requested units.

    Units are nanometres everywhere in connecto unless the caller says otherwise.
    ``voxel_size`` lives on the spec, and each backend declares what *it* returns
    (CAVE asks for nm outright via ``desired_resolution``; neuPrint hands back
    voxels), so no user ever has to think about this again.

    ``have`` overrides the backend's declaration, for the case where the frame did
    not come from the backend at all. A precomputed skeleton bucket is plain HTTPS
    and hands back nanometres no matter who fetched it - so run it through the
    neuPrint backend's ``voxel`` default and every coordinate is 4-40x too big.
    """
    have = have or getattr(ds, "_raw_position_units", "voxel")
    if have == units or ds.spec.voxel_size is None:
        return df

    factor = np.asarray(ds.spec.voxel_size, dtype="float64")
    if have == "voxel" and units == "nm":
        scale = factor
    elif have == "nm" and units == "voxel":
        scale = 1.0 / factor
    else:
        raise ValueError(f"Cannot convert positions from {have!r} to {units!r}.")

    for prefix in prefixes:
        for axis, s in zip(("x", "y", "z"), scale):
            col = f"{prefix}{axis}"
            if col in df.columns:
                df[col] = df[col].astype("float32") * float(s)
    return df


def add_transmitters(
    df: pd.DataFrame, nt_columns: dict[str, str], *, label: str
) -> pd.DataFrame:
    """Collapse per-synapse transmitter probabilities into a single call.

    ``nt_columns`` maps raw probability columns onto canonical transmitter names -
    ``{"ach": "acetylcholine"}`` on FlyWire's CAVE view, ``{"ntGabaProb": "gaba"}``
    on neuPrint. See :attr:`BackendSpec.nt_columns`.

    Adds ``nt`` (the winner), ``nt_confidence`` (its probability) and one
    ``nt_<transmitter>`` column per class. Lives here rather than in either backend
    because it is normalisation, and both backends need exactly the same argmax over
    differently-spelled columns.

    Missing columns raise. A transmitter class that the spec declares and the server
    did not return cannot be detected downstream - ``nt`` is an argmax, so losing a
    class does not produce a gap, it produces a confident wrong answer.
    """
    missing = [c for c in nt_columns if c not in df.columns]
    if not nt_columns or missing:
        raise CapabilityError(
            f"{label}: synapse frame is missing transmitter column(s) "
            f"{', '.join(sorted(missing)) or '(none declared)'}. `nt` is an argmax "
            f"across all of them, so continuing would silently return the best of "
            f"the wrong classes."
        )

    # Converted once, straight to the target dtype. Going through
    # `.apply(pd.to_numeric)` would build a float64 frame and convert it again -
    # two full passes and a transient twice the size of the result.
    probs = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce").astype("float32") for c in nt_columns},
        index=df.index,
    )
    values = probs.to_numpy()

    # A synapse with no prediction at all is normal - BANC's neuPrint copy has
    # them - and pandas' `idxmax` raises on an all-NA row rather than returning NA.
    # Filling to -inf makes the argmax total, and the winning probability is NaN
    # exactly on those rows, so it doubles as the mask to put them back: "not
    # predicted" and "predicted as X" are different, and both are true of some row
    # of this frame.
    winner = np.argmax(np.nan_to_num(values, nan=-np.inf), axis=1)
    conf = values[np.arange(len(values)), winner]
    names = np.array(list(nt_columns.values()), dtype=object)

    out = pd.concat(
        [df, probs.rename(columns={raw: f"nt_{t}" for raw, t in nt_columns.items()})],
        axis=1,
    )
    out["nt"] = np.where(np.isnan(conf), None, names[winner])
    out["nt_confidence"] = conf
    return out


def normalize_edges(
    raw: pd.DataFrame, ds, *, colmap: dict[str, str], version=None, extra: bool = False
) -> pd.DataFrame:
    """Normalize an edge list to ``pre, post, weight[, roi]``.

    Unlike annotations, the edge frame is a **closed** schema: exactly these
    columns, on every backend. This is the most-used object in the library, and
    "same code, two backends" is worth nothing if FlyWire hands back twenty
    columns (its edge view carries transmitter probabilities) and hemibrain hands
    back three.

    Whatever else the backend returned is therefore dropped - not lost: pass
    ``extra=True`` to keep it, or use ``.connectivity.transmitters()`` and
    ``.connectivity.synapses()`` for the per-connection detail.
    """
    df = _consume(raw.copy(), colmap)

    for col in ("pre", "post", "weight"):
        if col not in df.columns:
            raise KeyError(
                f"{ds.label}: backend returned no `{col}` column for edges "
                f"(got {list(raw.columns)}). This is a connecto bug."
            )

    df = _coerce(df, EDGE_DTYPES)

    if not extra:
        keep = [c for c in ("pre", "post", "weight", "roi") if c in df.columns]
        dropped = [c for c in df.columns if c not in keep]
        df = df[keep]
    else:
        dropped = []
        df = _order(df, ["pre", "post", "weight", "roi"])

    out = stamp(df, ds, query="edges", version=version, raw_columns=list(raw.columns))
    out.attrs["connecto"]["dropped_columns"] = dropped
    return out


def normalize_synapses(
    raw: pd.DataFrame,
    ds,
    *,
    colmap: dict[str, str],
    version=None,
    units: str = "nm",
) -> pd.DataFrame:
    """Normalize synapses; positions in nm unless ``units="voxel"``."""
    df = _consume(raw.copy(), colmap)
    df = _rescale(df, ("pre_", "post_"), ds, units)

    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df), dtype="int64"))

    df = _coerce(df, SYNAPSE_DTYPES)
    df = _order(
        df,
        [
            "id", "pre", "post",
            "pre_x", "pre_y", "pre_z",
            "post_x", "post_y", "post_z",
            "score", "roi", "nt", "nt_confidence",
        ],
    )
    out = stamp(
        df, ds, query="synapses", version=version, raw_columns=list(raw.columns)
    )
    if "nt" in out.columns:
        # Which model run made these calls. Not pedantry: BANC's CAVE door serves
        # `synapses_v2_nt_prediction_5` while its neuPrint door serves an earlier
        # run of the same eight-class model, and on one neuron they agree on only
        # 57% of shared synapses. Both are defensible; a frame that cannot say
        # which one it holds is not.
        out.attrs["connecto"]["transmitters"] = ds._transmitter_source
    return out


def _blank(s: pd.Series) -> pd.Series:
    """Null or empty string. Sources disagree about which one means "no value"."""
    return s.isna() | (s.astype("object") == "")


def _coalesce(df: pd.DataFrame, columns) -> tuple[pd.Series, pd.Series]:
    """Coalesce several columns in priority order; also say which one won.

    This is how `type` gets resolved: FlyWire looks at ``cell_type``, then
    ``hemibrain_type``; maleCNS looks at ``type``, then ``flywireType``... First
    non-null wins.

    Deliberately *mechanical*. connecto does not blacklist "bad" types the way
    cocoa's `_get_*_types` does - curation is analysis, and analysis lives
    upstream of here.

    The second return value is the name of the column each value came from. For
    most fields that is a curiosity; for `nt` it is the difference between a
    measurement and a guess - see :func:`normalize_annotations`.
    """
    present = [c for c in columns if c in df.columns]
    na = pd.Series(pd.NA, index=df.index, dtype="object")
    if not present:
        return na, na.copy()

    out, src = na.copy(), na.copy()
    for col in present:
        take = _blank(out) & ~_blank(df[col])
        out = out.mask(take, df[col])
        src = src.mask(take, col)
    return out, src





def normalize_annotations(
    raw: pd.DataFrame,
    ds,
    *,
    id_column: str,
    fields: dict[str, tuple[str, ...]] | None = None,
    version=None,
    units: str = "nm",
) -> pd.DataFrame:
    """Add canonical columns to an annotation frame; keep every raw column.

    Canonical: ``id, type, side, class, nt, nt_source, status, soma_x/y/z``.
    Derived by column priority from ``spec.fields``, overridable per call.

    ``nt`` gets a companion ``nt_source`` naming the column each value came from,
    and it is the one field that does. The reason is that a dataset's transmitter
    columns are not interchangeable opinions the way its type columns are - they
    are different *kinds* of claim. FlyWire's `known_nt` is somebody's
    immunostaining or RT-PCR; its `top_nt` is a CNN's argmax over a T-bar image.
    maleCNS's `consensusNt` reconciles a prediction with published evidence;
    `predictedNt` is the raw prediction. Coalescing those into one column and
    saying nothing would let "this neuron is GABAergic" mean either "we measured
    it" or "a model thinks so", with no way to tell which - and the two belong on
    different sides of an argument.
    """
    fields = dict(ds.spec.fields) | dict(fields or {})
    df = raw.copy()

    # `id` is a rename; the rest are derived, so raw sources stay in the frame.
    if id_column in df.columns:
        df = _consume(df, {"id": id_column})
    elif "id" not in df.columns:
        raise KeyError(
            f"{ds.label}: annotation source has no ID column {id_column!r} "
            f"(got {list(raw.columns)[:12]}...)."
        )

    # Pull values out of columns that hold them implicitly - e.g. hemibrain's
    # side, which lives as a suffix on the instance name ("DA1_lPN_R").
    for new_col, (src_col, pattern) in ds.spec.derive.items():
        if src_col in df.columns and new_col not in df.columns:
            df[new_col] = df[src_col].astype("string").str.extract(pattern, expand=False)

    # A source may declare a field explicitly empty - "this table does not have
    # one" (see `AnnotationSource.fields`). A raw column of the same name is then
    # not it, and must not be mistaken for it: leaving BANC's neuPrint `side` in
    # place would let `ids(side="left")` quietly answer from the 5% of neurons that
    # have one. Moved aside, so the canonical column is absent and the query says so.
    #
    # After `derive`, not before: `derive` fills a column only when it is missing,
    # so suppressing one first would invite it to be re-created from a regex two
    # lines later - reopening the hole this closes.
    for canon, cols in fields.items():
        if not cols and canon in df.columns:
            df = df.rename(columns={canon: f"{canon}_raw"})

    def _derive(canon, cols):
        # If none of the source columns exist, leave the canonical column *out*
        # rather than inventing an all-null one. A missing column is honest; a
        # column of NaNs looks like "we have this field and it's empty".
        if not cols or not any(c in df.columns for c in cols):
            return None, None
        return _coalesce(df, cols)

    for canon in ("type", "class", "nt", "status", "instance"):
        derived, source = _derive(canon, fields.get(canon))
        if derived is None:
            continue
        # Don't clobber a raw column of the same name that we're deriving *from*.
        if canon in df.columns and canon not in fields[canon]:
            df = df.rename(columns={canon: f"{canon}_raw"})
        df[canon] = derived
        if canon == "nt":
            # A source column of the same name would be a different thing entirely
            # - FlyWire and BANC both carry `known_nt_source`, which is a citation,
            # not a column name - so an incoming `nt_source` steps aside.
            if "nt_source" in df.columns:
                df = df.rename(columns={"nt_source": "nt_source_raw"})
            df["nt_source"] = source

    side_cols = fields.get("side")
    side, _ = _derive("side", side_cols)
    if side is not None:
        if ds.spec.side_map:
            side = side.map(
                lambda v: ds.spec.side_map.get(v, pd.NA) if pd.notna(v) else pd.NA
            )
        if "side" in df.columns and "side" not in side_cols:
            df = df.rename(columns={"side": "side_raw"})
        df["side"] = side

    soma_cols = fields.get("soma")
    if soma_cols and len(soma_cols) == 3:
        for axis, col in zip("xyz", soma_cols):
            if col in df.columns:
                df[f"soma_{axis}"] = pd.to_numeric(df[col], errors="coerce")
        df = _rescale(df, ("soma_",), ds, units)

    df = _coerce(df, ANNOTATION_DTYPES)

    # Root ID 0 means "no segment": CAVE annotation tables carry rows whose point
    # doesn't land on anything (MICrONS has thousands). A *null* ID means the neuron
    # has no root at this materialization - a live FlyTable row with no `root_783`,
    # say. Neither is a queryable neuron, and a surviving null is worse than useless:
    # it coerces the whole `id` column to float64, the very root-ID-as-float bug the
    # `caveclient>=8.0` pin exists to avoid. So drop both and keep `id` an honest int.
    invalid = 0
    if "id" in df.columns:
        ids = pd.to_numeric(df["id"], errors="coerce")
        keep = (ids.notna() & (ids > 0)).to_numpy()
        invalid = int((~keep).sum())
        if invalid:
            df = df[keep].reset_index(drop=True)
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")

    df = _order(
        df,
        [
            "id", "type", "side", "class", "nt", "nt_source", "status",
            "soma_x", "soma_y", "soma_z",
        ],
    )
    out = stamp(
        df, ds, query="annotations", version=version, raw_columns=list(raw.columns)
    )
    out.attrs["connecto"]["dropped_unassigned"] = invalid if "id" in df.columns else 0
    return out


def normalize_skeleton(
    raw: pd.DataFrame,
    ds,
    *,
    colmap: dict[str, str],
    units: str = "nm",
    source_units: str | None = None,
) -> pd.DataFrame:
    """Normalize skeleton nodes into navis' ``TreeNeuron.nodes`` layout."""
    df = _consume(raw.copy(), colmap)

    if "parent_id" in df.columns:
        # navis uses -1 for the root; SWC-ish sources use -1, 0 or NaN.
        df["parent_id"] = (
            pd.to_numeric(df["parent_id"], errors="coerce").fillna(-1).astype("int32")
        )
        df.loc[df["parent_id"] == 0, "parent_id"] = -1

    df = _rescale(df, ("",), ds, units, have=source_units)

    if "radius" not in df.columns:
        df["radius"] = -1.0

    dtypes = {
        "node_id": "int32",
        "parent_id": "int32",
        "x": "float32",
        "y": "float32",
        "z": "float32",
        "radius": "float32",
    }
    df = _coerce(df, dtypes)
    return _order(df, ["node_id", "parent_id", "x", "y", "z", "radius"])
