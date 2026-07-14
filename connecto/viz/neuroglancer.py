"""Neuroglancer scenes and URLs.

A scene is a plain dict. Build one with :func:`construct_scene`, push layers onto it
with :func:`add_annotation_layer` / :func:`add_skeleton_layer`, turn it into a link
with :func:`encode_url`. Nothing here holds state, and every function takes and
returns the dict, so an unusual scene is always one dict edit away.

**Two dialects.** Neuroglancer forked, and the two halves cannot read each other's
scenes. ``ngl.flywire.ai`` descends from a 2021 seung-lab build: it stores the camera
under ``navigation.pose.position.voxelCoordinates``, hides segments in a
``hiddenSegments`` list, and calls a graphene layer ``segmentation_with_graph``. Base
neuroglancer and spelunker went the other way: ``dimensions`` and ``position``, hidden
segments marked with a ``!`` prefix inside ``segments``, plain ``segmentation``. There
is no overlap and no negotiation - and, worse, no error: hand FlyWire's viewer a modern
scene and it opens perfectly well on an empty brain.

So the dataset declares which viewer it belongs to (``DatasetSpec.viewer``) and which
dialect that viewer speaks (``DatasetSpec.viewer_dialect``), and this module serialises
to match. Everything below is written twice, once per dialect, and the conformance suite
checks both against the keys the real viewers actually parse.

**Coordinates are nanometres.** Neuroglancer wants voxels; connecto speaks nanometres
everywhere else, and a soma position handed straight from ``ds.annotations.get()`` into
a scene must land on the soma. So every coordinate here - ``position``, annotations,
skeleton nodes - is nanometres by default and converted on the way in. Pass
``units="voxel"`` if you have viewer coordinates already.
"""

from __future__ import annotations

import copy
import json
import logging
import urllib.parse
import uuid

import numpy as np
import pandas as pd

from ..core.spec import DIALECTS

logger = logging.getLogger("connecto")

__all__ = [
    "construct_scene",
    "encode_url",
    "decode_url",
    "build_url",
    "add_annotation_layer",
    "add_skeleton_layer",
]

DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com"

# Viewers we know are stuck on the old schema. Used only to *refuse* an impossible
# combination, never to silently infer one: a modern scene sent to one of these opens
# on an empty viewer with no error, and that is the exact failure this library exists
# to make impossible.
_SEUNGLAB_HOSTS = frozenset({"ngl.flywire.ai", "neuromancer-seung-import.appspot.com"})

_GREY = "#999999"  # no label -> grey, and visibly so

# Where a link stops being shareable. Safari gives up around 80k, and Slack and email
# clients truncate well before a megabyte - which one skeleton layer easily reaches.
_URL_WARN_AT = 100_000


# --------------------------------------------------------------------------- scenes

def construct_scene(
    ds,
    ids=None,
    *,
    seg_colors=None,
    seg_groups=None,
    invis_segs=None,
    color_by=None,
    group_by=None,
    palette=None,
    position=None,
    units: str = "nm",
    layout: str = "xy-3d",
    image: bool = True,
    layers=None,
    viewer: str | None = None,
    dialect: str | None = None,
    version=None,
) -> dict:
    """Build a neuroglancer scene for this dataset.

    Parameters
    ----------
    ids :           IDs to select. Order is *meaningful*: ``seg_colors`` and
                    ``seg_groups``, when given as plain lists, zip onto it.
    seg_colors :    One colour for all of them, a list of colours (one per ID), or a
                    ``{id: colour}`` dict. Anything matplotlib understands is a colour:
                    ``"red"``, ``"#ff0000"``, ``(1, 0, 0)``.
    color_by :      Colour by a *label* instead: an annotation field (``"type"``,
                    ``"class"``, ...) or an array of labels aligned to ``ids``. Each
                    distinct label gets a colour from ``palette``. Mutually exclusive
                    with ``seg_colors``.
    seg_groups :    Split the IDs across separate layers, so they can be toggled
                    independently. A list of group names (one per ID), a
                    ``{id: group}`` dict, or a ``{group: [ids]}`` dict.
    group_by :      As ``color_by``, but for groups: an annotation field or an array.
    invis_segs :    Selected but *not* rendered - loaded and listed in the layer with
                    their visibility switched off, ready to be toggled on in the viewer.
    position :      Where to point the camera, in nanometres (see ``units``).
    units :         ``"nm"`` (the default, and what every other connecto method returns)
                    or ``"voxel"`` for coordinates already in viewer space.
    image :         Whether to include the EM image layer. Without it the segments float
                    in a void, which is rarely what anyone wants.
    layers :        Extra layer dicts to append verbatim.

    Returns
    -------
    scene :         dict
    """
    viewer, dialect = _viewer_and_dialect(ds, viewer, dialect)
    voxel = _voxel_size(ds)
    ids = _as_ids(ids)

    colors = _resolve_colors(ds, ids, seg_colors, color_by, palette, version)
    groups = _resolve_groups(ds, ids, seg_groups, group_by, version)
    invis = _as_ids(invis_segs)

    seg_source, img_source = _sources(ds, dialect)
    name = ds.name

    # With groups, the segments live in the group layers instead - otherwise every
    # neuron would be rendered twice, once by each layer, and the toggles would fight.
    base_segs = [] if groups else list(ids)

    seg_layer = _seg_layer(
        dialect, seg_source, name, base_segs, invis=invis, colors=colors, voxel=voxel
    )

    scene_layers = []
    if image and img_source:
        scene_layers.append({"type": "image", "source": img_source, "name": "image"})
    scene_layers.append(seg_layer)

    for group, members in (groups or {}).items():
        layer = _seg_layer(
            dialect, seg_source, group, members, invis=(), colors=colors, voxel=voxel
        )
        scene_layers.append(layer)

    scene_layers.extend(copy.deepcopy(list(layers or [])))

    scene = _base_scene(ds, dialect, voxel, layout, selected=name)
    scene["layers"] = scene_layers

    if position is not None:
        pos = _to_voxels(np.asarray(position, dtype=float).ravel(), voxel, units)
        if pos.size != 3:
            raise ValueError(f"`position` must be one x/y/z point, got {pos.size} values.")
        _set_position(scene, dialect, pos)

    return scene


def _base_scene(ds, dialect: str, voxel, layout: str, *, selected: str) -> dict:
    """The scene minus its layers - camera, units, and where to POST a shared state."""
    if dialect == "seunglab":
        scene = {
            "layers": [],
            "navigation": {
                # The old schema carries the voxel size *inside* the camera pose. It
                # is the only place it appears, so annotation layers read it back from
                # here (see `_scene_voxel_size`).
                "pose": {"position": {"voxelSize": [float(v) for v in voxel]}},
                "zoomFactor": 2.8,
            },
            "showAxisLines": False,
            "perspectiveZoom": 4800,
            "layout": layout,
            "selectedLayer": {"layer": selected, "visible": True},
        }
        state_server = _state_server(ds)
        if state_server:
            # Lets the viewer's own share button work, rather than only ours.
            scene["jsonStateServer"] = state_server
        return scene

    return {
        "layers": [],
        "dimensions": {axis: [float(v) * 1e-9, "m"] for axis, v in zip("xyz", voxel)},
        "layout": layout,
        "selectedLayer": {"layer": selected, "visible": True},
    }


def _seg_layer(dialect, source, name, segments, *, invis, colors, voxel) -> dict:
    """One segmentation layer, in whichever schema the target viewer parses."""
    segments = [str(int(i)) for i in segments]
    invis = [str(int(i)) for i in invis]

    # A flat `precomputed://` volume has no chunkedgraph, so it is a plain segmentation
    # layer even in the old schema; only graphene sources get the proofreading UI.
    graphene = str(source).startswith("graphene://")
    layer_type = (
        "segmentation_with_graph" if (dialect == "seunglab" and graphene) else "segmentation"
    )

    layer = {"type": layer_type, "source": source, "name": str(name)}

    if dialect == "seunglab":
        layer["segments"] = segments
        if invis:
            layer["hiddenSegments"] = invis
    else:
        # Modern neuroglancer keeps hidden segments in the same list, prefixed with
        # "!" - it parses them back out into `selectedSegments` minus `visibleSegments`.
        layer["segments"] = segments + [f"!{i}" for i in invis]

    if colors:
        shown = set(segments) | set(invis)
        subset = {i: c for i, c in colors.items() if i in shown}
        if subset:
            layer["segmentColors"] = subset

    return layer


def _set_position(scene: dict, dialect: str, pos) -> None:
    pos = [float(p) for p in pos]
    if dialect == "seunglab":
        # The old viewer wants whole voxels here.
        scene["navigation"]["pose"]["position"]["voxelCoordinates"] = [
            int(round(p)) for p in pos
        ]
    else:
        scene["position"] = pos


# ------------------------------------------------------------------ colours & groups

def _resolve_colors(ds, ids, seg_colors, color_by, palette, version) -> dict | None:
    """-> ``{"<id>": "#rrggbb"}``, or None."""
    if seg_colors is not None and color_by is not None:
        raise ValueError(
            "Give `seg_colors` (explicit colours) or `color_by` (colours derived from "
            "a label), not both."
        )

    if color_by is not None:
        labels = _labels(ds, ids, color_by, version)
        seg_colors = _palette_for(labels, palette)

    if seg_colors is None:
        return None

    if isinstance(seg_colors, dict):
        return {str(int(k)): _to_hex(v) for k, v in seg_colors.items()}

    if _is_single_color(seg_colors):
        return {str(int(i)): _to_hex(seg_colors) for i in ids}

    colors = list(seg_colors)
    if len(colors) != len(ids):
        raise ValueError(f"Got {len(colors)} colours for {len(ids)} segments.")
    return {str(int(i)): _to_hex(c) for i, c in zip(ids, colors)}


def _resolve_groups(ds, ids, seg_groups, group_by, version) -> dict | None:
    """-> ``{"group": [ids]}``, or None."""
    if seg_groups is not None and group_by is not None:
        raise ValueError("Give `seg_groups` or `group_by`, not both.")

    if group_by is not None:
        labels = _labels(ds, ids, group_by, version)
        seg_groups = [_clean(lbl) or "unlabelled" for lbl in labels]

    if seg_groups is None:
        return None

    if isinstance(seg_groups, dict):
        values = list(seg_groups.values())
        is_list = [isinstance(v, (list, tuple, set, np.ndarray, pd.Series)) for v in values]
        if values and all(is_list):
            return {str(g): _as_ids(v) for g, v in seg_groups.items()}
        if not any(is_list):
            groups: dict[str, list[int]] = {}
            for i, g in seg_groups.items():
                groups.setdefault(str(g), []).append(int(i))
            return groups
        raise ValueError(
            "`seg_groups` mixes {id: group} and {group: [ids]} - pick one shape."
        )

    values = list(seg_groups)
    if len(values) != len(ids):
        raise ValueError(f"Got {len(values)} groups for {len(ids)} segments.")
    groups = {}
    for i, g in zip(ids, values):
        groups.setdefault(str(g), []).append(int(i))
    return groups


def _labels(ds, ids, by, version=None) -> np.ndarray:
    """Labels for `ids`: either handed to us, or looked up in the annotations.

    This is the bit fafbseg cannot do - it has segments and colours, but no idea what
    any of them *are*. connecto does, so `color_by="type"` is one word.
    """
    if isinstance(by, str):
        ann = ds.annotations.get(ids, version=version)
        if by not in ann.columns:
            available = ", ".join(c for c in ann.columns if c != "id")
            raise ValueError(
                f"{ds.label} has no annotation field {by!r}. Available: {available}."
            )
        lut = ann.set_index("id")[by]
        lut = lut[~lut.index.duplicated()]
        return np.array([_clean(lut.get(int(i))) for i in ids], dtype=object)

    labels = np.asarray(by, dtype=object).ravel()
    if len(labels) != len(ids):
        raise ValueError(f"Got {len(labels)} labels for {len(ids)} segments.")
    return np.array([_clean(v) for v in labels], dtype=object)


def _palette_for(labels, palette=None) -> list:
    """One colour per label, distinct labels sharing a colour."""
    from matplotlib import colormaps

    distinct = [lbl for lbl in dict.fromkeys(labels) if lbl is not None]
    if palette is None:
        palette = "tab10" if len(distinct) <= 10 else "tab20" if len(distinct) <= 20 else "hsv"

    cmap = colormaps[palette]
    if getattr(cmap, "N", 256) >= 256:  # continuous: spread the labels across it
        picks = [cmap(i / max(len(distinct) - 1, 1)) for i in range(len(distinct))]
    else:  # categorical: take its colours in order
        picks = [cmap(i % cmap.N) for i in range(len(distinct))]

    lut = dict(zip(distinct, picks))
    return [lut.get(lbl, _GREY) for lbl in labels]


def _to_hex(color) -> str:
    from matplotlib.colors import to_hex

    try:
        return to_hex(color)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"{color!r} is not a colour. If you meant to colour *by* it - by cell type, "
            f"say - that is `color_by=`, which maps labels onto a palette for you."
        ) from e


def _is_single_color(value) -> bool:
    """One colour ("red", "#f00", (1, 0, 0)), as opposed to a sequence of them."""
    if isinstance(value, str):
        return True
    if isinstance(value, tuple) and len(value) in (3, 4):
        return all(isinstance(v, (int, float, np.floating, np.integer)) for v in value)
    return False


def _clean(value):
    """Label -> str, or None if it is missing. NaN is not a cell type."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------- layers

def add_annotation_layer(
    scene: dict, coords, *, name: str | None = None, color=None, units: str = "nm"
) -> dict:
    """Add points, lines or ellipsoids as a new annotation layer.

    The shape of ``coords`` picks the annotation type::

        (N, 3)      points          x/y/z
        (N, 2, 3)   line segments   start x/y/z, end x/y/z
        (N, 4)      ellipsoids      x/y/z + radius

    Coordinates are nanometres by default - so a synapse table from
    ``ds.connectivity.synapses()`` goes straight in - and are converted to the scene's
    own voxel size on the way. The scene is not modified; a new one comes back.
    """
    scene = copy.deepcopy(scene)
    dialect = _scene_dialect(scene)
    voxel = _scene_voxel_size(scene)

    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 2 and coords.shape[1] == 3:
        pts = _to_voxels(coords, voxel, units)
        records = [
            {"type": "point", "point": p.tolist(), "id": uuid.uuid4().hex}
            for p in pts
        ]
    elif coords.ndim == 3 and coords.shape[1:] == (2, 3):
        pts = _to_voxels(coords.reshape(-1, 3), voxel, units).reshape(-1, 2, 3)
        records = [
            {
                "type": "line",
                "pointA": a.tolist(),
                "pointB": b.tolist(),
                "id": uuid.uuid4().hex,
            }
            for a, b in pts
        ]
    elif coords.ndim == 2 and coords.shape[1] == 4:
        centers = _to_voxels(coords[:, :3], voxel, units)
        # One radius in nanometres becomes three, because a voxel is not a cube: 40nm
        # deep and 4nm wide on FlyWire, so a sphere is 10x fewer voxels in z.
        radii = _to_voxels(np.repeat(coords[:, 3:4], 3, axis=1), voxel, units)
        records = [
            {
                "type": "ellipsoid",
                "center": c.tolist(),
                "radii": r.tolist(),
                "id": uuid.uuid4().hex,
            }
            for c, r in zip(centers, radii)
        ]
    else:
        raise ValueError(
            "`coords` must be (N, 3) points, (N, 2, 3) lines or (N, 4) ellipsoids "
            f"(x/y/z + radius), got shape {coords.shape}."
        )

    if not name:
        n = sum(1 for lyr in scene["layers"] if lyr.get("type") == "annotation")
        name = f"annotations{n or ''}"

    layer = {"type": "annotation", "name": name, "annotations": records}
    if dialect == "seunglab":
        layer["voxelSize"] = [float(v) for v in voxel]
        layer["annotationTags"] = []
    else:
        # The modern viewer needs to be told what space these numbers are in. Same
        # dimensions as the scene, so they land where the segments are.
        layer["source"] = {
            "url": "local://annotations",
            "transform": {"outputDimensions": scene["dimensions"]},
        }
    if color is not None:
        layer["annotationColor"] = _to_hex(color)

    scene["layers"].append(layer)
    return scene


def add_skeleton_layer(scene: dict, x, *, name: str | None = None, color=None) -> dict:
    """Add skeletons as line annotations - one layer per neuron.

    Takes navis ``TreeNeuron``\\ s (a ``NeuronList`` is fine) and draws every
    parent-child edge as a line. Useful for showing a skeleton *next to* the
    segmentation it came from, or for putting a neuron from somewhere else - a CATMAID
    tracing, a transformed neuron - into the scene.

    Units come off the neuron: navis carries them, so a micron-scale neuron (from
    ``ds.l2.dotprops(units="um")``, say) is scaled correctly rather than landing a
    thousand times too close to the origin.
    """
    import navis

    neurons = x if isinstance(x, navis.NeuronList) else navis.NeuronList(x)

    colors = color
    if color is not None and not _is_single_color(color):
        colors = list(color)
        if len(colors) != len(neurons):
            raise ValueError(f"Got {len(colors)} colours for {len(neurons)} neurons.")

    for i, neuron in enumerate(neurons):
        if not isinstance(neuron, navis.TreeNeuron):
            raise TypeError(
                f"`add_skeleton_layer` needs skeletons (navis.TreeNeuron), got "
                f"{type(neuron).__name__}. Meshes have no edges to draw."
            )
        segments = _skeleton_segments(neuron)
        this_color = colors[i] if isinstance(colors, list) else colors
        scene = add_annotation_layer(
            scene,
            segments,
            name=name or str(neuron.name or neuron.id),
            color=this_color,
            units="nm",
        )
    return scene


def _skeleton_segments(neuron) -> np.ndarray:
    """Parent-child edges of a skeleton -> (N, 2, 3) nanometre line segments."""
    nodes = neuron.nodes
    children = nodes[nodes.parent_id >= 0]
    if not len(children):
        raise ValueError(f"Skeleton {neuron.id} has no edges to draw.")

    start = children[["x", "y", "z"]].to_numpy(dtype=float)
    end = (
        nodes.set_index("node_id")
        .loc[children.parent_id.to_numpy(), ["x", "y", "z"]]
        .to_numpy(dtype=float)
    )
    return np.stack([start, end], axis=1) * _neuron_scale_to_nm(neuron)


def _neuron_scale_to_nm(neuron) -> float:
    """What to multiply this neuron's coordinates by to get nanometres."""
    try:
        return float(neuron.units.to("nm").magnitude)
    except Exception:  # noqa: BLE001 - dimensionless, unset, or a unit pint dislikes
        return 1.0  # connecto's default, and navis's when nobody said otherwise


# ------------------------------------------------------------------------------ urls

def encode_url(scene: dict, *, viewer: str = DEFAULT_VIEWER) -> str:
    url = f"{viewer.rstrip('/')}/#!{urllib.parse.quote(json.dumps(scene))}"

    # A scene is a URL fragment, so a skeleton layer of a few thousand nodes is a few
    # thousand line annotations is a megabyte of link. Nothing rejects it: the browser
    # or the chat client simply truncates, and a truncated scene is a broken scene that
    # looks like a working one. Say so - and say what to do about it.
    if len(url) > _URL_WARN_AT:
        logger.warning(
            "This neuroglancer URL is %.1f MB. Browsers and chat clients truncate long "
            "links. Pass `shorten=True` to post the scene to the CAVE state server and "
            "get a short link back.",
            len(url) / 1e6,
        )
    return url


def decode_url(url: str) -> dict:
    _, _, fragment = url.partition("#!")
    if not fragment:
        raise ValueError("Not a neuroglancer URL.")
    return json.loads(urllib.parse.unquote(fragment))


def build_url(
    ds,
    ids=None,
    *,
    shorten: bool = False,
    viewer: str | None = None,
    dialect: str | None = None,
    **kwargs,
) -> str:
    """A neuroglancer URL showing the given neurons. See :func:`construct_scene`."""
    viewer, dialect = _viewer_and_dialect(ds, viewer, dialect)
    scene = construct_scene(ds, ids, viewer=viewer, dialect=dialect, **kwargs)

    if not shorten:
        return encode_url(scene, viewer=viewer)

    if ds.backend_kind != "cave":
        raise ValueError("Only CAVE datasets can shorten URLs (via the state server).")
    state_id = ds.client.state.upload_state_json(scene)
    return ds.client.state.build_neuroglancer_url(state_id, ngl_url=viewer)


# --------------------------------------------------------------------------- plumbing

def _viewer_and_dialect(ds, viewer=None, dialect=None) -> tuple[str, str]:
    spec = ds.spec

    if viewer is None:
        viewer = spec.viewer
    if viewer is None and ds.backend_kind == "cave":
        try:
            viewer = ds.client.info.viewer_site()
        except Exception:  # noqa: BLE001 - a viewer is not worth failing a query over
            viewer = None
    viewer = (viewer or DEFAULT_VIEWER).rstrip("/")

    if dialect is None:
        dialect = spec.viewer_dialect
    if dialect not in DIALECTS:
        raise ValueError(f"`dialect` must be one of {', '.join(DIALECTS)}, got {dialect!r}.")

    host = urllib.parse.urlparse(viewer).netloc
    if host in _SEUNGLAB_HOSTS and dialect != "seunglab":
        raise ValueError(
            f"{viewer} is an old-schema neuroglancer: it cannot read a {dialect!r} "
            f"scene, and it will not say so - it opens on an empty viewer. Pass "
            f"dialect='seunglab', or point `viewer` at a modern one."
        )
    return viewer, dialect


def _sources(ds, dialect: str) -> tuple[str, str | None]:
    """(segmentation, image) sources, formatted for the viewer we are targeting.

    The dataset resolves these - see `Dataset._segmentation_source`. `format_for` is
    what decides whether a graphene URL carries the `middleauth+` prefix, and that is
    not decoration: it tells a modern viewer to run CAVE's login flow, and a protected
    datastack will not load without it, while the old fork chokes on it.
    """
    fmt = "neuroglancer" if dialect == "seunglab" else "cave_explorer"
    seg = ds._segmentation_source(format_for=fmt)
    if seg is None:
        raise ValueError(f"{ds.label} declares no segmentation source.")
    return seg, ds._image_source()


def _state_server(ds) -> str | None:
    if ds.backend_kind != "cave":
        return None
    try:
        return ds.client.state.state_service_endpoint
    except Exception:  # noqa: BLE001
        return None


def _voxel_size(ds) -> tuple[float, float, float]:
    voxel = ds.spec.voxel_size
    if voxel is None:
        raise ValueError(
            f"{ds.label} declares no `voxel_size`, so there is no way to convert "
            f"nanometres into viewer coordinates."
        )
    return tuple(float(v) for v in voxel)


def _scene_dialect(scene: dict) -> str:
    return "seunglab" if "navigation" in scene else "modern"


def _scene_voxel_size(scene: dict) -> tuple[float, float, float]:
    """Read the voxel size back out of a scene, so layers can be added to any scene.

    Both dialects record it, in different places. Reading it from the scene rather than
    taking it as an argument means `add_annotation_layer` cannot disagree with the
    scene it is adding to.
    """
    dims = scene.get("dimensions")
    if dims:
        return tuple(float(dims[axis][0]) / 1e-9 for axis in "xyz")

    pose = scene.get("navigation", {}).get("pose", {}).get("position", {})
    if "voxelSize" in pose:
        return tuple(float(v) for v in pose["voxelSize"])

    raise ValueError("Scene declares no voxel size, so nanometres cannot be converted.")


def _to_voxels(coords, voxel, units: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if units == "nm":
        return coords / np.asarray(voxel, dtype=float)
    if units in ("voxel", "voxels"):
        return coords
    raise ValueError(f"`units` must be 'nm' or 'voxel', got {units!r}.")


def _as_ids(x) -> list[int]:
    """-> a list of ints, order preserved, duplicates dropped."""
    if x is None:
        return []
    values = np.atleast_1d(np.asarray(x)).ravel()
    return list(dict.fromkeys(int(v) for v in values))
