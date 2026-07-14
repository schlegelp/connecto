"""Neuroglancer scenes.

Offline: a stub dataset stands in for the servers, because everything interesting here
is a *serialisation* decision and none of it needs a live CAVE.

The thing under test is mostly the split between the two dialects. Neuroglancer forked,
and the halves cannot read each other's scenes - but neither half *says* so. Hand
`ngl.flywire.ai` a modern scene and it opens, cheerfully, on an empty brain. So the
tests below are less about JSON shape than about the fact that a wrong scene is
indistinguishable from a right one until a human looks at it, which means it has to be
right by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from connecto.core.spec import BackendSpec, Cap, DatasetSpec
from connecto.viz import (
    add_annotation_layer,
    add_skeleton_layer,
    construct_scene,
    decode_url,
    encode_url,
)

# --------------------------------------------------------------------------- the stub

IDS = [100, 200, 300]

_ANN = pd.DataFrame(
    {
        "id": [100, 200, 300],
        "type": ["DA1_lPN", "DA1_lPN", "DA2_lPN"],
        "side": ["left", "right", "right"],
    }
)


class _Info:
    """CAVE's info service, minus CAVE. Records what it was asked for."""

    def __init__(self):
        self.asked_for = []
        self.image_asked_for = []

    def segmentation_source(self, format_for="raw"):
        self.asked_for.append(format_for)
        prefix = "middleauth+" if format_for == "cave_explorer" else ""
        return f"graphene://{prefix}https://example.org/segmentation/table/stub"

    def image_source(self, format_for="raw"):
        # Mirrors the real quirk: caveclient hands back None for the viewer formats.
        self.image_asked_for.append(format_for)
        return "precomputed://gs://stub/em" if format_for == "raw" else None

    def viewer_site(self):
        return "https://spelunker.cave-explorer.org/"


class _State:
    state_service_endpoint = "https://example.org/nglstate/api/v1/post"


class _Client:
    def __init__(self):
        self.info = _Info()
        self.state = _State()


class _Annotations:
    def get(self, x=None, version=None):
        return _ANN[_ANN["id"].isin(list(x))] if x is not None else _ANN


class _StubDS:
    backend_kind = "cave"

    def __init__(self, **kwargs):
        self.spec = DatasetSpec(
            name="stub",
            label="Stub dataset",
            backends=(BackendSpec("cave", "stub_datastack"),),
            voxel_size=(4, 4, 40),
            capabilities=frozenset({Cap.NEUROGLANCER, Cap.ANNOTATIONS}),
            **kwargs,
        )
        self.client = _Client()
        self.annotations = _Annotations()

    name = "stub"
    label = "Stub dataset"

    # Mirrors CAVEDataset: the spec may override the volume, otherwise ask caveclient -
    # which is what decides whether the graphene URL carries `middleauth+`.
    def _segmentation_source(self, *, format_for: str = "raw"):
        if self.spec.segmentation_source is not None:
            return self.spec.segmentation_source
        return self.client.info.segmentation_source(format_for=format_for)

    def _image_source(self):
        return self.client.info.image_source(format_for="raw")


@pytest.fixture
def modern():
    return _StubDS()


@pytest.fixture
def legacy():
    return _StubDS(viewer="https://ngl.flywire.ai", viewer_dialect="seunglab")


def _seg(scene):
    return next(lyr for lyr in scene["layers"] if lyr["type"].startswith("segmentation"))


# ------------------------------------------------------------------------- the schema

def test_the_two_dialects_are_actually_different(modern, legacy):
    """Not a style preference. `ngl.flywire.ai`'s bundle contains `voxelCoordinates`
    and `hiddenSegments` and zero occurrences of `crossSectionScale`; base neuroglancer
    and spelunker are the exact mirror. A scene in the wrong schema opens on an empty
    viewer without erroring, so the schema is a promise like any other."""
    m = construct_scene(modern, IDS, position=(400, 400, 4000))
    lg = construct_scene(legacy, IDS, position=(400, 400, 4000))

    # Modern: a global coordinate space, camera in `position`.
    assert m["dimensions"] == {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]}
    assert m["position"] == [100.0, 100.0, 100.0]
    assert "navigation" not in m

    # Legacy: voxel size lives inside the camera pose, and it wants whole voxels.
    pose = lg["navigation"]["pose"]["position"]
    assert pose["voxelSize"] == [4.0, 4.0, 40.0]
    assert pose["voxelCoordinates"] == [100, 100, 100]
    assert "dimensions" not in lg


def test_hidden_segments_use_each_viewers_own_encoding(modern, legacy):
    """Modern neuroglancer parses "!123" out of `segments` into selected-but-not-visible
    (segmentation_user_layer.ts). The old fork keeps a separate `hiddenSegments` list.
    Same intent, no overlap."""
    m = _seg(construct_scene(modern, [100], invis_segs=[200]))
    assert m["segments"] == ["100", "!200"]
    assert "hiddenSegments" not in m

    lg = _seg(construct_scene(legacy, [100], invis_segs=[200]))
    assert lg["segments"] == ["100"]
    assert lg["hiddenSegments"] == ["200"]


def test_graphene_layers_get_the_graph_ui_only_on_the_old_fork(modern, legacy):
    """`segmentation_with_graph` is a seunglab type - the modern viewer has no such
    layer. But a *flat* precomputed volume has no chunkedgraph either way."""
    assert _seg(construct_scene(legacy, IDS))["type"] == "segmentation_with_graph"
    assert _seg(construct_scene(modern, IDS))["type"] == "segmentation"

    flat = _StubDS(segmentation_source="precomputed://gs://flat", viewer_dialect="seunglab")
    assert _seg(construct_scene(flat, IDS))["type"] == "segmentation"


def test_middleauth_is_asked_of_caveclient_not_spliced_in(modern, legacy):
    """Modern viewers need `graphene://middleauth+https://...` to know they must run
    CAVE's login flow; without it a protected datastack just fails to load. The old one
    chokes on the prefix. caveclient's `format_for` already encodes which is which, so
    the test pins that we *ask* rather than string-munge."""
    assert "middleauth+" in _seg(construct_scene(modern, IDS))["source"]
    assert modern.client.info.asked_for == ["cave_explorer"]

    assert "middleauth+" not in _seg(construct_scene(legacy, IDS))["source"]
    assert legacy.client.info.asked_for == ["neuroglancer"]

    # The image, though, must be asked for raw - caveclient returns None for the
    # viewer formats, and a None image source is just a missing layer.
    assert modern.client.info.image_asked_for == ["raw"]


def test_a_modern_scene_is_refused_for_a_legacy_viewer(legacy):
    """The one combination that fails silently in the wild is the one we refuse."""
    with pytest.raises(ValueError, match="old-schema"):
        construct_scene(legacy, IDS, dialect="modern")


def test_there_is_an_image_layer(modern):
    """Segments in a void are rarely what anyone wants - and `image_source()` returns
    None for every viewer format except "raw", which is easy to not notice."""
    scene = construct_scene(modern, IDS)
    assert [lyr["type"] for lyr in scene["layers"]] == ["image", "segmentation"]
    assert scene["layers"][0]["source"] == "precomputed://gs://stub/em"

    bare = construct_scene(modern, IDS, image=False)
    assert [lyr["type"] for lyr in bare["layers"]] == ["segmentation"]


# ------------------------------------------------------------------------ the colours

def test_colours_bind_to_the_order_the_caller_wrote(modern):
    """The whole reason `Viz` resolves IDs itself: `ds.ids()` ends in `np.unique`, so it
    sorts, and a positional colour list zipped onto sorted IDs paints the wrong neuron
    the wrong colour - a scene that is wrong but not visibly wrong."""
    scene = construct_scene(modern, [300, 100], seg_colors=["red", "blue"])
    assert _seg(scene)["segmentColors"] == {"300": "#ff0000", "100": "#0000ff"}


@pytest.mark.parametrize(
    "colors,expected",
    [
        ("red", {"100": "#ff0000", "200": "#ff0000", "300": "#ff0000"}),
        ((1.0, 0.0, 0.0), {"100": "#ff0000", "200": "#ff0000", "300": "#ff0000"}),
        ({100: "red"}, {"100": "#ff0000"}),
    ],
)
def test_seg_colors_takes_the_shapes_people_actually_pass(modern, colors, expected):
    assert _seg(construct_scene(modern, IDS, seg_colors=colors))["segmentColors"] == expected


def test_color_by_looks_the_label_up_in_the_annotations(modern):
    """The thing fafbseg cannot do: it has segments and colours but no idea what any of
    them *are*. Same type, same colour; different type, different colour."""
    scene = construct_scene(modern, IDS, color_by="type")
    colors = _seg(scene)["segmentColors"]

    assert colors["100"] == colors["200"]  # both DA1_lPN
    assert colors["300"] != colors["100"]  # DA2_lPN


def test_a_cell_type_passed_as_a_colour_says_so(modern):
    """`seg_colors=df.cell_type` is the obvious mistake, and "invalid RGBA argument"
    would not help anyone."""
    with pytest.raises(ValueError, match="color_by"):
        construct_scene(modern, IDS, seg_colors=["DA1_lPN", "DA1_lPN", "DA2_lPN"])


def test_colours_and_labels_are_not_both_accepted(modern):
    with pytest.raises(ValueError, match="not both"):
        construct_scene(modern, IDS, seg_colors="red", color_by="type")


def test_wrong_number_of_colours_raises(modern):
    with pytest.raises(ValueError, match="2 colours for 3 segments"):
        construct_scene(modern, IDS, seg_colors=["red", "blue"])


# ------------------------------------------------------------------------- the groups

def test_groups_become_layers_you_can_toggle(modern):
    scene = construct_scene(modern, IDS, group_by="type")
    layers = {lyr["name"]: lyr for lyr in scene["layers"] if lyr["type"] == "segmentation"}

    assert layers["DA1_lPN"]["segments"] == ["100", "200"]
    assert layers["DA2_lPN"]["segments"] == ["300"]
    # The base layer holds none of them, or every neuron would render twice.
    assert layers["stub"]["segments"] == []


@pytest.mark.parametrize(
    "groups",
    [
        ["a", "a", "b"],                    # one per segment
        {100: "a", 200: "a", 300: "b"},     # {id: group}
        {"a": [100, 200], "b": [300]},      # {group: [ids]}
    ],
)
def test_seg_groups_takes_the_shapes_people_actually_pass(modern, groups):
    scene = construct_scene(modern, IDS, seg_groups=groups)
    layers = {lyr["name"]: lyr for lyr in scene["layers"] if lyr["type"] == "segmentation"}
    assert layers["a"]["segments"] == ["100", "200"]
    assert layers["b"]["segments"] == ["300"]


def test_mixed_group_shapes_raise(modern):
    with pytest.raises(ValueError, match="mixes"):
        construct_scene(modern, IDS, seg_groups={100: "a", "b": [200, 300]})


# ------------------------------------------------------------------------- the layers

def test_annotations_are_nanometres_in_voxels_out(modern):
    """Every coordinate connecto hands you is nanometres - a soma from
    `annotations.get()`, a synapse from `connectivity.synapses()`. Neuroglancer wants
    voxels. Converting at the boundary is what makes the obvious thing work."""
    scene = construct_scene(modern, IDS)
    scene = add_annotation_layer(scene, [[400, 800, 4000]], name="somas", color="yellow")

    layer = scene["layers"][-1]
    assert layer["type"] == "annotation"
    assert layer["annotations"][0]["point"] == [100.0, 200.0, 100.0]
    assert layer["annotationColor"] == "#ffff00"
    # The modern viewer needs telling what space these numbers are in.
    assert layer["source"]["url"] == "local://annotations"
    assert layer["source"]["transform"]["outputDimensions"] == scene["dimensions"]


def test_annotation_coordinates_can_already_be_voxels(modern):
    scene = add_annotation_layer(construct_scene(modern, IDS), [[1, 2, 3]], units="voxel")
    assert scene["layers"][-1]["annotations"][0]["point"] == [1.0, 2.0, 3.0]


def test_the_annotation_shape_picks_the_annotation_type(modern):
    scene = construct_scene(modern, IDS)

    points = add_annotation_layer(scene, np.zeros((2, 3)))["layers"][-1]
    lines = add_annotation_layer(scene, np.zeros((2, 2, 3)))["layers"][-1]
    balls = add_annotation_layer(scene, np.zeros((2, 4)))["layers"][-1]

    assert points["annotations"][0]["type"] == "point"
    assert lines["annotations"][0]["type"] == "line"
    assert balls["annotations"][0]["type"] == "ellipsoid"

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        add_annotation_layer(scene, np.zeros((2, 7)))


def test_an_ellipsoid_radius_is_not_a_cube(modern):
    """A FlyWire voxel is 4nm wide and 40nm deep, so a 400nm sphere is 100 voxels
    across in x and 10 in z. fafbseg's ellipsoid branch was unreachable and emitted no
    radii at all."""
    scene = add_annotation_layer(construct_scene(modern, IDS), [[0, 0, 0, 400]])
    assert scene["layers"][-1]["annotations"][0]["radii"] == [100.0, 100.0, 10.0]


def test_annotations_are_added_to_a_copy(modern):
    scene = construct_scene(modern, IDS)
    before = len(scene["layers"])
    add_annotation_layer(scene, [[0, 0, 0]])
    assert len(scene["layers"]) == before


def test_a_layer_added_to_a_legacy_scene_is_a_legacy_layer(legacy):
    """`add_annotation_layer` reads the voxel size back out of the scene it is given,
    so it cannot disagree with it - and so it works on a hand-edited scene too."""
    scene = add_annotation_layer(construct_scene(legacy, IDS), [[400, 800, 4000]])
    layer = scene["layers"][-1]

    assert layer["voxelSize"] == [4.0, 4.0, 40.0]
    assert layer["annotations"][0]["point"] == [100.0, 200.0, 100.0]
    assert "source" not in layer  # local://annotations is a modern idea


def test_skeletons_become_lines_in_the_units_the_neuron_declares(modern):
    """navis carries units, so a micron-scale neuron - `l2.dotprops(units="um")` hands
    you one - must not land a thousand times too close to the origin."""
    navis = pytest.importorskip("navis")

    nodes = pd.DataFrame(
        {
            "node_id": [1, 2],
            "parent_id": [-1, 1],
            "x": [0.0, 4000.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "radius": [1.0, 1.0],
        }
    )
    nm = navis.TreeNeuron(nodes, id=1, units="1 nm")
    um = navis.TreeNeuron(nodes.assign(x=nodes.x / 1000), id=2, units="1 um")

    for neuron in (nm, um):
        scene = add_skeleton_layer(construct_scene(modern, IDS), neuron, color="red")
        line = scene["layers"][-1]["annotations"][0]
        # 4000nm along x, in 4nm voxels, whichever unit the neuron was in.
        assert line["type"] == "line"
        assert sorted([line["pointA"][0], line["pointB"][0]]) == pytest.approx([0, 1000])


# ---------------------------------------------------------------------------- the url

def test_a_url_round_trips(modern):
    scene = construct_scene(modern, IDS, seg_colors="red")
    url = encode_url(scene, viewer="https://spelunker.cave-explorer.org")

    assert url.startswith("https://spelunker.cave-explorer.org/#!")
    assert decode_url(url) == scene


def test_decoding_something_that_is_not_a_scene_raises():
    with pytest.raises(ValueError, match="Not a neuroglancer URL"):
        decode_url("https://example.org/")


def test_an_unshareably_long_url_says_so(modern, caplog):
    """A scene rides in the URL fragment, so one skeleton layer is a megabyte of link -
    and nothing rejects it. The browser or the chat client truncates, and a truncated
    scene is a broken scene that looks like a working one."""
    scene = construct_scene(modern, IDS)
    scene = add_annotation_layer(scene, np.zeros((20_000, 3)))

    with caplog.at_level("WARNING", logger="connecto"):
        url = encode_url(scene)

    assert len(url) > 100_000
    assert "shorten=True" in caplog.text
