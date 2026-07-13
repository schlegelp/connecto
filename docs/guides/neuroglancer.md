# Neuroglancer

## The one-liner

```python
import connecto as cn

fw = cn.FlyWire()
fw.viz.neuroglancer_url(fw.ids("DA1_lPN"), color_by="type")
```

That is a link, in FlyWire's own neuroglancer, with the EM image loaded, every DA1_lPN
selected, and each cell type in its own colour.

## Colouring

Colouring segments is the thing everyone actually wants, so it has two doors.

**`seg_colors`** when you know the colours. One for all of them, one each, or a dict —
anything matplotlib understands is a colour:

```python
fw.viz.neuroglancer_url(ids, seg_colors="red")
fw.viz.neuroglancer_url(ids, seg_colors=["red", "#00ffff", (0, 1, 0)])
fw.viz.neuroglancer_url(ids, seg_colors={720575940604407468: "red"})
```

**`color_by`** when you know the *label*. Give it an annotation field and each distinct
value gets a colour from the palette:

```python
fw.viz.neuroglancer_url(ids, color_by="type")            # a colour per cell type
fw.viz.neuroglancer_url(ids, color_by="side", palette="Set2")
fw.viz.neuroglancer_url(ids, color_by=my_cluster_labels)  # or your own array
```

This is the one thing connecto can do that a segmentation-only library cannot: it knows
what the neurons *are*, so `color_by="type"` is one word rather than a join.

!!! warning "Order is meaningful"

    `seg_colors=["red", "blue"]` zips onto the IDs **in the order you wrote them**.
    Note that `ds.ids()` sorts (it ends in `np.unique`), so if you hand it a query
    rather than a list, use a dict or `color_by=` — a positional list against a sorted
    result paints the wrong neuron the wrong colour, and the scene will not look wrong.

## Groups and hidden segments

`seg_groups` splits the selection across separate layers, so each can be toggled on and
off in the viewer. `group_by` does the same from an annotation field:

```python
fw.viz.neuroglancer_url(ids, group_by="class")           # one layer per super class
fw.viz.neuroglancer_url(ids, seg_groups={"mine": [a, b], "theirs": [c]})
```

`invis_segs` loads segments but leaves them switched off — they are listed in the layer,
ready to be ticked on, but do not render:

```python
fw.viz.neuroglancer_url(ids, invis_segs=maybe_partners)
```

## Adding layers

`ds.viz.scene()` hands back the raw dict, and the layer helpers take and return it. So
the canned scene is a starting point, never a ceiling:

```python
from connecto.viz import add_annotation_layer, add_skeleton_layer, encode_url

x = fw.ids("DA1_lPN", side="right")

scene = fw.viz.scene(x, color_by="type")
scene = add_annotation_layer(scene, fw.somas.get(x)[["x", "y", "z"]], name="somas",
                             color="yellow")
scene = add_skeleton_layer(scene, fw.skeletons.get(x[0]), color="magenta")

encode_url(scene)
```

The shape of the array picks the annotation type:

| shape       | annotation |
| ----------- | ---------- |
| `(N, 3)`    | points     |
| `(N, 2, 3)` | lines      |
| `(N, 4)`    | ellipsoids — x/y/z + radius |

## Coordinates are nanometres

Everything connecto returns is in nanometres. Neuroglancer works in voxels. connecto
converts at the boundary, so a coordinate from one call goes straight into another:

```python
soma = fw.somas.get(x).iloc[0]
fw.viz.neuroglancer_url(x, position=soma[["x", "y", "z"]])   # lands on the soma
```

If you already have viewer coordinates, say so — `units="voxel"`. This matters more than
it looks: a FlyWire voxel is 4 × 4 × 40 nm, so getting it wrong puts the camera an
order of magnitude away from the thing you asked to look at, and the scene still opens.

## Two neuroglancers

Neuroglancer forked, and the halves cannot read each other's scenes.

`ngl.flywire.ai` descends from a 2021 seung-lab build. It keeps the camera under
`navigation.pose.position.voxelCoordinates`, hides segments in a `hiddenSegments` list,
and calls a graphene layer `segmentation_with_graph`. Base neuroglancer and
[spelunker](https://spelunker.cave-explorer.org) went the other way: `dimensions` and
`position`, hidden segments marked with a `!` prefix inside `segments`, plain
`segmentation`, and `graphene://middleauth+https://...` sources so the viewer knows to
run CAVE's login flow.

There is no overlap — and, worse, no error. Hand FlyWire's viewer a modern scene and it
opens perfectly happily on an empty brain.

So each dataset declares which viewer it belongs to and which dialect that viewer
speaks, and connecto serialises to match:

```python
cn.get_spec("flywire").viewer          # 'https://ngl.flywire.ai'
cn.get_spec("flywire").viewer_dialect  # 'seunglab'
cn.get_spec("microns").viewer_dialect  # 'modern'
```

You rarely need to think about this. It surfaces in exactly two places: `viewer=` sends
a scene somewhere else, and the impossible combination is refused rather than emitted.

```python
fw.viz.neuroglancer_url(x, dialect="modern")
# ValueError: https://ngl.flywire.ai is an old-schema neuroglancer: it cannot read a
# 'modern' scene, and it will not say so - it opens on an empty viewer.
```

To view FlyWire in a modern viewer, point it at one:

```python
fw.viz.neuroglancer_url(x, viewer="https://spelunker.cave-explorer.org", dialect="modern")
```

## Shortening

Scenes get long — a skeleton of a few thousand nodes is a few thousand line annotations,
which is about a megabyte of URL. Nothing rejects that: the browser or the chat client
simply truncates it, and a truncated scene is a broken scene that looks like a working
one. connecto warns when a link crosses 100 kB.

CAVE datasets can post the state to the CAVE state server and hand back a short link
instead:

```python
fw.viz.neuroglancer_url(x, seg_colors="red", shorten=True)
# https://ngl.flywire.ai/?json_url=https://global.daf-apis.com/nglstate/api/v1/6616114959220736
```

neuPrint datasets have no state server, so `shorten=True` raises rather than quietly
returning the long URL.
