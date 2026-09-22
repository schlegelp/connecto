# 4. Morphology

Skeletons and meshes come back as [navis](https://navis-org.github.io/navis/) neurons.
Not dicts, not arrays, not a custom class you would have to learn — navis neurons, in
nanometres, so the entire navis ecosystem works without a conversion step.

```python
import connecto as cn
import navis

fw = cn.FlyWire()
```

## Skeletons

```python
skels = fw.skeletons.get(fw.ids("DA1_lPN")[:2])
skels
```

```
<class 'navis.core.neuronlist.NeuronList'> containing 2 neurons (175.8KiB)
               type                name                  id  n_nodes  n_branches  n_leafs  cable_length        units
0  navis.TreeNeuron  720575940603231916  720575940603231916     3588         586      645    2050971.75  1 nanometer
1  navis.TreeNeuron  720575940604407468  720575940604407468     3909         670      760    2367593.25  1 nanometer
```

A `navis.NeuronList` of `TreeNeuron`s. That means:

```python
navis.plot3d(skels)
navis.prune_by_strahler(skels, to_prune=1)
navis.resample_skeleton(skels, resample_to=1000)
skels[0].cable_length          # 2050971.75 nm = 2.05 mm
```

Cable length comes out in nanometres because `units` is set on the neuron — navis knows,
and will convert for you:

```python
skels[0].units.to("um")
```

For NBLAST, skip a step — **and ask for microns**:

```python
dp = fw.skeletons.dotprops("DA1_lPN", k=5, units="um")
navis.nblast(dp, dp)
```

!!! warning "NBLAST is calibrated in microns"

    `units` defaults to `"nm"`, because everything connecto returns is in nanometres.
    NBLAST is not: score it on nanometre dotprops and every pair collapses to the
    floor. Two right-side `DA1_lPN`s come back at **−0.88** instead of **+0.70** — which
    reads as *"nothing here is similar"* rather than as an error. navis notices and logs
    a warning, but a log line is easy to miss in a notebook.

    connecto cannot fix this from the outside — it does not sit between you and
    `navis.nblast`. So the rule is simply: **NBLAST gets microns.**

## The L2 cache — cheap skeletons and dotprops

Most CAVE datastacks (not FlyWire's public one) have a **level-2 chunk cache**: a
server-side summary of every chunk of the segmentation — a representative coordinate,
volume, surface area, and the chunk's principal axes. It is exposed as its own namespace,
present only where the cache actually exists:

```python
fw = cn.FlyWire("production")

hasattr(fw, "l2")                 # True
hasattr(cn.FlyWire(), "l2")       # False — the public stack has no L2 cache
hasattr(cn.Hemibrain(), "l2")     # False — neuPrint has no chunkedgraph at all
```

```python
fw.l2.info(ids)        # per-chunk table: position, volume, surface area
fw.l2.graph(ids)       # the level-2 chunk graph, as networkx
fw.l2.skeleton(ids)    # a skeleton, without running a skeletonisation
fw.l2.dotprops(ids)    # dotprops, without even building a skeleton
```

**`l2.dotprops` is the one to know about.** The cache already stores `rep_coord_nm` (a
point on the neurite) and `pca_0` (that chunk's principal axis) — a point and a
direction, which *is* a dotprop. So it needs one chunkedgraph call and one l2cache call,
and no skeleton at all. On six FlyWire `lPN`s it was **2.8× cheaper** than going via the
skeleton, and NBLAST told the two cell types apart just as well:

| | same-type | different-type | separation |
|---|---|---|---|
| `skeletons.dotprops` | +0.710 | +0.134 | **+0.576** |
| `l2.dotprops` | +0.706 | +0.076 | **+0.630** |

(Three `DA1_lPN` vs three `DA2_lPN`, right side, all-by-all NBLAST in microns. The gap
widens a lot when the skeleton service has not cached the neuron yet — then it is minutes
against seconds.)

Chunks too blobby to have a principal axis — about a third of them — come back with an
all-zero vector, and connecto **drops** them. A zero-length vector is not a direction, and
NBLAST would happily treat it as one.

!!! warning "Do not measure cable length on an L2 skeleton"

    Its nodes are chunk *representative coordinates*, so the path hops from one chunk's
    point to the next rather than following the neurite. It zig-zags, and over-estimates
    — consistently by about **2×** on FlyWire lPNs (1.9–2.2× against the same neurons
    from the skeleton service). Radii are volume-to-surface estimates, not measurements.

    L2 skeletons are for topology and shape at low cost. If the number matters, use
    `ds.skeletons.get()`.

## Meshes

```python
meshes = fw.meshes.get(fw.ids("DA1_lPN")[:1])
meshes
```

```
<class 'navis.core.neuronlist.NeuronList'> containing 1 neurons (33.0MiB)
               type                name                  id        units  n_vertices  n_faces
0  navis.MeshNeuron  720575940603231916  720575940603231916  1 nanometer      480548  961496
```

Note the size: **33 MB for one neuron**. Meshes are large. Ask for a lower level of
detail when you do not need every vertex:

```python
fw.meshes.get(ids, lod=2)
```

How many levels a neuron has depends on its size, so a level you name can be out of
range for small neurons. `lod_fallback=True` gives those the coarsest level they do
have instead of raising; `lod=-1` always means "the coarsest there is".

## Voxels

A skeleton is a neuron's centreline and a mesh is its surface. `ds.voxels` gives you the
thing itself — every voxel the segmentation assigns to that neuron.

```python
hb.voxels.get(1734350788, scale=4)
```

```
<class 'navis.core.neuronlist.NeuronList'> containing 1 neurons (3.6MiB)
                type        name          id            units               shape  dtype
0  navis.VoxelNeuron  1734350788  1734350788  128.0 nanometer  (1379, 2329, 1789)  int32
```

You get `navis.VoxelNeuron`s, so they plot and measure like any other navis neuron. The
coordinates stay in the (compact, integral) voxel grid and the neuron carries its
nm-per-voxel in `units`, which is why a 310,000-voxel neuron is 3.6 MB rather than
inflated into floats.

Two other output forms:

```python
hb.voxels.get(x, output="raw")   # {id: (N, 3) int32 array}
hb.voxels.get(x, output="rle")   # {id: (M, 4) runs — x, y, z, length}
```

`rle` is what the wire already carries, and it is much smaller — for this neuron, 53,779
runs (0.9 MB) instead of 310,252 voxels (3.7 MB). If you are storing volumes or moving
them around, store the runs.

!!! warning "`scale=` does not default to 0"

    Scale 0 is full resolution, and full resolution is enormous: this same hemibrain
    neuron is **1.17 billion voxels** at scale 0 — 14 GB as an `(N, 3)` int32 array.
    The default is a scale that returns a recognisable neuron in a couple of seconds.
    Ask for `scale=0` deliberately, or not at all.

    `ds.voxels.scales()` lists what a dataset serves — `(0, 1, ..., 7)` for hemibrain,
    but exactly `(1,)` for aedes, whose service publishes only that one. Note that a
    scale being *available* says nothing about it being *affordable*: every CAVE
    dataset lists scale 0, and none of them can deliver a whole neuron at it.

!!! note "aedes has a size ceiling"

    The aedes service refuses any neuron spanning more than 256 chunks, and it serves
    only one scale — so there is no coarser level to retreat to. The largest neurons
    in that dataset simply have no sparse volume. connecto says so rather than
    relaying the server's (impossible) suggestion to try a coarser scale. Their
    skeletons and meshes are unaffected.

### Where they come from, and why the cost differs so much

This is the part worth knowing, because the same call costs a second on one dataset and
several minutes on another.

| dataset | route | cost |
|---|---|---|
| hemibrain, maleCNS, MANC, fish2 | DVID `sparsevol` | one indexed request |
| aedes | a lookup service | one indexed request |
| FlyWire, BANC, FANC, MICrONS | dense read + mask | hundreds of block reads |

DVID keeps a live per-body index — body to blocks to runs — so extracting one neuron is
a single lookup. A chunkedgraph keeps **no such index**: it stores the graph, and the
voxels underneath are a static volume with nothing mapping a root ID to the blocks it
occupies. So the same question degrades to *read dense blocks, mask to one root,
sparsify*, and touches thousands of voxels for every one it keeps.

connecto does not hide that. Ask before you commit:

```python
fw.voxels.estimate(720575940604407468, scale=4)
```

```
           root_id  scale  l2_nodes  chunks  requests  voxels_transferred           resolution    block_shape
720575940604407468      4       705     247       288           603979776 (256.0, 256.0, 40.0) [256, 256, 32]
```

604 million voxels transferred to keep about 259,000. And a request that would be
genuinely unreasonable is refused rather than left running:

```python
fw.voxels.get(720575940604407468, scale=0)
# ValueError: Reading root 720575940604407468 at scale 0 would transfer
# 8,287,944,704 voxels (3952 block reads for 247 chunks), over the
# 2,000,000,000 ceiling.
# A chunkedgraph has no per-body index, so this is a dense read - see
# `voxels.estimate()`. Use a coarser `scale=`, pass `max_chunks=` to sample,
# or raise `max_voxels=`.
```

!!! note "Coarser is not automatically cheaper"

    A counter-intuitive one, and it is why the CAVE path plans its requests the way it
    does. Storage blocks are the *same shape at every scale* (256×256×32 for FlyWire),
    and fly pyramids downsample XY only. So asking for a coarser scale does not, by
    itself, move fewer bytes — it just discards more of each block. What makes coarse
    reads cheap is fetching each distinct block once instead of once per graph chunk,
    which is what connecto does.

### Nothing is hard-coded

For the DVID datasets, the server and node are **discovered at runtime** — hemibrain's
from neuPrint's own metadata, maleCNS's and MANC's from clio, fish2's from its
neuroglancer layers. None of those URLs appear in connecto's source, and the node is
always taken from neuPrint's `Meta.uuid`, so the voxels you get are from the same
snapshot as the annotations and connectivity you got alongside them.

maleCNS and MANC therefore need clio (`pip install connecto[clio]`) and a clio token for
this one namespace, even though their annotations and connectivity do not.

## Where skeletons actually come from

This is worth knowing, because it is the part most likely to surprise you.

neuPrint datasets simply *have* skeletons — they are stored, and you download them.

CAVE datasets mostly do not. connecto tries three sources, in order:

1. **Precomputed skeletons**, if the dataset declares a source. FlyWire has these
   (a published skeletonisation of every proofread neuron), and they are the best
   answer — fast and consistent with what everyone else in the field is using.
2. **CAVE's skeleton service**, which skeletonises on demand.
3. **The L2 cache**, a skeleton derived from the chunkedgraph's level-2 chunks. You can
   also ask for this one directly — see [above](#the-l2-cache-cheap-skeletons-and-dotprops).

The fallbacks matter. FlyWire's *public* stack has no L2 cache at all, and the CAVE
skeleton service requires one — so asking it to skeletonise a public FlyWire neuron
returns an HTTP 500. connecto knows this (the dataset does not claim the `l2cache`
capability) and goes to the precomputed skeletons instead.

Which source you got is not always obvious, and it matters: whether the L2 cache is
*reachable* is read from the dataset's `l2cache` capability, not by asking the l2cache
service. That is deliberate. Asking the service would couple skeletons to l2cache uptime,
and the failure is nasty — a transient 503 from the l2cache reads as "this datastack has
no L2 cache", which quietly downgrades you to the L2 skeleton path: the one route
guaranteed to be broken, because the l2cache is exactly what just went down. FANC's
l2cache does this regularly.

There is also a trap it explicitly checks for: the skeleton service returns a
**one-vertex skeleton** for neurons it has not generated yet. That is not an error, it
is a `200 OK` with a valid-looking payload — and it would sail straight through into
your analysis as a neuron with 0 µm of cable. connecto detects the degenerate case and
falls back, rather than handing you a neuron-shaped object that is not a neuron.

```python
fw.skeletons.get(ids, output="raw")   # the node table, if you'd rather have that
```

## Somas

```python
fw.somas.get("DA1_lPN")
```

Soma positions also come back in nanometres, and appear as `soma_x/y/z` in the
annotation frame:

```python
fw.annotations.get("DA1_lPN")[["id", "type", "side", "soma_x", "soma_y", "soma_z"]].head(3)
```

```
                   id     type   side    soma_x   soma_y  soma_z
0  720575940604407468  DA1_lPN  right  158400.0  56848.0   539.0
1  720575940623543881  DA1_lPN  right  159208.0  57008.0   695.0
2  720575940637469254  DA1_lPN  right  160304.0  57248.0   679.0
```

!!! warning "A soma column of NaNs is not a soma column"

    hemibrain's `soma_x/y/z` are `NaN` for most neurons — the volume is cropped and
    many somata are simply outside it. connecto does not fabricate positions to fill
    the schema. If you need somata, check before you rely on them.

## Combining with connectivity

The IDs are the same objects throughout, so morphology and connectivity join trivially:

```python
ids   = fw.ids("DA1_lPN", side="left")
skels = fw.skeletons.get(ids)
edges = fw.connectivity.edges(ids, upstream=False)

# cable length vs. output synapses, per neuron
counts = fw.connectivity.synapse_counts(ids).set_index("id")
for n in skels:
    print(n.id, round(n.cable_length / 1000), "um", counts.loc[n.id, "pre"], "outputs")
```

Next: [versions and root IDs](versions.md) — the one that will save you.
