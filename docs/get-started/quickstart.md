# Quickstart

Ten minutes, end to end. Every output on this page is copied from a real run.

## 1. Open a dataset

```python
import connecto as cn

fw = cn.FlyWire()      # neuPrint - FlyWire (FAFB) public release
hb = cn.Hemibrain()    # neuPrint - hemibrain
```

FlyWire is served by *both* backends, and neuPrint is the default — it is the same
release, and it answers faster. Pass `backend="cave"` for the chunkedgraph, proofreading
and per-synapse transmitters, which the neuPrint mirror does not have. connecto will tell
you when you need it.

A dataset is an immutable **handle**. It holds no neurons, no selection, no cached
frame of annotations — just "which connectome, which backend, which version".

```python
fw
```

```
<CAVEDataset 'FlyWire (FAFB) public release' backend=cave version=783>
```

It pinned itself to a version on first access. FlyWire's public release ships
materializations `630` and `783`; connecto took the newest.

## 2. Select some neurons

Everything takes the same mini-language, and it always returns IDs:

```python
fw.ids("DA1_lPN")
```

```
array([720575940603231916, 720575940604407468, 720575940605102694,
       720575940613345442, 720575940614309535, 720575940617229632,
       720575940619385765, 720575940621185050, 720575940621239679,
       720575940623303108, 720575940623543881, 720575940626034819,
       720575940630066007, 720575940637208718, 720575940637469254])
```

Fifteen DA1 projection neurons. `ids()` is a pure function — it returns an array
and mutates nothing.

```python
fw.ids("DA1_lPN", side="left")        # 8 of them
fw.ids("/^AOTU00.*")                  # regex -> 35 neurons
fw.ids("superclass:visual_projection")    # any annotation column
```

## 3. Ask what they connect to

```python
edges = fw.connectivity.edges("DA1_lPN")
edges.head()
```

```
                  pre                post  weight
0  720575940605102694  720575940646122804      64
1  720575940603231916  720575940629163931      50
2  720575940604407468  720575940631143213      49
3  720575940623303108  720575940612355507      48
4  720575940603231916  720575940635945919      46
```

Three columns — `pre`, `post`, `weight` — as `int64, int64, int32`. Always. This is
a **closed schema**: FlyWire's underlying edge view carries seventeen extra columns
of neurotransmitter probabilities, and connecto drops them so that the frame you get
from FlyWire is the frame you get from hemibrain.

It tells you what it dropped, rather than pretending they never existed:

```python
edges.attrs["connecto"]["dropped_columns"]
```

```
['gaba', 'ach', 'glut', 'oct', 'ser', 'da', 'connection_score', 'cleft_score',
 'gaba_std', 'ach_std', 'glut_std', 'oct_std', 'ser_std', 'da_std',
 'connection_score_std', 'cleft_score_std', 'valid_nt']
```

Pass `extra=True` to keep them.

## 4. Do it again on a different brain

Same line, different animal:

```python
hb.connectivity.edges("DA1_lPN").head()
```

```
          pre        post  weight
0  1765040289  1671620613      84
1   754538881  1671620613      76
2   754534424  1704347707      74
3  1734350908  1671620613      73
4   754534424  1671620613      70
```

Different backend (neuPrint, not CAVE), different ID space (9-digit body IDs, not
18-digit root IDs), different specimen. Identical frame.

## 5. Get the morphology

```python
skels = fw.skeletons.get(fw.ids("DA1_lPN")[:2])
skels
```

```
<class 'navis.core.neuronlist.NeuronList'> containing 2 neurons (175.8KiB)
               type                name                  id  n_nodes  n_branches  cable_length        units
0  navis.TreeNeuron  720575940603231916  720575940603231916     3588         586    2050971.75  1 nanometer
1  navis.TreeNeuron  720575940604407468  720575940604407468     3909         670    2367593.25  1 nanometer
```

These are ordinary [navis](https://navis-org.github.io/navis/) `TreeNeuron`s, in
nanometres, so the whole navis ecosystem works with no conversion step:

```python
import navis
navis.plot3d(skels)
```

Meshes work the same way and give you `MeshNeuron`s.

## 6. Notice what it refuses to do

```python
mic = cn.MICrONS()
mic.connectivity.synapses(ids, min_score=100)
```

```
CapabilityError: MICrONS (minnie65) public does not support `min_score`
(no synapse_scores). Drop the argument, or use a dataset that has it.
```

MICrONS has no per-synapse cleft scores. A library that quietly ignored `min_score`
here would hand you an unfiltered result while you believed you had filtered it —
so connecto raises instead. Defaults never raise; only an *explicit* request for
something the dataset lacks does.

Whole namespaces are absent rather than broken, so you can check first:

```python
hasattr(cn.FlyWire(backend="cave"), "proofreading")   # True  - FlyWire is still edited
hasattr(hb, "proofreading")                           # False - hemibrain is frozen
```

Where one word covers two promises, connecto splits it. Both datasets have a
segmentation volume; only FlyWire has a chunkedgraph under it:

```python
hb.segmentation.locs_to_segments(tbars)   # fine - what body is at this point?
hb.segmentation.update_ids([1734350788])  # CapabilityError: no chunkedgraph
```

And a capability belongs to a *(dataset, backend)* pair, not to a dataset — so the same
FlyWire has a chunkedgraph through one door and not the other, and the error says which:

```python
cn.FlyWire().segmentation.update_ids(old_ids)
```

```
CapabilityError: FlyWire (FAFB) public release (neuprint) does not support
chunkedgraph. Available: annotations, connectivity, meshes, neuroglancer,
roi_connectivity, rois, segmentation, skeletons, somas, synapse_scores, synapses.
The cave backend does: cn.get_dataset("flywire", backend="cave").
```

## Where to go next

- [Tutorials](../tutorials/index.md) — the same ground, slower and deeper.
- [Capabilities](../guides/capabilities.md) — why the errors above are the point.
- [Versions and root IDs](../tutorials/versions.md) — read this **before** you cache
  a root ID anywhere.
