# 3. Connectivity

Everything lives under `ds.connectivity`.

```python
import connecto as cn

fw = cn.FlyWire()
hb = cn.Hemibrain()
```

## Edges

```python
fw.connectivity.edges("DA1_lPN")            # both directions   -> (40581, 3)
fw.connectivity.edges("DA1_lPN", upstream=False)     # outputs only
fw.connectivity.edges("DA1_lPN", downstream=False)   # inputs only
fw.connectivity.edges("DA1_lPN", min_weight=10)      # >= 10 synapses
```

`weight` is a synapse count. Always `int32`, always the same column name.

Edges between two of your query neurons appear once, not twice — connecto
de-duplicates across the up- and downstream fetches for you.

## Adjacency

When you want a matrix rather than a list:

```python
fw.connectivity.adjacency("DA1_lPN")
```

```
post                720575940603231916  720575940604407468  720575940605102694  ...
pre
720575940603231916                   0                   0                   2
720575940604407468                   0                   0                   0
720575940605102694                  10                   0                   0
720575940613345442                   4                   0                   6
...
```

Rows are sources, columns are targets, `int32`, zeros filled in. Different sets on the
two axes if you want them:

```python
fw.connectivity.adjacency(sources="DA1_lPN", targets="cell_class:ALLN")
```

The row and column order is exactly the order of the IDs you passed, so it lines up
with anything else you built from the same selection.

## Synapse counts

```python
fw.connectivity.synapse_counts("DA1_lPN")
```

```
                   id   pre  post
0  720575940603231916  4571   887
1  720575940604407468  4719   852
2  720575940605102694  5252  1155
3  720575940613345442  4381   852
4  720575940614309535  4308   876
```

`pre` = outgoing synapses, `post` = incoming. Add `by_roi=True` to split by neuropil.

## Individual synapses

```python
syn = fw.connectivity.synapses("DA1_lPN")
syn.shape
```

```
(79127, 21)
```

```python
syn[["id", "pre", "post", "pre_x", "pre_y", "pre_z", "score", "roi"]].head(3)
```

```
          id                 pre                post     pre_x     pre_y     pre_z  score      roi
0  191366973  720575940621185050  720575940622964735  350712.0  152184.0  147120.0  176.0     LH_L
1    1200702  720575940621185050  720575940626999114  431108.0  146448.0  198600.0   71.0  MB_CA_L
2    1201179  720575940630066007  720575940588344515  615904.0  146364.0  199560.0  133.0  MB_CA_R
```

**Positions are in nanometres.** Not voxels, not "whatever the backend used". CAVE
hands connecto nanometres; neuPrint hands it voxels and connecto multiplies by the
dataset's voxel size. You never have to remember which.

```python
fw.connectivity.synapses("DA1_lPN", units="um")   # or ask for micrometres
```

### Filtering synapses

```python
fw.connectivity.synapses("DA1_lPN", min_score=50)        # cleft score
fw.connectivity.synapses("DA1_lPN", rois=["LH_L"])       # by neuropil
fw.connectivity.synapses("DA1_lPN", pre=True, post=False) # outgoing only
```

`min_score` is the interesting one:

```python
cn.MICrONS().connectivity.synapses(ids, min_score=100)
```

```
CapabilityError: MICrONS (minnie65) public does not support `min_score`
(no synapse_scores). Drop the argument, or use a dataset that has it.
```

MICrONS has no per-synapse cleft score. It would be *easy* for connecto to accept the
argument and ignore it — and you would then believe you had filtered when you had not.
That is the single most dangerous thing a normalisation layer can do, so it does not do
it. See [Capabilities](../guides/capabilities.md).

## Per-ROI connectivity

On datasets with neuropil annotations, split the edges by region:

```python
hb.connectivity.edges("DA1_lPN", by_roi=True).head()
```

```
         pre       post  weight    roi
0  722817260  296199149       1  LH(R)
1  722817260  297520036       1  LH(R)
2  722817260  297869179       2  LH(R)
3  722817260  329206392       1  LH(R)
4  722817260  329906396       7  LH(R)
```

Same three columns, plus `roi`. What ROIs are there?

```python
hb.rois.list()
```

```
         roi  primary
0      AB(L)     True
1      AB(R)     True
2      AL(L)     True
3      AL(R)     True
4    AL-D(L)    False
...
[231 rows x 2 columns]
```

`primary=True` marks the non-overlapping set — the one to use if you want the parts to
sum to the whole. neuPrint's ROIs nest (`AL-DA1(R)` is inside `AL(R)`), and summing
across all 231 would double-count. There is a containment tree if you need it:

```python
hb.rois.hierarchy()     # networkx.DiGraph
```

## Transmitters

FlyWire has per-synapse neurotransmitter predictions. Rolling them up per neuron:

```python
fw.connectivity.transmitters("DA1_lPN")
```

```
                   id             nt  confidence
0  720575940603231916  acetylcholine    0.972872
1  720575940604407468  acetylcholine    0.960161
2  720575940605102694  acetylcholine    0.942879
3  720575940613345442  acetylcholine    0.958457
4  720575940614309535  acetylcholine    0.964485
```

The vote is taken over that neuron's *presynapses*, and `confidence` is the winning
fraction — so you can see how convincing the call was, rather than just being handed a
label. Cholinergic, and confidently so, which is what an olfactory PN should be.

hemibrain has no per-synapse predictions, so:

```python
hb.connectivity.transmitters("DA1_lPN")
```

```
CapabilityError: hemibrain (neuprint) does not support nt_per_synapse.
Available: annotations, connectivity, meshes, roi_connectivity, rois, skeletons,
somas, synapse_scores, synapses.
```

## Caching

Edge fetches can be slow. `cache=True` writes the raw response to disk and reuses it:

```python
fw.connectivity.edges(big_list, cache=True)   # slow once, instant after
```

The cache key includes the dataset, the version and the query — so a cached result can
never be served for a different materialization. See [Caching](../guides/caching.md).

Next: [morphology](morphology.md).
