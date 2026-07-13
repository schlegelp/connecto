# 6. One script, two datasets

The promise is that the same code runs on any dataset. This tutorial is about what that
actually buys you — and, just as importantly, where it stops.

## The easy part

```python
import connecto as cn

def summarise(ds, query):
    ids   = ds.ids(query)
    edges = ds.connectivity.edges(ids, upstream=False, min_weight=5)
    return {
        "dataset": ds.label,
        "version": str(ds.version),
        "neurons": len(ids),
        "targets": edges["post"].nunique(),
        "synapses": int(edges["weight"].sum()),
    }

for ds in (cn.FlyWire(), cn.Hemibrain()):
    print(summarise(ds, "DA1_lPN"))
```

```
{'dataset': 'FlyWire (FAFB) public release', 'version': '783',             'neurons': 15, ...}
{'dataset': 'hemibrain',                     'version': 'hemibrain:v1.2.1', 'neurons': 7,  ...}
```

Nothing in `summarise` knows which backend it is on. It does not construct a Cypher
query or a `filter_in_dict`; it does not convert voxels to nanometres; it does not
rename `bodyId_pre` or `pre_pt_root_id`. Those are all real differences, and they are all
handled below the line you are writing.

## The interesting part: 15 vs 7

Look again at the neuron counts. FlyWire has 15 DA1 projection neurons; hemibrain has 7.

That is not a bug, and connecto is right not to hide it. **hemibrain is a hemibrain** —
it contains one side of the brain, so it has the right-side DA1 PNs and not the left.
FlyWire is a whole brain and has both.

```python
cn.FlyWire().annotations.get("DA1_lPN")["side"].value_counts()
```

```
right    7
left     8
```

```python
cn.Hemibrain().annotations.get("DA1_lPN")["side"].value_counts()
```

```
right    7
```

The `side` column is doing real work here. hemibrain has no `somaSide` field at all —
the side is encoded as a **suffix on the instance name**, `DA1_lPN_R`. connecto derives
it with a rule the dataset declares, and normalises `R` to `right`, so the comparison
above can be written at all.

This is the difference between a normalisation layer and a wrapper. A wrapper would have
handed you `instance` and let you write the regex yourself, differently, in every script.

## Where it stops

Now try to be ambitious:

```python
for ds in (cn.FlyWire(), cn.MICrONS()):
    print(summarise(ds, "DA1_lPN"))
```

```
ValueError: MICrONS (minnie65) public has no `type` column configured. Pass
`fields={'type': (...)}` to the dataset, or query a raw column with 'column:value'.
```

Good. There are no DA1 projection neurons in a mouse. The code is portable; the
*question* is not, and connecto declines to pretend otherwise.

This is the boundary worth internalising. connecto normalises **structure** — column
names, dtypes, units, ID handling, versioning, errors. It does not normalise **biology**.
A cell type is not a portable concept across species, and a library that let
`ds.ids("DA1_lPN")` return something plausible on a mouse would be actively lying to you.

So portable code is code that does not assume the biology:

```python
def hub_neurons(ds, seed, n=5):
    """The n strongest downstream partners of `seed`. Works anywhere."""
    edges = ds.connectivity.edges(seed, upstream=False)
    return (edges.groupby("post")["weight"].sum()
                 .sort_values(ascending=False)
                 .head(n))

hub_neurons(cn.FlyWire(), "DA1_lPN")
hub_neurons(cn.MICrONS(), mic_ids)      # fine - IDs, not a fly type
hub_neurons(cn.Fish2(), fish_ids)       # also fine
```

## The test that keeps it honest

There is one place where "the same frame" is not a claim but a *test*.

BANC is served by **both** backends, at the same snapshot: CAVE materialization 888 is
neuPrint `banc:v888`, and its neuPrint body IDs are literally valid CAVE root IDs. So the
two paths can be compared directly:

```python
import pandas as pd

ids  = cn.BANC().ids("DNa02")
cave = cn.BANC(backend="cave").connectivity.edges(ids)
np_  = cn.BANC(backend="neuprint").connectivity.edges(ids)

pd.testing.assert_frame_equal(cave, np_)   # passes
```

This runs in CI. It is the difference between a library that *says* it normalises and one
that is checked.

It has already earned its keep twice:

- **15 edges vs 5.** neuprint-python's `NeuronCriteria` defaults to matching the `:Neuron`
  label, which quietly excludes `:Segment` fragments. CAVE draws no such distinction. The
  two backends disagreed about which neurons existed. Fixed by defaulting to `Segment`.
- **1503 edges vs 1501.** The two extras were **autapses** — `pre == post`. neuPrint omits
  them; CAVE's synapse table has them. Fixed by excluding autapses by default (they are
  nearly always segmentation errors) and documenting the switch.

Both would have shipped as quiet wrongness. Neither was findable by reading the code —
only by having two independent answers to the same question and insisting they match.

## The general shape

Write functions that take a `Dataset` and IDs:

```python
def cable_vs_outputs(ds, ids):
    skels  = ds.skeletons.get(ids)
    counts = ds.connectivity.synapse_counts(ids).set_index("id")
    return pd.DataFrame({
        "cable_um": {n.id: n.cable_length / 1000 for n in skels},
        "outputs":  counts["pre"],
    })
```

Positions are nanometres everywhere, so `/1000` means micrometres everywhere. `id`,
`pre`, `post` are canonical everywhere, so the join is written once.

Then check what a dataset can do before you assume it:

```python
if ds.supports(cn.Cap.NT_PER_SYNAPSE):
    nt = ds.connectivity.transmitters(ids)
```

Which is the subject of the next page: [Capabilities](../guides/capabilities.md).
