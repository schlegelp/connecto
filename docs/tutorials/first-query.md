# 1. Your first query

## Open a dataset

```python
import connecto as cn

fw = cn.FlyWire()
fw
```

```
<CAVEDataset 'FlyWire (FAFB) public release' backend=cave version=783>
```

Three things are worth noticing in that one line.

**It is a handle, not a container.** There is no `fw.neurons`, no `fw.edges_`, no
`use_types` flag to set. A dataset knows *which connectome, which backend, which
version* and nothing else. Your selection lives in your code, where you can see it.

**It picked a backend.** FlyWire is served by CAVE, so you got a `CAVEDataset`. Some
datasets are served by both — `cn.BANC(backend="neuprint")` is a real thing — and the
`backend=` argument is how you choose.

**It resolved a version.** `783` is a CAVE *materialization*: a frozen snapshot of the
segmentation and its tables. connecto pinned the newest one available. Everything you
ask this handle from now on is answered at 783, which is what makes a session
internally consistent.

## Ask for an edge list

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

```python
edges.shape, edges.dtypes.to_dict()
```

```
((40581, 3), {'pre': dtype('int64'), 'post': dtype('int64'), 'weight': dtype('int32')})
```

You passed a *cell type*, not IDs. connecto resolved `"DA1_lPN"` to fifteen root IDs
and fetched everything up- and downstream of them: 40,581 edges.

By default you get **both** directions. To ask one way:

```python
fw.connectivity.edges("DA1_lPN", upstream=False, min_weight=10).shape
```

```
(906, 3)
```

## The frame is a closed schema

`pre, post, weight` — as `int64, int64, int32`. That is the whole contract, on every
backend, forever.

This matters more than it looks. FlyWire's underlying edge view (`valid_connection_v2`)
carries **twenty** columns: the two IDs, the synapse count, and seventeen columns of
neurotransmitter probabilities and confidence scores. hemibrain's has three. If
connecto passed both through untouched, "the same query returns the same frame" would
be false the moment you switched datasets, and every downstream script would need a
per-dataset special case.

So edges are a *closed* schema: the canonical columns are produced, and everything else
is dropped.

## Nothing is dropped silently

Every frame carries its provenance:

```python
edges.attrs["connecto"]
```

```python
{'dataset': 'flywire',
 'backend': 'cave',
 'source': 'flywire_fafb_public',
 'version': '783',
 'version_timestamp': datetime.datetime(2023, 9, 30, 5, 10, 1, tzinfo=timezone.utc),
 'query': 'edges',
 'fetched': datetime.datetime(2026, 7, 13, 13, 14, 13, tzinfo=timezone.utc),
 'raw_columns': ['pre_pt_root_id', 'post_pt_root_id', 'n_syn', 'gaba', 'ach', 'glut',
                 'oct', 'ser', 'da', 'connection_score', 'cleft_score', ...],
 'dropped_columns': ['gaba', 'ach', 'glut', 'oct', 'ser', 'da', 'connection_score',
                     'cleft_score', 'gaba_std', 'ach_std', 'glut_std', 'oct_std',
                     'ser_std', 'da_std', 'connection_score_std', 'cleft_score_std',
                     'valid_nt']}
```

Two things you can act on:

- **`version_timestamp`** — the edges are as of 30 September 2023, not today. Worth
  knowing before you compare them to something else.
- **`dropped_columns`** — exactly what the closed schema threw away, by name. Not a
  vague "some columns may be omitted" in the docs; the list, in the frame.

If you want them:

```python
fw.connectivity.edges("DA1_lPN", extra=True).shape
```

```
(40581, 20)
```

Now you are holding a FlyWire-shaped frame, and you know it, because you asked for it.

!!! tip "Autapses are off by default"

    Edges where `pre == post` are excluded unless you pass `autapses=True`. They are
    nearly always segmentation errors; neuPrint omits them entirely and caveclient's
    own `synapse_query` drops them. Leaving them in made the *same neuron* have
    different connectivity depending on which backend you asked — which is exactly
    the class of bug this library exists to prevent.

## Now do it somewhere else

```python
hb = cn.Hemibrain()
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

Different backend. Different specimen. Different ID space — hemibrain body IDs are
nine digits and immutable; FlyWire root IDs are eighteen digits and change whenever
someone edits the neuron. Same three columns, same dtypes.

Next: [selecting neurons](selecting-neurons.md), which is where most of the real work
happens.
