# 2. Selecting neurons

There is one mini-language for saying "these neurons", and it works everywhere.
Anything you can pass to `ids()` you can pass to *any* query — `edges()`,
`synapses()`, `skeletons.get()`, all of it.

```python
import connecto as cn
fw = cn.FlyWire()
```

## The five forms

=== "An ID"

    ```python
    fw.ids(720575940604407468)
    fw.ids([720575940604407468, 720575940623543881])
    ```

    Root IDs, body IDs, ints, strings of digits, lists, arrays, Series — all fine.

=== "A type"

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

    Matched across whichever columns the dataset declares as type columns — see
    [below](#what-counts-as-a-type).

=== "A regex"

    ```python
    fw.ids("/^AOTU00.*")   # 35 neurons
    ```

    A leading `/` means regex. Nothing else does.

=== "A column filter"

    ```python
    fw.ids("super_class:visual_projection")
    ```

    `column:value` filters any column in the annotation table — not just the ones
    connecto knows about.

=== "Criteria"

    ```python
    from connecto import NeuronCriteria

    fw.ids(NeuronCriteria(type="DA1_lPN", side="left"))
    ```

    The explicit form. Everything above desugars into this.

## Combining

`side=` is separate, because you want it constantly:

```python
fw.ids("DA1_lPN", side="left")
```

```
array([720575940603231916, 720575940605102694, 720575940613345442,
       720575940614309535, 720575940619385765, 720575940621185050,
       720575940626034819, 720575940637208718])
```

Eight of the fifteen. `side` is always `left` / `right` / `center` regardless of what
the dataset natively calls it — FlyWire says `left`/`right`, hemibrain encodes it as
a suffix on `instance` (`DA1_lPN_R`), MANC says something else again. You get the
same three words.

A list is a union:

```python
fw.ids(["DA1_lPN", "DA1_vPN"])          # both types
fw.ids(["/^AOTU00.*", 720575940604407468])   # regex OR a specific neuron
```

## It is a pure function

`ids()` returns an array and mutates nothing. There is no `add_neurons`, no
accumulating selection on the dataset object, no hidden state that a later call will
read.

This is a deliberate reaction to how the previous attempt (`cocoa`) worked: a mutable
selection plus a memoised annotation frame is precisely how you end up joining
annotations from one materialization onto edges from another, and never finding out.

If you want the IDs, you hold them:

```python
ids = fw.ids("DA1_lPN", side="left")
edges = fw.connectivity.edges(ids)
skels = fw.skeletons.get(ids)
```

## What counts as a "type"

Different datasets have wildly different ideas about this, so each declares its own
priority order. FlyWire's is `cell_type`, then `hemibrain_type`; hemibrain's is just
`type`. `ds.annotations.fields` tells you:

```python
fw.annotations.fields
```

```python
{'type': ('cell_type', 'hemibrain_type'),
 'side': ('side',),
 'class': ('super_class', 'cell_class'),
 'nt': ('known_nt', 'top_nt'),
 ...}
```

`ds.ids("DA1_lPN")` coalesces those in order: use `cell_type` if it is set, else fall
back to `hemibrain_type`.

If a dataset has *no* type column configured, connecto does not guess:

```python
cn.MICrONS().ids("BC")
```

```
ValueError: MICrONS (minnie65) public has no `type` column configured. Pass
`fields={'type': (...)}` to the dataset, or query a raw column with 'column:value'.
```

MICrONS ships with `fields={}` on purpose. Its cell types are spread across several
tables with genuinely different semantics (`cell_type_local`, `aibs_metamodel_mtypes`,
`nucleus_svm`), and picking one to be *the* type would be a scientific judgement
disguised as a default. So it refuses, and tells you how to say what you meant:

```python
mic = cn.MICrONS()
mic.ids("cell_type:BC")                                    # be explicit, or
mic = cn.MICrONS(fields={"type": ("cell_type",)})          # declare it once
mic.ids("BC")
```

## Finding out what is there

```python
fw.annotations.search("DA1")     # 143 rows: DA1_lPN, DA1_vPN, ...
```

Searches `type`, `class` and `instance`. To see everything:

```python
ann = fw.annotations.get("DA1_lPN")
ann.columns.tolist()
```

```
['id', 'type', 'side', 'class', 'nt', 'nt_source', 'status', 'soma_x', 'soma_y',
 'soma_z', 'pre', 'post', 'downstream', 'upstream', 'statusLabel', 'areaNm',
 'connectivityTag', 'crossVersionConsistentName', 'dimorphism', 'flow', 'fruDsx',
 'hartensteinHemilineage', 'hemibrainType', 'itoLeeHemilineage', 'lengthNm',
 'location', 'nerve', 'ntAcetylcholineProb', 'ntDopamineProb', 'ntGabaProb',
 'ntGlutamateProb', 'ntOctopamineProb', 'ntSerotoninProb', 'opticColumnId',
 'opticColumnP', 'opticColumnQ', 'opticColumnX', 'opticColumnY', 'outlier',
 'predictedNt', 'predictedNtProb', 'sizeNm', 'subclass', 'superclass',
 'supertype', 'supervoxelId', 'synonyms', 'synweight', 'vfbId']
```

The first ten are **canonical** — the same names, same dtypes, same units on every
dataset. The remaining thirty-nine are FlyWire's own, passed through untouched. Pick a
different source and the tail changes completely while the first ten do not:
`fw.annotations.get("DA1_lPN", source="public")` hands back the published TSV's
snake_case columns instead.

### `nt` is coalesced too — and says so

`nt` follows the same first-non-null rule as `type`, but it is the one canonical column
that comes with a companion saying where each value came from:

```python
fw.annotations.get("DA1_lPN")[["id", "nt", "nt_source"]]
```

```
                   id             nt    nt_source
0  720575940604407468  acetylcholine  predictedNt
1  720575940623543881  acetylcholine  predictedNt
```

The reason is that a dataset's transmitter columns are not interchangeable opinions the
way its type columns are — they are different *kinds* of claim. A prediction is a CNN's
argmax over a T-bar image; a *measurement* is somebody's immunostaining or RT-PCR. The
default neuPrint source has only the first, which is what `nt_source` is telling you
above. FlyWire's published TSV has both:

```python
ann = fw.annotations.get(source="public")
ann["nt_source"].value_counts()
```

```
known_nt    87837       # measured
top_nt      51248       # predicted
```

So filtering on evidence is a column comparison:

```python
measured = ann[ann["nt_source"] == "known_nt"]
```

Without `nt_source`, "this neuron is GABAergic" would mean either "we measured it" or "a
model thinks so", with no way to tell which — and the two belong on different sides of
an argument.

It is there even when a dataset has only one transmitter column, because *which* one it
is still matters: aedes has `neurotransmitter_verified` and no predictions at all, and
`nt_source` is how you find that out without reading the spec.

Annotations are an *open* schema, and that is the opposite choice from edges. Edges
are closed because you compare them across datasets. Annotations are open because
`ito_lee_hemilineage` has no hemibrain equivalent and no mouse equivalent, and
throwing it away would be worse than the asymmetry.

Any of those columns works as a filter:

```python
fw.ids("ito_lee_hemilineage:ALl1_ventral")
fw.ids("nerve:AN", side="left")
```

Misspell one and it guesses:

```python
fw.ids("super_clas:visual_projection")
```

```
ValueError: FlyWire (FAFB) public release annotations have no column 'super_clas'.
Did you mean 'super_class'?
```

## A worked example

Left-side olfactory projection neurons, and what they talk to:

```python
alpn = fw.ids("cell_class:ALPN", side="left")
len(alpn)
```

```
164
```

```python
edges = fw.connectivity.edges(alpn, upstream=False, min_weight=5)

# Who do they target, by type?
ann = fw.annotations.get()[["id", "type"]]
top = (edges.merge(ann, left_on="post", right_on="id")
            .groupby("type", observed=True)["weight"].sum()
            .sort_values(ascending=False)
            .head())
```

Note the join key: `post` on the left, `id` on the right. `id` is canonical — it is
called `id` in every annotation frame from every dataset, which is what lets that line
be written once.

Next: [connectivity](connectivity.md) in earnest.
