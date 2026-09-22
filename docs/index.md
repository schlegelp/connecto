---
template: home.html
title: connecto
hide:
  - navigation
  - toc
  - path
---

<!-- The logo *is* the page heading. The theme only injects its own title heading
     when the content has none, so this replaces it rather than sitting under a
     duplicate. `__skip` is the skip-link target; `alt` carries the name. -->
<h1 id="__skip" class="cn-logo">
  <img src="_static/logo.png" alt="connecto" width="320" height="320">
</h1>

<div class="cn-section" markdown>

## The same query, the same frame

Connectomic datasets each speak their own dialect. CAVE has materialization
versions, root IDs that change under you, and per-datastack table names. neuPrint
has immutable body IDs, Cypher, and dataset strings like `hemibrain:v1.2.1`. And
every dataset has its own idea of what "type" or "side" means.

`connecto` is one interface over all of it.

```python
import connecto as cn

fw = cn.FlyWire()      # neuPrint (backend="cave" for the other door)
hb = cn.Hemibrain()    # neuPrint

fw.connectivity.edges("DA1_lPN")     # pre, post, weight
hb.connectivity.edges("DA1_lPN")     # the same frame - different backend, different species conventions
```

Both calls return `pre, post, weight` as `int64, int64, int32`. Positions are
always nanometres. Side is always `left` / `right` / `center`. Skeletons and
meshes are always [navis](https://navis-org.github.io/navis/) neurons.

</div>

<div class="cn-section" markdown>

## Two promises

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

<span class="cn-card__icon">:material-check-all:</span>

### It is the same frame, and that is tested

BANC is served by *both* backends at the same snapshot — CAVE materialization 888
is neuPrint `banc:v888`, and its neuPrint body IDs are valid CAVE root IDs. So the
promise is checkable rather than aspirational:

```python
cave = cn.BANC(backend="cave").connectivity.edges(ids)
np_  = cn.BANC(backend="neuprint").connectivity.edges(ids)

pd.testing.assert_frame_equal(cave, np_)   # passes
```

That test has already caught two real divergences.
[How &rarr;](guides/capabilities.md#the-test-that-keeps-it-honest)

</div>

<div class="cn-card" markdown>

<span class="cn-card__icon">:material-alert-octagon:</span>

### Nothing degrades silently

If a dataset cannot do what you asked, it says so. It does not quietly hand you an
unfiltered result.

```python
fw.connectivity.synapses(x, min_score=100)   # fine
mic.connectivity.synapses(x, min_score=100)  # CapabilityError
```

Defaults never raise; only an *explicit* request for something the dataset lacks
does. Whole namespaces are absent rather than broken, so feature detection works:

```python
hasattr(fw, "segmentation")   # True
hasattr(hb, "segmentation")   # False
```

[How &rarr;](guides/capabilities.md)

</div>

</div>
</div>

<div class="cn-section" markdown>

## One mini-language for selecting neurons

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

<span class="cn-card__icon">:material-cursor-default-click:</span>

### `ids()` takes anything

```python
ds.ids(720575940604407468)   # a root/body ID
ds.ids("DA1_lPN")            # a type
ds.ids("/^AOTU00.*")         # a regex
ds.ids("superclass:visual_projection")
ds.ids("DA1_lPN", side="left")
```

Anything you can pass to `ids()` you can pass to any query. It is a pure function:
it returns IDs and mutates nothing.
[More &rarr;](tutorials/selecting-neurons.md)

</div>

<div class="cn-card" markdown>

<span class="cn-card__icon">:material-clock-time-four-outline:</span>

### `version="auto"` is correct on both backends

```python
ds.connectivity.edges(ids, version="auto")
```

Finds the newest materialization in which *all* your IDs are jointly valid — which
matters on CAVE, where root IDs change every time someone edits a neuron. On
neuPrint body IDs are immutable, so it is an identity. The same line is right
either way. [More &rarr;](tutorials/versions.md)

</div>

<div class="cn-card" markdown>

<span class="cn-card__icon">:material-brain:</span>

### Morphology is just navis

```python
skels = ds.skeletons.get("DA1_lPN")   # navis.NeuronList
navis.plot3d(skels)
```

Skeletons are `TreeNeuron`s, meshes are `MeshNeuron`s, always in nanometres — so
the whole navis ecosystem works without a conversion step.
[More &rarr;](tutorials/morphology.md)

</div>

</div>
</div>

<div class="cn-section" markdown>

## Datasets

| dataset | species | backend | annot | conn | syn | scores | NT/syn | NT/neuron | roi-conn | rois | skel | mesh | seg | cgraph | vox | proof | soma | live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flywire` | fly | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | · | · | ✅ | · |
|  |  | cave | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · |
| `flywire-production` | fly | **cave** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `banc` | fly | **neuprint** | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | · | · | · | · | · | · | ✅ | · |
|  |  | cave | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | · |
| `fanc` | fly | **cave** | ✅ | ✅ | ✅ | ✅ | · | · | · | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hemibrain` | fly | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | · | ✅ | · |
| `malecns` | fly | **neuprint** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | · | ✅ | · |
| `manc` | fly | **neuprint** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | · | ✅ | · |
| `microns` | mouse | **cave** | ✅ | ✅ | ✅ | · | · | · | · | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | · |
| `aedes` | mosquito | **cave** | ✅ | ✅ | ✅ | · | · | ◐ | · | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ |
| `fish2` | zebrafish | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | · | ✅ | · |

The capability columns come from `cn.capability_matrix()`, and a `·` there means the
call **raises** — not that it returns something subtly wrong.

**NT/syn** and **NT/neuron** are different claims, and the difference matters. *NT/syn*
is a capability: a prediction per synapse, so `synapses(x, transmitters=True)` and
`connectivity.transmitters(x)` work, and raise where they can't. *NT/neuron* is one call
per neuron and is **not** a capability — it arrives as the `nt` column of
`annotations.get()`, and it depends on the annotation *source* rather than the backend,
so it has no cell in `capability_matrix()`. Ask `ds.annotations.fields["nt"]` for the
columns behind it.

`NT/neuron` has a third mark, because "this dataset has transmitters" hides a difference
you cannot afford to miss when you aggregate:

| | |
|---|---|
| ✅ | **dense** — a classifier was run over the volume, so essentially every neuron has a call, and most of them are model output |
| ◐ | **sparse** — curated only: immunostaining, RT-PCR and the literature, transferred by cell type. Real evidence, but it covers a *fraction* of the dataset and a null means "nobody has looked", not "no transmitter" |
| · | nothing at all |

`aedes` is the ◐: 6,981 of its 17,397 neurons carry a transmitter, every one of them
traceable to a paper, and no model has ever run on that volume. `hemibrain` is the `·`.
Treating those two the same — or treating aedes's 40% like FlyWire's 99% — is how a
"GABAergic fraction" ends up being a statement about annotation effort. `nt_source`
names the exact column behind every value, so the distinction survives into the frame.

`aedes`'s CAVE datastack ships no annotation table — so its cell typing comes from
FlyTable (the lab's SeaTable) instead, keyed by root ID. It is lab-internal, so it
needs a `SEATABLE_TOKEN`: without one you get a clear missing-token error, not a
silently empty frame. What aedes still has no `·` for is a per-synapse **score** — its
synapse table records `size`, not a confidence — so `min_score=` raises.

Anything not listed works too, without writing a class:

```python
ca3  = cn.CAVE("zheng_ca3", fields={"type": ("cell_type",)})
wasp = cn.NeuPrint("wasp3:v0.8", server="neuprint-pre.janelia.org")
```

[Adding a dataset &rarr;](guides/custom-datasets.md)

</div>

<div class="cn-section" markdown>

## Get going

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

### :material-download: Install

```bash
pip install connecto
```

Then point it at your existing CAVE and neuPrint tokens — connecto reads from
wherever they already live.

[Installation &rarr;](get-started/installation.md) ·
[Credentials &rarr;](get-started/credentials.md)

</div>

<div class="cn-card" markdown>

### :material-school: Learn

Six short tutorials, from a first query to running one script unchanged across
two species.

[Tutorials &rarr;](tutorials/index.md)

</div>

<div class="cn-card" markdown>

### :material-api: Look things up

The API reference is generated from the docstrings, so there is no second copy of
the documentation to drift out of sync.

[Reference &rarr;](reference/index.md)

</div>

</div>
</div>

<div class="cn-section" markdown>

## What it doesn't do

Data fetching only. No clustering, no matching, no connectivity vectors, no
curation policy — connecto will backfill a `type` from the columns your dataset
declares, but it will never decide that some types are "bad" and null them out.
That is analysis, and it belongs upstream (see
[cocoa](https://github.com/flyconnectome/cocoa)).

</div>
