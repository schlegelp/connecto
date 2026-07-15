# Capabilities

## The rule

> **Defaults never raise. An explicit request for something the dataset cannot do
> always raises.**

That is the whole design, and everything below is a consequence of it.

## Why it exists

`fafbseg` — connecto's most direct ancestor — can be pointed at datasets other than
FlyWire. When you do, deep inside, it does something like this:

```python
if dataset != "flywire":
    min_score = None
    filtered = False
    transmitters = False
```

Which means: you asked for a minimum cleft score, and you did not get one. You asked
for filtered synapses, and you got unfiltered ones. No error, no warning. Your script
runs, produces a plot, and the plot is wrong in a way that looks exactly like a real
result.

This is the single most dangerous failure mode available to a normalisation layer,
because normalising *is* smoothing over differences, and the temptation to smooth over
one more is always there.

connecto's answer:

```python
fw  = cn.FlyWire()
mic = cn.MICrONS()

fw.connectivity.synapses(x, min_score=100)    # fine - FlyWire has cleft scores
mic.connectivity.synapses(x, min_score=100)   # CapabilityError
```

```
CapabilityError: MICrONS (minnie65) public does not support `min_score`
(no synapse_scores). Drop the argument, or use a dataset that has it.
```

But the default path still works everywhere:

```python
mic.connectivity.synapses(x)    # fine - you did not ask for scoring
```

The mechanism is a sentinel. Arguments default to `UNSET`, not to `None` — so connecto
can distinguish *"the user did not mention this"* from *"the user explicitly asked for
None"*. Only the second is an error.

## Namespaces are absent, not broken

Capabilities are not just per-argument. Whole namespaces either exist or do not:

```python
hasattr(fw, "proofreading")   # True  - FlyWire is still being edited
hasattr(hb, "proofreading")   # False - hemibrain is frozen; there is no edit history
```

This is also how a capability earns its keep. `L2CACHE` used to gate nothing you could
call — it was an internal hint to the skeleton fallback, so five datasets could claim it
and no test could hold them to the claim. It now carries the [`ds.l2`
namespace](../tutorials/morphology.md#the-l2-cache-cheap-skeletons-and-dotprops), which
means `test_capabilities_are_honest` checks it like any other:

```python
hasattr(cn.FlyWire("production"), "l2")   # True
hasattr(cn.FlyWire(), "l2")               # False - the public stack has no L2 cache
```

A capability that gates nothing is not a promise. It is a comment.

`hasattr` genuinely returns `False`, because `CapabilityError` subclasses
`AttributeError`. That is deliberate: it means ordinary Python feature detection works,
and you do not have to learn a connecto-specific idiom to write adaptive code.

Ask anyway, and you get told why — not a bare `AttributeError`:

```python
hb.proofreading
```

```
CapabilityError: hemibrain (neuprint) has no `proofreading` - it does not support
proofreading. Available: annotations, connectivity, meshes, roi_connectivity, rois,
segmentation, skeletons, somas, synapse_scores, synapses.
```

The error lists what the dataset *can* do. An error that only says "no" makes you go
and read the source; this one answers the question you were about to ask next.

## One word, two promises: `SEGMENTATION` and `CHUNKEDGRAPH`

"Does this dataset have a segmentation?" turns out to be two questions, and hemibrain
answers them differently. It *does* have a segmentation volume — a flat `precomputed://`
bucket you can ask "what body is at this point". It does *not* have a chunkedgraph:
its body IDs were frozen at publication, there are no supervoxels beneath them, and
nobody is proofreading them any more.

So there are two capabilities, and hemibrain has exactly one of them:

```python
hb.segmentation.locs_to_segments(tbars)   # fine - reads the volume
hb.segmentation.update_ids([1734350788])  # CapabilityError: no chunkedgraph
```

Collapsing them into one would force a choice between two lies: deny that hemibrain has
a segmentation (it does), or promise `update_ids` on IDs that cannot change (it isn't).
Splitting them lets both datasets tell the truth:

```python
fw.supports(Cap.SEGMENTATION), fw.supports(Cap.CHUNKEDGRAPH)   # (True, True)
hb.supports(Cap.SEGMENTATION), hb.supports(Cap.CHUNKEDGRAPH)   # (True, False)
```

## A capability belongs to a *door*, not to a dataset

FlyWire has a chunkedgraph. You cannot reach it through neuPrint.

Both statements are true, which means "does FlyWire support `CHUNKEDGRAPH`?" is not
actually a well-formed question. A dataset served by two backends is one dataset behind
two doors, and the doors are different widths — so a capability lives on the *pair*:

```python
fw = cn.FlyWire()                   # neuPrint, the default
fw.supports(cn.Cap.CHUNKEDGRAPH)    # False

fw = cn.FlyWire(backend="cave")
fw.supports(cn.Cap.CHUNKEDGRAPH)    # True
```

This is not pedantry. Before capabilities were resolved per backend, a neuPrint-backed
FlyWire claimed `CHUNKEDGRAPH` and then raised on every call that used it — and, far
worse:

```python
cn.FlyWire(backend="neuprint").connectivity.synapses(x, transmitters=True)
```

…returned a frame with **no `nt` column and no error**. The neuPrint copy of FlyWire
simply does not carry per-synapse transmitters. That is the fafbseg bug at the top of
this page, reproduced exactly, inside the library written to prevent it. It now raises:

```
CapabilityError: FlyWire (FAFB) public release (neuprint) does not support
`transmitters` (no nt_per_synapse). Drop the argument, or use a dataset that has it.
The cave backend does: cn.get_dataset("flywire", backend="cave").
```

Note the last sentence. A refusal that names the door which *would* have worked is the
difference between a dead end and a next step.

The narrowing is declared once, on the backend, rather than per dataset
(`spec.BACKEND_LIMITS`) — neuPrint serves frozen snapshots, so it can never have
supervoxels, an edit history, an L2 cache, or a "right now" query. The *widening* is
declared per dataset, because it is a fact about that pairing: both neuPrint copies ship
an ROI hierarchy their CAVE datastack has no equivalent of, so `ds.rois` appears when you
come in through neuPrint and vanishes when you don't.

```python
cn.get_spec("flywire").backends_with(cn.Cap.CHUNKEDGRAPH)   # ('cave',)
cn.get_spec("flywire").capabilities_for("neuprint")          # what you can actually reach
```

## Checking up front

```python
import connecto as cn

ds.supports(cn.Cap.NT_PER_SYNAPSE)    # bool
ds.capabilities                       # frozenset[Cap] - for the backend it is bound to
```

So adaptive code reads naturally:

```python
def describe(ds, ids):
    out = {"n": len(ids)}
    out["synapses"] = ds.connectivity.synapse_counts(ids)["pre"].sum()

    if ds.supports(cn.Cap.NT_PER_SYNAPSE):
        out["nt"] = ds.connectivity.transmitters(ids)

    if hasattr(ds, "proofreading"):
        out["proofread"] = ds.proofreading.is_proofread(ids).mean()

    return out
```

## The full matrix

```python
cn.capability_matrix()
```

```
                     backend  annotations  connectivity  synapses  synapse_scores  nt_per_synapse  roi_connectivity   rois  skeletons  meshes  l2cache  segmentation  chunkedgraph  proofreading  somas   live  neuroglancer
aedes                   cave         True          True      True           False           False             False  False       True    True     True          True          True         False   True   True          True
banc                neuprint         True          True      True           False           False              True   True      False   False    False         False         False         False   True  False         False
banc                    cave         True          True      True           False           False              True  False       True    True     True          True          True         False   True  False          True
fanc                    cave         True          True      True            True           False             False  False       True    True     True          True          True          True   True   True          True
fish2               neuprint         True          True      True            True           False              True   True       True    True    False         False         False         False   True  False         False
flywire             neuprint         True          True      True            True           False              True   True       True    True    False          True         False         False   True  False          True
flywire                 cave         True          True      True            True            True              True  False       True    True    False          True          True          True   True  False          True
flywire-production      cave         True          True      True            True            True              True  False       True    True     True          True          True          True   True   True          True
hemibrain           neuprint         True          True      True            True           False              True   True       True    True    False          True         False         False   True  False         False
malecns             neuprint         True          True      True            True           False              True   True       True    True    False          True         False         False   True  False         False
manc                neuprint         True          True      True            True           False              True   True       True    True    False          True         False         False   True  False         False
microns                 cave         True          True      True           False           False             False  False       True    True     True          True          True         False   True  False          True
```

One row per **door**, not per dataset — `banc` and `flywire` each appear twice, because
that is where a capability actually lives. The first row for a dataset is its default
backend: the one `get_dataset(name)` hands you.

A `False` means the call **raises**. It does not mean it returns something subtly
wrong, or an empty frame, or a frame with a column of `NaN`.

That last one is worth stating on its own. connecto will not fabricate a canonical
column just to satisfy the schema. hemibrain has no `class` and no per-synapse
transmitter, so those columns are **absent** from a hemibrain annotation frame — because
a column of `NaN` looks like *"we have this field and it happens to be empty"*, which is
a different and much more misleading statement than *"we do not have this field"*.

## The capabilities

| capability | means |
|---|---|
| `ANNOTATIONS` | there is a metadata table (types, sides, classes) |
| `CONNECTIVITY` | edge lists |
| `SYNAPSES` | synapse-resolution data with positions |
| `SYNAPSE_SCORES` | a per-synapse confidence / cleft score (`min_score=`) |
| `NT_PER_SYNAPSE` | per-synapse neurotransmitter predictions |
| `ROI_CONN` | edges can be broken down by neuropil (`by_roi=`, `rois=`) |
| `ROIS` | an ROI hierarchy and meshes |
| `SKELETONS` / `MESHES` | morphology |
| `L2CACHE` | CAVE's level-2 chunk cache — carries the `ds.l2` namespace |
| `SEGMENTATION` | a segmentation volume you can query — "what body is at this point", cutouts, voxels |
| `CHUNKEDGRAPH` | that volume is a *proofreadable graph* — supervoxels, root-ID history, `update_ids` |
| `PROOFREADING` | edit history, proofreading status |
| `SOMAS` | soma / nucleus positions |
| `LIVE` | non-materialized "right now" queries |
| `NEUROGLANCER` | can build a Neuroglancer scene URL |

## The test that keeps it honest

Capabilities are a *claim*, and a claim needs checking — a dataset that advertises
`SKELETONS` but 500s when you ask for one is worse than one that admits it cannot.

So the conformance suite runs, for every registered dataset with a pinned example
neuron:

```python
def test_capabilities_are_honest(ds):
    """Every capability claimed must actually work; every one denied must raise."""
```

This is how the FlyWire skeleton bug was found. FlyWire's spec claimed `L2CACHE`; the
public stack has no L2 cache at all, and CAVE's skeleton service *requires* one — so
`fw.skeletons.get(...)` returned an HTTP 500. The spec was lying, the test caught it,
and the fix was to stop claiming the capability and add a precomputed skeleton source
instead.

Related: BANC is served by *both* backends at the same snapshot, so the two can be
compared directly. See
[The test that keeps it honest](../tutorials/cross-dataset.md#the-test-that-keeps-it-honest) —
it has caught two real divergences that no amount of reading the code would have found.
