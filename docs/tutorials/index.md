# Tutorials

Six of them, in order. Each is short and stands alone, but they build.

Everything shown is a real output from a real run — if a number looks oddly
specific, that is because it is.

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

### [1. Your first query](first-query.md)

Open a dataset, pull an edge list, and learn to read the provenance stamped on
every frame you get back.

</div>

<div class="cn-card" markdown>

### [2. Selecting neurons](selecting-neurons.md)

The `ids()` mini-language: IDs, types, regexes, annotation columns, and how to
find out what a dataset will even accept.

</div>

<div class="cn-card" markdown>

### [3. Connectivity](connectivity.md)

Edges, adjacency matrices, individual synapses, per-ROI breakdowns and predicted
transmitters.

</div>

<div class="cn-card" markdown>

### [4. Morphology](morphology.md)

Skeletons and meshes as navis neurons, and where they come from on each backend.

</div>

<div class="cn-card" markdown>

### [5. Versions and root IDs](versions.md)

The one that will save you. Root IDs change when neurons are edited; this is how
you stop that silently corrupting your analysis.

</div>

<div class="cn-card" markdown>

### [6. One script, two datasets](cross-dataset.md)

Write it once, run it on a fly and on a mouse. What genuinely transfers, and what
does not.

</div>

</div>

## Before you start

You will need working credentials — see [Credentials](../get-started/credentials.md).
The tutorials use FlyWire and hemibrain, both of which need only a token you can get
yourself in about a minute.

```python
import connecto as cn

cn.auth_status()   # should say OK for cave and neuprint
```
