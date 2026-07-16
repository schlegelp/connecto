# Adding a dataset

Three levels, in increasing order of effort. Most people never leave the first.

## 1. Just point at it

```python
import connecto as cn

ca3  = cn.CAVE("zheng_ca3")       # mouse hippocampus; connecto has never heard of it
wasp = cn.NeuPrint("wasp3:v0.8", server="neuprint-pre.janelia.org")
```

connecto **probes the server** for the spec: which tables exist, which synapse table to
use, what the voxel size is, which versions are available. You get a working dataset with
no registration and no class.

```python
ca3.connectivity.edges(some_ids)
ca3.skeletons.get(some_ids)
```

A probed dataset is deliberately modest about what it claims. It has no `fields`, so:

```python
ca3.ids("pyr")
```

```
ValueError: zheng_ca3 has no `type` column configured. Pass `fields={'type': (...)}`
to the dataset, or query a raw column with 'column:value'.
```

It does not know which of the columns it found means "type" — and rather than guessing
(and being wrong in a way you would not notice) it says so. Tell it:

```python
ca3 = cn.CAVE("zheng_ca3", fields={"type": ("cell_type",)})
ca3.ids("pyr")     # now works
```

`fields` maps a canonical column to the raw columns it may be coalesced from, **in
priority order**. `("cell_type", "hemibrain_type")` means "use `cell_type`; where it is
null, fall back to `hemibrain_type`".

## 2. Register it

To make it available by name everywhere, including `get_dataset()` and the capability
matrix:

```python
cn.register(ca3.spec)

cn.get_dataset("zheng_ca3")
cn.list_datasets()          # now includes it
```

Put that in your `sitecustomize.py` or the top of your analysis package and the dataset
behaves exactly like a built-in.

## 3. Write the spec out

For anything you will use repeatedly — or contribute back — declare it properly. A
`DatasetSpec` is pure data; there is no code to write.

```python
from connecto import DatasetSpec, BackendSpec, AnnotationSource, Cap

SPEC = DatasetSpec(
    name="ca3",
    label="Zheng CA3 (mouse hippocampus)",
    species="Mus musculus",
    backends=(
        BackendSpec(
            kind="cave",
            source="zheng_ca3",
            default_version="latest",
            synapse_table="synapses_ca3_v1",
            nucleus_table="c3_nuclei_v1",
            # edge_view=...      # a pre-aggregated fast path, if the stack has one
            # score_column=...   # if the synapse table calls it something other
            #                    # than `cleft_score` (FANC says plain `score`)
        ),
    ),
    annotation_sources=(
        AnnotationSource(
            name="cave",
            kind="cave_table",
            location="ca3_cell_type",
            id_column="pt_root_id",
            public=True,
        ),
    ),
    fields={
        "type": ("cell_type",),
    },
    voxel_size=(18, 18, 45),
    capabilities=frozenset({
        Cap.ANNOTATIONS, Cap.CONNECTIVITY, Cap.SYNAPSES,
        Cap.SKELETONS, Cap.MESHES, Cap.SEGMENTATION, Cap.CHUNKEDGRAPH, Cap.SOMAS,
    }),
    example_ids=(648518346448625630,),             # <- do not skip this
)

cn.register(SPEC)
cn.get_dataset("ca3")
```

Every table name above came from asking the server (`cn.datasets.probe_cave("zheng_ca3")`
prints what it found). Do not guess them — a spec that names a table which does not exist
fails at query time, not at import.

### Claim only what is true

`capabilities` is a **promise**, and the conformance suite will hold you to it: every
capability you claim must actually work, and every one you omit must raise. Do not add
`Cap.SYNAPSE_SCORES` because the synapse table happens to have a `score` column; add it
because filtering on that column does something meaningful.

This is not hypothetical. FlyWire's spec used to claim `Cap.L2CACHE`. The public stack
has no L2 cache, CAVE's skeleton service requires one, and so `skeletons.get()` returned
an HTTP 500 — from a dataset whose capability matrix promised skeletons worked. The lie
was in the spec.

### `example_ids` is what makes the test run

The conformance suite is parameterised over every registered dataset that has
`example_ids`. A dataset with none is **silently skipped** — it will appear in the
capability matrix, claim whatever it likes, and never be checked. Two or three real,
long-lived neurons is enough.

### Derived columns

Some datasets do not have the field you need as a column, but do have it hidden in
another. hemibrain has no `somaSide`; the side is a suffix on `instance`
(`DA1_lPN_R`). Rather than making every user write that regex, declare it:

```python
DatasetSpec(
    ...,
    derive={"side_from_instance": ("instance", r"_([LRM])$")},
    fields={"side": ("side_from_instance",)},
)
```

`derive` maps a new column to `(source column, regex)`; the first capture group becomes
the value. connecto then maps `R` → `right` through the normal side vocabulary, so the
dataset is comparable with everything else.

### Long-format annotation tables

Some tables are **long**: one row per `(neuron, key, value)`, where other datasets would
have one column per key. BANC's `codex_annotations` is (32 classification systems), and so
is FANC's `neuron_information` (34 tag categories). Declare the key and value columns and
connecto pivots it wide:

```python
AnnotationSource(
    name="cave", kind="cave_table", location="neuron_information",
    id_column="pt_root_id",
    pivot=("tag2", "tag"),      # (key column, value column)
)
```

Each distinct key becomes a column, so `fields` can then point at it like any other.

A neuron may carry **several values for one key** — each FANC MDN is tagged `MDN`, `MDN3`
*and* `moonwalker descending neuron` — so the pivot stores the set in one cell, joined
with `", "`. Selection understands that: `ds.ids("MDN")`, `ds.ids("MDN3")` and
`ds.ids("moonwalker descending neuron")` all find it. Membership, not substring —
`ds.ids("MDN")` does not match a neuron whose only tag is `MDN3`.

## Annotation sources are separate from backends

An `AnnotationSource` says *where the metadata lives*, which is often nowhere near the
connectivity:

| kind | where |
|---|---|
| `cave_table` | a table in the CAVE datastack |
| `neuprint` | the neuPrint `:Neuron` node properties |
| `github_tsv` | a published TSV (this is how FlyWire's public annotations work) |
| `clio` | Clio (`pip install "connecto[clio]"`) |
| `seatable` | SeaTable / "flytable" (`pip install "connecto[flytable]"`) |

A dataset can have several, and you pick:

```python
cn.FlyWire(annotations="public")     # the published TSV
cn.FlyWire(annotations="flytable")   # the lab's internal table
cn.FlyWire().annotations.sources     # ['public', 'flytable']
```

The two are independent axes, and they genuinely cross: `cn.BANC(backend="neuprint")`
reads its **edges** from neuPrint and its **annotations** from CAVE, because that is
where each actually lives.

A `seatable` source names its table as `base.table`, and *names the base for a
reason*: without it, SeaTable has to search every base your token can see to find the
table (~20s for a big one); with it, resolution is ~1s. A source can even be several
tables — `location="main.info,optic_lobes.optic"` — which connecto concatenates, which
is exactly how FlyWire's live annotations span a central-brain and an optic-lobe table.
Its freshness token folds in every table, so an edit to any of them re-fetches.

SeaTable is also not one server. The lab runs its own deployment (`instance="flytable"`,
the default, reached via `SEATABLE_SERVER`) and there is the official `cloud.seatable.io`
(`instance="seatable"`) — and a base named on one does not exist on the other. FlyWire
and aedes are on flytable; BANC's `banc_meta` is on the cloud, so it sets
`instance="seatable"`. A new deployment is one line in `connecto.sources.seatable._INSTANCES`.

```python
# BANC's live annotations live on the official cloud, not the lab instance
AnnotationSource("flytable", "seatable", "banc_meta.banc_meta",
                 instance="seatable", id_column="root_888", public=False)
```

## Adding a whole backend

If you need to support something that is neither CAVE nor neuPrint, subclass `Dataset`
and implement ten `_fetch_*` hooks. That is the entire surface:

```python
_resolve_version   _list_versions      _find_version    _ids_exist
_fetch_annotations _fetch_edges        _fetch_synapses
_fetch_skeletons   _fetch_meshes       _fetch_somas
```

Each returns a **raw** frame — whatever the service gave you, under whatever column
names. You then declare how to read it:

```python
_edge_colmap = {"pre": "presynaptic_id", "post": "postsynaptic_id", "weight": "n_syn"}
_raw_position_units = "voxel"     # or "nm"
```

and connecto does the renaming, the dtype coercion, the unit conversion and the
provenance stamping. **A backend never builds a user-facing DataFrame.** That is the
rule that makes the normalisation trustworthy: there is exactly one implementation of it,
so there is nothing to drift.
