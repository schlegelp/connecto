# Get started

Three pages, in order:

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

### :material-download: [Installation](installation.md)

`pip install connecto`, plus the optional extras for private annotation sources.

</div>

<div class="cn-card" markdown>

### :material-key: [Credentials](credentials.md)

connecto reads the CAVE and neuPrint tokens you already have. This page is also
where to look when something 401s.

</div>

<div class="cn-card" markdown>

### :material-rocket-launch: [Quickstart](quickstart.md)

Ten minutes: pick a dataset, select some neurons, pull their connectivity and
their morphology.

</div>

</div>

## The idea in one paragraph

Connectomic datasets each speak their own dialect. CAVE (FlyWire, BANC, MICrONS)
has materialization versions, root IDs that change under you, and per-datastack
table names. neuPrint (hemibrain, maleCNS, MANC, zebrafish) has immutable body
IDs, Cypher, and dataset strings like `hemibrain:v1.2.1`. `connecto` is a
**normalisation layer** over both: the backends fetch raw frames, and everything
you actually touch — column names, dtypes, units, errors — is produced in one
place that both backends share.

The practical consequence is that this works:

```python
import connecto as cn

for ds in (cn.FlyWire(), cn.Hemibrain()):
    edges = ds.connectivity.edges("DA1_lPN")
    print(ds.label, edges.shape, list(edges.columns))
```

```
FlyWire (FAFB) public release (40581, 3) ['pre', 'post', 'weight']
hemibrain                     (24898, 3) ['pre', 'post', 'weight']
```

Different backend, different specimen, different ID space — same frame.

!!! note "What connecto is not"

    A data-fetching library, and nothing else. No clustering, no matching, no
    connectivity vectors, no curation policy. That is analysis, and it belongs
    upstream — see [cocoa](https://github.com/flyconnectome/cocoa).
