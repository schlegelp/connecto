# Datasets

```python
import connecto as cn

cn.list_datasets()
```

```
                  name                          label                  species        backends                     annotations
0                aedes         Aedes (mosquito brain)            Aedes aegypti            cave
1                 banc    BANC (brain and nerve cord)  Drosophila melanogaster  cave, neuprint                  cave, flytable
2                 fanc              FANC (female VNC)  Drosophila melanogaster            cave                            cave
3                fish2       fish2 (larval zebrafish)              Danio rerio        neuprint                        neuprint
4              flywire  FlyWire (FAFB) public release  Drosophila melanogaster  cave, neuprint                public, flytable
5   flywire-production      FlyWire (FAFB) production  Drosophila melanogaster            cave                public, flytable
6            hemibrain                      hemibrain  Drosophila melanogaster        neuprint                        neuprint
7              malecns                       male CNS  Drosophila melanogaster        neuprint                  neuprint, clio
8                 manc                MANC (male VNC)  Drosophila melanogaster        neuprint                        neuprint
9              microns      MICrONS (minnie65) public             Mus musculus            cave  celltypes, mtypes, nucleus_svm
10          optic-lobe                     optic lobe  Drosophila melanogaster        neuprint                        neuprint
```

`aedes` has an empty `annotations` cell, and that is not a gap in the table — the
datastack genuinely has no annotation table. See below.

Every one of these is also reachable by name:

```python
cn.get_dataset("hemibrain", version="hemibrain:v1.1")
```

## Backend is not the same as dataset

This trips people up, so it is worth being explicit: **a dataset can be served by more
than one backend.** BANC and FlyWire both are.

```python
cn.BANC(backend="cave")       # CAVE materialization 888
cn.BANC(backend="neuprint")   # neuPrint banc:v888  - the same snapshot
```

These are not two datasets. They are two doors into one. BANC's neuPrint body IDs *are*
valid CAVE root IDs — which is what makes them directly comparable, and is the basis of
connecto's cross-backend conformance test.

---

## Drosophila

### `flywire` — FlyWire (FAFB) public release

The public whole-brain adult fly connectome. Backend: CAVE (`flywire_fafb_public`).
Versions `630` and `783`.

- **Annotations** come from a published GitHub TSV, not from CAVE. This is load-bearing:
  it means `ds.ids("DA1_lPN")` genuinely *cannot* compile to a CAVE server-side filter,
  and must be resolved client-side against the cached annotation frame. 34 columns,
  including `super_class`, `ito_lee_hemilineage`, `top_nt`.
- **Skeletons** come from a precomputed source, not from CAVE's skeleton service — the
  public stack has no L2 cache, and the service needs one.
- Has per-synapse neurotransmitter predictions (`nt_per_synapse`), which is rare.
- No `live` queries. The public release is a frozen release.

### `flywire-production` — FlyWire (FAFB) production

The live, actively-proofread stack (`flywire_fafb_production`). Same brain, moving
target.

- Adds `live` and `l2cache`.
- **Needs group membership**, not just a token. If you are not in the FlyWire group you
  get a 403, which connecto reports as a *permission* problem rather than sending you
  off to fetch a new token that would not help. See
  [Credentials](../get-started/credentials.md#403-is-a-different-problem).

### `banc` — brain and nerve cord

Fly brain *and* ventral nerve cord in one volume. Served by **both** CAVE (mat 888) and
neuPrint (`banc:v888`).

- Annotations live in a CAVE table (`codex_annotations`) which is **long-format** —
  1.84 M rows across 32 classification systems. connecto pivots it to wide and fetches it
  in chunks, because it will not come down in one request.
- The neuPrint-backed handle still reads its annotations from CAVE, because that is where
  they are. Backend and annotation source are independent choices.

### `fanc` — female adult nerve cord

The female counterpart to `manc`, but a CAVE dataset rather than a neuPrint one — an
editable segmentation with a chunkedgraph, so root IDs move under you and
`version="auto"` earns its keep.

- **Not public.** Needs the `FANC_edit` group; without it you get a 403, which connecto
  reports as a permission problem rather than a bad token.
- Its synapse table calls the confidence score `score`, not `cleft_score`. That is a
  column name, not a capability — `min_score=` works exactly as it does on FlyWire.
- Annotations are **long-format tags**: one row per (neuron, tag), with the category in
  `tag2` and the value in `tag`. connecto pivots it wide, so each category
  (`hemilineage`, `soma segment`, `publication`, …) becomes a column.
- A neuron carries **several tags per category**. Each MDN is tagged `MDN`, `MDN3` *and*
  `moonwalker descending neuron`, so its `type` reads
  `"MDN, MDN3, moonwalker descending neuron"`. All three are individually searchable —
  `ds.ids("MDN")`, `ds.ids("MDN3")` and `ds.ids("moonwalker descending neuron")` all find
  it.
- It pins to `latest`, not to the one permanent materialization (v840). v840 never
  expires, but it predates most of the annotation work: 95 neurons have a cell type there
  against 4,265 today, and `hemilineage` does not exist at all. Current annotations win.
- Its `l2cache` service is flaky and 503s on its own. That does not take skeletons with
  it — connecto uses the CAVE skeleton service, which is a different service.

### `hemibrain`

The dataset everyone has used. neuPrint, `hemibrain:v1.2.1`.

- **It is a hemibrain.** One side of the brain, cropped. Expect roughly half the neurons
  of a whole-brain dataset, and expect somata to be missing — they are frequently outside
  the volume, so `soma_x/y/z` is `NaN` for many neurons.
- Has no `somaSide` field. The side is a suffix on `instance` (`DA1_lPN_R`), which
  connecto derives and normalises to `left`/`right`.
- Has no `class` column, and connecto does not invent one.
- Rich ROI hierarchy (231 ROIs, nested — use `primary=True` for the non-overlapping set).

### `malecns` — male CNS

neuPrint, plus optional [Clio](https://clio.janelia.org) annotations
(`pip install "connecto[clio]"`). Not fully public.

### `manc` — male adult nerve cord

neuPrint. The male ventral nerve cord, and the counterpart to CAVE-backed [`fanc`](#fanc-female-adult-nerve-cord).

- Uses `LHS`/`RHS` for side, where hemibrain and maleCNS use `L`/`R`. connecto normalises
  both to `left`/`right`.
- `type` coalesces `type` → `systematicType` → `instance`, so neurons with only a
  systematic name still get one.

### `optic-lobe`

neuPrint. The optic lobe, in detail.

- Like hemibrain, it has **no `somaSide`** — side is a suffix on `instance` (`Tm1_R`), and
  connecto derives it.
- It has **no `class`** column at all, and connecto does not invent one.

---

## Mus musculus

### `microns` — MICrONS (minnie65) public

A cubic millimetre of mouse visual cortex. Backend: CAVE (`minnie65_public`).

Two things are different enough to call out:

**It ships with `fields={}`.** No `type`, no `side`, no `class`. This is deliberate.
MICrONS's cell types live across several tables with genuinely different semantics —
`cell_type_local`, `aibs_metamodel_mtypes_v661`, `nucleus_svm` — and they disagree.
Choosing one to be *the* type would be a scientific judgement smuggled in as a library
default. So connecto declines:

```python
cn.MICrONS().ids("BC")
```

```
ValueError: MICrONS (minnie65) public has no `type` column configured. Pass
`fields={'type': (...)}` to the dataset, or query a raw column with 'column:value'.
```

Say what you mean, and it is a one-liner:

```python
mic = cn.MICrONS(fields={"type": ("cell_type",)})
mic.ids("BC")
```

**A root ID can have more than one nucleus.** Merge errors are common in a dataset this
size, so do not assume `id` is unique in the annotation frame.

---

## Danio rerio

### `fish2` — larval zebrafish

neuPrint, on its **own server** (`neuprint-fish2.janelia.org`). A separate deployment —
a token that works for `neuprint.janelia.org` is not guaranteed to work here, which is
why `auth_status()` probes every server rather than one.

Unversioned: the dataset string is just `fish2`, with no `:vN` suffix. connecto handles
both forms.

---

## Aedes aegypti

### `aedes` — mosquito brain

CAVE (`wclee_aedes_brain`), from the Wei-Chung Lee lab.

**It has no annotations.** The datastack ships a synapse table, a nucleus table, and
nothing else — so there is no column that means "type" or "side", and connecto does not
pretend otherwise. The whole namespace is absent:

```python
aedes = cn.Aedes()

aedes.connectivity.edges(root_ids)   # fine
hasattr(aedes, "annotations")        # False
aedes.ids("SomeType")                # CapabilityError: Aedes has no `annotations`
```

An empty `fields` is a description, not a stub waiting to be filled in. If a cell-type
table appears upstream, that is the day `fields` gets populated — not before.

Two things worth knowing:

- **It uses the `synapses` table, not the newer `synapses_v2`.** `synapses_v2` is 23%
  larger and more recent, but CAVE has it registered at the datastack's 16×16×45 voxel
  size while its coordinates are already in nanometres. connecto asks CAVE for nanometres,
  so the server scales them up *again* and hands back a mosquito brain 9.9 mm deep. Edges
  would be fine; every synapse position would be silently wrong by 16–45×. Correct beats
  bigger. If the registration is fixed upstream, this becomes a one-line change.
- **Every materialization expires within weeks**, so there is no stable snapshot to pin
  to and the default version is `latest`. Root IDs churn (~2%/month), which is what
  `version="auto"` is for.

---

## Anything else

You do not need a class. See [Adding a dataset](custom-datasets.md):

```python
ca3  = cn.CAVE("zheng_ca3", fields={"type": ("cell_type",)})
wasp = cn.NeuPrint("wasp3:v0.8", server="neuprint-pre.janelia.org")
```
