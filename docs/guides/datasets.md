# Datasets

```python
import connecto as cn

cn.list_datasets()
```

```
                 name                          label                  species        backends                     annotations  public
0               aedes         Aedes (mosquito brain)            Aedes aegypti            cave                        flytable   False
1                banc    BANC (brain and nerve cord)  Drosophila melanogaster  neuprint, cave                  cave, flytable    True
2                fanc              FANC (female VNC)  Drosophila melanogaster            cave                            cave   False
3               fish2       fish2 (larval zebrafish)              Danio rerio        neuprint                        neuprint   False
4             flywire  FlyWire (FAFB) public release  Drosophila melanogaster  neuprint, cave                public, flytable    True
5  flywire-production      FlyWire (FAFB) production  Drosophila melanogaster            cave                public, flytable   False
6           hemibrain                      hemibrain  Drosophila melanogaster        neuprint                        neuprint    True
7             malecns                       male CNS  Drosophila melanogaster        neuprint                  neuprint, clio    True
8                manc                MANC (male VNC)  Drosophila melanogaster        neuprint                        neuprint    True
9             microns      MICrONS (minnie65) public             Mus musculus            cave  celltypes, mtypes, nucleus_svm    True
```

`aedes`'s only annotation source is `flytable` — its CAVE datastack has no annotation
table, so the cell typing lives in the lab's SeaTable instead. It is the only source,
and it is non-public, so `annotations="auto"` still resolves to it. See below.

Every one of these is also reachable by name:

```python
cn.get_dataset("hemibrain", version="hemibrain:v1.1")
```

## Who made it, and may you have it?

A dataset is somebody's years of work, and connecto is the last thing that knows which
one produced the numbers in your figure. So it carries the provenance:

```python
print(cn.Hemibrain().cite())
```

```
hemibrain (Drosophila melanogaster)

Dense FIB-SEM reconstruction of much of the central brain of a female Drosophila -
one hemisphere, cropped, and no optic lobes. ~25,000 typed neurons and ~20M synapses.

Please cite:
  Scheffer LK, Xu CS, Januszewski M, Lu Z, et al. (2020) A connectome and analysis of
  the adult Drosophila central brain. eLife. doi:10.7554/eLife.57443
  Xu CS, Hayworth KJ, Lu Z, Grob P, et al. (2017) Enhanced FIB-SEM systems for
  large-volume 3D imaging. eLife. doi:10.7554/eLife.25916

See also:
  website    https://www.janelia.org/project-team/flyem/hemibrain
  neuprint   https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1
```

Every field is data, so you can use it rather than read it:

```python
ds.description      # the blurb
ds.publications     # tuple[Publication] - authors, year, title, journal, doi, url
ds.links            # {"website": ..., "neuprint": ..., "codex": ...}
ds.public           # False if a token alone will not get you in
ds.access           # ...and what will
```

**`public=False` is not a secret.** Every dataset here is one you may read *about*; four
of them are ones connecto cannot fetch for you on a fresh token — `flywire-production`,
`fanc`, `aedes` and `fish2`. Saying so up front is kinder than letting you find out from
a 403 halfway through a script, and when the 403 does come, the dataset's `access` note
is in the error:

```
ConnectoAuthError: Your CAVE token is valid, but you do not have permission
for this resource (HTTP 403).
  https://cave.fanc-fly.com
  [fanc_production_mar2021]

Authenticated as: you@lab.org
Your groups:      default

FANC (female VNC): Needs the `FANC_edit` group: join the FANC community and have
your Google account authorized before you generate a CAVE token. There is no public
FANC datastack - an unprivileged token gets a 403, and a new one will not help.
```

Two datasets cite **nothing**, and that is deliberate. `aedes` and `fish2` have no paper,
and it would be easy to give them a plausible neighbour's — the Lee lab's mosquito
preprint describes a *different, partial* volume; there are two published larval
zebrafish connectomes and neither is known to be this one. Both would look right in a
methods section and both would be wrong, so connecto says it does not know.

## Backend is not the same as dataset

This trips people up, so it is worth being explicit: **a dataset can be served by more
than one backend.** BANC and FlyWire both are, and **neuPrint is the default for both**.

```python
cn.BANC()                 # neuPrint banc:v888
cn.BANC(backend="cave")   # CAVE materialization 888 - the same snapshot
```

These are not two datasets. They are two doors into one. BANC's neuPrint body IDs *are*
valid CAVE root IDs — which is what makes them directly comparable, and is the basis of
connecto's cross-backend conformance test.

But the doors are **different widths**, and connecto will not pretend otherwise. See
[A capability belongs to a door](capabilities.md#a-capability-belongs-to-a-door-not-to-a-dataset).
The short version:

| | neuPrint | CAVE |
|---|---|---|
| ROI hierarchy, `by_roi=` edges | ✅ | · |
| chunkedgraph (`update_ids`, supervoxels) | · | ✅ |
| proofreading, L2 cache | · | ✅ |
| FlyWire per-synapse transmitters | · | ✅ |
| BANC skeletons, meshes, segmentation | · | ✅ |

Everything a door cannot do **raises**, and the error names the door that can.

---

## Drosophila

### `flywire` — FlyWire (FAFB) public release

The whole brain of an adult female *Drosophila* at synapse resolution, proofread by the
FlyWire community from the FAFB serial-section TEM volume. The v783 release has 139,255
neurons and ~50 M synapses. Backends: **neuPrint** (`flywire-fafb:v783b`, the default)
and CAVE (`flywire_fafb_public`, versions `630` and `783`).

> **Cite** Dorkenwald et al. (2024) *Neuronal wiring diagram of an adult brain*, Nature,
> [10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) · Schlegel et
> al. (2024) *Whole-brain annotation and multi-connectome cell typing of Drosophila*,
> Nature, [10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5) ·
> Zheng et al. (2018) *A complete electron microscopy volume of the brain of adult
> Drosophila melanogaster*, Cell,
> [10.1016/j.cell.2018.06.019](https://doi.org/10.1016/j.cell.2018.06.019)
> — the segmentation is FlyWire's, the electrons are FAFB's.
>
> **Find it** [flywire.ai](https://flywire.ai/) ·
> [Codex](https://codex.flywire.ai/) ·
> [annotations](https://github.com/flyconnectome/flywire_annotations)

- **Annotations** come from a published GitHub TSV, not from CAVE. This is load-bearing:
  it means `ds.ids("DA1_lPN")` genuinely *cannot* compile to a CAVE server-side filter,
  and must be resolved client-side against the cached annotation frame. 34 columns,
  including `super_class`, `ito_lee_hemilineage`, `top_nt`. Same on either backend.
- **Skeletons** come from a precomputed bucket, on *both* backends — the CAVE public
  stack has no L2 cache (and CAVE's skeleton service needs one), and the neuPrint mirror
  has no skeleton store at all. It is a plain HTTPS bucket, so both doors read it and
  return the identical skeleton.
- **Per-synapse neurotransmitters are CAVE-only.** The neuPrint mirror's `Synapse` nodes
  do not carry them, so `synapses(x, transmitters=True)` raises there and tells you to
  use `backend="cave"`. It used to return a frame with no `nt` column and no error.
- No `live` queries. The public release is a frozen release.

### `flywire-production` — FlyWire (FAFB) production

The live, actively-proofread stack (`flywire_fafb_production`). Same brain, moving
target: root IDs change with every edit.

- Adds `live` and `l2cache`. Uses CAVE's skeleton service, not the precomputed bucket —
  those are published per materialization of the *frozen* release, and a live root ID is
  in none of them.
- **Not public.** Needs group membership, not just a token. If you are not in the FlyWire
  group you get a 403, which connecto reports as a *permission* problem rather than
  sending you off to fetch a new token that would not help. See
  [Credentials](../get-started/credentials.md#403-is-a-different-problem).

### `banc` — brain and nerve cord

The first synapse-resolution connectome to hold the brain **and** the ventral nerve cord
of one animal — a female adult *Drosophila*, imaged by GridTape-TEM. 158,262 neurons at
v888. Served by **both** neuPrint (`banc:v888`, the default) and CAVE (mat 888).

> **Cite** Bates et al. (2026) *Distributed control circuits across a brain-and-cord
> connectome*, Nature,
> [10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w)
>
> **Find it** [wiki](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki) ·
> [Codex](https://codex.flywire.ai/banc) ·
> [neuroglancer](https://ng.banc.community/view)

- Annotations live in a CAVE table (`codex_annotations`) which is **long-format** —
  1.84 M rows across 32 classification systems. connecto pivots it to wide and fetches it
  in chunks, because it will not come down in one request.
- The neuPrint-backed handle still reads its annotations from CAVE, because that is where
  they are. Backend and annotation source are independent choices.
- **The neuPrint door is connectivity-only, for now.** `banc:v888` hosts no skeleton
  store, and BANC publishes no flat volume — its only segmentation is CAVE's graphene one
  — so skeletons, meshes, cutouts and neuroglancer scenes are all absent there and raise,
  naming the CAVE door. (Left as a one-line deletion for when the server gains the
  stores.) Connectivity, synapses, annotations, somas and ROIs all work.

### `fanc` — female adult nerve cord

GridTape-TEM volume and connectome of the ventral nerve cord of an adult **female**
*Drosophila*: ~45 M synapses and ~14,600 cell bodies, with leg and wing motor neurons
mapped to the muscles they drive. The female counterpart to `manc`, but a CAVE dataset
rather than a neuPrint one — an editable segmentation with a chunkedgraph, so root IDs
move under you and `version="auto"` earns its keep.

> **Cite** Azevedo et al. (2024) *Connectomic reconstruction of a female Drosophila
> ventral nerve cord*, Nature,
> [10.1038/s41586-024-07389-x](https://doi.org/10.1038/s41586-024-07389-x) · Phelps et
> al. (2021) *Reconstruction of motor control circuits in adult Drosophila using
> automated transmission electron microscopy*, Cell,
> [10.1016/j.cell.2020.12.013](https://doi.org/10.1016/j.cell.2020.12.013) · Lesser et
> al. (2024) *Synaptic architecture of leg and wing premotor control networks in
> Drosophila*, Nature,
> [10.1038/s41586-024-07600-z](https://doi.org/10.1038/s41586-024-07600-z)
>
> **Find it** [FANC_auto_recon](https://github.com/htem/FANC_auto_recon)

- **Not public.** Needs the `FANC_edit` group: join the FANC community and have your
  Google account authorized before you generate a CAVE token. There is no public FANC
  datastack. Without it you get a 403, which connecto reports as a permission problem
  rather than a bad token.
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

The dataset everyone has used. Dense FIB-SEM reconstruction of much of the central brain
of a female *Drosophila*: ~25,000 typed neurons, ~20 M synapses. neuPrint,
`hemibrain:v1.2.1`.

> **Cite** Scheffer et al. (2020) *A connectome and analysis of the adult Drosophila
> central brain*, eLife, [10.7554/eLife.57443](https://doi.org/10.7554/eLife.57443) · Xu
> et al. (2017) *Enhanced FIB-SEM systems for large-volume 3D imaging*, eLife,
> [10.7554/eLife.25916](https://doi.org/10.7554/eLife.25916)
>
> **Find it** [Janelia FlyEM](https://www.janelia.org/project-team/flyem/hemibrain) ·
> [neuPrint](https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1)

- **It is a hemibrain.** One side of the brain, cropped. Expect roughly half the neurons
  of a whole-brain dataset, and expect somata to be missing — they are frequently outside
  the volume, so `soma_x/y/z` is `NaN` for many neurons.
- Has no `somaSide` field. The side is a suffix on `instance` (`DA1_lPN_R`), which
  connecto derives and normalises to `left`/`right`.
- Has no `class` column, and connecto does not invent one.
- Rich ROI hierarchy (231 ROIs, nested — use `primary=True` for the non-overlapping set).

### `malecns` — male CNS

The complete central nervous system — brain *and* ventral nerve cord — of a male
*Drosophila*: 166,691 neurons in 11,691 cell types, annotated with *fruitless*/*doublesex*
expression. The counterpart that makes synapse-resolution male-vs-female comparison
possible. neuPrint, plus optional [Clio](https://clio.janelia.org) annotations
(`pip install "connecto[clio]"`; Clio itself needs a login).

> **Cite** Berg et al. (2025) *Sexual dimorphism in the complete connectome of the
> Drosophila male central nervous system*, bioRxiv,
> [10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999) — still a
> preprint; there is no journal version yet.
>
> **Find it** [male-cns.janelia.org](https://male-cns.janelia.org/) ·
> [neuPrint](https://neuprint.janelia.org/?dataset=male-cns:v1.0)

!!! note "There is no `optic-lobe` dataset"

    There used to be. It was the optic lobes of *this* specimen, released on their own —
    so it was two names for one dataset, and the narrower one could only ever hand you
    fewer neurons under a different set of body IDs. Registering both invited a silent
    mismatch of exactly the kind connecto exists to prevent. Use `malecns` and select the
    optic-lobe ROIs.

### `manc` — male adult nerve cord

Dense FIB-SEM connectome of the complete ventral nerve cord of an adult male
*Drosophila* (~23,000 neurons), with motor, descending and ascending neurons,
hemilineages and predicted transmitters. neuPrint. The counterpart to CAVE-backed
[`fanc`](#fanc-female-adult-nerve-cord).

> **Cite** Takemura et al. (2024) *A connectome of the male Drosophila ventral nerve
> cord*, eLife, [10.7554/eLife.97769](https://doi.org/10.7554/eLife.97769) · Marin et al.
> (2024) *Systematic annotation of a complete adult male Drosophila nerve cord
> connectome*, eLife, [10.7554/eLife.97766](https://doi.org/10.7554/eLife.97766) · Cheong
> et al. (2025) *Transforming descending input into motor output*, eLife,
> [10.7554/eLife.96084](https://doi.org/10.7554/eLife.96084)
>
> **Find it** [Janelia FlyEM](https://www.janelia.org/project-team/flyem/manc-connectome) ·
> [neuPrint](https://neuprint.janelia.org/?dataset=manc:v1.2.3)

- Uses `LHS`/`RHS` for side, where hemibrain and maleCNS use `L`/`R`. connecto normalises
  both to `left`/`right`.
- `type` coalesces `type` → `systematicType` → `instance`, so neurons with only a
  systematic name still get one.

---

## Mus musculus

### `microns` — MICrONS (minnie65) public

A cubic millimetre of mouse visual cortex — ~1.4 × 0.87 × 0.84 mm of serial-section EM
spanning all six layers of VISp and neighbouring higher visual areas, with >200,000 cells
and ~0.5 billion synapses, co-registered with two-photon calcium imaging of ~75,000
neurons in the same awake animal. Backend: CAVE (`minnie65_public`).

> **Cite** The MICrONS Consortium (2025) *Functional connectomics spanning multiple areas
> of mouse visual cortex*, Nature,
> [10.1038/s41586-025-08790-w](https://doi.org/10.1038/s41586-025-08790-w) ·
> Schneider-Mizell et al. (2025) *Inhibitory specificity from a connectomic census of
> mouse visual cortex*, Nature,
> [10.1038/s41586-024-07780-8](https://doi.org/10.1038/s41586-024-07780-8)
>
> **Find it** [microns-explorer.org](https://www.microns-explorer.org/cortical-mm3) ·
> [BossDB](https://bossdb.org/project/microns-minnie)

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

A larval zebrafish connectome, on its **own neuPrint server**
(`neuprint-fish2.janelia.org`). A separate deployment — a token that works for
`neuprint.janelia.org` is *not* valid here, which is why `auth_status()` probes every
server rather than one.

> **Cite** — nothing. This is the one dataset with no publication to point at: no paper,
> no preprint, no project page names it. There *is* a published larval zebrafish
> connectome and another forthcoming one, and it would be easy to attach either; neither
> is known to be this. connecto would rather say it does not know than put the wrong
> paper in your methods section.
>
> **Find it** [neuprint-fish2.janelia.org](https://neuprint-fish2.janelia.org/)

- **Not public.** The server issues its own tokens and access appears to be granted per
  account. Ask Janelia.
- Unversioned: the dataset string is just `fish2`, with no `:vN` suffix. connecto handles
  both forms.

---

## Aedes aegypti

### `aedes` — mosquito brain

An EM volume of the brain of the yellow fever mosquito, from the Wei-Chung Lee lab — the
substrate for an in-progress whole-brain connectome (a Wellcome Discovery Award, with
Jefferis, Marin and Younger). CAVE (`wclee_aedes_brain`).

> **Cite** — nothing yet. The Lee lab has a published mosquito preprint, but its data
> availability statement describes a *different, partial* volume (the posterior antennal
> lobes, served over CATMAID) and never mentions this datastack. So connecto cites none
> rather than the wrong one.
>
> **Find it**
> [project announcement](https://flyconnecto.me/2025/05/07/new-project-an-aedes-aegypti-brain-connectome/)

- **Not public.** The datastack is undocumented and appears to be restricted to project
  members. Ask the Lee lab or the whole-brain connectome project.

**Its annotations come from FlyTable, not CAVE.** The datastack ships a synapse table
and a nucleus table and nothing else — no column that means "type" or "side". The
project keeps its cell typing in FlyTable, the lab's SeaTable database: the `aedes_main`
table of the `aedes` base, keyed by root ID. So connecto reads annotations from there:

```python
aedes = cn.Aedes()

aedes.connectivity.edges(root_ids)   # from CAVE
aedes.ids("class:KC")                # from FlyTable — Kenyon cells
aedes.ids("superclass:visual_projection")
aedes.annotations.get().columns      # type, side, class, nt, status, + every raw column
```

FlyTable is lab-internal: it needs a `SEATABLE_TOKEN` on top of CAVE access, and there
is no public alternative. Because it is the *only* source, `annotations="auto"` still
resolves to it rather than giving up — and if you have no token, the fetch says so
plainly rather than handing back an empty frame:

```python
aedes.ids("class:KC")                # ConnectoAuthError: No SeaTable credentials found.
                                     #   Set one with: cn.set_token("seatable", "<token>")
```

`aedes_main` carries both a fine `class` (KC, CX, ALSN, …) and a coarse `superclass`
(cb_intrinsic, cb_sensory, …); the fine one is the canonical `class`, the coarse one
falls in behind it and both stay queryable. Somas still come from the nucleus table in
nanometres (`aedes.somas`), not from the table's voxel-space `soma_xyz` column, which
passes through untouched.

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
