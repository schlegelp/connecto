# Connecto :crystal_ball:

Unified Interface for Querying Connectomes.

**📖 [Documentation](https://schlegelp.github.io/connecto/)** — tutorials, guides and API reference.

![logo](docs/_static/logo.png)

Connectomic datasets each speak their own dialect. CAVE has materialization
versions, root IDs that change under you, and per-datastack table names. neuPrint
has immutable body IDs, Cyphers, and dataset strings like `hemibrain:v1.2.1`. To make life even harder,
every dataset has its own idea of what "type" or "side" means.

`connecto` is one interface over all of it.

```python
import connecto as co

fw = co.FlyWire()      # neuPrint (pass backend="cave" for the CAVE door)
hb = co.Hemibrain()    # neuPrint

fw.connectivity.edges("DA1_lPN")     # pre, post, weight
hb.connectivity.edges("DA1_lPN")     # the same frame — different backend, different conventions
```

## The two things it promises

**1. The same query returns the same shape.** Edges are always `pre, post, weight`
as `int64, int64, int32`. Positions are always nanometres. Side is always
`left`/`right`/`center`. Skeletons, meshes and voxels are always `navis` neurons.

This is checkable rather than aspirational: BANC is served by *both* backends at
the same snapshot (CAVE materialization 888 ≡ neuPrint `banc:v888`, and its
neuPrint body IDs are valid CAVE root IDs), so the test suite asserts the two
return byte-identical frames.

```python
cave     = co.BANC(backend="cave").connectivity.edges(ids)
neuprint = co.BANC(backend="neuprint").connectivity.edges(ids)
pd.testing.assert_frame_equal(cave, neuprint)   # passes
```

**2. Nothing degrades silently.** If a dataset cannot do what you asked, it says
so. It does not quietly hand you an unfiltered result.

```python
fw.connectivity.synapses(x, min_score=100)    # fine — FlyWire has cleft scores
mic.connectivity.synapses(x, min_score=100)   # CapabilityError, not a shrug
```

Defaults never raise; only an *explicit* request for something the dataset lacks
does. Whole namespaces are absent rather than broken, so feature detection works:

```python
hasattr(co.FlyWire(backend="cave"), "proofreading")   # True  — FlyWire is still being edited
hasattr(hb, "proofreading")                           # False — hemibrain is frozen
```

A capability belongs to a *(dataset, backend)* pair, not to a dataset. FlyWire has a
chunkedgraph; you cannot reach it through neuPrint — so the neuPrint handle does not
claim it, and the refusal names the door that can:

```python
co.FlyWire().connectivity.synapses(x, transmitters=True)
```

```
CapabilityError: FlyWire (FAFB) public release (neuprint) does not support
`transmitters` (no nt_per_synapse). Drop the argument, or use a dataset that has it.
The cave backend does: co.get_dataset("flywire", backend="cave").
```

And where a word means two things, there are two capabilities. Both FlyWire and
hemibrain have a segmentation volume you can query; only FlyWire has a *chunkedgraph*
underneath it, so only FlyWire's IDs can go stale:

```python
hb.segmentation.locs_to_segments(tbars)   # fine — what body is at this point?
hb.segmentation.update_ids([1734350788])  # CapabilityError: no chunkedgraph
```

The same principle covers *cost*, not just presence. `ds.voxels` gives you a neuron's
sparse volume everywhere, but hemibrain is backed by DVID, which keeps a per-body index
and answers in one request, while FlyWire's chunkedgraph keeps none and has to read
dense blocks and mask them. Both work; one is three orders of magnitude dearer, so
connecto says so instead of appearing to hang:

```python
hb.voxels.get(1734350788, scale=4)          # one request, ~2s → navis.VoxelNeuron
fw.voxels.estimate(root)                    # ask before you commit
fw.voxels.get(root, scale=0)                # ValueError: would transfer 8,287,944,704 voxels
```

None of the DVID servers or nodes are hard-coded — they are parsed at runtime out of
neuPrint and clio metadata, and the node always matches the snapshot the rest of your
query came from.

## Selecting neurons

One mini-language, everywhere:

```python
ds.ids(720575940604407468)          # a root/body ID
ds.ids("DA1_lPN")                   # a type, matched across the dataset's type columns
ds.ids("/^AOTU00.*")                # a regex
ds.ids("super_class:visual_projection")   # any annotation column
ds.ids("DA1_lPN", side="left")
```

Anything you can pass to `ids()` you can pass to any query. It's a pure function —
it returns IDs and mutates nothing.

## Datasets

| dataset | backend | annot | conn | syn | scores | NT | roi-conn | rois | skel | mesh | seg | cgraph | proof | soma | live |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flywire` | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | · |
|  | cave | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · |
| `flywire-production` | **cave** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `banc` | **neuprint** | ✅ | ✅ | ✅ | · | · | ✅ | ✅ | · | · | · | · | · | ✅ | · |
|  | cave | ✅ | ✅ | ✅ | · | · | ✅ | · | ✅ | ✅ | ✅ | ✅ | · | ✅ | · |
| `fanc` | **cave** | ✅ | ✅ | ✅ | ✅ | · | · | · | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hemibrain` | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | · |
| `malecns` | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | · |
| `manc` | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | · |
| `microns` | **cave** | ✅ | ✅ | ✅ | · | · | · | · | ✅ | ✅ | ✅ | ✅ | · | ✅ | · |
| `aedes` | **cave** | ✅ | ✅ | ✅ | · | · | · | · | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ |
| `fish2` | **neuprint** | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | · | · | · | ✅ | · |

Generated by `co.capability_matrix()`. A `·` means the call raises, not that it
returns something subtly wrong.

One row per **door**. `flywire` and `banc` are each served by two backends, and the
doors are different widths — neuPrint adds ROIs and takes away the chunkedgraph — so the
capability lives on the *pair*, not on the dataset. **Bold** is the default backend, the
one `co.get_dataset(name)` gives you; `backend="cave"` gets you the other.

`aedes` is the honest case: its CAVE datastack ships a synapse table and a nucleus
table and *nothing else*, so its annotations come from FlyTable (the lab's SeaTable)
instead — lab-internal, keyed by root ID, and if you have no `SEATABLE_TOKEN` the fetch
says so plainly rather than handing back an empty frame. Its `·` is `scores`: the
synapse table records `size`, not a confidence, so `min_score=` raises.

Anything not listed works too, without writing a class:

```python
ca3  = co.CAVE("zheng_ca3", fields={"type": ("cell_type",)})
wasp = co.NeuPrint("wasp3:v0.8", server="neuprint-pre.janelia.org")
co.register(ca3.spec)       # now co.get_dataset("zheng_ca3") works everywhere
```

The spec is probed from the server. A probed dataset claims fewer capabilities and
has no `fields`, so `ds.ids("DA1_lPN")` tells you it doesn't know which column
means "type" rather than guessing.

## Versions

```python
co.FlyWire(version=783)                       # pinned
ds.connectivity.edges(ids, version="live")    # per-call override
ds.at(630)                                    # a new handle, same client

ds.connectivity.edges(ids, version="auto")    # <- the good one
```

`version="auto"` finds the newest materialization in which *all* your IDs are
jointly valid — which matters on CAVE, where root IDs change every time someone
edits a neuron. On neuPrint, body IDs are immutable so it's an identity. The same
line is correct on both.

## What it doesn't do

Data fetching only. No clustering, no matching, no connectivity vectors, no
curation policy — connecto will backfill a `type` from the columns your dataset
declares, but it will never decide that some types are "bad" and null them out.
That's analysis, and it belongs upstream (see
[`cocoa`](https://github.com/flyconnectome/cocoa)).

## Install

```bash
pip install connecto
pip install connecto[clio,flytable]"   # private annotation sources
```

## Credentials

Read from wherever they already live — `~/.cloudvolume/secrets/`,
`$NEUPRINT_APPLICATION_CREDENTIALS`, `~/.connecto/config.toml`. connecto is a
facade over your existing setup, not a competing secret store, so a token that
works today keeps working.

`auth_status()` **validates** — it asks each service who you are, rather than
merely noting that some string exists. "Present" and "works" are different claims,
and this is the function you reach for when nothing works:

```python
co.auth_status()
```

```
 service   server                      status    source                              identity       access
 cave      global.daf-apis.com         OK        ~/.cloudvolume/…/cave-secret.json   you@lab.org    flywire_train, FANC_edit, flywire
 neuprint  neuprint.janelia.org        OK        $NEUPRINT_APPLICATION_CREDENTIALS   you@lab.org    readwrite
 neuprint  neuprint-fish2.janelia.org  INVALID   $NEUPRINT_APPLICATION_CREDENTIALS   —              server rejected this token
 clio      —                           MISSING   —                                   —              —
```

Every neuPrint server gets checked, not just one — `neuprint-fish2` is a separate
deployment, so a token good for the main server isn't guaranteed good there. Pass
`validate=False` for an instant presence-only check.

When something *is* wrong, you get one `ConnectoAuthError` that names which of the
five possible sources the rejected token came from — because that's the one you
have to go and fix:

```
ConnectoAuthError: CAVE rejected your token (HTTP 401).
  [flywire_fafb_public]

The token came from: $CONNECTO_CAVE_TOKEN
It is present, but the server says it is invalid or expired.

Fix:
  1. Get a new token: https://global.daf-apis.com/auth/api/v1/create_token
  2. co.set_token("cave", "<token>")
```

A 403 is reported as a *different* problem, because it has a different fix: your
token is fine, you're just not in the right group — and a new token won't help.

```python
co.set_token("cave", "<token>")   # writes to connecto's config *and* the shared
                                  # CAVE secret file, so the two can never disagree
```

## When CAVE is down

CAVE answers `503` for **hours** while it builds a new materialization version. That
is scheduled, normal, and — until now — indistinguishable from a crash: forty lines
of stack ending in a page of nginx HTML.

The useful question at that moment is not "what is a 503". It is *"is it only CAVE, or
is it me?"* — so connecto probes every service the dataset depends on and tells you:

```
ConnectoServerError: CAVE returned 503 (Service Unavailable).

  dataset:   flywire_fafb_public
  service:   materialize

CAVE services, checked just now:
  info          OK           client setup, datastack metadata
  materialize   UNAVAILABLE  annotations, connectivity, synapses, somas, ROIs, version lookup
  chunkedgraph  OK           segmentation, edit history
  skeleton      OK           skeletons

Only materialize is down; the rest of CAVE is fine. That is exactly what a
materialization in progress looks like from outside.

CAVE serves 503 while it builds a new materialization version. That runs for
*hours*, not seconds, so a retry loop will not outlast it - and caveclient has
already retried this three times.

Next:
  co.wait_until_available("flywire")   # block until it is back
```

One service down means a materialization; *every* service `UNREACHABLE` means your
VPN, not their server. Those have different fixes, and the message says which one you
are looking at rather than making you guess. The explanation follows the evidence: if
the servers are already answering again by the time it looks, it says so and tells you
to just retry.

connecto never silently retries a 503 — no honest backoff outlasts a multi-hour
materialization, and a library that sleeps for three hours pretending to work is worse
than one that raises. Waiting is available, but you have to ask for it:

```python
co.server_status("flywire")          # what is answering, right now
co.wait_until_available("flywire")   # blocks; polls every 5 min, gives up after 6 h
```

## Tests

```bash
pytest                  # offline, fast
pytest -m network       # against the live servers; needs tokens
```

## Docs

The site is built with [Zensical](https://zensical.org). The API reference is
generated from the docstrings, so `connecto` itself has to be importable — which is
what the `docs` extra installs alongside the site generator.

```bash
uv run --extra docs zensical serve -o    # live preview at localhost:8000
uv run --extra docs zensical build       # -> site/
```

`docs/index.md` uses a custom template (`overrides/home.html`) for the landing page;
everything else is ordinary Markdown. CI builds with `--strict`, so a dead
cross-reference fails the build rather than shipping.
