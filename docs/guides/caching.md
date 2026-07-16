# Caching

## What is cached automatically

**Annotation tables.** They are fetched whole, they are needed by nearly every query
(resolving `"DA1_lPN"` to IDs means searching them), and re-fetching one is slow. So the
first call downloads and every call after reads from disk — within one handle from an
in-process memo (a handle is pinned to a version, so it reads a source once and reuses
it), and across runs from `~/.connecto/cache`.

That is the only thing cached without being asked. When the source is **live** — FlyTable,
or a neuPrint instance still being curated — connecto re-validates it cheaply so an edit
upstream is never served stale; see [Live sources](#live-sources).

Because that decision is invisible otherwise, `annotations.get()` says it out loud —
which source, and whether the data came from the remote server, the local cache, a cache
refresh, or the session memo:

```pycon
>>> fw.annotations.get(source="flytable")
FlyWire (FAFB) public release: 'flytable' annotations [FlyTable main.info,optic_lobes.optic] - downloading from remote (caching for next time)...
>>> fw.annotations.get(source="flytable")          # a new handle, next day
FlyWire (FAFB) public release: 'flytable' annotations [FlyTable main.info,optic_lobes.optic] - up to date in local cache (re-validated against the source).
```

Pass `verbose=False` to silence it. Only a direct `annotations.get()` speaks; the same
table loaded behind the scenes to resolve `ds.ids("DA1_lPN")` stays quiet.

## What is cached on request

Edge fetches, which can be slow:

```python
fw.connectivity.edges(big_id_list, cache=True)   # slow once, instant after
```

Not the default, because an edge query is parameterised by six things (the IDs, both
directions, `by_roi`, `rois`, `min_weight`) and caching every variant of every query
anybody ever runs would fill a disk quietly. You opt in for the ones you will repeat.

## The cache key cannot betray you

```python
CacheEntry(ds.name, version, "edges", ids, upstream, downstream, by_roi, rois, min_weight)
```

The **version is part of the key**. This is the whole reason to trust it.

A cache that keyed only on the query would happily serve you materialization 630's edges
for a handle pinned at 783 — and you would never find out, because the frame would look
completely normal. Version-in-the-key makes that unrepresentable: a different snapshot is
a different entry.

The same applies across datasets and backends. `cn.BANC(backend="cave")` and
`cn.BANC(backend="neuprint")` do not share cache entries, even though they describe the
same snapshot, because their raw frames differ.

## Live sources

Version-in-the-key is enough only when the source is *frozen at that version*. Most are:
a CAVE annotation table is materialization-scoped, and FlyWire's public GitHub TSV is a
per-release file. For those, the version is the whole story and the cache is permanent.

Some annotation sources are not frozen, and this is the one place a version key is not
enough:

- **FlyTable / SeaTable** is a live curation database, edited daily, with no version of
  its own. FlyWire is pinned at materialization 783 *forever*, so its `(dataset, version)`
  never turns over — a naïve cache would serve the first snapshot until you cleared it by
  hand.
- **neuPrint** instances like maleCNS and fish2 are re-curated in place, under the *same*
  version tag.

So for these, connecto folds a **freshness token** into the key — one the source cheaply
reports and that changes exactly when its content does: SeaTable's `COUNT(*)` and
`MAX(_mtime)` (a delete drops the count, any edit bumps the mtime), neuPrint's
`lastDatabaseEdit`. The token is checked once per handle. A frozen dataset yields a
constant token and caches forever; a live one turns the key over the moment it is edited,
and the superseded entry is deleted so it does not accumulate. The upshot: **you never
have to `clear()` to see an upstream edit** — a new handle picks it up on its own.

If a freshness probe can't run (no SeaTable token, say), connecto falls back to the
version-scoped key rather than failing the query — no worse than before, and loud when it
matters because the *fetch* itself would then raise a credential error.

## Managing it

```python
import connecto as cn

cn.cache.cache_dir()          # where it lives
cn.cache.size()               # bytes
cn.cache.clear()              # all of it
cn.cache.clear("flywire")     # just one dataset
```

The default location is `~/.connecto/cache`. Override it:

```bash
export CONNECTO_CACHE_DIR=/scratch/me/connecto
export CONNECTO_NO_CACHE=1     # disable entirely
```

Frames are stored as Feather, which preserves dtypes — a cache that round-tripped
through CSV would give you back `float64` root IDs, and you would be back to the very
bug the `caveclient>=8.0` pin exists to avoid. The few frames Arrow cannot hold — a
SeaTable table with an object column that mixes `int` and `str`, say — fall back to
pickle rather than silently not caching at all; either way a cache hit is exactly what
a miss would have returned.

## Meshes

Meshes are cached by `cloud-volume`, in its own directory, not by connecto. They are
large — one FlyWire neuron is around 33 MB at full resolution — so if you are hunting for
disk space, look there as well as at `cn.cache.size()`.

## The rule

The cache is keyed on `(dataset, backend, version, query)` — plus a freshness token for
the annotation sources that are not frozen at a version (see [Live sources](#live-sources)).
If any of those differ, you get a fresh fetch. If they are identical, the answer genuinely
cannot have changed — because either the materialization is frozen, or the token would have
moved.

That is also why caching a *live connectivity query* is not possible, and asking for it
raises rather than silently handing you something stale.
