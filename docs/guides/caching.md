# Caching

## What is cached automatically

**Annotation tables.** They are fetched whole, they are needed by nearly every query
(resolving `"DA1_lPN"` to IDs means searching them), and they change only when the
dataset does. So the first call downloads, and every call after that reads from disk.

That is the only thing cached without being asked.

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

Frames are stored as Parquet, which preserves dtypes — a cache that round-tripped
through CSV would give you back `float64` root IDs, and you would be back to the very
bug the `caveclient>=8.0` pin exists to avoid.

## Meshes

Meshes are cached by `cloud-volume`, in its own directory, not by connecto. They are
large — one FlyWire neuron is around 33 MB at full resolution — so if you are hunting for
disk space, look there as well as at `cn.cache.size()`.

## The rule

The cache is keyed on `(dataset, backend, version, query)`. If any of those differ, you
get a fresh fetch. If they are identical, the answer genuinely cannot have changed —
because a materialization is, by definition, frozen.

That is why caching a *live* query is not possible, and asking for it raises rather than
silently handing you something stale.
