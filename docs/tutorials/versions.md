# 5. Versions and root IDs

This is the tutorial that will save you an afternoon, and possibly a result.

## The problem

On CAVE, a **root ID is not a stable name for a neuron.** It is the name of a
particular *agglomeration* of supervoxels. The moment somebody merges a missing branch
onto that neuron, the agglomeration changes, and so does its ID. The old root ID does
not point at a worse version of the neuron — it points at nothing. It is retired.

So the root ID you wrote in your notebook last March may simply not exist today.

On neuPrint, none of this is true: body IDs are immutable, the dataset is frozen, and a
body ID from 2020 is the same body today. Two backends, two completely different
contracts.

connecto's job is to let you write one line that is correct on both.

## Materializations

A CAVE **materialization** is a frozen snapshot: the segmentation, plus every table,
as of one instant.

```python
import connecto as cn

fw = cn.FlyWire()
fw.versions()
```

```
[630, 783]
```

```python
fw.version
```

```
Version(783, backend='cave')
```

Every handle pins itself to one version and answers everything from it. That is what
makes a session internally consistent — you cannot accidentally join annotations from
630 onto edges from 783, because a handle only ever speaks one of them.

neuPrint's equivalent is the dataset string:

```python
cn.Hemibrain().version
```

```
Version('hemibrain:v1.2.1', backend='neuprint')
```

## Pinning

```python
cn.FlyWire(version=630)                      # pin at construction
fw.at(630)                                   # a new handle, same client
fw.connectivity.edges(ids, version=630)      # just this call
```

`at()` returns a **new** handle. It does not mutate `fw`. A mutable version setter, on
an object that also memoises an annotation frame, is the exact mechanism by which you
end up silently mixing snapshots.

## `version="auto"`, the good one

Here is the line that matters:

```python
fw.connectivity.edges(ids, version="auto")
```

**Find the newest materialization in which *all* of these IDs are jointly valid.**

Not "the newest version". Not "the version each ID was last seen in". The newest one
where the whole set works *together* — because an edge list is only meaningful if every
neuron in it is contemporaneous.

```python
fw.find_version(fw.ids("DA1_lPN"))
```

```
Version(783, backend='cave')
```

On CAVE, this scans materializations newest-first and checks validity. On neuPrint it
is an **identity** — body IDs are immutable, so every version trivially contains them.

That is the whole point. The same line is correct on both backends, and you do not have
to remember which one you are talking to.

## What goes wrong without it

Suppose you have a list of root IDs from a colleague, collected some months ago.

```python
ids = [720575940603231916, 720575940604407468, ...]
fw.connectivity.edges(ids)         # at the latest materialization
```

If half of those IDs were retired by subsequent edits, CAVE does not raise. It returns
the edges for the ones that still exist. You get a smaller edge list, no warning, and a
result that is quietly wrong — a connectivity matrix missing the neurons that happened
to have been proofread.

```python
fw.connectivity.edges(ids, version="auto")
```

This finds the newest snapshot where the whole set is valid, and answers there.

If no such snapshot exists, you get told:

```python
fw.find_version(mixed_ids)
```

```
AmbiguousVersionError: No materialization contains all 12 IDs. The newest version
containing any of them is 783 (9/12 valid). IDs not valid in 783: [...]
```

That is an error you can act on, rather than a silently truncated result.

## Bringing old IDs forward

If the IDs are genuinely stale and you want *today's* equivalents:

```python
fw.segmentation.update_ids(old_ids)
```

```
              old_id             new_id  confidence
0  720575940604407468  720575940604407468        1.00
1  720575940612345678  720575940655554321        0.97
```

This walks the chunkedgraph's lineage to find what each old neuron became. `confidence`
is the fraction of the old neuron's supervoxels that ended up in the new one — a low
value means the old neuron was split, and "the equivalent" is a judgement call you
should look at.

This is CAVE-only, obviously:

```python
hasattr(cn.Hemibrain(), "segmentation")   # False
```

## Live queries

Some CAVE stacks let you query the *current* state, ahead of any materialization:

```python
fwp = cn.FlyWire("flywire-production")
fwp.connectivity.edges(ids, version="live")
```

Live queries are slower, are not reproducible, and are not available on every dataset —
the public FlyWire release has no live endpoint at all, and asking for one raises rather
than silently giving you 783 and letting you believe it was live.

## Rules of thumb

- **Pin a version in anything you will run twice.** `cn.FlyWire(version=783)`.
- **Record it.** Every frame carries `attrs["connecto"]["version"]` and
  `version_timestamp`. Put them in your figure caption.
- **Use `version="auto"` for IDs you did not generate yourself**, e.g. from a
  collaborator or an old notebook.
- **Never cache a root ID without its version.** A root ID alone is not a neuron; a root
  ID *plus a materialization* is.

Next: [one script, two datasets](cross-dataset.md).
