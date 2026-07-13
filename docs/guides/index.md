# Guides

Reference-shaped pages on the things that are more design than procedure.

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

### [Datasets](datasets.md)

Every dataset that ships with connecto, what it is, what it can do, and its
particular idiosyncrasies.

</div>

<div class="cn-card" markdown>

### [Capabilities](capabilities.md)

Why `min_score=100` raises on MICrONS instead of being ignored, and how to write
code that adapts.

</div>

<div class="cn-card" markdown>

### [Adding a dataset](custom-datasets.md)

Point connecto at any CAVE datastack or neuPrint server, with or without writing a
class.

</div>

<div class="cn-card" markdown>

### [Caching](caching.md)

What is cached, where, and why the cache key can never serve you the wrong
materialization.

</div>

<div class="cn-card" markdown>

### [When the server is down](outages.md)

CAVE answers `503` for hours while it materializes. What that looks like, how to tell
it apart from a dead VPN, and how to wait it out.

</div>

</div>

## The one-paragraph architecture

Backends implement ten `_fetch_*` hooks that return **raw** frames. Everything you
actually touch — version resolution, capability checking, caching, column renaming,
dtype coercion, unit conversion, error translation — happens *above* them, once, in
code both backends share.

A backend never constructs a user-facing DataFrame. That is not a style preference;
it is the only thing that structurally guarantees FlyWire and hemibrain hand back the
same frame. If normalisation lived in the backends, there would be two
implementations of it, and they would drift.

```
          ds.connectivity.edges("DA1_lPN")
                      |
        [ capability gate + auth translation ]     core/dataset.py
                      |
        [ version resolution, ID resolution ]      core/criteria.py
                      |
              ds._fetch_edges(...)                 <- the ONLY backend code
                      |
        [ rename, coerce, rescale, stamp ]         core/schemas.py
                      |
              pre, post, weight
```

Adding a backend means writing ten methods. It does not mean touching any of the
normalisation.
