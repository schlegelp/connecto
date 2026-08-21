---
title: Changelog
---

# Changelog

All notable changes to `connecto`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Below `1.0` a minor bump can still change behaviour you were relying on; those changes
are listed under **Changed** rather than left for you to find.

## [0.2.0] - 2026-08-21

### Added

- `connecto.precomputed`: an in-house reader for neuroglancer precomputed and graphene
  volumes over plain HTTPS. Sharded containers (both hash functions, delta-encoded
  minishard indices, compressed morton codes, per-shard index caching),
  `compressed_segmentation`, multi-resolution draco meshes, and graphene - watershed
  reads, label decoding, and meshes assembled from static shards plus the fragments
  re-meshed since a proofreader last touched the neuron, with the seams welded.
  Read-only by design: no writing, no image codecs, no cloud SDK credentials.
- Per-synapse neurotransmitters for BANC, male-CNS and MANC through neuPrint. Their
  `Synapse` nodes carry the eight-class probability vector, so `Cap.NT_PER_SYNAPSE` is
  no longer denied backend-wide - `synapses(transmitters=True)` and `transmitters()`
  work on all three.
- `BackendSpec.nt_columns` and `BackendSpec.nt_table` declare *where* transmitters live
  for a given (dataset, backend). A dataset that claims `NT_PER_SYNAPSE` without either
  is now rejected at import rather than returning a frame with no `nt` column.
- Synapse frames carry a transmitter provenance stamp in
  `df.attrs["connecto"]["transmitters"]`. BANC's two doors serve different runs of the
  same model - they agree on the neuron-level call but on only ~57% of shared synapses -
  so a frame now says which run it holds.
- `nt_source` column on neuron-level annotations, naming the column each value came
  from. `known_nt` is somebody's immunostaining, `top_nt` is a CNN's argmax; coalescing
  them silently let "this neuron is GABAergic" mean either.
- `AnnotationSource.fields`: per-source column spellings, and a way for a source to
  declare that it has not got a field at all.
- neuPrint annotation sources for FlyWire and BANC. `fetch_neuprint` now borrows the
  spec's neuPrint door instead of assuming the query backend *is* neuPrint, so
  annotation source and backend are a genuine cross-product:
  `FlyWire(backend="cave", annotations="neuprint")` is a sentence, not a crash.
- `DracoPy` as a direct dependency (draco has no reasonable pure-Python fallback), and
  `compressed-segmentation` in the `voxels` extra - about 25x quicker per chunk than the
  numpy decoder in `connecto.precomputed.codecs`, which stays exact and is used
  otherwise.
- The capability tables in the README and on the docs landing page are now checked
  against the registry, per cell and per door, so a table that says it is generated has
  to prove it. `NT` is split into `NT/syn` (the capability) and `NT/neuron` (an
  annotation column, marked dense / sparse / none), and the `vox` column was added.

### Changed

- **FlyWire and BANC default to their neuPrint annotations.** For BANC the mirror is the
  better table almost everywhere: 153,984 neurons with a type against codex's 118,446,
  154,830 with a transmitter against 115,601, plus somas and status codex has not got.
  Both datasets' previous sources are one `annotations=` argument away.
- BANC's neuPrint source declares `side` absent - it covers 8,153 bodies against codex's
  158,250. `ids("PFNd", side="left")` now refuses and names the source that can answer,
  instead of returning 0 where the truth is 18.
- Annotation field precedence is stated in one resolver, in one order: spec, then
  source, then the handle's `fields=`, then the call's. The handle's `fields=` is the
  escape hatch `resolve_criteria` names by name, so it now outranks the source.
- `Annotations.fields` reports the priorities in force for the source the handle will
  actually read, not `spec.fields`.
- male-CNS's neuron-level `nt` prefers `consensusNt` over `predictedNt`.
- neuPrint meshes are read from the published bucket directly rather than through navis,
  and mesh fetching is threaded instead of one neuron at a time.
- `PrecomputedMeta.resolution` no longer rounds to int. FANC's mip 0 is 17.2 x 17.2 x 45
  nm, and rounding to 17 is a 1.2% error - a hundred voxels at the far edge of the
  volume, far enough to land a point lookup inside the neighbouring neuron.
- Detecting list-valued annotation columns only scans object columns and stops at the
  first hit: 0.9 s -> 0.3 s on a 175k x 50 frame, which two datasets now read by default.
- `borrow(ds, kind)` in `connecto.sources` replaces two hand-rolled copies of "reach the
  dataset's other backend". The CAVE side now gets the cached client, the real version
  rules and connecto's error translation rather than its own token lookup and a version
  guess.

### Fixed

- neuPrint `synapses()` and `synapse_counts()` silently undercounted, by up to 5x: the
  unconstrained side was passed as `None`, which neuprint-python turns into
  `:Neuron`-only criteria, dropping every `:Segment` fragment partner (hemibrain
  2,018 -> 4,903; male-CNS 1,186 -> 4,491; BANC 187 -> 994; MANC 3,190 -> 8,044). They
  now match `edges()` exactly.
- `soma_x/y/z` were all-null in neuPrint annotation frames on hemibrain, male-CNS, MANC
  and fish2. `somaLocation` comes back as a raw Neo4j point on the unfiltered query, not
  the list `fetch_neurons` hands back for a body-ID list.
- FlyWire's neuPrint door claimed `NT_PER_SYNAPSE` without having the columns behind it,
  so `transmitters=True` returned a frame with no `nt` column and no error. It now denies
  the capability on its own `BackendSpec`.
- A source's "declared empty" field rename runs *after* `spec.derive`, so a suppressed
  column is no longer recreated from a regex two lines later.
- The "another source has this column" suggestion carries the caller's `backend=`
  through; annotation source and backend are independent axes, and it was resetting one.
- `get_dataset("banc", fields={"side": (...)})` was silently overruled by the annotation
  source that declared `side` absent.

### Removed

- `cloud-volume` is no longer a runtime dependency. Reading is a much smaller job than
  reading *and writing to* half a dozen cloud backends, and paying for the latter cost
  56 packages and ~78 MB of a default install (boto3, the Google Cloud SDK, gevent,
  protobuf and the rest). It stays a *dev* dependency: `tests/test_precomputed.py` reads
  the same cutouts and meshes through both readers and asserts they agree, so
  cloud-volume is the oracle the new reader is checked against.

## [0.1.1] - 2026-07-20

### Added

- `ds.voxels`: sparse volumes - every voxel belonging to a neuron - as
  `navis.VoxelNeuron`s, or as `raw` `(N, 3)` arrays or `rle` runs, in voxel or nm
  coordinates. `ds.voxels.scales()` lists the pyramid levels a dataset actually serves.
- `Cap.VOXELS`, kept separate from `SEGMENTATION` because "there is a segmentation
  volume" and "you can get one neuron's voxels out of it without reading the whole
  brain" are different claims. DVID keeps a live per-body index and answers in one
  request; a chunkedgraph keeps none, so the same question degrades to reading dense
  blocks and masking them - 100-1000x more voxels touched than kept.
- `ds.voxels.estimate()`, and a refusal rather than an apparent hang when a read would
  transfer billions of voxels. `voxels.get()` also picks a sensible default scale, since
  scale 0 for a hemibrain neuron is 1.17 billion voxels.
- `SparseVolSource`, for datasets served by an external per-body index - with the scales
  it really serves and a per-axis `downsample` factor, because fly pyramids routinely
  halve X and Y while leaving Z alone.
- DVID servers and nodes are discovered at runtime from neuPrint and clio metadata, so
  nothing is hard-coded and the node always matches the snapshot the rest of the query
  came from.
- `NoSuchBodyError`, which is also a `KeyError` so `except KeyError` around a per-neuron
  loop keeps working.
- The `voxels` extra, installing `fastremap` to speed up masking a dense cutout down to
  one root. `connecto.voxels.pcg` falls back to numpy without it.

## [0.1.0] - 2026-07-16

Initial release: one interface over CAVE and neuPrint.

### Added

- Dataset handles for FlyWire, Hemibrain, BANC, FANC, MANC, male-CNS, MICrONS, Aedes and
  fish2, plus generic `CAVE` and `NeuPrint` constructors and `register()` for pointing
  connecto at any datastack or neuPrint server. Handles are immutable; `ds.at(version)`
  gives you another one.
- Namespaces with normalised output: `annotations`, `connectivity`, `skeletons`,
  `meshes`, `rois`, `somas`, `viz`, and - where the backend has them - `segmentation`,
  `proofreading` and `l2`. Edges are always `pre, post, weight` as `int64, int64,
  int32`, positions are always nanometres, side is always `left`/`right`/`center`, and
  skeletons and meshes are always navis neurons.
- The capability model: `Cap`, `DatasetSpec`/`BackendSpec`, `ds.supports()` and
  `capability_matrix()`. A capability belongs to a *(dataset, backend)* pair, so
  `FlyWire(backend="neuprint")` no longer claims a chunkedgraph it cannot reach. Nothing
  degrades silently - an explicit request a dataset cannot honour raises
  `CapabilityError`, and whole namespaces are absent rather than broken, so `hasattr`
  works as feature detection.
- `NeuronCriteria` and the mini-language it desugars from, so the same selection reads
  the same on every dataset.
- Version handling: materialization versions and neuPrint dataset tags behind one
  `Version`, `ds.versions()`, `ds.find_version()`, and an error rather than a guess when
  a request is ambiguous.
- Annotation sources as a separate axis from the backend - CAVE tables, neuPrint,
  GitHub TSVs, SeaTable/FlyTable and clio - with per-source freshness tokens.
- An on-disk cache under `~/.connecto/cache` whose key includes the materialization, so
  it can never serve you the wrong version; live sources join a freshness token to the
  key and keep one entry rather than one per edit.
- Neuroglancer scene building via `ds.viz`.
- Token handling (`set_token`, `get_token`, `auth_status`) and server status helpers
  (`server_status`, `wait_until_available`), which wait only when you ask - connecto
  never silently retries a 503.
- `ds.cite()`, because connecto knows exactly which dataset produced the numbers in your
  figure.

[0.2.0]: https://github.com/schlegelp/connecto/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/schlegelp/connecto/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/schlegelp/connecto/releases/tag/v0.1.0
