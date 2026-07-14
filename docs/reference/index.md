# Reference

Generated from the docstrings, so there is no second copy of the documentation to drift
out of sync with the code.

<div class="cn-grid" markdown>

<div class="cn-card" markdown>

### [Dataset](dataset.md)

The handle: versions, capabilities, `ids()`, `at()`. Plus the dataset constructors —
`FlyWire`, `Hemibrain`, `BANC`, `FANC`, `MANC`, `MICrONS`, `Aedes`, `CAVE`, `NeuPrint`.

</div>

<div class="cn-card" markdown>

### [Namespaces](namespaces.md)

`ds.annotations`, `ds.connectivity`, `ds.skeletons`, `ds.meshes`, `ds.rois`,
`ds.somas`, `ds.viz`, `ds.segmentation`, `ds.proofreading`.

</div>

<div class="cn-card" markdown>

### [Selecting neurons](criteria.md)

`NeuronCriteria` and the mini-language it desugars from.

</div>

<div class="cn-card" markdown>

### [Registry and specs](registry.md)

`DatasetSpec`, `BackendSpec`, `AnnotationSource`, `Cap`, and the registry functions.

</div>

<div class="cn-card" markdown>

### [Auth and cache](auth.md)

`auth_status()`, `set_token()`, `ConnectoAuthError`, and the cache.

</div>

</div>

## The shape of the API

```python
import connecto as cn

ds = cn.FlyWire(version=783, backend="cave", annotations="public")

# identity
ds.name, ds.label, ds.backend_kind, ds.source, ds.capabilities

# versions
ds.version, ds.versions(), ds.at(630), ds.find_version(ids)

# selection - one pure function, used by everything else
ds.ids("DA1_lPN", side="left")

# namespaces
ds.annotations.get(x)         ds.annotations.search(term)   ds.annotations.ids()
ds.connectivity.edges(x)      ds.connectivity.adjacency(x)  ds.connectivity.synapses(x)
ds.connectivity.synapse_counts(x)                           ds.connectivity.transmitters(x)
ds.skeletons.get(x)           ds.skeletons.dotprops(x)
ds.meshes.get(x)
ds.rois.list()                ds.rois.hierarchy()           ds.rois.mesh(roi)
ds.somas.get(x)
ds.viz.scene(x)               ds.viz.neuroglancer_url(x, color_by="type")

# wherever there is a segmentation volume - CAVE *and* neuPrint
ds.segmentation.locs_to_segments(locs)   ds.segmentation.neuron_to_segments(n)
ds.segmentation.get_voxels(x)            ds.segmentation.get_segmentation_cutout(bbox)

# CAVE only (a chunkedgraph: supervoxels, and IDs that change)
ds.segmentation.update_ids(x)        ds.segmentation.locs_to_supervoxels(locs)
ds.proofreading.is_proofread(x)      ds.proofreading.edit_history(x)

# CAVE, and only where the datastack has an L2 cache
ds.l2.info(x)                        ds.l2.graph(x)
ds.l2.skeleton(x)                    ds.l2.dotprops(x)
```

Every namespace method accepts the same neuron query `x`, and every one takes
`version=` to override the handle's pin for that call.

## Exceptions

| exception | when |
|---|---|
| `CapabilityError` | the dataset cannot do what you explicitly asked. Subclasses `NotImplementedError` **and** `AttributeError`, so `hasattr()` works |
| `ConnectoAuthError` | a credential problem. `.kind` is `missing` / `invalid` / `forbidden` |
| `NoSuchVersionError` | that version does not exist |
| `AmbiguousVersionError` | no single materialization contains all your IDs |
| `NoSuchDatasetError` | not in the registry |
| `MissingDependencyError` | an optional extra is not installed. Tells you the `pip install` |

All of them subclass `ConnectoError`.
