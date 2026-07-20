# Namespaces

Every dataset exposes its functionality through namespaces. Which ones exist depends on
what the dataset can actually do — `ds.segmentation` is simply **absent** on neuPrint,
and `hasattr(ds, "segmentation")` is `False`. See [Capabilities](../guides/capabilities.md).

Every method below takes the same neuron query (see [Selecting neurons](criteria.md)) and
a `version=` override.

## Annotations

::: connecto.core.namespaces.Annotations

## Connectivity

::: connecto.core.namespaces.Connectivity

## Skeletons

::: connecto.core.namespaces.Skeletons

## Meshes

::: connecto.core.namespaces.Meshes

## Voxels

Sparse volumes: every voxel belonging to a neuron, as an `(N, 3)` array.

The one namespace whose cost varies by three orders of magnitude between datasets, so
it is also the one that talks about cost. hemibrain, maleCNS, MANC and fish2 are backed
by DVID, which keeps a live per-body index and answers in a single request; aedes has a
lookup service that does the same. FlyWire, BANC, FANC and MICrONS have a chunkedgraph,
which keeps **no** such index — so the same question means reading dense blocks and
masking them, touching thousands of voxels for every one it keeps. `estimate()` tells
you which you are in for before you commit.

`scale=` never defaults to 0. A hemibrain neuron at scale 0 is 1.17 billion voxels; on
the CAVE route the equivalent request is refused outright rather than left to look like
a hang.

::: connecto.core.namespaces.Voxels

## ROIs

::: connecto.core.namespaces.ROIs

## Somas

::: connecto.core.namespaces.Somas

## Visualisation

See the [Neuroglancer guide](../guides/neuroglancer.md) for colouring, groups and layers.

::: connecto.core.namespaces.Viz

::: connecto.viz.neuroglancer.construct_scene

::: connecto.viz.neuroglancer.add_annotation_layer

::: connecto.viz.neuroglancer.add_skeleton_layer

::: connecto.viz.neuroglancer.build_url

::: connecto.viz.neuroglancer.encode_url

::: connecto.viz.neuroglancer.decode_url

## Segmentation

Present wherever the dataset declares `Cap.SEGMENTATION` — which includes neuPrint,
whose flat `precomputed://` volumes read as well as CAVE's graphene ones. The methods
that need a *chunkedgraph* (supervoxels, `update_ids`, root-ID history) additionally
require `Cap.CHUNKEDGRAPH` and raise on a flat volume rather than inventing an answer.

Coordinates are nanometres, like everywhere else in connecto; pass `units="voxel"` if
yours are not.

::: connecto.core.segmentation.Segmentation

## Proofreading (CAVE only)

::: connecto.backends.cave.proofreading.Proofreading

## L2 cache (CAVE only, and not every datastack)

Present only where the dataset declares `Cap.L2CACHE`. Cheap skeletons and dotprops
straight from the chunkedgraph's level-2 summaries — see
[Morphology](../tutorials/morphology.md#the-l2-cache-cheap-skeletons-and-dotprops).

::: connecto.backends.cave.l2.L2
