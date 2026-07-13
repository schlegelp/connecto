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

## Segmentation (CAVE only)

::: connecto.backends.cave.segmentation.Segmentation

## Proofreading (CAVE only)

::: connecto.backends.cave.proofreading.Proofreading

## L2 cache (CAVE only, and not every datastack)

Present only where the dataset declares `Cap.L2CACHE`. Cheap skeletons and dotprops
straight from the chunkedgraph's level-2 summaries — see
[Morphology](../tutorials/morphology.md#the-l2-cache-cheap-skeletons-and-dotprops).

::: connecto.backends.cave.l2.L2
