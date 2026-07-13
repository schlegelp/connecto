# Registry and specs

A `DatasetSpec` is **pure data** — everything connecto knows about a dataset, with no
code. Registering one makes it available by name everywhere. See
[Adding a dataset](../guides/custom-datasets.md).

## Registry

::: connecto.core.registry
    options:
      members:
        - register
        - get_dataset
        - get_spec
        - list_datasets
        - capability_matrix

## Specs

::: connecto.core.spec.DatasetSpec

::: connecto.core.spec.BackendSpec

::: connecto.core.spec.AnnotationSource

## Capabilities

::: connecto.core.spec.Cap
