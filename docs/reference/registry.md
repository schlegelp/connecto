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

## Provenance

Who made the dataset, and whether you may have it. See
[Who made it, and may you have it?](../guides/datasets.md#who-made-it-and-may-you-have-it).

::: connecto.core.spec.Publication

## Capabilities

A capability belongs to a *(dataset, backend)* pair, not to a dataset —
`DatasetSpec.capabilities_for` is the only one to ask. `BACKEND_LIMITS` says what a
backend can never do; `BackendSpec.missing_capabilities` says what one server happens not
to host. See [Capabilities](../guides/capabilities.md#a-capability-belongs-to-a-door-not-to-a-dataset).

::: connecto.core.spec.Cap
