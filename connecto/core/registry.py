"""The dataset registry.

Datasets are data, so they live in a dict. Anyone can add one - including you, at
runtime, without editing connecto:

    spec = co.CAVE("zheng_ca3", fields={"type": ("cell_type",)}).spec
    co.register(spec)
    co.get_dataset("zheng_ca3")
"""

from __future__ import annotations

import pandas as pd

from ..exceptions import NoSuchDatasetError
from .spec import Cap, DatasetSpec

__all__ = ["register", "get_spec", "get_dataset", "list_datasets", "capability_matrix"]

REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec, *, overwrite: bool = True) -> DatasetSpec:
    """Add a dataset to the registry."""
    if spec.name in REGISTRY and not overwrite:
        raise ValueError(f"Dataset {spec.name!r} is already registered.")
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> DatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise NoSuchDatasetError(
            f"No dataset {name!r}. Registered: {known}."
        ) from None


def get_dataset(name: str, **kwargs):
    """Build a dataset by name, e.g. ``get_dataset("flywire", version=783)``."""
    from ..backends import build

    return build(get_spec(name), **kwargs)


def list_datasets() -> pd.DataFrame:
    """Every registered dataset, which backends serve it, and whether you can read it.

    The first backend listed is the default - the one ``get_dataset(name)`` gives
    you. ``public=False`` is not a secret; it means a fresh token is not enough, and
    ``get_spec(name).access`` says what is.
    """
    rows = [
        {
            "name": s.name,
            "label": s.label,
            "species": s.species,
            "backends": ", ".join(s.backend_kinds),
            "annotations": ", ".join(a.name for a in s.annotation_sources),
            "public": s.public,
        }
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]
    return pd.DataFrame(rows)


def capability_matrix() -> pd.DataFrame:
    """Datasets x capabilities - one row per (dataset, backend).

    The executable form of the "no silent degradation" promise: if a cell is
    False, the corresponding call raises rather than quietly returning something
    subtly wrong.

    There is a row per *door*, not per dataset, because that is where a capability
    actually lives. BANC and FlyWire are each served by both backends and the doors
    are different widths - neuPrint adds ROIs and takes away the chunkedgraph - so a
    single row per dataset could only be true by being vague. The default backend
    (what you get from ``get_dataset(name)``) comes first.
    """
    caps = list(Cap)
    rows, index = [], []
    for s in sorted(REGISTRY.values(), key=lambda s: s.name):
        for b in s.backends:
            have = s.capabilities_for(b.kind)
            rows.append({"backend": b.kind} | {str(c): c in have for c in caps})
            index.append(s.name)
    return pd.DataFrame(rows, index=index)
