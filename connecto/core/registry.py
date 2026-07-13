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
    """Every registered dataset, and which backends serve it."""
    rows = [
        {
            "name": s.name,
            "label": s.label,
            "species": s.species,
            "backends": ", ".join(s.backend_kinds),
            "annotations": ", ".join(a.name for a in s.annotation_sources),
        }
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]
    return pd.DataFrame(rows)


def capability_matrix() -> pd.DataFrame:
    """Datasets x capabilities.

    The executable form of the "no silent degradation" promise: if a cell is
    False, the corresponding call raises rather than quietly returning something
    subtly wrong.
    """
    caps = list(Cap)
    data = {
        s.name: [c in s.capabilities for c in caps]
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    }
    return pd.DataFrame(data, index=[str(c) for c in caps]).T
