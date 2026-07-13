"""Backends.

A backend is ten ``_fetch_*`` methods. That is the whole porting surface: adding
a third one (a local edge dump, CATMAID, ...) means writing those ten and nothing
else.
"""

from __future__ import annotations

from ..core.spec import DatasetSpec

__all__ = ["build", "BACKENDS"]


def _cave():
    from .cave import CAVEDataset

    return CAVEDataset


def _neuprint():
    from .neuprint import NeuPrintDataset

    return NeuPrintDataset


BACKENDS = {"cave": _cave, "neuprint": _neuprint}


def build(spec: DatasetSpec, *, backend: str | None = None, **kwargs):
    """Instantiate the right Dataset subclass for a spec."""
    b = spec.backend(backend)
    cls = BACKENDS[b.kind]()
    return cls(spec, backend=b.kind, **kwargs)
