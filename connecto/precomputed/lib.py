"""Small value types shared by the precomputed readers.

``Bbox`` is deliberately API-compatible with ``cloudvolume.lib.Bbox`` for the
handful of operations connecto uses - ``minpt``/``maxpt``, ``clamp``, ``subvoxel``,
``astype`` - because :mod:`connecto.voxels.pcg` was written against that shape and
the geometry there is subtle enough to be worth not re-deriving. Only those, and
what they need: this is not a general geometry type.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Bbox"]


class Bbox:
    """A half-open 3D voxel box: ``minpt`` inclusive, ``maxpt`` exclusive."""

    __slots__ = ("minpt", "maxpt")

    def __init__(self, minpt, maxpt, dtype=None):
        self.minpt = np.asarray(minpt, dtype=dtype if dtype else None).reshape(3)
        self.maxpt = np.asarray(maxpt, dtype=dtype if dtype else None).reshape(3)

    # ---------------------------------------------------------------- geometry

    @property
    def size3(self) -> np.ndarray:
        return self.maxpt - self.minpt

    @property
    def volume(self) -> int:
        return int(np.prod(np.maximum(self.size3, 0).astype(np.int64)))

    def subvoxel(self) -> bool:
        """True if the box encloses no whole voxel - i.e. it is empty."""
        return bool(np.any(self.size3 <= 0))

    def contains_bbox(self, other: Bbox) -> bool:
        return bool(np.all(other.minpt >= self.minpt) and np.all(other.maxpt <= self.maxpt))

    def astype(self, dtype) -> Bbox:
        return Bbox(self.minpt.astype(dtype), self.maxpt.astype(dtype))

    @classmethod
    def clamp(cls, box: Bbox, limit: Bbox) -> Bbox:
        """`box` cropped to `limit`. Empty (subvoxel) if they do not overlap."""
        return cls(
            np.maximum(box.minpt, limit.minpt),
            np.minimum(box.maxpt, limit.maxpt),
        )

    # ------------------------------------------------------------------ dunders

    def __eq__(self, other) -> bool:
        return isinstance(other, Bbox) and bool(
            np.array_equal(self.minpt, other.minpt) and np.array_equal(self.maxpt, other.maxpt)
        )

    def __hash__(self) -> int:
        return hash((tuple(self.minpt.tolist()), tuple(self.maxpt.tolist())))

    def __repr__(self) -> str:
        return f"Bbox({self.minpt.tolist()}, {self.maxpt.tolist()})"
