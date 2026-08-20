"""The ``info`` file, and the arithmetic that hangs off it.

Every question connecto asks about a volume's geometry - what is the resolution at
scale 4, which chunks does this box touch, where does the volume end - is answered
from this JSON and nothing else. No network, no state.

Scale identifiers follow cloud-volume's convention, because connecto's callers
already speak it: an ``int`` is an index into ``scales``, a ``str`` is a scale key
(``"16_16_40"``), and a 3-vector is a resolution to look up.
"""

from __future__ import annotations

import numpy as np

from .lib import Bbox
from .sharding import ShardingSpec

__all__ = ["PrecomputedMeta"]


class PrecomputedMeta:
    """Geometry and encoding of one precomputed volume."""

    def __init__(self, info: dict, path: str = ""):
        self.info = info
        self.path = path

    # ------------------------------------------------------------------- basics

    @property
    def scales(self) -> list:
        return self.info["scales"]

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    @property
    def available_mips(self) -> range:
        """The scale indices this volume defines."""
        return range(len(self.scales))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.info["data_type"])

    @property
    def num_channels(self) -> int:
        return int(self.info.get("num_channels", 1))

    @property
    def layer_type(self) -> str:
        return self.info.get("type", "segmentation")

    @property
    def mesh_path(self) -> str | None:
        return self.info.get("mesh")

    # -------------------------------------------------------------- scale lookup

    def to_mip(self, mip) -> int:
        """Resolve an int / scale key / resolution triple to a scale index."""
        if isinstance(mip, (int, np.integer)):
            mip = int(mip)
            if mip < 0:
                mip += self.num_scales
            if not 0 <= mip < self.num_scales:
                raise ValueError(
                    f"Scale {mip} out of range; {self.num_scales} available "
                    f"({', '.join(s['key'] for s in self.scales)})."
                )
            return mip

        if isinstance(mip, str):
            for i, scale in enumerate(self.scales):
                if scale["key"] == mip:
                    return i
            raise ValueError(
                f"No scale keyed {mip!r}. Available: "
                f"{', '.join(s['key'] for s in self.scales)}."
            )

        want = np.asarray(mip, dtype="float64").reshape(3)
        for i, scale in enumerate(self.scales):
            if np.array_equal(np.asarray(scale["resolution"], dtype="float64"), want):
                return i
        raise ValueError(
            f"No scale at resolution {tuple(want)}. Available: "
            f"{[tuple(s['resolution']) for s in self.scales]}."
        )

    def scale(self, mip=0) -> dict:
        return self.scales[self.to_mip(mip)]

    def key(self, mip=0) -> str:
        return self.scale(mip)["key"]

    # ---------------------------------------------------------------- geometry

    def resolution(self, mip=0) -> np.ndarray:
        """Nanometres per voxel. **Not** necessarily integral.

        FANC's mip-0 is 17.2 x 17.2 x 45, and rounding that to 17 is a 1.2% error -
        which at the far edge of the volume is a hundred voxels, far enough to land a
        point lookup inside the neighbouring neuron. So the info's own dtype is kept,
        unlike the voxel counts below, which really are integers.
        """
        return np.asarray(self.scale(mip)["resolution"])

    def chunk_size(self, mip=0) -> np.ndarray:
        return np.asarray(self.scale(mip)["chunk_sizes"][0], dtype=np.int64)

    def voxel_offset(self, mip=0) -> np.ndarray:
        return np.asarray(self.scale(mip).get("voxel_offset", [0, 0, 0]), dtype=np.int64)

    def volume_size(self, mip=0) -> np.ndarray:
        return np.asarray(self.scale(mip)["size"], dtype=np.int64)

    def bounds(self, mip=0) -> Bbox:
        offset = self.voxel_offset(mip)
        return Bbox(offset, offset + self.volume_size(mip))

    def grid_size(self, mip=0) -> np.ndarray:
        """Chunks along each axis. What the compressed morton code is sized by."""
        size = self.volume_size(mip)
        chunk = self.chunk_size(mip)
        return np.ceil(size / chunk).astype(np.int64)

    def encoding(self, mip=0) -> str:
        return self.scale(mip).get("encoding", "raw")

    def compressed_segmentation_block_size(self, mip=0):
        return self.scale(mip).get("compressed_segmentation_block_size")

    def sharding(self, mip=0) -> ShardingSpec | None:
        spec = self.scale(mip).get("sharding")
        return ShardingSpec.from_dict(spec) if spec else None

    # ------------------------------------------------------------- conversions

    def downsample_ratio(self, mip, to_mip) -> np.ndarray:
        """How much coarser `to_mip` is than `mip`, per axis."""
        return self.resolution(to_mip).astype("float64") / self.resolution(mip).astype("float64")

    def bbox_to_mip(self, bbox: Bbox, mip, to_mip) -> Bbox:
        """Re-express a box at another scale, rounding *outward*.

        Outward matters: a box rounded inward silently drops the voxels at its
        edge, and on the sparse-volume path those are exactly the ones that make a
        neuron look chopped off at chunk boundaries.
        """
        ratio = self.downsample_ratio(mip, to_mip)
        return Bbox(
            np.floor(np.asarray(bbox.minpt, dtype="float64") / ratio).astype(np.int64),
            np.ceil(np.asarray(bbox.maxpt, dtype="float64") / ratio).astype(np.int64),
        )

    def __repr__(self) -> str:
        return (
            f"PrecomputedMeta({self.path!r}, {self.num_scales} scales, "
            f"{self.dtype}, {self.layer_type})"
        )
