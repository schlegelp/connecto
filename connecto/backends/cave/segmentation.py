"""The CAVE segmentation namespace: root IDs, supervoxels, the chunkedgraph.

CAVE-only, and that is the whole reason it is a namespace rather than sixteen
methods on the dataset: ``hb.segmentation`` simply does not exist, and
``hasattr(hb, "segmentation")`` is False.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.namespaces import _Namespace

__all__ = ["Segmentation"]


class Segmentation(_Namespace):
    """Query and update the segmentation."""

    @property
    def _cg(self):
        return self._ds.client.chunkedgraph

    def _timestamp(self, version=None):
        ds = self._ds
        v = ds._resolve_version_arg(version)
        if v.is_live:
            return None
        return ds.client.materialize.get_timestamp(v.value)

    # ------------------------------------------------------------------ lookups

    def locs_to_supervoxels(self, locs, *, coordinates: str = "voxel") -> np.ndarray:
        """Supervoxel ID at each xyz location."""
        from .points import lookup_points

        return lookup_points(self._ds, locs, coordinates=coordinates, agglomerate=False)

    def locs_to_segments(self, locs, *, coordinates: str = "voxel", version=None) -> np.ndarray:
        """Root ID at each xyz location."""
        sv = self.locs_to_supervoxels(locs, coordinates=coordinates)
        return self.supervoxels_to_roots(sv, version=version)

    def supervoxels_to_roots(self, supervoxels, *, version=None) -> np.ndarray:
        sv = np.asarray(supervoxels, dtype="int64").ravel()
        out = np.zeros(len(sv), dtype="int64")
        mask = sv != 0
        if mask.any():
            out[mask] = self._cg.get_roots(sv[mask], timestamp=self._timestamp(version))
        return out

    def roots_to_supervoxels(self, x, *, version=None) -> dict:
        ds = self._ds
        ids = ds.ids(x, version=version)
        return {int(i): np.asarray(self._cg.get_leaves(int(i), stop_layer=1)) for i in ids}

    def neuron_to_segments(self, x, *, coordinates: str = "voxel", version=None) -> pd.DataFrame:
        """Which root IDs a neuron's nodes fall into. Useful for spotting merges."""
        import navis

        neurons = navis.NeuronList(x)
        rows = []
        for n in neurons:
            nodes = n.nodes[["x", "y", "z"]].to_numpy()
            roots = self.locs_to_segments(nodes, coordinates=coordinates, version=version)
            counts = pd.Series(roots).value_counts()
            for root, count in counts.items():
                rows.append({"neuron": n.id, "root_id": int(root), "n_nodes": int(count)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ validity

    def is_latest_root(self, x, *, version=None) -> np.ndarray:
        ids = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_latest_roots(ids, timestamp=self._timestamp(version)))

    def is_valid_root(self, x) -> np.ndarray:
        ids = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_valid_nodes(ids))

    def is_valid_supervoxel(self, x) -> np.ndarray:
        sv = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_valid_nodes(sv))

    # ------------------------------------------------------------------- updates

    def update_ids(self, x, *, version=None, supervoxels=None) -> pd.DataFrame:
        """Map outdated root IDs onto their current equivalents.

        Returns ``old_id, new_id, confidence, changed`` - confidence being the
        fraction of the old neuron that ended up in the new one, so you can see
        when an ID has been split rather than merely renamed.
        """
        ids = np.asarray(x, dtype="int64").ravel()
        ts = self._timestamp(version)

        rows = []
        for i, old in enumerate(ids):
            if supervoxels is not None:
                new = int(self._cg.get_root_id(int(supervoxels[i]), timestamp=ts))
                rows.append({"old_id": int(old), "new_id": new, "confidence": 1.0})
                continue
            if self._cg.is_latest_roots([int(old)], timestamp=ts)[0]:
                rows.append({"old_id": int(old), "new_id": int(old), "confidence": 1.0})
                continue
            suggested, frac = self._cg.suggest_latest_roots(
                int(old), timestamp=ts, return_fraction_overlap=True
            )
            if np.isscalar(suggested):
                new, conf = int(suggested), 1.0
            else:
                best = int(np.argmax(list(frac.values()))) if isinstance(frac, dict) else 0
                new = int(np.atleast_1d(suggested)[best])
                conf = float(np.atleast_1d(list(frac.values()) if isinstance(frac, dict) else frac)[best])
            rows.append({"old_id": int(old), "new_id": new, "confidence": conf})

        out = pd.DataFrame(rows)
        out["changed"] = out["old_id"] != out["new_id"]
        return out

    def find_common_time(self, x) -> pd.Timestamp:
        """The latest timestamp at which all the given root IDs co-existed."""
        ids = np.asarray(x, dtype="int64").ravel()
        stamps = self._cg.get_root_timestamps(ids)
        return pd.Timestamp(max(stamps))

    # -------------------------------------------------------------------- voxels

    def get_voxels(self, x, *, mip: int = 0):
        from .points import get_voxels

        return get_voxels(self._ds, int(x), mip=mip)

    def get_segmentation_cutout(self, bbox, *, mip: int = 0, coordinates: str = "voxel"):
        from .points import segmentation_cutout

        return segmentation_cutout(self._ds, bbox, mip=mip, coordinates=coordinates)
