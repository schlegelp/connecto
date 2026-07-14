"""The segmentation namespace: what body is at this point, and - on CAVE - the graph.

Shared by both backends, because both have a segmentation volume. What they do not
share is a *chunkedgraph*, so the methods split in two:

``Cap.SEGMENTATION`` - reading the volume. ``locs_to_segments``, ``get_voxels``,
``get_segmentation_cutout``. Works on FlyWire's graphene source and on hemibrain's
flat ``precomputed://`` bucket alike.

``Cap.CHUNKEDGRAPH`` - the proofreading graph underneath. Supervoxels, root IDs that
change when someone makes an edit, ``update_ids``, ``is_latest_root``. CAVE only, and
asking a neuPrint dataset for any of it raises rather than inventing an answer -
hemibrain's body IDs are immutable, there are no supervoxels under them, and there is
no timeline to be out of date with respect to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import CapabilityError
from .dataset import requires
from .namespaces import _Namespace
from .spec import Cap

__all__ = ["Segmentation"]


class Segmentation(_Namespace):
    """Query the segmentation volume, and the chunkedgraph where there is one."""

    @property
    def _cg(self):
        return self._ds.client.chunkedgraph

    def _timestamp(self, version=None):
        ds = self._ds
        v = ds._resolve_version_arg(version)
        if v.is_live:
            return None
        return ds.client.materialize.get_timestamp(v.value)

    # ---------------------------------------------------------- reading the volume

    @requires(Cap.SEGMENTATION)
    def locs_to_segments(
        self, locs, *, units: str = "nm", version=None, progress: bool = True
    ) -> np.ndarray:
        """The segment ID at each xyz location. ``locs`` is Nx3, in nanometres.

        On a chunkedgraph dataset this goes via supervoxels, so it honours
        ``version=`` and answers as of that materialization. On a flat volume there
        is only one answer - the volume *is* the version - and ``version=`` is
        therefore refused rather than ignored.
        """
        from . import volume

        ds = self._ds
        if not ds.supports(Cap.CHUNKEDGRAPH):
            if version is not None:
                raise CapabilityError(
                    f"{ds.label} has a flat segmentation volume, not a chunkedgraph: "
                    f"its segment IDs are immutable, so there is nothing for "
                    f"`version=` to select. Drop the argument."
                )
            return volume.lookup_points(ds, locs, units=units, progress=progress)

        sv = self.locs_to_supervoxels(locs, units=units, progress=progress)
        return self.supervoxels_to_roots(sv, version=version)

    @requires(Cap.SEGMENTATION)
    def neuron_to_segments(
        self, x, *, units: str = "nm", version=None, progress: bool = True
    ) -> pd.DataFrame:
        """Which segments a neuron's nodes fall into. Useful for spotting merges."""
        import navis

        rows = []
        for n in navis.NeuronList(x):
            nodes = n.nodes[["x", "y", "z"]].to_numpy()
            segs = self.locs_to_segments(
                nodes, units=units, version=version, progress=progress
            )
            for seg, count in pd.Series(segs).value_counts().items():
                rows.append({"neuron": n.id, "segment": int(seg), "n_nodes": int(count)})
        return pd.DataFrame(rows)

    @requires(Cap.SEGMENTATION)
    def get_voxels(self, x, *, mip: int = 0) -> np.ndarray:
        """Every voxel belonging to one segment. Expensive; use sparingly."""
        from . import volume

        return volume.get_voxels(self._ds, int(x), mip=mip)

    @requires(Cap.SEGMENTATION)
    def get_segmentation_cutout(self, bbox, *, mip: int = 0, units: str = "nm"):
        """Raw segmentation in a bounding box. ``bbox`` is ``[[x0,y0,z0],[x1,y1,z1]]``."""
        from . import volume

        return volume.segmentation_cutout(self._ds, bbox, mip=mip, units=units)

    # ----------------------------------------------------------- the chunkedgraph

    @requires(Cap.CHUNKEDGRAPH)
    def locs_to_supervoxels(
        self, locs, *, units: str = "nm", progress: bool = True
    ) -> np.ndarray:
        """The supervoxel ID at each xyz location.

        Reads the *graph* source, never the spec's display volume. FlyWire's spec
        points at the flat v783 bucket - excellent for meshes and for neuroglancer,
        useless here: a flat volume hands back root IDs, and a root ID passed off as
        a supervoxel is a lie `get_roots` will happily accept, because it is
        idempotent on roots. It would be right at v783 and quietly wrong everywhere
        else.
        """
        from . import volume

        ds = self._ds
        return volume.lookup_points(
            ds, locs, units=units, source=ds._graph_source(), progress=progress
        )

    @requires(Cap.CHUNKEDGRAPH)
    def supervoxels_to_roots(self, supervoxels, *, version=None) -> np.ndarray:
        sv = np.asarray(supervoxels, dtype="int64").ravel()
        out = np.zeros(len(sv), dtype="int64")
        mask = sv != 0
        if mask.any():
            out[mask] = self._cg.get_roots(sv[mask], timestamp=self._timestamp(version))
        return out

    @requires(Cap.CHUNKEDGRAPH)
    def roots_to_supervoxels(self, x, *, version=None) -> dict:
        ds = self._ds
        ids = ds.ids(x, version=version)
        return {int(i): np.asarray(self._cg.get_leaves(int(i), stop_layer=1)) for i in ids}

    @requires(Cap.CHUNKEDGRAPH)
    def is_latest_root(self, x, *, version=None) -> np.ndarray:
        ids = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_latest_roots(ids, timestamp=self._timestamp(version)))

    @requires(Cap.CHUNKEDGRAPH)
    def is_valid_root(self, x) -> np.ndarray:
        ids = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_valid_nodes(ids))

    @requires(Cap.CHUNKEDGRAPH)
    def is_valid_supervoxel(self, x) -> np.ndarray:
        sv = np.asarray(x, dtype="int64").ravel()
        return np.asarray(self._cg.is_valid_nodes(sv))

    @requires(Cap.CHUNKEDGRAPH)
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
                conf = float(
                    np.atleast_1d(list(frac.values()) if isinstance(frac, dict) else frac)[best]
                )
            rows.append({"old_id": int(old), "new_id": new, "confidence": conf})

        out = pd.DataFrame(rows)
        out["changed"] = out["old_id"] != out["new_id"]
        return out

    @requires(Cap.CHUNKEDGRAPH)
    def find_common_time(self, x) -> pd.Timestamp:
        """The latest timestamp at which all the given root IDs co-existed."""
        ids = np.asarray(x, dtype="int64").ravel()
        stamps = self._cg.get_root_timestamps(ids)
        return pd.Timestamp(max(stamps))
