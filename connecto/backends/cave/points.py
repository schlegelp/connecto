"""Point -> segment lookups.

``GSPointLoader`` batches point queries by storage chunk instead of hitting the
volume once per point, which is the difference between "usable" and "not". Based
on an implementation by Peter Li:
https://gist.github.com/chinasaur/5429ef3e0a60aa7a1c38801b0cbfe9bb
"""

from __future__ import annotations

import collections

import numpy as np

from ...exceptions import MissingDependencyError

__all__ = ["GSPointLoader", "lookup_points", "get_voxels", "segmentation_cutout"]


class GSPointLoader:
    """Accumulate points, then load them batched by storage chunk."""

    def __init__(self, cloud_volume):
        self._volume = cloud_volume
        self._chunk_map = collections.defaultdict(set)
        self._points = None

    def add_points(self, points):
        """Queue points (Nx3, in the volume's own resolution)."""
        points = np.asarray(points)

        if self._points is None:
            self._points = points
        else:
            self._points = np.concatenate((self._points, points))

        resolution = np.array(self._volume.scale["resolution"])
        chunk_size = np.array(self._volume.scale["chunk_sizes"])
        chunk_starts = (points // resolution).astype(int) // chunk_size * chunk_size
        for point, chunk_start in zip(points, chunk_starts):
            self._chunk_map[tuple(chunk_start)].add(tuple(point))

    def _load_chunk(self, chunk_start, chunk_end):
        return self._volume[
            chunk_start[0] : chunk_end[0],
            chunk_start[1] : chunk_end[1],
            chunk_start[2] : chunk_end[2],
        ]

    def _load_points(self, chunk_map_key):
        chunk_start = np.array(chunk_map_key)
        points = np.array(list(self._chunk_map[chunk_map_key]))

        resolution = np.array(self._volume.scale["resolution"])
        indices = (points // resolution).astype(int) - chunk_start

        # Subset the chunk to just the part containing our points - saves a lot of
        # memory on sparse queries.
        mn, mx = indices.min(axis=0), indices.max(axis=0)
        chunk_end = chunk_start + mx + 1
        chunk_start = chunk_start + mn
        indices -= mn

        chunk = self._load_chunk(chunk_start, chunk_end)
        return points, chunk[indices[:, 0], indices[:, 1], indices[:, 2]]

    def load_all(self, max_workers: int = 4, return_sorted: bool = True, progress: bool = True):
        from tqdm.auto import tqdm

        progress_state = self._volume.progress
        self._volume.progress = False

        try:
            with tqdm(
                total=len(self._chunk_map),
                desc="Segmentation IDs",
                leave=False,
                disable=not progress,
            ) as pbar:
                results = []
                if max_workers > 1:
                    try:
                        from pathos.pools import _ProcessPool as Pool
                    except ModuleNotFoundError as e:
                        raise MissingDependencyError.for_extra(
                            "pathos", "points", "Parallel point lookup"
                        ) from e
                    with Pool(max_workers) as pool:
                        for result in pool.imap_unordered(self._load_points, self._chunk_map):
                            results.append(result)
                            pbar.update(1)
                else:
                    for key in self._chunk_map:
                        results.append(self._load_points(key))
                        pbar.update(1)
        finally:
            self._volume.progress = progress_state

        if return_sorted:
            lookup = dict(
                zip(
                    [tuple(p) for r in results for p in r[0]],
                    [i for r in results for i in r[1]],
                )
            )
            data = np.array([lookup[tuple(p)] for p in self._points])
            points = self._points
        else:
            points = np.concatenate([r[0] for r in results])
            data = np.concatenate([r[1] for r in results])

        return points, data.flatten()


def lookup_points(ds, locs, *, coordinates: str = "voxel", agglomerate: bool = False,
                  max_workers: int = 4, progress: bool = True) -> np.ndarray:
    """Supervoxel (or root) ID at each location."""
    from .meshes import get_cloudvolume

    vol = get_cloudvolume(ds)
    locs = np.asarray(locs, dtype="float64").reshape(-1, 3)

    if coordinates == "nm":
        locs = locs / np.array(vol.scale["resolution"], dtype="float64")
    locs = (locs * np.array(vol.scale["resolution"])).astype("int64")

    loader = GSPointLoader(vol)
    loader.add_points(locs)
    _, svids = loader.load_all(max_workers=max_workers, progress=progress)
    return np.asarray(svids, dtype="int64")


def get_voxels(ds, root_id: int, *, mip: int = 0) -> np.ndarray:
    """Every voxel belonging to a neuron. Expensive; use sparingly."""
    from .meshes import get_cloudvolume

    vol = get_cloudvolume(ds)
    mesh = vol.mesh.get(root_id)
    if isinstance(mesh, dict):
        mesh = mesh[root_id]
    bounds = np.vstack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    cutout = segmentation_cutout(ds, bounds, mip=mip, coordinates="nm")
    return np.argwhere(cutout == root_id)


def segmentation_cutout(ds, bbox, *, mip: int = 0, coordinates: str = "voxel"):
    """Raw segmentation in a bounding box."""
    from .meshes import get_cloudvolume

    vol = get_cloudvolume(ds)
    bbox = np.asarray(bbox, dtype="float64").reshape(2, 3)

    if coordinates == "nm":
        bbox = bbox / np.array(vol.mip_resolution(mip), dtype="float64")
    bbox = bbox.astype("int64")

    return vol.download(
        bbox=[bbox[0].tolist(), bbox[1].tolist()], mip=mip, agglomerate=True
    )
