"""Reading a segmentation volume, via CloudVolume.

Backend-agnostic on purpose. A segmentation volume is a segmentation volume: the
difference between FlyWire's graphene source and hemibrain's flat ``precomputed://``
bucket is what the *values* mean (root IDs sitting on supervoxels, versus body IDs
and nothing underneath), not how you read a voxel out of one. So the reading lives
here and the meaning lives in :mod:`connecto.core.segmentation`.

Two sources, not one
--------------------
``ds._segmentation_source()`` is the volume to *read and display*. It is what the
spec may override, and FlyWire does override it - with the flat v783 volume, which
loads without a CAVE login and is what the public release actually is.

``ds._graph_source()`` is the chunkedgraph. It is never overridden, because a flat
volume has no graph beneath it: reading FlyWire's flat volume gives you root IDs,
and a root ID passed off as a supervoxel is a lie that `get_roots` will cheerfully
accept (it is idempotent on roots) and answer wrongly for any version but 783.

Nanometres in, nanometres out
-----------------------------
Every coordinate connecto hands you is in nanometres, so ``units="nm"`` is the
default here too. CloudVolume works in voxels at its own mip-0 resolution - which is
not necessarily the dataset's ``voxel_size``: FlyWire's spec says 4x4x40, its volume
is 16x16x40 - so the conversion always uses the volume's own scale, never the spec's.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor

import cloudvolume as cv
import numpy as np

__all__ = [
    "GSPointLoader",
    "PRECOMPUTED_SKELETON_COLMAP",
    "PRECOMPUTED_SKELETON_INFO",
    "get_cloudvolume",
    "get_voxels",
    "lookup_points",
    "precomputed_skeleton",
    "segmentation_cutout",
]

# A precomputed skeleton arrives canonical and in nanometres. It is *not* backend
# data, so it must not be run through the backend's column map or its voxel scaling
# - which is exactly what happened when FlyWire's neuPrint door first read this
# bucket: neuPrint's map looked for `rowId`/`link` and found neither, and its
# `voxel` units would have multiplied every coordinate by (4, 4, 40).
PRECOMPUTED_SKELETON_COLMAP = {
    "node_id": "node_id",
    "parent_id": "parent_id",
    "x": "x", "y": "y", "z": "z", "radius": "radius",
}
PRECOMPUTED_SKELETON_UNITS = "nm"

_VOLUMES: dict = {}

# What a neuroglancer precomputed skeleton bucket looks like. Passed explicitly
# because these buckets typically ship no `info` of their own.
PRECOMPUTED_SKELETON_INFO = {
    "@type": "neuroglancer_skeletons",
    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "vertex_attributes": [
        {"id": "radius", "data_type": "float32", "num_components": 1}
    ],
}


def precomputed_skeleton(url: str, root: int):
    """Read a published neuroglancer skeleton. Already in nanometres.

    `url` is the resolved bucket (see `Dataset._skeleton_source`).

    Backend-agnostic, and that is the point: this is a plain HTTPS bucket, needing
    no login and no client of any kind, so the same skeleton comes back whichever
    door you came in by. It used to live in the CAVE backend, which meant the
    neuPrint-backed FlyWire could not see it and fell through to neuPrint's own
    skeleton store - which, for `flywire-fafb:v783b`, does not exist.
    """
    import navis

    tn = navis.read_precomputed(
        f"{url}/{int(root)}", datatype="skeleton", info=PRECOMPUTED_SKELETON_INFO
    )
    return tn.nodes


def get_cloudvolume(ds, source: str | None = None):
    """A CloudVolume onto ``source`` (default: this dataset's segmentation), cached."""
    if source is None:
        source = ds._segmentation_source()
    if source is None:
        raise ValueError(f"{ds.label} declares no segmentation source.")

    if source not in _VOLUMES:
        _VOLUMES[source] = cv.CloudVolume(
            source, use_https=True, progress=False, fill_missing=True
        )
    return _VOLUMES[source]


def _to_volume_nm(vol, locs, units: str) -> np.ndarray:
    """Caller coordinates -> nanometres, in the volume's own frame."""
    locs = np.asarray(locs, dtype="float64").reshape(-1, 3)
    if units == "nm":
        return locs
    if units == "voxel":
        return locs * np.asarray(vol.scale["resolution"], dtype="float64")
    raise ValueError(f"`units` must be 'nm' or 'voxel', got {units!r}.")


class GSPointLoader:
    """Accumulate points, then load them batched by storage chunk.

    One request per chunk rather than one per point, which is the difference between
    "usable" and "not". Based on an implementation by Peter Li:
    https://gist.github.com/chinasaur/5429ef3e0a60aa7a1c38801b0cbfe9bb
    """

    def __init__(self, cloud_volume):
        self._volume = cloud_volume
        self._chunk_map = collections.defaultdict(set)
        self._points = None

    def add_points(self, points):
        """Queue points, in **nanometres**."""
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
        chunk_start += mn
        indices -= mn

        chunk = self._load_chunk(chunk_start, chunk_end)
        return points, chunk[indices[:, 0], indices[:, 1], indices[:, 2]]

    def load_all(self, max_workers: int = 4, return_sorted: bool = True, progress: bool = True):
        """Fetch every queued chunk.

        Threads, not processes. The work is one HTTPS GET per chunk - network-bound,
        with the GIL released for the whole of it - so a thread pool is both the right
        tool and the cheap one. The process pool this used to use had to pickle a
        CloudVolume per task, and on macOS (spawn, not fork) it simply deadlocked:
        every lookup hung forever at the default `max_workers=4`.
        """
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
                workers = max(1, min(max_workers, len(self._chunk_map)))
                if workers > 1:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for result in pool.map(self._load_points, list(self._chunk_map)):
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


def lookup_points(
    ds,
    locs,
    *,
    units: str = "nm",
    source: str | None = None,
    max_workers: int = 4,
    progress: bool = True,
) -> np.ndarray:
    """Whatever the volume holds at each location.

    On a graphene source that is a supervoxel; on a flat one it is the segment ID
    itself. The caller knows which volume it asked for, so it knows which it got.
    """
    vol = get_cloudvolume(ds, source)
    locs = _to_volume_nm(vol, locs, units)
    if not len(locs):
        return np.array([], dtype="int64")

    loader = GSPointLoader(vol)
    loader.add_points(locs)
    _, ids = loader.load_all(max_workers=max_workers, progress=progress)
    return np.asarray(ids, dtype="int64")


def segmentation_cutout(ds, bbox, *, mip: int = 0, units: str = "nm", source: str | None = None):
    """Raw segmentation in a bounding box, as a 3D array."""
    from cloudvolume import Bbox

    source = source or ds._segmentation_source()
    vol = get_cloudvolume(ds, source)
    bbox = np.asarray(bbox, dtype="float64").reshape(2, 3)

    if units == "nm":
        bbox = bbox / np.asarray(vol.mip_resolution(mip), dtype="float64")
    elif units != "voxel":
        raise ValueError(f"`units` must be 'nm' or 'voxel', got {units!r}.")
    bbox = bbox.astype("int64")

    # `download` wants a Bbox. Handing it a list of two lists raises
    # `AttributeError: 'list' object has no attribute 'start'` - it mistakes the
    # list for a sequence of slices - which is neither obviously about the bbox nor
    # obviously our fault, so it is worth not doing.
    box = Bbox(bbox[0], bbox[1])

    # `agglomerate` is a graphene notion: roll supervoxels up into root IDs. A flat
    # volume stores the segment IDs outright and has nothing to agglomerate.
    opts = {"agglomerate": True} if str(source).startswith("graphene://") else {}

    return vol.download(box, mip=mip, **opts)


def get_voxels(ds, seg_id: int, *, mip: int = 0, source: str | None = None) -> np.ndarray:
    """Every voxel belonging to one segment. Expensive; use sparingly."""
    vol = get_cloudvolume(ds, source)
    mesh = vol.mesh.get(seg_id)
    if isinstance(mesh, dict):
        mesh = mesh[seg_id]

    # The mesh bounds tell us where to cut, so we download a box around the neuron
    # rather than the whole brain.
    bounds = np.vstack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    cutout = segmentation_cutout(ds, bounds, mip=mip, units="nm", source=source)
    return np.argwhere(np.asarray(cutout)[..., 0] == seg_id)
