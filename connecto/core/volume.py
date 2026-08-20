"""Reading a segmentation volume.

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
default here too. A volume works in voxels at its own mip-0 resolution - which is
not necessarily the dataset's ``voxel_size``: FlyWire's spec says 4x4x40, its volume
is 16x16x40 - so the conversion always uses the volume's own scale, never the spec's.

The reading itself is :mod:`connecto.precomputed`, connecto's own precomputed and
graphene reader. It used to be cloud-volume, whose write path and multi-cloud
storage layer cost some 78 MB of dependencies that reading never touches.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..precomputed import Bbox, Volume, is_graphene

__all__ = [
    "GSPointLoader",
    "PRECOMPUTED_SKELETON_COLMAP",
    "PRECOMPUTED_SKELETON_INFO",
    "fetch_meshes",
    "get_voxels",
    "get_volume",
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


def _volume_kwargs(ds, source: str) -> dict:
    """Which credentials, if any, this source wants.

    Only a graphene source gets the CAVE session, and it is the *service* that gets
    it - the graph and meshing endpoints. Everything else is a public object store,
    and sending a CAVE bearer token there is not merely unnecessary: Google Storage
    tries to authenticate with it and answers 401 on an object anyone can read
    anonymously. FlyWire is exactly this case, since a CAVE-backed FlyWire reads its
    *flat* v783 bucket for meshes while its graph lives behind CAVE.
    """
    graph = getattr(getattr(ds, "client", None), "chunkedgraph", None)
    if graph is None or not is_graphene(source):
        return {}
    return {
        "session": getattr(graph, "session", None),
        "get_roots": getattr(graph, "get_roots", None),
    }


def get_volume(ds, source: str | None = None):
    """A volume onto ``source`` (default: this dataset's segmentation), cached."""
    if source is None:
        source = ds._segmentation_source()
    if source is None:
        raise ValueError(f"{ds.label} declares no segmentation source.")

    if source not in _VOLUMES:
        _VOLUMES[source] = Volume(
            source, fill_missing=True, **_volume_kwargs(ds, source)
        )
    return _VOLUMES[source]


def _to_volume_nm(vol, locs, units: str) -> np.ndarray:
    """Caller coordinates -> nanometres, in the volume's own frame."""
    locs = np.asarray(locs, dtype="float64").reshape(-1, 3)
    if units == "nm":
        return locs
    if units == "voxel":
        return locs * np.asarray(vol.mip_resolution(0), dtype="float64")
    raise ValueError(f"`units` must be 'nm' or 'voxel', got {units!r}.")


class GSPointLoader:
    """Accumulate points, then load them batched by storage chunk.

    One request per chunk rather than one per point, which is the difference between
    "usable" and "not". Based on an implementation by Peter Li:
    https://gist.github.com/chinasaur/5429ef3e0a60aa7a1c38801b0cbfe9bb
    """

    def __init__(self, volume):
        self._volume = volume
        self._chunk_map = collections.defaultdict(set)
        self._points = None

    def add_points(self, points):
        """Queue points, in **nanometres**."""
        points = np.asarray(points)

        if self._points is None:
            self._points = points
        else:
            self._points = np.concatenate((self._points, points))

        resolution = self._volume.mip_resolution(0)
        chunk_size = self._volume.meta.chunk_size(0)
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

        resolution = self._volume.mip_resolution(0)
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
        volume per task, and on macOS (spawn, not fork) it simply deadlocked: every
        lookup hung forever at the default `max_workers=4`.
        """
        from tqdm.auto import tqdm

        with tqdm(
            total=len(self._chunk_map),
            desc="Segmentation IDs",
            leave=False,
            disable=not progress,
        ) as pbar:
            results = []
            workers = max(1, min(max_workers, len(self._chunk_map)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(self._load_points, list(self._chunk_map)):
                    results.append(result)
                    pbar.update(1)

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
    vol = get_volume(ds, source)
    locs = _to_volume_nm(vol, locs, units)
    if not len(locs):
        return np.array([], dtype="int64")

    loader = GSPointLoader(vol)
    loader.add_points(locs)
    _, ids = loader.load_all(max_workers=max_workers, progress=progress)
    return np.asarray(ids, dtype="int64")


def segmentation_cutout(ds, bbox, *, mip: int = 0, units: str = "nm", source: str | None = None):
    """Raw segmentation in a bounding box, as a 3D array."""
    source = source or ds._segmentation_source()
    vol = get_volume(ds, source)
    bbox = np.asarray(bbox, dtype="float64").reshape(2, 3)

    if units == "nm":
        bbox = bbox / np.asarray(vol.mip_resolution(mip), dtype="float64")
    elif units != "voxel":
        raise ValueError(f"`units` must be 'nm' or 'voxel', got {units!r}.")
    bbox = bbox.astype("int64")

    # `download` wants a Bbox, not a list of two lists.
    box = Bbox(bbox[0], bbox[1])

    # `agglomerate` is a graphene notion: roll supervoxels up into root IDs. A flat
    # volume stores the segment IDs outright and has nothing to agglomerate, and it
    # knows that about itself - no need to re-read the protocol off its URL.
    opts = {"agglomerate": True} if vol.agglomerable else {}

    return vol.download(box, mip=mip, **opts)


def fetch_meshes(
    ds,
    ids,
    *,
    source: str | None = None,
    lod=None,
    progress: bool = True,
    max_workers: int = 4,
):
    """Yield ``(id, trimesh.Trimesh)`` for each segment, in the order asked for.

    Backend-agnostic, because reading a mesh no longer depends on which door you
    came in by: a flat multi-resolution bucket and a chunkedgraph's meshing service
    both answer :meth:`mesh.get` and both hand back a ``Trimesh``.

    Threaded, because a mesh is a handful of sequential HTTPS GETs (shard index,
    manifest, fragments) and fetching one neuron at a time leaves the link idle for
    most of the wall clock.
    """
    from tqdm.auto import tqdm

    vol = get_volume(ds, source)
    ids = [int(i) for i in ids]
    # Only pass `lod` when the caller meant one. Graphene meshes have no levels of
    # detail - the graph layer sets the resolution - and the number would end up in
    # the manifest URL, asking the service for something that does not exist.
    kwargs = {} if lod is None else {"lod": int(lod)}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(ids) or 1))) as pool:
        meshes = pool.map(lambda i: vol.mesh.get(i, **kwargs), ids)
        yield from tqdm(
            zip(ids, meshes),
            desc="Meshes",
            total=len(ids),
            disable=not progress or len(ids) < 2,
            leave=False,
        )


def get_voxels(ds, seg_id: int, *, mip: int = 0, source: str | None = None) -> np.ndarray:
    """Every voxel belonging to one segment. Expensive; use sparingly."""
    vol = get_volume(ds, source)
    mesh = vol.mesh.get(seg_id)

    # The mesh bounds tell us where to cut, so we download a box around the neuron
    # rather than the whole brain.
    bounds = np.vstack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    cutout = segmentation_cutout(ds, bounds, mip=mip, units="nm", source=source)
    return np.argwhere(np.asarray(cutout)[..., 0] == seg_id)
