"""CAVE meshes, via CloudVolume."""

from __future__ import annotations

from ...exceptions import MissingDependencyError

__all__ = ["fetch_meshes", "get_cloudvolume"]

_VOLUMES: dict = {}


def get_cloudvolume(ds):
    """A CloudVolume onto this dataset's segmentation (cached)."""
    try:
        import cloudvolume as cv
    except ModuleNotFoundError as e:
        raise MissingDependencyError.for_extra(
            "cloud-volume", "points", "Fetching meshes"
        ) from e

    key = ds.source
    if key not in _VOLUMES:
        source = ds.spec.segmentation_source or ds.client.info.segmentation_source()
        _VOLUMES[key] = cv.CloudVolume(
            source, use_https=True, progress=False, fill_missing=True
        )
    return _VOLUMES[key]


def fetch_meshes(ds, ids, version, *, lod=None, progress: bool = True, **opts):
    """Yield ``(root_id, trimesh.Trimesh)``."""
    from tqdm.auto import tqdm

    vol = get_cloudvolume(ds)
    kwargs = {} if lod is None else {"lod": lod}

    for root in tqdm(ids, desc="Meshes", disable=not progress or len(ids) < 2, leave=False):
        root = int(root)
        mesh = vol.mesh.get(root, **kwargs)
        # CloudVolume returns {id: mesh} for some sources and a bare mesh for others.
        if isinstance(mesh, dict):
            mesh = mesh[root]
        yield root, mesh
