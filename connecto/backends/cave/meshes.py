"""CAVE meshes, via CloudVolume."""

from __future__ import annotations

from ...core.volume import get_cloudvolume

__all__ = ["fetch_meshes"]


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
