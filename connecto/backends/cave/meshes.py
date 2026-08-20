"""CAVE meshes.

Which reader runs depends on what the datastack points at, and the two are not
alike. FlyWire overrides its segmentation source with the flat v783 bucket, whose
meshes are a neat multi-resolution octree. BANC, FANC and MICrONS have no flat
volume at all, so their meshes come from the chunkedgraph's meshing service:
static shards for whatever was meshed in bulk, plus a loose fragment for every
piece re-meshed since a proofreader touched it.

Both live behind :mod:`connecto.precomputed`, and both answer the same call, so
the difference does not surface here at all - which is why there is nothing left
in this module but the door.
"""

from __future__ import annotations

from ...core.volume import fetch_meshes

__all__ = ["fetch_meshes"]
