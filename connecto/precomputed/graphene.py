"""Graphene: a chunkedgraph in front of a precomputed volume.

A graphene source is not a volume of its own. It is a *service* that owns a graph,
plus a pointer (``data_dir``) at an ordinary precomputed bucket holding the
watershed supervoxels the graph is built over. So reading voxels from a graphene
source means reading that bucket - which is public even where the graph service is
not - and everything the graph contributes is arithmetic over 64-bit labels.

A graphene label packs its layer and chunk position into the high bits::

    | layer (8) | X (ct) | Y (ct) | Z (ct) | segid (rest) |

where ``ct`` (``spatial_bit_count``) shrinks as you go up the layers, because a
higher layer has fewer, larger chunks. That is why the chunks a neuron occupies
can be derived from its L2 IDs alone, with no volume access at all - the trick
:mod:`connecto.voxels.pcg` is built on.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

from .meta import PrecomputedMeta
from .store import Store, normalize_url

__all__ = ["GrapheneMeta", "GraphenePath", "parse_graphene_path"]

# The server tells us which of these it speaks, as indices into this list.
VERSION_ORDERING = ["1.0", "v1"]

_API_VX = re.compile(r"^/?(\w+)/api/(v[\d.]+)/table/([\w\d._\-]+)/?$")
_TABLE = re.compile(r"^/?(\w+)/table/([\w\d._\-]+)/?$")
_LEGACY = re.compile(r"^/?(\w+)/([\d.]+)/([\w\d._\-]+)/?$")


@dataclass(frozen=True)
class GraphenePath:
    scheme: str
    fqdn: str
    modality: str
    version: str
    dataset: str

    @property
    def base(self) -> str:
        return f"{self.scheme}://{self.fqdn}"

    def path_for(self, modality: str, api_version: str) -> str:
        """The URL of one service on this server, in the dialect it speaks."""
        if api_version == "1.0":
            tail = posixpath.join(modality, "1.0", self.dataset)
        else:
            tail = posixpath.join(modality, "api", api_version, "table", self.dataset)
        return posixpath.join(self.base, tail)


def parse_graphene_path(path: str) -> GraphenePath:
    """Pull the server, modality and datastack out of any graphene URL form."""
    url = normalize_url(path)
    parsed = urlparse(url)
    for pattern, version in ((_API_VX, None), (_TABLE, "table"), (_LEGACY, None)):
        match = pattern.match(parsed.path)
        if not match:
            continue
        groups = match.groups()
        return GraphenePath(
            parsed.scheme,
            parsed.netloc,
            groups[0],
            version or groups[1],
            groups[-1],
        )
    raise ValueError(
        f"Cannot parse {path!r} as a graphene source. Expected something like "
        f"graphene://https://server/segmentation/table/<datastack>."
    )


class GrapheneMeta(PrecomputedMeta):
    """The graphene ``info``: scales, plus the graph's own geometry."""

    def __init__(self, info: dict, graphene_path: GraphenePath, path: str = ""):
        super().__init__(info, path)
        self.server = graphene_path

    # -------------------------------------------------------------- construction

    @classmethod
    def fetch(cls, path: str, session=None, timeout: float = 120) -> GrapheneMeta:
        """Read the graphene info. Needs whatever auth the server wants."""
        server = parse_graphene_path(path)
        # The `/table/` form answers on every deployment, including the ones that
        # only advertise the legacy `1.0` paths.
        url = posixpath.join(server.base, server.modality, "table", server.dataset)
        info = Store(url, session=session, timeout=timeout).get_json("info")
        return cls(info, server, path)

    # ------------------------------------------------------------------- service

    @property
    def api_version(self) -> str:
        """Which dialect this server's URLs are in.

        The URL you were handed already says: a deployment that still publishes
        ``/segmentation/1.0/`` wants ``/meshing/1.0/`` too, and asking it for the
        newer path 404s. Only the modern ``/table/`` form leaves it open, and there
        the server names the versions it supports.
        """
        version = self.server.version
        if version != "table":
            return version
        supported = self.info.get("app", {}).get("supported_api_versions", [0])
        return VERSION_ORDERING[max(int(i) for i in supported)]

    @property
    def manifest_endpoint(self) -> str:
        return posixpath.join(
            self.server.path_for("meshing", self.api_version), "manifest"
        )

    @property
    def data_dir(self) -> str:
        """The precomputed bucket the graph sits on top of."""
        return self.info["data_dir"]

    # --------------------------------------------------------------------- graph

    @property
    def graph(self) -> dict:
        return self.info["graph"]

    @property
    def n_layers(self) -> int:
        return int(self.graph["n_layers"])

    @property
    def n_bits_for_layer_id(self) -> int:
        return int(self.graph.get("n_bits_for_layer_id", 8))

    @property
    def graph_chunk_size(self) -> np.ndarray:
        return np.asarray(self.graph["chunk_size"], dtype=np.int64)

    @property
    def watershed_mip(self) -> int:
        """The scale the graph's chunk grid is defined at."""
        return int(self.graph.get("cv_mip", 0))

    @property
    def chunks_start_at_voxel_offset(self) -> bool:
        return bool(self.info.get("chunks_start_at_voxel_offset", False))

    # ------------------------------------------------------------- label decoding

    def spatial_bit_count(self, level) -> int:
        return int(self.graph["spatial_bit_masks"][str(int(level))])

    def segid_bits(self, level) -> int:
        return 64 - self.n_bits_for_layer_id - 3 * self.spatial_bit_count(level)

    def decode_layer_id(self, label) -> int:
        return int(label) >> (64 - self.n_bits_for_layer_id)

    def decode_segid(self, label) -> int:
        return int(label) & ((1 << self.segid_bits(self.decode_layer_id(label))) - 1)

    def decode_chunk_position(self, label) -> np.ndarray:
        """``(x, y, z)`` of the chunk this label lives in."""
        label = int(label)
        ct = self.spatial_bit_count(self.decode_layer_id(label))
        shift = self.segid_bits(self.decode_layer_id(label))
        mask = (1 << ct) - 1
        return np.array(
            [
                (label >> (shift + 2 * ct)) & mask,
                (label >> (shift + ct)) & mask,
                (label >> shift) & mask,
            ],
            dtype=np.int64,
        )

    def decode_chunk_position_number(self, label) -> int:
        """X, Y and Z as one packed integer - how shard files are named."""
        label = int(label) & 0x00FFFFFFFFFFFFFF
        return label >> self.segid_bits(self.decode_layer_id(label))

    def encode_label(self, layer, x, y, z, segid) -> int:
        ct = self.spatial_bit_count(layer)
        shift = self.segid_bits(layer)
        return (
            (int(layer) << (64 - self.n_bits_for_layer_id))
            | (int(x) << (shift + 2 * ct))
            | (int(y) << (shift + ct))
            | (int(z) << shift)
            | int(segid)
        )

    # ---------------------------------------------------------------------- mesh

    @property
    def mesh_metadata(self) -> dict:
        return self.info.get("mesh_metadata") or {}

    @property
    def uniform_draco_grid_size(self):
        return self.mesh_metadata.get("uniform_draco_grid_size")

    @property
    def max_meshed_layer(self) -> int:
        return int(self.mesh_metadata.get("max_meshed_layer", self.n_layers))

    @property
    def unsharded_mesh_dir(self) -> str:
        return self.mesh_metadata.get("unsharded_mesh_dir", "dynamic")

    @property
    def sharded_mesh_dir(self) -> str:
        return "initial"

    def get_draco_grid_size(self, level) -> float:
        """The quantization step draco used at this graph layer.

        Needed to recognise a vertex sitting on a chunk boundary: quantization
        means "on the boundary" is a tolerance, not an equality.
        """
        if self.uniform_draco_grid_size is not None:
            return float(self.uniform_draco_grid_size)
        sizes = self.mesh_metadata.get("draco_grid_sizes")
        if not sizes or str(int(level)) not in sizes:
            raise ValueError(
                f"{self.server.dataset} declares no draco grid size for layer {level}."
            )
        return float(sizes[str(int(level))])

    def __repr__(self) -> str:
        return (
            f"GrapheneMeta({self.server.dataset!r} on {self.server.fqdn}, "
            f"{self.n_layers} layers, data_dir={self.data_dir!r})"
        )
