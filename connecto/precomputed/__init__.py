"""A read-only reader for neuroglancer precomputed and graphene volumes.

This is connecto's replacement for cloud-volume. It reads exactly what connecto
needs and nothing else - segmentation cutouts, meshes, and the geometry that hangs
off an ``info`` file - over plain HTTPS.

    from connecto.precomputed import Volume

    vol = Volume("precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation")
    vol.download(Bbox([0, 0, 0], [64, 64, 64]), mip=3)
    vol.mesh.get(1734350788, lod=2)

Why not cloud-volume? Because cloud-volume also *writes*, to Google Cloud, S3, and
half a dozen other backends, and pays for that in dependencies: boto3, the Google
Cloud SDK, gevent, protobuf and the rest come to some 78 MB and around 37 packages,
all of it reachable from connecto and none of it used. Reading is a much smaller
job: resolve a URL, GET some bytes, decode them.

What is deliberately *not* here: writing, image volumes and their codecs (connecto
never reads EM imagery - ``_image_source`` only ever builds a neuroglancer URL),
annotations, spatial indices, and anything that needs cloud SDK credentials.
Everything connecto reads is either a public bucket or a CAVE-authenticated
service, and ``requests`` handles both.

Correctness is checked against cloud-volume itself: ``tests/test_precomputed.py``
reads the same boxes and meshes both ways and compares them.
"""

from __future__ import annotations

import functools

import numpy as np

from .graphene import GrapheneMeta, GraphenePath, parse_graphene_path
from .image import ImageSource, compressed_morton_code
from .lib import Bbox
from .mesh import mesh_source
from .meta import PrecomputedMeta
from .sharding import ShardingSpec, ShardReader
from .store import Store, is_graphene, normalize_url, split_protocol

__all__ = [
    "Bbox",
    "GrapheneMeta",
    "GraphenePath",
    "GrapheneVolume",
    "ImageSource",
    "PrecomputedMeta",
    "PrecomputedVolume",
    "ShardReader",
    "ShardingSpec",
    "Store",
    "Volume",
    "compressed_morton_code",
    "is_graphene",
    "normalize_url",
    "parse_graphene_path",
    "split_protocol",
]


class _BaseVolume:
    """What both volume kinds have in common."""

    meta: PrecomputedMeta
    image: ImageSource

    #: Whether ``download(agglomerate=True)`` means anything here. A flat volume
    #: stores its segment IDs outright and has nothing underneath to roll up, so
    #: callers ask the volume rather than re-reading the protocol off its URL.
    agglomerable = False

    def __init__(self, mip: int = 0):
        self.mip = mip

    # ------------------------------------------------------------------ geometry

    @property
    def scale(self) -> dict:
        return self.meta.scale(self.mip)

    def mip_resolution(self, mip) -> np.ndarray:
        return self.meta.resolution(mip)

    @property
    def bounds(self) -> Bbox:
        return self.meta.bounds(self.mip)

    # ------------------------------------------------------------------- reading

    def download(self, bbox: Bbox, mip=None, **kwargs) -> np.ndarray:
        return self.image.download(bbox, mip=self.mip if mip is None else mip, **kwargs)

    def __getitem__(self, slices) -> np.ndarray:
        """``vol[x0:x1, y0:y1, z0:z1]``, at the volume's current mip."""
        if not isinstance(slices, tuple):
            slices = (slices,)
        if len(slices) < 3:
            slices = slices + (slice(None),) * (3 - len(slices))
        bounds = self.meta.bounds(self.mip)
        starts, stops = [], []
        for i, sl in enumerate(slices[:3]):
            starts.append(int(bounds.minpt[i]) if sl.start is None else int(sl.start))
            stops.append(int(bounds.maxpt[i]) if sl.stop is None else int(sl.stop))
        return self.download(Bbox(starts, stops), mip=self.mip)


class PrecomputedVolume(_BaseVolume):
    """A flat ``precomputed://`` volume."""

    def __init__(
        self,
        source: str,
        *,
        session=None,
        mip: int = 0,
        fill_missing: bool = True,
        bounded: bool = False,
        parallel: int = 8,
    ):
        super().__init__(mip=mip)
        self.source = source
        self.store = Store(source, session=session)
        self.meta = PrecomputedMeta(self.store.get_json("info"), source)
        self.image = ImageSource(
            self.store,
            self.meta,
            fill_missing=fill_missing,
            bounded=bounded,
            parallel=parallel,
        )

    @functools.cached_property
    def mesh(self):
        path = self.meta.mesh_path
        if path is None:
            raise ValueError(f"{self.source} declares no mesh directory.")
        return mesh_source(self.store, path)

    def __repr__(self) -> str:
        return f"PrecomputedVolume({self.source!r}, mip={self.mip})"


class GrapheneVolume(_BaseVolume):
    """A ``graphene://`` source: a chunkedgraph over a watershed volume.

    Two endpoints, two identities. The graph and meshing services want the CAVE
    session; the bucket underneath is public and must be read *without* it - a CAVE
    bearer token sent to Google Storage comes back 401 on an object anyone can read.
    Nothing is passed in for the bucket, so that stays true by construction.
    """

    agglomerable = True

    def __init__(
        self,
        source: str,
        *,
        session=None,
        get_roots=None,
        mip: int = 0,
        fill_missing: bool = True,
        bounded: bool = False,
        parallel: int = 8,
    ):
        super().__init__(mip=mip)
        self.source = source
        self.session = session
        self._get_roots = get_roots

        self.meta = GrapheneMeta.fetch(source, session=session)
        # No session for the bucket: it is public, and Google Storage answers 401 to
        # a CAVE bearer token on an object anyone can read anonymously.
        self.store = Store(self.meta.data_dir)
        self.image = ImageSource(
            self.store,
            self.meta,
            fill_missing=fill_missing,
            bounded=bounded,
            parallel=parallel,
        )

    @functools.cached_property
    def mesh(self):
        from .graphene_mesh import GrapheneMeshSource

        return GrapheneMeshSource(self.meta, session=self.session)

    def download(self, bbox: Bbox, mip=None, *, agglomerate: bool = False, **kwargs):
        """Supervoxels, or root IDs if `agglomerate`.

        Agglomerating is a graph question, so it costs a chunkedgraph round trip -
        one, for the whole cutout, rather than one per chunk.
        """
        out = self.image.download(bbox, mip=self.mip if mip is None else mip, **kwargs)
        if not agglomerate:
            return out
        return self.agglomerate(out)

    def agglomerate(self, supervoxels: np.ndarray) -> np.ndarray:
        """Map an array of supervoxel IDs onto their current root IDs.

        Remapped through a lookup table indexed by ``np.unique``'s own inverse, so
        the cost is one pass over the array rather than one pass per distinct
        supervoxel. A 256^3 cutout holds tens of thousands of them, and scanning the
        whole array once for each is minutes of work and gigabytes of transient
        masks for something a single gather answers.
        """
        if self._get_roots is None:
            raise ValueError(
                "Agglomerating needs the chunkedgraph. Construct this volume with "
                "`get_roots=client.chunkedgraph.get_roots`."
            )
        unique, inverse = np.unique(supervoxels, return_inverse=True)
        present = unique != 0
        if not present.any():
            return supervoxels

        # Background stays background: index 0 of the table is left at zero.
        table = np.zeros(len(unique), dtype=supervoxels.dtype)
        table[present] = np.asarray(
            self._get_roots(unique[present].astype(np.uint64)), dtype=supervoxels.dtype
        )
        return table[np.asarray(inverse).reshape(supervoxels.shape)]

    def __repr__(self) -> str:
        return f"GrapheneVolume({self.source!r}, mip={self.mip})"


def Volume(source: str, **kwargs):
    """Open `source`, picking the reader from its protocol."""
    if is_graphene(source):
        return GrapheneVolume(source, **kwargs)
    if split_protocol(source)[0] in ("precomputed", ""):
        # Graphene-only keywords are *not* swallowed here: a caller asking a flat
        # volume to agglomerate has misunderstood something, and a TypeError says so
        # where a silent no-op would hand back un-agglomerated supervoxels.
        return PrecomputedVolume(source, **kwargs)
    raise ValueError(
        f"connecto reads precomputed:// and graphene:// sources; got {source!r}."
    )
