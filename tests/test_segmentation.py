"""The segmentation namespace.

Offline: a stub volume stands in for the real reader, because what is under test is the
*routing* - which volume gets read, which coordinates get sent, and which questions get
refused - and none of that needs a live bucket.

The through-line is that "does this dataset have a segmentation?" is two questions.
hemibrain has a segmentation volume (a flat `precomputed://` bucket) and no
chunkedgraph (its body IDs are frozen; there are no supervoxels beneath them). Every
test here is about not conflating those, because each way of conflating them produces
an answer that is wrong without looking wrong: a supervoxel that is really a root ID, a
`version=` that is silently ignored, a lookup 8x off because someone assumed voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

from connecto.core import volume
from connecto.core.dataset import Dataset
from connecto.core.segmentation import Segmentation
from connecto.core.spec import BackendSpec, Cap, DatasetSpec
from connecto.exceptions import CapabilityError

FLAT = "precomputed://gs://flyem/seg"
GRAPHENE = "graphene://https://cave/segmentation/table/x"

# The volume is 8nm isotropic, so a point at 800nm is voxel 100. Any confusion between
# the two shows up as a factor of eight, which the assertions below can see.
RES = [8, 8, 8]


class _Meta:
    """Just the geometry `volume.py` asks a volume's metadata for."""

    def resolution(self, mip=0):
        return np.array(RES)

    def chunk_size(self, mip=0):
        return np.array([64, 64, 64])


class _Vol:
    """The least volume that `volume.py` will accept."""

    def __init__(self, source):
        self.source = source
        self.meta = _Meta()
        # A flat volume has nothing to roll up; a graphene one does. This is the
        # property `segmentation_cutout` reads instead of re-parsing the URL.
        self.agglomerable = str(source).startswith("graphene://")
        self.reads: list = []
        self.downloads: list = []

    def mip_resolution(self, mip):
        return np.array(RES)

    def __getitem__(self, key):
        # GSPointLoader slices the volume; record the voxel window it asked for.
        self.reads.append(tuple((s.start, s.stop) for s in key))
        shape = tuple(s.stop - s.start for s in key)
        return np.full((*shape, 1), 42, dtype="uint64")

    def download(self, bbox, mip=0, **opts):
        self.downloads.append((bbox, mip, opts))
        return np.zeros((2, 2, 2, 1), dtype="uint64")


def _spec(name, caps, source):
    return DatasetSpec(
        name=name,
        backends=(BackendSpec("cave", "stack"),),
        voxel_size=(8, 8, 8),
        segmentation_source=source,
        capabilities=frozenset(caps),
        example_ids=(1,),
    )


class _DS(Dataset):
    """A dataset with no server behind it."""

    segmentation = None  # set per-instance below

    def __init__(self, spec, *, graph_source=None):
        self._spec = spec
        self._backend = spec.backends[0]
        self._version_request = "latest"
        self._resolved_version = None
        self._annotation_source = None
        self._graph = graph_source
        self.cg_calls: list = []

    # identity
    @property
    def spec(self):
        return self._spec

    @property
    def name(self):
        return self._spec.name

    @property
    def label(self):
        return self._spec.name

    @property
    def capabilities(self):
        return self._spec.capabilities

    def _graph_source(self):
        if self._graph is None:
            return super()._graph_source()
        return self._graph

    # the chunkedgraph, faked
    @property
    def client(self):
        return self

    @property
    def chunkedgraph(self):
        return self

    def get_roots(self, sv, timestamp=None):
        self.cg_calls.append(("get_roots", list(sv), timestamp))
        return np.full(len(sv), 999, dtype="int64")

    def _resolve_version_arg(self, version=None):
        class V:
            is_live = True
            value = None

        return V()

    # abstract hooks we never reach
    def _resolve_version(self, v): ...
    def _list_versions(self): ...
    def _find_version(self, ids, *, raise_missing): ...
    def _ids_exist(self, ids, version): ...
    def _fetch_annotations(self, source, version): ...
    def _fetch_edges(self, pre, post, version, **kw): ...
    def _fetch_synapses(self, pre, post, version, **opts): ...
    def _fetch_skeletons(self, ids, version, **opts): ...
    def _fetch_meshes(self, ids, version, **opts): ...


def _flat_ds():
    return _DS(_spec("hemibrain", {Cap.SEGMENTATION}, FLAT))


def _graph_ds():
    return _DS(
        _spec("flywire", {Cap.SEGMENTATION, Cap.CHUNKEDGRAPH}, FLAT),
        graph_source=GRAPHENE,
    )


@pytest.fixture(autouse=True)
def _stub_volumes(monkeypatch):
    """Hand out a fake volume, and remember which source was asked for."""
    made: dict = {}

    def fake(ds, source=None):
        source = source or ds._segmentation_source()
        return made.setdefault(source, _Vol(source))

    monkeypatch.setattr(volume, "get_volume", fake)
    return made


# ------------------------------------------------------------------ the capability split


def test_a_flat_volume_is_a_segmentation_but_not_a_chunkedgraph():
    ds = _flat_ds()
    assert ds.supports(Cap.SEGMENTATION)
    assert not ds.supports(Cap.CHUNKEDGRAPH)


def test_reading_the_volume_works_without_a_chunkedgraph(_stub_volumes):
    seg = Segmentation(_flat_ds())
    out = seg.locs_to_segments([[800, 800, 800]], progress=False)
    assert list(out) == [42]


def test_chunkedgraph_methods_refuse_on_a_flat_volume():
    seg = Segmentation(_flat_ds())
    for call in (
        lambda: seg.locs_to_supervoxels([[800, 800, 800]]),
        lambda: seg.update_ids([1]),
        lambda: seg.is_latest_root([1]),
        lambda: seg.roots_to_supervoxels([1]),
    ):
        with pytest.raises(CapabilityError, match="chunkedgraph"):
            call()


def test_version_is_refused_on_a_flat_volume_not_ignored():
    """The volume *is* the version. Accepting `version=` and quietly ignoring it would
    hand back v1.2 answers to someone who asked for v1.1 and thought they got them."""
    seg = Segmentation(_flat_ds())
    with pytest.raises(CapabilityError, match="version"):
        seg.locs_to_segments([[800, 800, 800]], version=630)


# ------------------------------------------------------------------------ which volume


def test_supervoxels_are_read_from_the_graph_not_the_display_volume(_stub_volumes):
    """FlyWire's spec points at the flat v783 bucket - right for meshes and for
    neuroglancer, and catastrophic here: a flat volume hands back *root IDs*, and
    `get_roots` is idempotent on roots, so the wrong answer would survive every sanity
    check and be wrong only for versions other than 783."""
    ds = _graph_ds()
    Segmentation(ds).locs_to_supervoxels([[800, 800, 800]], progress=False)

    assert GRAPHENE in _stub_volumes, "supervoxel lookup did not read the graph source"
    assert FLAT not in _stub_volumes, "supervoxel lookup read the flat display volume"


def test_locs_to_segments_on_a_chunkedgraph_goes_through_the_graph(_stub_volumes):
    ds = _graph_ds()
    out = Segmentation(ds).locs_to_segments([[800, 800, 800]], progress=False)

    assert ds.cg_calls, "did not consult the chunkedgraph"
    name, svs, _ = ds.cg_calls[0]
    assert name == "get_roots" and svs == [42]
    assert list(out) == [999]


def test_reading_the_volume_uses_the_display_source(_stub_volumes):
    """Meshes, cutouts and 'what is at this point' all want the volume the spec names."""
    Segmentation(_graph_ds()).get_segmentation_cutout([[0, 0, 0], [16, 16, 16]])
    assert FLAT in _stub_volumes


# ----------------------------------------------------------------------------- units


def test_coordinates_are_nanometres_by_default(_stub_volumes):
    """Every other connecto call speaks nanometres. If this one defaulted to voxels,
    feeding it `ds.connectivity.synapses()` output - the obvious thing to do - would
    look up a point eight times too far out and quietly return the wrong body."""
    seg = Segmentation(_flat_ds())
    seg.locs_to_segments([[800, 800, 800]], progress=False)

    vol = _stub_volumes[FLAT]
    (start, _), *_ = vol.reads[0]
    assert start == 100, "800 nm should be voxel 100 at 8 nm/voxel"


def test_voxels_can_be_passed_explicitly(_stub_volumes):
    seg = Segmentation(_flat_ds())
    seg.locs_to_segments([[100, 100, 100]], units="voxel", progress=False)

    vol = _stub_volumes[FLAT]
    (start, _), *_ = vol.reads[0]
    assert start == 100


def test_a_nonsense_unit_is_rejected():
    seg = Segmentation(_flat_ds())
    with pytest.raises(ValueError, match="'nm' or 'voxel'"):
        seg.locs_to_segments([[1, 1, 1]], units="microns")


# --------------------------------------------------------------------------- cutouts


def test_the_cutout_passes_a_bbox_not_a_list(_stub_volumes):
    """`download(bbox=[[...],[...]])` raises `AttributeError: 'list' object has no
    attribute 'start'` - it takes the list for a sequence of slices. Which is how this
    method came to have never worked, on any backend."""
    from connecto.precomputed import Bbox

    Segmentation(_flat_ds()).get_segmentation_cutout([[0, 0, 0], [800, 800, 800]])

    bbox, _, _ = _stub_volumes[FLAT].downloads[0]
    assert isinstance(bbox, Bbox)
    assert list(bbox.minpt) == [0, 0, 0]
    assert list(bbox.maxpt) == [100, 100, 100]  # nm -> voxels


def test_agglomerate_is_only_sent_to_graphene(_stub_volumes):
    """`agglomerate` rolls supervoxels up into roots. A flat volume has nothing to roll
    up, and no way to answer the question if asked."""
    Segmentation(_flat_ds()).get_segmentation_cutout([[0, 0, 0], [16, 16, 16]])
    _, _, opts = _stub_volumes[FLAT].downloads[0]
    assert "agglomerate" not in opts

    ds = _graph_ds()
    volume.segmentation_cutout(ds, [[0, 0, 0], [16, 16, 16]], source=GRAPHENE)
    _, _, opts = _stub_volumes[GRAPHENE].downloads[0]
    assert opts["agglomerate"] is True
