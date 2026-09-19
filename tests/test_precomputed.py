"""connecto's own precomputed reader.

Two halves. The offline half pins down the bit-twiddling - hashes, morton codes,
delta-encoded shard indices, the compressed_segmentation decoder - against fixtures
and synthetic data, because those are the parts where a wrong answer is still a
well-formed array and nothing looks broken.

The network half checks the reader against **cloud-volume**, which is the reference
implementation and the thing this module replaced. Reading the same box or the same
mesh both ways and comparing is the only test that can catch a misunderstanding of
the format, as opposed to a mistake in implementing what we understood.
"""

from __future__ import annotations

import gzip
import warnings

import numpy as np
import pytest

from connecto.precomputed import Bbox, Store, mesh, mmh3
from connecto.precomputed import meta as meta_mod
from connecto.precomputed.codecs import _decompress_segmentation_numpy, decode_chunk
from connecto.precomputed.graphene import (
    GrapheneMeta,
    GraphenePath,
    parse_graphene_path,
)
from connecto.precomputed.image import ImageSource, compressed_morton_code
from connecto.precomputed.sharding import ShardingSpec, ShardReader
from connecto.precomputed.store import normalize_url, split_protocol

# --------------------------------------------------------------------------- hashes

# Straight out of cloud-volume's mmh3, which is the implementation the neuroglancer
# sharded spec is defined by. Any drift here silently sends reads to the wrong shard.
MURMUR_VECTORS = {
    0: 5148371408780832321,
    1: 16770674756601302682,
    42: 13982433266630259834,
    2**32: 13524640716595723620,
    2**63: 11063714688786943912,
    2**64 - 1: 6291360166951214362,
    720575940621039145: 6760157900999300077,
    1734350788: 7715152132933788294,
}


@pytest.mark.parametrize(("value", "expected"), list(MURMUR_VECTORS.items()))
def test_murmurhash3_matches_the_reference_implementation(value, expected):
    assert mmh3.hash64_low(value) == expected


def test_murmurhash_handles_every_tail_length():
    """The tail is 15 separate fall-through cases; an off-by-one in any of them only
    shows up for inputs of that exact length."""
    seen = {mmh3.hash128(bytes(range(n))) for n in range(0, 48)}
    assert len(seen) == 48  # all distinct, none crashed


# ----------------------------------------------------------------------- morton code


def test_compressed_morton_code_interleaves_bits():
    grid = [8, 8, 8]
    assert compressed_morton_code([0, 0, 0], grid) == 0
    assert compressed_morton_code([1, 0, 0], grid) == 0b001
    assert compressed_morton_code([0, 1, 0], grid) == 0b010
    assert compressed_morton_code([0, 0, 1], grid) == 0b100
    assert compressed_morton_code([1, 1, 1], grid) == 0b111


def test_compressed_morton_code_skips_axes_that_have_run_out():
    """The "compressed" part: a short axis stops contributing bits, so the code stays
    dense. Interleaving naively would leave gaps and address the wrong chunk."""
    # x needs 3 bits, y and z need 1 each. After bit 0, only x is still contributing.
    assert compressed_morton_code([0b100, 0, 0], [8, 2, 2]) == 0b10000
    assert compressed_morton_code([0, 0, 0], [1, 1, 1]) == 0


# ------------------------------------------------------------------------------ URLs


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("precomputed://gs://bucket/path", "https://storage.googleapis.com/bucket/path"),
        ("gs://bucket/path", "https://storage.googleapis.com/bucket/path"),
        ("precomputed://s3://bucket/path", "https://bucket.s3.amazonaws.com/path"),
        ("graphene://https://server/segmentation/table/x", "https://server/segmentation/table/x"),
        (
            "graphene://middleauth+https://server/segmentation/table/x",
            "https://server/segmentation/table/x",
        ),
    ],
)
def test_urls_resolve_to_plain_https(source, expected):
    assert normalize_url(source) == expected


def test_middleauth_is_stripped_not_resolved():
    """`middleauth+` tells a *viewer* to run CAVE's login flow. Left in the hostname
    it is a DNS failure, and the error says nothing about why."""
    assert "middleauth" not in normalize_url("graphene://middleauth+https://s/segmentation/table/d")


def test_split_protocol():
    assert split_protocol("precomputed://gs://b") == ("precomputed", "gs://b")
    assert split_protocol("graphene://https://s/x") == ("graphene", "https://s/x")
    assert split_protocol("https://s/x") == ("", "https://s/x")


def test_an_unresolvable_source_says_so():
    with pytest.raises(ValueError, match="Cannot resolve"):
        normalize_url("dvid://server/uuid")


# ------------------------------------------------------------------------------ Bbox


def test_bbox_is_half_open():
    assert Bbox([0, 0, 0], [10, 10, 10]).volume == 1000
    assert not Bbox([0, 0, 0], [1, 1, 1]).subvoxel()
    assert Bbox([5, 0, 0], [5, 10, 10]).subvoxel()


def test_bbox_clamp_can_produce_an_empty_box():
    """Chunks on a volume's edge clip away to nothing, and a box that has become
    empty must report itself as such rather than wrap around to a negative size."""
    clipped = Bbox.clamp(Bbox([100, 100, 100], [200, 200, 200]), Bbox([0, 0, 0], [50, 50, 50]))
    assert clipped.subvoxel()


# ------------------------------------------------------------- sharding arithmetic


def _spec(**kw):
    base = dict(
        preshift_bits=0,
        hash="identity",
        minishard_bits=2,
        shard_bits=4,
        minishard_index_encoding="raw",
        data_encoding="raw",
    )
    base.update(kw)
    return ShardingSpec(**base)


def test_shard_location_splits_the_hashed_key():
    spec = _spec(minishard_bits=2, shard_bits=4)
    # 0b110101 -> minishard 0b01, shard 0b1101
    assert spec.locate(0b110101) == (0b1101, 0b01)


def test_preshift_bits_are_applied_before_hashing():
    spec = _spec(preshift_bits=3)
    assert spec.locate(0b111_000) == spec.locate(0b111_111)


def test_shard_filenames_are_zero_padded_to_the_bit_width():
    assert _spec(shard_bits=16).shard_filename(1) == "0001.shard"
    assert _spec(shard_bits=4).shard_filename(1) == "1.shard"
    assert _spec(shard_bits=17).shard_filename(255) == "000ff.shard"


def test_a_minishard_index_is_delta_decoded():
    """Ids, offsets and sizes are all deltas, and an offset is a *gap* from the end of
    the previous value - so the sizes have to be added back in. Getting that wrong
    reads plausible bytes from the wrong place."""
    ids = [10, 5, 1]          # -> 10, 15, 16
    offsets = [0, 4, 0]       # -> 0, (0+100)+4=104, (104+7)+0=111
    sizes = [100, 7, 3]
    raw = np.array(ids + offsets + sizes, dtype="<u8").tobytes()

    spec = _spec(minishard_bits=1)  # index_length = 32
    keys, offsets, sizes = ShardReader(Store("gs://x"), spec)._decode_minishard(raw)

    assert list(keys) == [10, 15, 16]                 # sorted by key
    assert list(offsets) == [0 + 32, 104 + 32, 111 + 32]
    assert list(sizes) == [100, 7, 3]


# ------------------------------------------------------ a whole shard, synthesised


class _FakeStore(Store):
    """A store over an in-memory dict of bytes."""

    def __init__(self, files: dict):
        super().__init__("gs://fake")
        self.files = files
        self.reads: list = []

    def at(self, *parts):
        return self

    def get(self, key="", *, missing_ok=True):
        self.reads.append(key)
        return self.files.get(key)

    def get_range(self, key, start, end, *, missing_ok=True):
        self.reads.append(key)
        blob = self.files.get(key)
        return None if blob is None else blob[int(start) : int(end)]


def _build_shard(values: dict, spec: ShardingSpec) -> bytes:
    """Pack ``{key: payload}`` into one shard file, per the spec."""
    by_minishard: dict = {}
    for key, payload in sorted(values.items()):
        by_minishard.setdefault(spec.locate(key)[1], []).append((key, payload))

    data, minishard_indices = b"", {}
    for minishard, entries in sorted(by_minishard.items()):
        # Offsets restart from the top of the data region for every minishard - they
        # are absolute there, not relative to whatever the previous minishard wrote.
        ids, offsets, sizes, prev_key, prev_end = [], [], [], 0, 0
        for key, payload in entries:
            ids.append(key - prev_key)
            offsets.append(len(data) - prev_end)
            sizes.append(len(payload))
            prev_key, prev_end = key, len(data) + len(payload)
            data += payload
        table = np.array(ids + offsets + sizes, dtype="<u8").tobytes()
        if spec.minishard_index_encoding == "gzip":
            table = gzip.compress(table)
        minishard_indices[minishard] = table

    index = np.zeros((1 << spec.minishard_bits, 2), dtype="<u8")
    tail, cursor = b"", len(data)
    for minishard, table in sorted(minishard_indices.items()):
        index[minishard] = (cursor, cursor + len(table))
        tail += table
        cursor += len(table)
    return index.tobytes() + data + tail


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_a_synthesised_shard_round_trips(encoding):
    spec = _spec(minishard_bits=2, shard_bits=4, minishard_index_encoding=encoding)
    values = {k: bytes([k % 251]) * (10 + k) for k in range(1, 40)}
    store = _FakeStore({spec.shard_filename(spec.locate(k)[0]): b"" for k in values})
    for shard in {spec.locate(k)[0] for k in values}:
        mine = {k: v for k, v in values.items() if spec.locate(k)[0] == shard}
        store.files[spec.shard_filename(shard)] = _build_shard(mine, spec)

    reader = ShardReader(store, spec)
    for key, payload in values.items():
        assert reader.get(key) == payload, key
    assert reader.get(9999) is None


def test_shard_indices_are_read_once_per_shard():
    """Without this cache every chunk of a cutout costs three GETs instead of one,
    which is the difference between a usable reader and an unusable one."""
    spec = _spec(minishard_bits=0, shard_bits=4)
    values = {k: b"payload" for k in range(0, 200, 16)}   # all in shard 0
    store = _FakeStore({spec.shard_filename(0): _build_shard(values, spec)})
    reader = ShardReader(store, spec)

    for key in values:
        reader.get(key)
    # 1 shard index + 1 minishard index + one data read per value.
    assert len(store.reads) == 2 + len(values)


# --------------------------------------------------------- compressed_segmentation

try:
    import compressed_segmentation as _cseg
except ImportError:  # pragma: no cover
    _cseg = None

requires_cseg = pytest.mark.skipif(
    _cseg is None, reason="compressed-segmentation not installed (it is the encoder here)"
)


@requires_cseg
@pytest.mark.parametrize("shape", [(64, 64, 64), (8, 8, 8), (11, 7, 5), (84, 256, 32), (1, 1, 1)])
@pytest.mark.parametrize("n_labels", [1, 3, 60, 5000])
def test_the_numpy_decoder_matches_the_c_one(shape, n_labels):
    """Including shapes that are not a multiple of the block size: a block is always
    *encoded* at its full nominal size and clipped on the way out, so a partial block
    is the case most likely to be got wrong."""
    rng = np.random.default_rng(abs(hash((shape, n_labels))) % 2**32)
    arr = np.asfortranarray(rng.integers(0, n_labels, size=shape, dtype=np.uint64))
    blob = _cseg.compress(arr, block_size=(8, 8, 8), order="F")

    mine = _decompress_segmentation_numpy(blob, shape, np.uint64, (8, 8, 8))
    assert np.array_equal(mine.reshape(shape, order="F"), arr)


@requires_cseg
def test_the_numpy_decoder_handles_uint32_and_odd_block_sizes():
    rng = np.random.default_rng(0)
    arr = np.asfortranarray(rng.integers(0, 400, size=(37, 13, 29), dtype=np.uint32))
    blob = _cseg.compress(arr, block_size=(16, 16, 2), order="F")
    mine = _decompress_segmentation_numpy(blob, (37, 13, 29), np.uint32, (16, 16, 2))
    assert np.array_equal(mine.reshape((37, 13, 29), order="F"), arr)


def test_a_missing_chunk_reads_as_empty_not_as_an_error():
    """`fill_missing`. A chunkedgraph's chunk list routinely names chunks that were
    never written, and those are empty space, not failures."""
    out = decode_chunk(None, "compressed_segmentation", (4, 4, 4), np.uint64, block_size=(8, 8, 8))
    assert out.shape == (4, 4, 4, 1)
    assert not out.any()


def test_an_image_encoding_is_refused_with_an_explanation():
    with pytest.raises(ValueError, match="segmentation volumes only"):
        decode_chunk(b"\x00" * 16, "jpeg", (2, 2, 2), np.uint8)


# ------------------------------------------------------------------------- metadata


INFO = {
    "type": "segmentation",
    "data_type": "uint64",
    "num_channels": 1,
    "mesh": "mesh",
    "scales": [
        {"key": "8_8_8", "resolution": [8, 8, 8], "size": [1000, 800, 600],
         "voxel_offset": [10, 20, 30], "chunk_sizes": [[64, 64, 64]],
         "encoding": "compressed_segmentation", "compressed_segmentation_block_size": [8, 8, 8]},
        {"key": "16_16_16", "resolution": [16, 16, 16], "size": [500, 400, 300],
         "voxel_offset": [5, 10, 15], "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
    ],
}


def test_scales_can_be_named_by_index_key_or_resolution():
    m = meta_mod.PrecomputedMeta(INFO)
    assert m.to_mip(1) == m.to_mip("16_16_16") == m.to_mip([16, 16, 16]) == 1
    assert m.to_mip(-1) == 1


def test_an_unknown_scale_lists_the_ones_that_exist():
    m = meta_mod.PrecomputedMeta(INFO)
    with pytest.raises(ValueError, match="8_8_8"):
        m.to_mip("32_32_32")
    with pytest.raises(ValueError, match="2 available"):
        m.to_mip(7)


def test_a_fractional_resolution_is_not_rounded():
    """FANC's mip 0 is 17.2 x 17.2 x 45 nm. Rounding that to 17 is a 1.2% error, and
    1.2% of the far edge of the volume is a hundred voxels - far enough to land a
    point lookup inside the neighbouring neuron, with nothing about the answer
    looking wrong."""
    info = {**INFO, "scales": [{**INFO["scales"][0], "resolution": [17.2, 17.2, 45]}]}
    m = meta_mod.PrecomputedMeta(info)

    assert list(m.resolution(0)) == [17.2, 17.2, 45]
    # Voxel counts, unlike resolutions, really are integers.
    assert m.chunk_size(0).dtype == np.int64
    assert m.voxel_offset(0).dtype == np.int64
    assert m.volume_size(0).dtype == np.int64


def test_bounds_start_at_the_voxel_offset():
    m = meta_mod.PrecomputedMeta(INFO)
    assert list(m.bounds(0).minpt) == [10, 20, 30]
    assert list(m.bounds(0).maxpt) == [1010, 820, 630]


def test_bbox_to_mip_rounds_outward():
    """Rounding inward drops the voxels at a box's edge, which on the sparse-volume
    path is what makes a neuron look chopped off at chunk boundaries."""
    m = meta_mod.PrecomputedMeta(INFO)
    out = m.bbox_to_mip(Bbox([1, 1, 1], [3, 3, 3]), mip=0, to_mip=1)
    assert list(out.minpt) == [0, 0, 0]
    assert list(out.maxpt) == [2, 2, 2]


# ------------------------------------------------------------- assembling a cutout


def _fake_volume(monkeypatch, fill=7):
    """An ImageSource whose chunks are constant, so pasting is what is under test."""
    m = meta_mod.PrecomputedMeta(INFO)
    src = ImageSource(_FakeStore({}), m, fill_missing=True)

    def fake_fetch(box, pos, mip, source):
        shape = tuple(int(v) for v in box.size3)
        return np.full((*shape, 1), fill, dtype="uint64")

    monkeypatch.setattr(src, "_fetch_chunk", fake_fetch)
    return src


def test_a_cutout_returns_the_shape_that_was_asked_for(monkeypatch):
    src = _fake_volume(monkeypatch)
    out = src.download(Bbox([100, 100, 100], [150, 160, 170]), mip=0)
    assert out.shape == (50, 60, 70, 1)
    assert (out == 7).all()


def test_a_cutout_running_off_the_volume_is_zero_padded(monkeypatch):
    """cloud-volume's semantics, and the useful ones: the caller gets the shape it
    asked for, and the region outside the volume is empty by definition."""
    src = _fake_volume(monkeypatch)
    bounds = src.meta.bounds(0)
    out = src.download(Bbox(bounds.maxpt - 10, bounds.maxpt + 10), mip=0)
    assert out.shape == (20, 20, 20, 1)
    assert (out[:10, :10, :10] == 7).all()
    assert not out[10:, 10:, 10:].any()


def test_a_cutout_entirely_outside_the_volume_is_all_zeros(monkeypatch):
    src = _fake_volume(monkeypatch)
    bounds = src.meta.bounds(0)
    out = src.download(Bbox(bounds.maxpt + 100, bounds.maxpt + 150), mip=0)
    assert out.shape == (50, 50, 50, 1)
    assert not out.any()


def test_bounded_refuses_instead_of_clipping(monkeypatch):
    m = meta_mod.PrecomputedMeta(INFO)
    src = ImageSource(_FakeStore({}), m, bounded=True)
    with pytest.raises(ValueError, match="outside the volume's bounds"):
        src.download(Bbox([0, 0, 0], [5000, 5000, 5000]), mip=0)


def test_unsharded_chunks_are_addressed_by_filename():
    m = meta_mod.PrecomputedMeta(INFO)
    store = _FakeStore({})
    ImageSource(store, m).download(Bbox([10, 20, 30], [20, 30, 40]), mip=0)
    assert store.reads == ["10-74_20-84_30-94"]


# ------------------------------------------------------------------ mesh manifests


def test_a_truncated_mesh_manifest_is_rejected():
    """A short manifest would otherwise decode into plausible-looking fragment
    positions and read arbitrary bytes as draco."""
    with pytest.raises(ValueError, match="truncated"):
        mesh.MultiResManifest.from_binary(b"\x00" * 64)


def test_a_mesh_manifest_round_trips():
    header = (
        np.array([32, 32, 32], dtype="<f4").tobytes()          # chunk_shape
        + np.array([1, 2, 3], dtype="<f4").tobytes()           # grid_origin
        + np.array([2], dtype="<u4").tobytes()                 # num_lods
        + np.array([1, 2], dtype="<f4").tobytes()              # lod_scales
        + np.array([[0, 0, 0], [0, 0, 0]], dtype="<f4").tobytes()   # vertex_offsets
        + np.array([2, 1], dtype="<u4").tobytes()              # num_fragments_per_lod
    )
    body = (
        np.array([0, 1, 0, 1, 0, 1], dtype="<u4").tobytes()    # lod 0 positions (F order)
        + np.array([10, 20], dtype="<u4").tobytes()            # lod 0 sizes
        + np.array([0, 0, 0], dtype="<u4").tobytes()           # lod 1 positions
        + np.array([30], dtype="<u4").tobytes()                # lod 1 sizes
    )
    manifest = mesh.MultiResManifest.from_binary(header + body)

    assert manifest.num_lods == 2
    assert manifest.lod_byte_sizes() == [30, 30]
    assert list(manifest.grid_origin) == [1, 2, 3]
    assert manifest.fragment_positions[0].tolist() == [[0, 0, 0], [1, 1, 1]]


# ------------------------------------------------------------------------ graphene

GRAPHENE_INFO = {
    "data_type": "uint64",
    "data_dir": "gs://bucket/watershed",
    "mesh": "meshes",
    "chunks_start_at_voxel_offset": True,
    "app": {"supported_api_versions": [0, 1]},
    "mesh_metadata": {"max_meshed_layer": 6, "uniform_draco_grid_size": 21},
    "graph": {
        "chunk_size": [256, 256, 512],
        "cv_mip": 0,
        "n_bits_for_layer_id": 8,
        "n_layers": 10,
        "spatial_bit_masks": {str(i): b for i, b in enumerate([0, 10, 8, 7, 6, 5, 4, 3, 2, 1, 1])},
    },
    "scales": INFO["scales"],
}


@pytest.mark.parametrize(
    ("url", "modality", "version", "dataset"),
    [
        ("graphene://https://s.org/segmentation/table/ds", "segmentation", "table", "ds"),
        ("graphene://https://s.org/segmentation/1.0/ds", "segmentation", "1.0", "ds"),
        ("graphene://https://s.org/segmentation/api/v1/table/ds", "segmentation", "v1", "ds"),
    ],
)
def test_every_graphene_url_form_parses(url, modality, version, dataset):
    path = parse_graphene_path(url)
    assert (path.modality, path.version, path.dataset) == (modality, version, dataset)


def test_the_url_form_decides_the_api_dialect():
    """A deployment still publishing `/segmentation/1.0/` wants `/meshing/1.0/` too;
    asking such a server for the modern path 404s."""
    legacy = GrapheneMeta(GRAPHENE_INFO, parse_graphene_path("graphene://https://s.org/segmentation/1.0/ds"))
    assert legacy.manifest_endpoint == "https://s.org/meshing/1.0/ds/manifest"

    modern = GrapheneMeta(GRAPHENE_INFO, parse_graphene_path("graphene://https://s.org/segmentation/table/ds"))
    assert modern.manifest_endpoint == "https://s.org/meshing/api/v1/table/ds/manifest"


def _graphene_meta():
    return GrapheneMeta(GRAPHENE_INFO, GraphenePath("https", "s.org", "segmentation", "table", "ds"))


def test_a_graphene_label_decodes_to_layer_and_chunk():
    m = _graphene_meta()
    label = m.encode_label(2, 5, 6, 7, 1234)
    assert m.decode_layer_id(label) == 2
    assert list(m.decode_chunk_position(label)) == [5, 6, 7]
    assert m.decode_segid(label) == 1234


def test_chunk_positions_come_from_the_label_alone():
    """No volume access at all - which is what makes it possible to know which chunks
    a neuron occupies from its L2 ids, and is the whole basis of the sparse-volume
    path."""
    m = _graphene_meta()
    labels = [m.encode_label(2, x, y, z, 1) for x, y, z in [(1, 2, 3), (1, 2, 3), (9, 9, 9)]]
    positions = np.unique([m.decode_chunk_position(x) for x in labels], axis=0)
    assert positions.tolist() == [[1, 2, 3], [9, 9, 9]]


def test_the_graphene_reader_knows_its_watershed_bucket():
    m = _graphene_meta()
    assert m.data_dir == "gs://bucket/watershed"
    assert m.n_layers == 10
    assert list(m.graph_chunk_size) == [256, 256, 512]
    assert m.chunks_start_at_voxel_offset is True
    assert m.get_draco_grid_size(2) == 21


# ============================================================ against cloud-volume

pytestmark_network = pytest.mark.network

HEMIBRAIN = "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
WATERSHED = "precomputed://gs://seunglab2/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200"
FLYWIRE_FLAT = "precomputed://gs://flywire_v141_m783"


@pytest.fixture(scope="module")
def cv():
    return pytest.importorskip("cloudvolume", reason="cloud-volume is the test oracle")


def _both(source, mip):
    import cloudvolume

    from connecto.precomputed import Volume

    mine = Volume(source, mip=mip)
    ref = cloudvolume.CloudVolume(
        source, use_https=True, progress=False, fill_missing=True, bounded=False, mip=mip
    )
    return mine, ref


@pytest.mark.network
@pytest.mark.parametrize(
    ("source", "mip", "label"),
    [
        (HEMIBRAIN, 4, "sharded, compressed_segmentation + gzip"),
        (WATERSHED, 2, "unsharded"),
        (FLYWIRE_FLAT, 3, "sharded, minishard_bits=0"),
    ],
)
def test_cutouts_are_identical_to_cloud_volume(cv, source, mip, label):
    from cloudvolume.lib import Bbox as CVBbox

    mine, ref = _both(source, mip)
    centre = ((mine.bounds.minpt + mine.bounds.maxpt) // 2).astype(int)

    for offset, size in [(0, 64), (13, 100)]:   # aligned-ish, then deliberately not
        lo = centre + offset
        box = Bbox(lo, lo + size)
        got = np.asarray(mine.download(box, mip=mip))
        want = np.asarray(ref.download(CVBbox(lo, lo + size), mip=mip))
        assert np.array_equal(got, want), f"{label} @ offset {offset}"


@pytest.mark.network
def test_geometry_matches_cloud_volume_at_every_scale(cv):
    mine, ref = _both(HEMIBRAIN, 0)
    assert list(mine.meta.available_mips) == list(ref.meta.available_mips)
    for mip in mine.meta.available_mips:
        assert np.array_equal(mine.meta.resolution(mip), ref.meta.resolution(mip))
        assert np.array_equal(mine.meta.chunk_size(mip), ref.meta.chunk_size(mip))
        assert np.array_equal(mine.meta.voxel_offset(mip), ref.meta.voxel_offset(mip))
        assert np.array_equal(mine.meta.bounds(mip).minpt, ref.meta.bounds(mip).minpt)
        assert np.array_equal(mine.meta.bounds(mip).maxpt, ref.meta.bounds(mip).maxpt)


@pytest.mark.network
def test_a_cutout_off_the_end_matches_cloud_volume(cv):
    from cloudvolume.lib import Bbox as CVBbox

    mine, ref = _both(HEMIBRAIN, 4)
    lo = mine.bounds.maxpt - 30
    hi = mine.bounds.maxpt + 50
    got = np.asarray(mine.download(Bbox(lo, hi), mip=4))
    want = np.asarray(ref.download(CVBbox(lo, hi), mip=4))
    assert got.shape == want.shape == (80, 80, 80, 1)
    assert np.array_equal(got, want)


@pytest.mark.network
@pytest.mark.parametrize("lod", [2, 3])
def test_multiresolution_meshes_are_identical_to_cloud_volume(cv, lod):
    mine, ref = _both(HEMIBRAIN, 0)
    segid = 1734350788

    got = mine.mesh.get(segid, lod=lod)
    want = ref.mesh.get(segid, lod=lod)
    want = want[segid] if isinstance(want, dict) else want

    assert np.allclose(got.vertices, np.asarray(want.vertices, dtype="float64"), atol=1e-6)
    assert np.array_equal(got.faces, np.asarray(want.faces))


@pytest.mark.network
def test_asking_for_a_level_of_detail_that_does_not_exist_says_so(cv):
    mine, _ = _both(HEMIBRAIN, 0)
    with pytest.raises(ValueError, match="out of range"):
        mine.mesh.get(1734350788, lod=99)


@pytest.mark.network
def test_a_missing_mesh_raises_rather_than_returning_an_empty_one(cv):
    mine, _ = _both(HEMIBRAIN, 0)
    with pytest.raises(KeyError):
        mine.mesh.get(1, lod=0)


# ----------------------------------------------------------------- fan-out limits


def test_pool_ceiling_covers_the_mesh_fan_out():
    """The connection pool must hold what the default mesh fan-out asks for.

    These three numbers are one decision split across two modules, and breaking the
    relationship does not fail: urllib3 discards the connections it cannot keep and
    pays a fresh TLS handshake per read, which reads as a slow network rather than as
    a bug. That is what this asserts, and it is why `POOL_MAXSIZE` is derived.
    """
    from connecto.precomputed import limits
    from connecto.precomputed.store import _default_session

    assert limits.POOL_MAXSIZE >= limits.DEFAULT_MESH_WORKERS * limits.DEFAULT_MESH_PARALLEL

    # The defaults were never the case that would break it - a maintainer changes
    # those deliberately. A caller tuning `meshes.get(max_workers=8, parallel=64)` is,
    # and has no way to raise the ceiling themselves, so it has to cover them too.
    assert limits.POOL_MAXSIZE >= 8 * 64

    # And the session actually built from it agrees - the constant is not decorative.
    adapter = _default_session().get_adapter("https://storage.googleapis.com")
    assert adapter._pool_maxsize == limits.POOL_MAXSIZE


@pytest.mark.parametrize(
    ("version", "threaded"),
    [
        ("1.4.0", False),
        ("1.7.0", False),
        # Published before the GIL work landed, and measured to behave exactly like
        # 1.7 - the case a major-version check got wrong.
        ("2.0.0", False),
        ("2.0.9", False),
        ("2.1.0", True),
        ("2.1.3", True),
        ("2.10.0", True),
        ("3.0.0", True),
        # The one that tells numbers from strings: "10.0.0" < "2.1" lexically.
        ("10.0.0", True),
        ("2.1.0rc1", True),
        ("weird", False),
    ],
)
def test_decode_workers_follows_the_installed_dracopy(version, threaded, monkeypatch):
    """Decode threads only where decoding can actually overlap.

    DracoPy holds the GIL through `decode` until 2.1 (seung-lab/DracoPy#67); before
    that, pointing threads at it costs a few percent and gains nothing. connecto
    declares `DracoPy>=1.4.0`, so that is most installs, and they must come out at 1
    - which `MultiResMeshSource.get` turns into the plain serial loop rather than a
    pool of one.
    """
    from connecto.precomputed import limits

    monkeypatch.setattr(limits.importlib.metadata, "version", lambda _: version)
    limits.decode_workers.cache_clear()
    try:
        assert (limits.decode_workers() > 1) is threaded
    finally:
        limits.decode_workers.cache_clear()


# ------------------------------------------------------------------- seam welding


def _dedup_reference(vertices, faces, is_chunk_aligned):
    """The straightforward reading of the seam-welding rule, done with `np.unique`.

    Kept as the yardstick for the fast path in `graphene_mesh`, which reaches the
    same answer by folding each row into one integer and renumbering through a lookup
    table. This is the version that is obviously correct; that one is the quick one.
    """
    _, inverse, counts = np.unique(
        vertices, return_inverse=True, return_counts=True, axis=0
    )
    inverse = np.asarray(inverse).reshape(-1)
    doubled = np.isin(inverse, np.flatnonzero(counts == 2))
    merge = doubled & np.asarray(is_chunk_aligned, dtype=bool)
    key = np.where(merge, inverse, np.arange(len(vertices), dtype=np.int64) + len(counts))
    corners = faces.reshape(-1)
    _, first, new_faces = np.unique(
        key[corners], return_index=True, return_inverse=True
    )
    return vertices[corners[first]], np.asarray(new_faces).reshape(-1, 3)


def test_deduplicate_vertices_welds_only_aligned_pairs():
    """The rule itself, on a case small enough to read.

    Rows 0 and 1 are the same coordinate, both on a boundary: one seam, welded.
    Rows 2-4 are the same coordinate three times over - a real feature of the
    surface, not a seam - and must survive whatever their flags say. Rows 5 and 6
    are a duplicate pair with only one of them on a boundary, which is not a seam
    either: a fragment's interior vertex that happens to coincide with its
    neighbour's edge is not evidence that the two edges are the same edge.
    """
    from connecto.precomputed.graphene_mesh import _deduplicate_vertices

    vertices = np.array(
        [[0.0, 0, 0], [0.0, 0, 0],           # aligned pair      -> welded
         [1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0],  # tripled        -> kept apart
         [2.0, 0, 0], [2.0, 0, 0]],          # half-aligned pair -> kept apart
        dtype="float64",
    )
    aligned = np.array([True, True, True, True, True, True, False])
    faces = np.array([[0, 2, 5], [1, 3, 6], [4, 5, 6]], dtype="int64")

    verts, out_faces = _deduplicate_vertices(vertices, faces, aligned)

    # 7 vertices in, one pair fused, so 6 out - and the tripled coordinate is still
    # three separate vertices.
    assert len(verts) == 6
    assert (verts == [1.0, 0, 0]).all(axis=1).sum() == 3
    assert (verts == [2.0, 0, 0]).all(axis=1).sum() == 2
    # Corners 0 and 1 named different rows of `vertices` and now name one vertex.
    assert out_faces[0, 0] == out_faces[1, 0]
    # The surface is unchanged: every corner still sits where it did.
    assert np.array_equal(verts[out_faces], vertices[faces])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("integral", [True, False], ids=["packed", "lexsort-fallback"])
def test_deduplicate_vertices_matches_reference(seed, integral):
    """The fast path and the obvious path agree, exactly, on messy input.

    Random rather than fixed because the interesting cases are combinations - a
    coordinate group of the wrong size, a group split across the aligned flag, a
    vertex no face names - and enumerating them by hand is how you miss one. Small
    coordinate range so duplicates are common, and unused vertices so the
    drop-what-the-faces-never-name path is exercised too.

    Run twice over: whole-number coordinates take the packed-integer sort, halves
    cannot be packed and fall back to `lexsort`. Both have to be exact, and the
    fallback is the one no real dataset exercises - so if it is ever wrong, this is
    the only place that would say so.
    """
    from connecto.precomputed.graphene_mesh import (
        _deduplicate_vertices,
        _packed_row_key,
    )

    rng = np.random.default_rng(seed)
    vertices = rng.integers(0, 12, size=(200, 3)).astype("float64")
    if not integral:
        vertices = vertices / 2
    aligned = rng.random(200) < 0.5
    faces = rng.integers(0, 200, size=(120, 3)).astype("int64")

    assert (_packed_row_key(vertices) is not None) is integral

    got_v, got_f = _deduplicate_vertices(vertices, faces, aligned)
    want_v, want_f = _dedup_reference(vertices, faces, aligned)

    assert np.array_equal(got_v, want_v)
    assert np.array_equal(got_f, want_f)
    # Whatever the renumbering did, it did not move the surface.
    assert np.array_equal(got_v[got_f], vertices[faces])


@pytest.mark.parametrize(
    ("why", "vertices"),
    [
        ("no rows at all", np.zeros((0, 3))),
        ("fractional", [[0.5, 0, 0], [1, 2, 3]]),
        ("nan", [[float("nan"), 0, 0], [1, 2, 3]]),
        ("inf", [[float("inf"), 0, 0], [1, 2, 3]]),
        ("too wide to pack", [[0, 0, 0], [2.0**40, 2.0**40, 2.0**40]]),
        ("past int64", [[0, 0, 0], [2.0**63, 0, 0]]),
    ],
)
def test_packed_row_key_declines_what_it_cannot_represent(why, vertices):
    """Every input the packing cannot hold exactly must be refused, not approximated.

    Fractional coordinates are the one that would silently corrupt: casting them to
    integers would make two distinct vertices compare equal and weld a seam that is
    not there. The rest are refused before the cast, because `astype(int64)` of a
    NaN or of something past the integer range is undefined - numpy warns and hands
    back a sentinel - so `filterwarnings("error")` is part of the assertion.
    """
    from connecto.precomputed.graphene_mesh import _packed_row_key

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _packed_row_key(np.array(vertices, dtype="float64")) is None


@pytest.mark.parametrize("seed", range(4))
def test_packed_row_key_orders_exactly_like_lexsort(seed):
    """The property the swap rests on: same order, so the output does not move.

    Packing x into the high bits makes an integer sort reproduce the coordinate
    order `lexsort` gives. If that ever stops holding the meshes stay correct but
    every vertex is renumbered, which is the kind of change that surfaces as a
    confusing diff somewhere downstream rather than as a failure here.
    """
    from connecto.precomputed.graphene_mesh import _packed_row_key

    rng = np.random.default_rng(seed)
    vertices = rng.integers(-500, 500, size=(1000, 3)).astype("float64")

    packed = np.argsort(_packed_row_key(vertices), kind="stable")
    lex = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    assert np.array_equal(vertices[packed], vertices[lex])


def test_deduplicate_vertices_empty():
    """An all-empty fragment set must not IndexError on `starts[0]`."""
    from connecto.precomputed.graphene_mesh import _deduplicate_vertices

    verts, faces = _deduplicate_vertices(
        np.zeros((0, 3)), np.zeros((0, 3), dtype="int64"), np.zeros(0, dtype=bool)
    )
    assert verts.shape == (0, 3)
    assert faces.shape == (0, 3)


@pytest.mark.network
@pytest.mark.parametrize(
    ("dataset", "root"),
    [
        ("flywire", 720575940621039145),      # unsharded / dynamic fragments
        ("microns", 864691135506210290),      # sharded, byte offsets from the service
        ("banc", 720575941509145950),         # sharded + dynamic, and stitched
    ],
)
def test_graphene_meshes_match_cloud_volume(cv, dataset, root):
    """The hard case: fragments come from shards *and* from the loose files written
    by re-meshing after a proofreading edit, and the seams between them have to be
    welded. Vertex order is not deterministic on either side, so compare as sets."""
    import connecto as co
    from connecto.precomputed.graphene import GrapheneMeta
    from connecto.precomputed.graphene_mesh import GrapheneMeshSource

    ds = {
        "flywire": lambda: co.FlyWire(backend="cave"),
        "microns": co.MICrONS,
        "banc": lambda: co.BANC(backend="cave"),
    }[dataset]()
    session = ds.client.chunkedgraph.session

    mine = GrapheneMeshSource(
        GrapheneMeta.fetch(ds._graph_source(), session=session), session=session
    ).get(root)

    # The reference has to come from cloud-volume itself, not from connecto.
    import cloudvolume

    cvol = cloudvolume.CloudVolume(
        ds._graph_source(), use_https=True, progress=False, fill_missing=True
    )
    want = cvol.mesh.get(root)
    want = want[root] if isinstance(want, dict) else want
    want_v = np.asarray(want.vertices, dtype="float64")

    assert mine.vertices.shape == want_v.shape
    assert mine.faces.shape == np.asarray(want.faces).shape
    assert np.array_equal(
        np.unique(np.round(mine.vertices, 4), axis=0),
        np.unique(np.round(want_v, 4), axis=0),
    )
