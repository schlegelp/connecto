"""The level-2 chunk cache. CAVE-only, and not every datastack has one.

CAVE's chunkedgraph builds the segmentation out of chunks, and the L2 cache stores a
summary of each level-2 chunk *already computed, server-side*: a representative
coordinate on the segment, its volume and surface area, and its principal axes. Two
useful things fall straight out of that, without ever running a skeletonisation:

``skeleton()``
    the level-2 chunk graph, with ``rep_coord_nm`` for node positions.

``dotprops()``
    ``rep_coord_nm`` is a point on the neurite and ``pca_0`` is that chunk's principal
    axis - which is to say, a point and a direction. That is a dotprop. It needs one
    chunkedgraph call and one l2cache call, and no skeleton at all.

This is a **different representation**, not a cheaper route to the same one, which is
why it lives in its own namespace rather than behind a ``method=`` flag on
``skeletons.get()``. L2 chunks sample the neuron more finely than CAVE's skeleton
service does but carry no radius worth the name, and roughly a third of chunks are too
blobby to have a principal axis at all - those are dropped from the dotprops, because a
zero-length vector is not a direction and NBLAST would quietly treat it as one.

Only datasets that declare ``Cap.L2CACHE`` have this namespace at all. ``flywire``
(the public release) does not: that datastack genuinely has no L2 cache, which is why
it ships precomputed skeletons instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.dataset import requires
from ...core.namespaces import _Namespace
from ...core.spec import Cap
from ...exceptions import CapabilityError
from .skeletons import l2_skeleton

__all__ = ["L2"]

# The dotprop columns, in the order the l2cache hands them back.
_XYZ = ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]
_PCA0 = ["pca_0_x", "pca_0_y", "pca_0_z"]

_SCALE = {"nm": 1.0, "um": 1e-3}
_UNIT_NAME = {"nm": "nanometer", "um": "micrometer"}  # as pint spells them


class L2(_Namespace):
    """Level-2 chunk data: cheap skeletons and dotprops."""

    @requires(Cap.L2CACHE)
    def info(self, x, *, version=None) -> pd.DataFrame:
        """Per-chunk table: position (nm), volume, surface area.

        One row per level-2 chunk, with the ``id`` of the neuron it belongs to. This is
        the raw material the other methods are built from - useful on its own for cheap
        volume/area estimates without fetching a mesh.
        """
        ds = self._ds
        ids = ds.ids(x, version=version)

        frames = []
        for root in ids:
            df = _l2_table(ds, int(root), attributes=["rep_coord_nm", "size_nm3", "area_nm2"])
            if not len(df):
                continue
            out = pd.DataFrame(
                {
                    "id": np.int64(root),
                    "l2_id": df.index.to_numpy(dtype="int64"),
                    "x": df["rep_coord_nm_x"].to_numpy("float32"),
                    "y": df["rep_coord_nm_y"].to_numpy("float32"),
                    "z": df["rep_coord_nm_z"].to_numpy("float32"),
                }
            )
            for col in ("size_nm3", "area_nm2"):
                if col in df.columns:
                    out[col] = df[col].to_numpy("float64")
            frames.append(out)

        if not frames:
            return pd.DataFrame(columns=["id", "l2_id", "x", "y", "z", "size_nm3", "area_nm2"])
        return pd.concat(frames, ignore_index=True)

    @requires(Cap.L2CACHE)
    def graph(self, x, *, version=None):
        """The level-2 chunk graph, as ``networkx.Graph`` (one per neuron).

        Returns a dict keyed by root ID. This is the connectivity of the chunks
        themselves - the thing :meth:`skeleton` walks to find parents.
        """
        import networkx as nx

        ds = self._ds
        out = {}
        for root in ds.ids(x, version=version):
            root = int(root)
            edges = ds.client.chunkedgraph.level2_chunk_graph(root)
            g = nx.Graph()
            g.add_edges_from([(int(a), int(b)) for a, b in edges])
            out[root] = g
        return out

    @requires(Cap.L2CACHE)
    def skeleton(self, x, *, version=None):
        """Skeletons built from the L2 chunk graph -> ``navis.NeuronList``.

        Positions are nanometres. Nodes are chunk *representative coordinates*, and two
        consequences follow that you have to know about:

        **Do not measure cable length on these.** The path hops from one chunk's
        representative point to the next, which zig-zags rather than following the
        neurite, so it over-estimates - on FlyWire DA1_lPNs, consistently by about
        **2x** (1.9-2.2x against the same neurons from the skeleton service). Use
        ``ds.skeletons.get()`` if the number matters.

        **Radii are estimates.** They come from the chunk's volume-to-surface ratio, not
        from a measurement of the neurite.

        What the L2 skeleton *is* good for is topology and shape at low cost - and, via
        :meth:`dotprops`, NBLAST.
        """
        import navis

        # No version resolution here beyond `ids`: the L2 cache is keyed on the *root
        # ID*, not on a materialization. The ID already carries the state of the
        # segmentation, so there is nothing left for a version to pin.
        ds = self._ds
        neurons = []
        for root in ds.ids(x, version=version):
            root = int(root)
            n = navis.TreeNeuron(l2_skeleton(ds, root), id=root, units="1 nm")
            n.name = str(root)
            neurons.append(n)
        return navis.NeuronList(neurons)

    @requires(Cap.L2CACHE)
    def dotprops(self, x, *, units: str = "nm", version=None):
        """Dotprops straight from the L2 cache -> ``navis.NeuronList``.

        No skeletonisation: ``rep_coord_nm`` is the point and ``pca_0`` the vector, both
        computed server-side.

        Chunks with a degenerate (all-zero) principal axis are **dropped** - they are
        too blobby to have a direction, and passing a zero vector to NBLAST would have
        it silently score them as if they did.

        ``units`` defaults to ``"nm"``, like everything else connecto returns. **NBLAST
        is calibrated in microns**: hand it nanometre dotprops and every score collapses
        to the floor, which looks like "nothing matches" rather than like an error. So
        for NBLAST, ask for microns::

            dp = ds.l2.dotprops(x, units="um")
            navis.nblast(dp, dp)
        """
        import navis

        if units not in _SCALE:
            raise ValueError(f"`units` must be 'nm' or 'um', got {units!r}.")
        scale = _SCALE[units]

        ds = self._ds
        out = []
        for root in ds.ids(x, version=version):
            root = int(root)
            df = _l2_table(ds, root, attributes=["rep_coord_nm", "pca"])
            pts = df[_XYZ].to_numpy("float32")
            vect = df[_PCA0].to_numpy("float32")

            keep = np.linalg.norm(vect, axis=1) > 0
            if not keep.any():
                raise CapabilityError(
                    f"{ds.label}: no level-2 chunk of {root} has a principal axis, so "
                    f"no dotprops can be built. Try `ds.l2.skeleton({root})`."
                )

            dp = navis.Dotprops(
                pts[keep] * scale,
                k=None,  # vectors are given, not fitted
                vect=vect[keep],
                id=root,
                units=f"1 {_UNIT_NAME[units]}",
            )
            dp.name = str(root)
            out.append(dp)
        return navis.NeuronList(out)


def _l2_table(ds, root: int, *, attributes) -> pd.DataFrame:
    """Level-2 attributes for one neuron, with the rows that have no position dropped."""
    l2_ids = ds.client.chunkedgraph.get_leaves(root, stop_layer=2)
    df = ds.client.l2cache.get_l2data_table(
        l2_ids.tolist(), attributes=attributes, split_columns=True
    )
    if "rep_coord_nm_x" in df.columns:
        df = df.dropna(subset=["rep_coord_nm_x"])
    return df
