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
from ...core.parallel import DEFAULT_NEURON_WORKERS, map_ordered
from ...core.spec import Cap
from ...exceptions import CapabilityError
from .skeletons import L2_SKELETON_ATTRIBUTES, l2_info, l2_skeleton

__all__ = ["L2"]

# The dotprop columns, in the order the l2cache hands them back.
_XYZ = ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]
_PCA0 = ["pca_0_x", "pca_0_y", "pca_0_z"]

_SCALE = {"nm": 1.0, "um": 1e-3}
_UNIT_NAME = {"nm": "nanometer", "um": "micrometer"}  # as pint spells them


class L2(_Namespace):
    """Level-2 chunk data: cheap skeletons and dotprops."""

    @requires(Cap.L2CACHE)
    def info(
        self, x, *, version=None, progress: bool = True,
        max_workers: int = DEFAULT_NEURON_WORKERS,
    ) -> pd.DataFrame:
        """Per-chunk table: position (nm), volume, surface area.

        One row per level-2 chunk, with the ``id`` of the neuron it belongs to. This is
        the raw material the other methods are built from - useful on its own for cheap
        volume/area estimates without fetching a mesh.
        """
        ds = self._ds
        ids = ds.ids(x, version=version)

        frames = []
        for root, df in map_ordered(
            (int(i) for i in ids),
            lambda root: _l2_table(ds, root, attributes=L2_SKELETON_ATTRIBUTES),
            workers=max_workers, desc="L2 info", progress=progress,
        ):
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
    def graph(
        self, x, *, version=None, progress: bool = True,
        max_workers: int = DEFAULT_NEURON_WORKERS,
    ):
        """The level-2 chunk graph, as ``networkx.Graph`` (one per neuron).

        Returns a dict keyed by root ID. This is the connectivity of the chunks
        themselves - the thing :meth:`skeleton` walks to find parents.
        """
        import networkx as nx

        ds = self._ds

        def one(root: int) -> nx.Graph:
            g = nx.Graph()
            g.add_edges_from(
                [(int(a), int(b)) for a, b in ds.client.chunkedgraph.level2_chunk_graph(root)]
            )
            return g

        return dict(
            map_ordered(
                (int(i) for i in ds.ids(x, version=version)),
                one,
                workers=max_workers, desc="L2 graphs", progress=progress,
            )
        )

    @requires(Cap.L2CACHE)
    def skeleton(
        self, x, *, version=None, progress: bool = True,
        max_workers: int = DEFAULT_NEURON_WORKERS,
    ):
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

        def one(root: int):
            n = navis.TreeNeuron(l2_skeleton(ds, root), id=root, units="1 nm")
            n.name = str(root)
            return n

        return navis.NeuronList(
            [
                n
                for _, n in map_ordered(
                    (int(i) for i in ds.ids(x, version=version)),
                    one,
                    workers=max_workers, desc="L2 skeletons", progress=progress,
                )
            ]
        )

    @requires(Cap.L2CACHE)
    def dotprops(
        self, x, *, units: str = "nm", version=None, progress: bool = True,
        max_workers: int = DEFAULT_NEURON_WORKERS,
    ):
        """Dotprops straight from the L2 cache -> ``navis.NeuronList``.

        No skeletonisation: ``rep_coord_nm`` is the point and ``pca_0`` the vector, both
        computed server-side.

        Chunks the cache has no principal axis for are **dropped** - they are too
        blobby to have a direction, and passing a non-direction to NBLAST would have
        it silently score them as if they did. See :func:`_has_principal_axis` for
        why "no axis" is not simply "all zero".

        **Expect the point count to move a little between calls.** The L2 cache
        computes an attribute it does not have on demand and answers with what it
        has ready, so how many chunks come back with a principal axis depends on how
        busy it is - measurably so: the same MICrONS neuron yields 2541 points
        fetched on its own and 2540 with five other requests in flight. That is the
        service, not this code, and it is why these dotprops are for shape
        comparison rather than for anything that needs to be identical twice.

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

        def one(root: int):
            df = _l2_table(ds, root, attributes=["rep_coord_nm", "pca"])
            pts = df[_XYZ].to_numpy("float32")
            vect = df[_PCA0].to_numpy("float32")

            keep = _has_principal_axis(vect)
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
            return dp

        return navis.NeuronList(
            [
                dp
                for _, dp in map_ordered(
                    (int(i) for i in ds.ids(x, version=version)),
                    one,
                    workers=max_workers, desc="L2 dotprops", progress=progress,
                )
            ]
        )


def _has_principal_axis(vect: np.ndarray) -> np.ndarray:
    """Which of these ``pca_0`` rows are a real principal axis, by unit length.

    Not ``norm > 0``, which is the obvious test and is wrong. caveclient fills a
    *missing* attribute with ``np.empty`` - uninitialised memory - rather than with
    NaN (see ``l2cache._flatten_pca``), so a chunk the cache has no axis for arrives
    as whatever bytes happened to be lying around. Freshly mapped pages are
    zero-filled, which is why ``norm > 0`` mostly worked; reused ones are not, and
    then a chunk with no axis arrives with a norm of 3e-30, passes the test, and is
    handed to NBLAST as though it were a direction.

    A real principal axis is a unit vector; on MICrONS the populated ones measure
    0.99971 to 1.00036 after the float32 round trip, and the missing ones separate
    cleanly (278 of 278 kept, 0 of 122). Garbage is never plausibly unit length.

    This does *not* make ``dotprops`` reproducible run to run, and it was a mistake
    to think it would: see the note on cache readiness in :meth:`L2.dotprops`. It
    fixes a different thing - a chunk with no axis being kept as if it had one.
    """
    return np.abs(np.linalg.norm(vect, axis=1) - 1.0) < 1e-3


def _l2_table(ds, root: int, *, attributes) -> pd.DataFrame:
    """Level-2 attributes for one neuron, with the rows that have no position dropped.

    The fetch itself is :func:`~connecto.backends.cave.skeletons.l2_info`; what is
    added here is dropping the positionless chunks. `l2_skeleton` wants them kept
    long enough to tell "the cache has nothing for this neuron" apart from "the cache
    has chunks but no coordinates", which is a different error message.
    """
    df = l2_info(ds, root, attributes=attributes)
    if "rep_coord_nm_x" in df.columns:
        df = df.dropna(subset=["rep_coord_nm_x"])
    return df
