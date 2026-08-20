"""Per-neuron sparse volumes, from whichever source a dataset actually has.

Three routes to the same answer, and they cost wildly different amounts:

===============  ==========================================  ================
route            datasets                                    cost
===============  ==========================================  ================
``dvid``         hemibrain, maleCNS, MANC, fish2             one indexed request
``service``      aedes                                      one indexed request
``pcg``          FlyWire, BANC, FANC, MICrONS               a dense read
===============  ==========================================  ================

The first two hand back a per-body index somebody else maintains. The third has no
index to consult and must read dense blocks and mask them - which is why
``voxels.get`` on a CAVE dataset can refuse outright rather than appear to hang.

Every route returns the *same* thing: ``(M, 4)`` run-length encoded voxels
``(x, y, z, length)`` along +X, plus the nm-per-voxel of the grid they are in.
:mod:`connecto.voxels.rle` decodes all of them, because DVID's wire format, the
lookup service's, and what the CAVE path encodes into are one format.
"""

from __future__ import annotations

import numpy as np

from . import dvid, pcg, rle, service

__all__ = ["fetch", "estimate", "route", "rle", "dvid", "service", "pcg"]


def route(ds) -> str:
    """Which of the three routes ``ds`` uses. Data, not a guess."""
    if ds.spec.sparsevol_source is not None:
        return "service"
    if ds.backend_kind == "neuprint":
        return "dvid"
    return "pcg"


# The scale `voxels.get` picks when the caller does not.
#
# Never 0. A hemibrain neuron at scale 0 is 1.17 billion voxels - 14 GB as int32
# triples - and on the CAVE route the equivalent request is refused outright. The
# default has to return a recognisable neuron in seconds, not the finest thing that
# might just fit.
#
# The two routes get different numbers because they cost different amounts. An
# indexed lookup is one request whatever the scale, so 4 is a comfortable middle. The
# dense read is not: at scale 3 a *mid-sized* FlyWire neuron transfers ~1.2 billion
# voxels and a MICrONS cell exceeds the 2-billion ceiling outright and raises - which
# is what the conformance suite caught. 6 is the coarsest useful level and stays well
# inside the ceiling for every dataset here.
_DEFAULT_SCALE = {"dvid": 4, "pcg": 6}


def default_scale(ds) -> int:
    """A scale that returns something usable without reading the whole brain."""
    src = ds.spec.sparsevol_source
    if src is not None:
        # A service serves what it serves, and the cost is the server's problem, so
        # take the finest available rather than second-guessing it.
        return min(src.scales)

    # Clamped to what the pyramid actually has. Pyramid depth is a property of the
    # volume, not of the route: FlyWire stops at 6, BANC at 7, MICrONS at 9, and FANC
    # at *5* - so a fixed preference of 6 asked FANC for a scale that does not exist
    # and got `IndexError: list index out of range` out of the volume's metadata.
    return min(_DEFAULT_SCALE[route(ds)], max(scales(ds)))


# Which extra keywords each route understands. Kept explicit so an option meant for
# a different route is refused by name instead of surfacing as
# `TypeError: fetch_sparsevol() got an unexpected keyword argument 'max_chunks'`,
# naming a private function the caller has never heard of. This is not hypothetical:
# the dense route's own ceiling message suggests `max_chunks=`, and carrying that
# call over to hemibrain would otherwise crash.
_ROUTE_OPTS = {
    "dvid": {"timeout"},
    "service": {"timeout"},
    "pcg": {"max_chunks", "max_voxels", "parallel"},
}


def _check_opts(kind: str, opts: dict) -> None:
    unknown = set(opts) - _ROUTE_OPTS[kind]
    if not unknown:
        return
    elsewhere = {
        opt: k for k, allowed in _ROUTE_OPTS.items() for opt in allowed if opt in unknown
    }
    bits = []
    for opt in sorted(unknown):
        where = elsewhere.get(opt)
        bits.append(f"{opt!r}" + (f" (only on the {where!r} route)" if where else ""))
    raise TypeError(
        f"This dataset fetches voxels via the {kind!r} route, which does not take "
        f"{', '.join(bits)}. It accepts: "
        f"{', '.join(sorted(_ROUTE_OPTS[kind])) or 'no extra options'}."
    )


def fetch(ds, body: int, scale: int, **opts):
    """One neuron's voxels: ``(runs, resolution_nm, stats)``.

    ``runs`` is ``(M, 4)``; ``resolution_nm`` is the nm-per-voxel of the grid the
    runs are expressed in, which is what turns them into physical space.
    """
    kind = route(ds)
    _check_opts(kind, opts)

    if kind == "service":
        src = ds.spec.sparsevol_source
        runs, stats = service.fetch_sparsevol(src, body, scale, **opts)
        return runs, src.resolution(scale, ds.spec.voxel_size), stats

    if kind == "dvid":
        target = dvid.resolve(ds)
        runs = dvid.fetch_sparsevol(target, body, scale, **opts)
        return runs, dvid.resolution(target, scale), {"source": str(target)}

    return pcg.fetch_sparsevol(ds, body, scale, **opts)


def estimate(ds, body: int, scale: int) -> dict:
    """What fetching ``body`` at ``scale`` would cost, without fetching it.

    Only the CAVE route can answer usefully - it is the only one doing work
    proportional to anything the caller controls. The indexed routes report the
    route and let the server worry about it.
    """
    kind = route(ds)
    if kind == "pcg":
        return pcg.estimate(ds, body, scale)
    return {"root_id": int(body), "scale": scale, "route": kind, "requests": 1}


def scales(ds) -> tuple[int, ...]:
    """The scales this dataset can serve.

    Every route can answer this, so it always returns a tuple. The dense route reads
    the pyramid straight off the volume - that these are *available* says nothing
    about whether they are *affordable*, which is what `estimate` is for.
    """
    src = ds.spec.sparsevol_source
    if src is not None:
        return tuple(sorted(src.scales))
    if route(ds) == "dvid":
        return tuple(range(dvid.max_scale(dvid.resolve(ds)) + 1))
    return tuple(range(len(pcg.get_volume(ds).meta.available_mips)))


def to_nm(coords: np.ndarray, resolution) -> np.ndarray:
    """Voxel coordinates to nanometres."""
    return coords.astype(np.int64) * np.asarray(resolution, dtype=np.float64)
