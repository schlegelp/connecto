"""How wide connecto is allowed to fan out, and why the numbers are what they are.

One module, because these three numbers are not independent. Fetching meshes nests a
pool inside a pool - neurons in flight, and within each neuron its fragments - and it
is the *product* that arrives at one host's connection pool. Get that wrong in the
generous direction and urllib3 does not fail: it discards the connections it cannot
keep and silently pays a fresh TLS handshake per read, so the concurrency is
requested, charged for, and spent on setup. That is a performance bug that looks like
a slow network.

They lived in three modules once, each with a comment pointing at the other two.
``POOL_MAXSIZE`` is derived here instead, so raising a fan-out cannot outrun it.

A leaf module on purpose: ``store`` needs the ceiling and ``mesh`` needs the
fan-out, and ``mesh`` already imports ``store``, so neither can own all three.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import re

__all__ = [
    "GIL_RELEASING_DRACOPY",
    "DEFAULT_MESH_PARALLEL",
    "DEFAULT_MESH_WORKERS",
    "POOL_MAXSIZE",
    "decode_workers",
]

#: Neurons in flight in :func:`connecto.core.volume.fetch_meshes`.
#:
#: Eight Aedes neurons (40M vertices), interleaved runs, medians:
#:
#: * DracoPy 2.1: 8.9 s at 4, 7.8 s at 8, 8.0 s at 12 and 16.
#: * DracoPy 1.7: 21.3 s at 4, 20.8 s at 8.
#:
#: So 8, where the 2.1 curve flattens, and which costs the older decoder nothing.
#: It used to be 4, on the reasoning that what a neuron costs once its bytes arrive
#: - draco decoding, seam welding - holds the GIL and serialises however many
#: neurons are in flight. With a decoder that releases it, the biggest share of that
#: work overlaps, and that reasoning describes only the old DracoPy. What still
#: serialises is the numpy seam welding, around 0.1 s per large neuron.
#:
#: The product with ``DEFAULT_MESH_PARALLEL`` is 256 fragment reads at once, which
#: ``POOL_MAXSIZE`` below is sized to keep.
DEFAULT_MESH_WORKERS = 8

#: Fragment reads in flight within one mesh.
#:
#: High, because the cost of a fragment is a round trip and not a download - tens of
#: kB each - so wall clock divides almost linearly by this number. Measured on one
#: Aedes neuron (71 fragments, 2.8 MB): 3.3 s at 8, 1.6 s at 16, 1.3 s at 32, 0.8 s
#: at 64.
#:
#: 32 rather than 64 because of the product below: one neuron gives up ~0.5 s, and
#: four neurons at 64 apiece would ask for 256 connections at once.
DEFAULT_MESH_PARALLEL = 32

#: Live connections one host may hold, for the shared anonymous session.
#:
#: Generous rather than exact. It caps *retained idle* connections and urllib3 opens
#: them lazily, so headroom costs nothing until it is used - whereas running out is
#: the silent-handshake failure this module opens by describing, and a caller who
#: tunes ``meshes.get(max_workers=8, parallel=64)`` can ask for 512 without any way
#: to raise the ceiling themselves. So: room for a tuned fan-out, with the derivation
#: kept as a floor so the shipped defaults can never outgrow it either.
POOL_MAXSIZE = max(512, DEFAULT_MESH_WORKERS * DEFAULT_MESH_PARALLEL + 64)


#: The first DracoPy that releases the GIL during decode (seung-lab/DracoPy#67).
GIL_RELEASING_DRACOPY = (2, 1)


@functools.cache
def decode_workers() -> int:
    """Threads for decoding draco fragments that have *already* been fetched.

    A different budget from the two above, sized by a different thing: those wait on
    a network and want far more workers than cores, this is CPU work and wants about
    as many as there are cores.

    It only pays with a DracoPy that releases the GIL during decode, and that is a
    capability, not a constant - so it is tested rather than assumed. The same 215
    fragments (two Aedes neurons, two hemibrain LOD-0 meshes), 14-core machine:

    ========  =========  ===========  ===========
    DracoPy   serial     8 threads    output
    ========  =========  ===========  ===========
    1.7.0     1994 ms    0.95-1.00x   identical
    2.0.0     1972 ms    0.97-1.00x   identical
    2.1.0      510 ms    3.5-5.6x     identical
    ========  =========  ===========  ===========

    So below 2.1 this answers ``1`` and callers take a plain serial loop: pointing
    threads at a decoder that holds the GIL buys nothing and the handoffs cost a few
    percent. connecto declares ``DracoPy>=1.4.0``, so that is still most installs.

    Note it is 2.1, not 2.0. A 2.0.0 was published before the GIL work landed and
    behaves exactly like 1.7 - this gate used to check the major version only and
    got that wrong, which is the case its own docstring warned about. The test is
    the installed version rather than a timing probe, because timing is flaky and
    test-hostile; an unparseable version string is treated as the old decoder,
    which costs a little speed and never correctness.

    Capped at 8 because the curve is flat past there and connecto is a library that
    runs inside other people's pipelines.
    """
    try:
        version = importlib.metadata.version("dracopy")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return 1

    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < GIL_RELEASING_DRACOPY:
        return 1
    return min(os.cpu_count() or 4, 8)
