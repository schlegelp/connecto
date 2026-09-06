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

__all__ = [
    "DEFAULT_MESH_PARALLEL",
    "DEFAULT_MESH_WORKERS",
    "POOL_MAXSIZE",
    "decode_workers",
]

#: Neurons in flight in :func:`connecto.core.volume.fetch_meshes`.
#:
#: Low, and worth less than it looks: what a neuron costs after its bytes arrive -
#: draco decoding, seam deduplication - is C and numpy holding the GIL, so it
#: serialises no matter what this is set to. Measured on three Aedes neurons, 4
#: workers against 1 is 7.97 s against 8.24 s. The fan-out that pays is the one
#: below.
DEFAULT_MESH_WORKERS = 4

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


@functools.cache
def decode_workers() -> int:
    """Threads for decoding draco fragments that have *already* been fetched.

    A different budget from the two above, sized by a different thing: those wait on
    a network and want far more workers than cores, this is CPU work and wants about
    as many as there are cores.

    It only pays with a DracoPy that releases the GIL during decode, and that is a
    capability, not a constant - so it is tested rather than assumed. Measured on one
    hemibrain neuron's 60 LOD-0 fragments, 14-core machine:

    * GIL-releasing DracoPy: 163 ms at 1 worker, 48 ms at 4, 37 ms at 8, 34 ms at 16.
    * Stock DracoPy 1.7: 628 ms at 1, 637 ms at 4, 675 ms at 8. Decode serialises
      whatever is pointed at it, and the handoffs cost ~6%.

    Which is why this answers ``1`` there, and callers take a plain serial loop:
    connecto declares ``DracoPy>=1.4.0``, so most installs are the second case, and a
    default that is knowingly 6% slow for them is not made acceptable by a comment
    saying so.

    The test is the installed version, not a timing probe - timing is flaky and
    test-hostile. It is a guess about someone else's release, and it can be wrong: a
    published DracoPy 2.0 that does *not* release the GIL would put us back to the
    6%, which is exactly today's behaviour and no worse. Capped at 8 because the
    curve is flat past there - another 8 threads buy 3 ms - and connecto is a library
    that runs inside other people's pipelines.
    """
    try:
        version = importlib.metadata.version("dracopy")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return 1

    major = version.split(".")[0]
    if not (major.isdigit() and int(major) >= 2):
        return 1
    return min(os.cpu_count() or 4, 8)
