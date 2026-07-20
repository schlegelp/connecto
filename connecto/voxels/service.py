"""Sparse volumes from a lookup service.

Some datasets have somebody standing between connecto and the voxels: a service
that keeps the per-body index a chunkedgraph does not, and answers the same
question in one request instead of several hundred. Where a dataset declares a
:class:`~connecto.core.spec.SparseVolSource`, this is what serves it, and the
expensive :mod:`connecto.voxels.pcg` path is skipped entirely.

The wire format is DVID's, minus the header - little-endian int32
``(x, y, z, run_length)`` runs along +X - so :mod:`connecto.voxels.rle` decodes
these and DVID's alike.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_service_response

__all__ = ["fetch_sparsevol", "SERVICE_STATS_PREFIX"]

# The aedes service reports what it did in response headers: how many chunks it
# read, how many voxels it touched versus kept, how long it spent. Surfaced via
# `voxels.get(..., verbose=True)` because "12 million voxels out of 1.5 billion
# read" is the single most useful number for deciding what scale to ask for next.
SERVICE_STATS_PREFIX = "x-sparsevol-"


def _session():
    import requests

    global _SESSION
    try:
        return _SESSION
    except NameError:
        _SESSION = requests.Session()
        return _SESSION


def fetch_sparsevol(
    source, body: int, scale: int, *, timeout: int = 600
) -> tuple[np.ndarray, dict]:
    """One neuron's sparse volume from a lookup service.

    Returns ``(runs, stats)`` - an ``(M, 4)`` run array and whatever the service
    chose to report about the work it did.
    """
    from ..exceptions import ConnectoError

    body, scale = int(body), int(scale)

    # Refuse an unserved scale here rather than relaying the server's answer. The
    # aedes service returns a helpful 400 for scale 0 but a bare `500 Internal
    # Server Error` for scales 2 and 3, which tells the user nothing at all.
    if scale not in source.scales:
        served = ", ".join(str(s) for s in sorted(source.scales))
        raise ValueError(
            f"This sparse-volume service serves scale {served} only, got scale "
            f"{scale}. Coarser or finer levels are not available for this dataset."
        )

    url = source.url.format(id=body, scale=scale)
    r = _session().get(url, timeout=timeout)

    if r.status_code != 200:
        detail = r.text[:300]
        # The service refuses segments above a size ceiling and advises a coarser
        # scale. Where it is the *only* scale, that advice cannot be taken, and
        # relaying it verbatim sends people looking for an option that does not
        # exist. Say what is actually true instead.
        if r.status_code == 400 and len(source.scales) == 1:
            raise ConnectoError(
                f"The sparse-volume service refused body {body}: {detail}\n"
                f"This service serves scale {scale} only, so there is no coarser "
                f"level to fall back to - this neuron is too large for it. Its "
                f"skeleton and mesh are unaffected."
            )
        raise ConnectoError(
            f"Sparse-volume service returned HTTP {r.status_code} for body {body} "
            f"at scale {scale}: {detail}"
        )

    # Header case is not guaranteed on the wire, so normalise before keying.
    stats = {
        k.lower()[len(SERVICE_STATS_PREFIX) :]: v
        for k, v in r.headers.items()
        if k.lower().startswith(SERVICE_STATS_PREFIX)
    }
    return decode_service_response(r.content), stats
