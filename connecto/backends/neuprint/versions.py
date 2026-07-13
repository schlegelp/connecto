"""neuPrint dataset versions.

neuPrint has no materialization: the version is baked into the dataset string
(``hemibrain:v1.2.1``) and body IDs are immutable. Some datasets carry no version
at all (``fish2``, ``cns``, ``mushroombody``), so nothing here may assume the
``name:vX.Y`` shape.
"""

from __future__ import annotations

import functools

import requests

from ...core.version import Version, sort_versions
from ...exceptions import NoSuchVersionError

__all__ = ["parse_source", "resolve", "available", "server_datasets"]


def parse_source(source: str) -> tuple[str, str, str | None]:
    """``"server/dataset:version"`` -> ``(server, dataset, version | None)``."""
    server, _, rest = source.partition("/")
    if not rest:  # no server given
        server, rest = "neuprint.janelia.org", server
    name, sep, version = rest.partition(":")
    return server, name, (version if sep else None)


@functools.lru_cache(maxsize=32)
def server_datasets(server: str, token: str | None = None) -> tuple[str, ...]:
    """Every dataset on a neuPrint server.

    Hits ``/api/dbmeta/datasets`` directly rather than going through
    ``neuprint.Client``, whose constructor *validates* the dataset name - so you
    cannot use it to find out which names are valid. Chicken, meet egg.

    This is usually the *first* call connecto makes to neuPrint, so whoever has a
    problem - no token, or a server in maintenance - meets this function before they
    meet anything else. Left bare, its ``raise_for_status()`` produced a naked 401 or
    503 that mentioned no token, no server and no fix - and pre-empted
    neuprint-python's own much better "set NEUPRINT_APPLICATION_CREDENTIALS" message,
    which never got to fire.
    """
    from ...servers import upstream_errors

    host = server
    if not server.startswith("http"):
        server = f"https://{server}"

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with upstream_errors("neuprint", server=host):
        r = requests.get(f"{server}/api/dbmeta/datasets", headers=headers, timeout=30)
        r.raise_for_status()
    return tuple(r.json())


def available(server: str, name: str, token: str | None = None) -> list[str]:
    """Every version string for a dataset on this server, oldest first."""
    datasets = server_datasets(server, token)
    hits = [d for d in datasets if d == name or d.startswith(f"{name}:")]
    if not hits:
        raise NoSuchVersionError(
            f"{server} has no dataset {name!r}. "
            f"Available: {', '.join(sorted(datasets))}."
        )
    return sort_versions(hits)


def resolve(server: str, name: str, request, *, token=None, backend="neuprint") -> Version:
    """Turn a version request into a concrete :class:`Version`.

    ``value`` is the full neuPrint dataset string - that is what identifies the
    snapshot, and it is what the Client must be constructed with.
    """
    if isinstance(request, Version):
        return request

    options = available(server, name, token)

    if request in (None, "latest", "auto"):
        # Semantic sort, not a string sort: "v0.9" must come before "v0.13".
        return Version(options[-1], backend)

    if request == "live":
        raise NoSuchVersionError(
            "neuPrint datasets are immutable snapshots; there is no live query. "
            f"Available: {', '.join(options)}."
        )

    request = str(request)
    for candidate in (request, f"{name}:{request}", f"{name}:v{request.lstrip('v')}"):
        if candidate in options:
            return Version(candidate, backend)

    raise NoSuchVersionError(
        f"No version {request!r} of {name!r}. Available: {', '.join(options)}."
    )
