"""Exceptions raised by connecto."""

from __future__ import annotations

__all__ = [
    "ConnectoError",
    "CapabilityError",
    "ConnectoAuthError",
    "ConnectoServerError",
    "NoSuchVersionError",
    "NoSuchDatasetError",
    "AmbiguousVersionError",
    "MissingDependencyError",
]


class ConnectoError(Exception):
    """Base class for all connecto errors."""


class ConnectoServerError(ConnectoError):
    """The server could not serve the request - and it is not your fault.

    Three kinds, because they call for three different responses:

    ``UNAVAILABLE``
        HTTP 502/503/504. The service is deliberately refusing work. On CAVE this
        is overwhelmingly a *materialization in progress*, which runs for **hours**,
        not seconds. The single most useful thing to tell someone here is that
        waiting in a tight loop will not outlast it - go and do something else -
        which is the opposite of what a bare 503 traceback suggests.
    ``ERROR``
        HTTP 500. Something broke server-side. Retrying rarely helps; this is worth
        reporting to whoever runs the server.
    ``UNREACHABLE``
        No HTTP response at all: DNS, TLS, timeout, no route. Very often this is the
        user's own network rather than the server, and saying so saves them filing a
        bug against the wrong project.
    """

    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __init__(self, message: str, *, kind: str, service: str, server=None, status=None):
        super().__init__(message)
        self.kind = kind
        self.service = service
        self.server = server
        self.status = status  # the HTTP status, when there was one


class ConnectoAuthError(ConnectoError):
    """A credential problem, stated in terms the user can act on.

    Three kinds, because they have three different fixes:

    ``MISSING``
        No token found anywhere. Go and get one.
    ``INVALID``
        A token *was* found and the server rejected it (HTTP 401). The useful
        thing to say here is *which* of the five possible sources it came from -
        an env var, a config file, or one of three secret files - because that is
        the one the user has to go and fix.
    ``FORBIDDEN``
        The token is fine; you just aren't allowed near this resource (HTTP 403).
        Usually a missing group membership. A new token will not help, and telling
        someone to get one sends them down the wrong path entirely.
    """

    MISSING = "missing"
    INVALID = "invalid"
    FORBIDDEN = "forbidden"

    def __init__(self, message: str, *, kind: str, service: str, server=None):
        super().__init__(message)
        self.kind = kind
        self.service = service
        self.server = server


class CapabilityError(ConnectoError, NotImplementedError, AttributeError):
    """A dataset does not support a requested feature.

    Subclasses ``AttributeError`` on purpose: this makes ``hasattr(ds, "segmentation")``
    return ``False`` for datasets without a segmentation, so duck-typed feature
    detection works, while direct access still raises with a useful message.
    """


class NoSuchDatasetError(ConnectoError, KeyError):
    """No dataset with that name is registered."""

    def __str__(self):  # KeyError otherwise repr()s the message
        return self.args[0] if self.args else ""


class NoSuchVersionError(ConnectoError, ValueError):
    """The requested version does not exist for this dataset."""


class AmbiguousVersionError(ConnectoError, ValueError):
    """No single version contains all the requested IDs.

    Raised by ``version="auto"`` when the given IDs never co-existed in any
    materialization.
    """


class MissingDependencyError(ConnectoError, ImportError):
    """An optional dependency is required for this operation."""

    @classmethod
    def for_extra(cls, package: str, extra: str, what: str):
        return cls(
            f"{what} requires the `{package}` package, which is not installed. "
            f"Install it with `pip install connecto[{extra}]`."
        )
