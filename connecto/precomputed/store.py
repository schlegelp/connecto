"""HTTPS access to a precomputed bucket.

Everything connecto reads - Google Storage, S3, a plain web server, a CAVE
graphene service - answers ordinary HTTPS GETs, most of them anonymous. So this is
a thin layer over :mod:`requests`: resolve the ``gs://``/``s3://`` shorthand to a
URL, GET a whole object or a byte range, and turn a 404 into ``None`` rather than
an exception (which is what ``fill_missing`` needs).

No write path, by design. cloud-volume's storage layer carries boto3, the Google
Cloud SDK and gevent precisely because it *can* write to all of them; connecto
never does.
"""

from __future__ import annotations

import functools
import gzip
import json
import posixpath
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..servers import server_errors
from .limits import POOL_MAXSIZE

__all__ = ["Store", "is_graphene", "normalize_url", "split_protocol", "decompress"]

# Retried because object storage occasionally 503s under a fan-out of a few hundred
# concurrent chunk reads, and one lost chunk would otherwise fail a whole cutout.
_RETRY = Retry(
    total=4,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "HEAD"]),
    raise_on_status=False,
)


def split_protocol(path: str) -> tuple[str, str]:
    """``"precomputed://gs://bucket/x"`` -> ``("precomputed", "gs://bucket/x")``."""
    path = str(path)
    for proto in ("precomputed", "graphene", "n5", "zarr"):
        if path.startswith(proto + "://"):
            return proto, path[len(proto) + 3 :]
    return "", path


def is_graphene(source) -> bool:
    """Is this a chunkedgraph service, in any of the forms connecto sees?

    One predicate, because the answer decides three separate things - which volume
    class to build, whether the CAVE session travels with the request, and whether a
    cutout can be agglomerated - and three copies of the test would eventually
    disagree in a way that fails silently rather than loudly.
    """
    source = str(source)
    return split_protocol(source)[0] == "graphene" or "middleauth+" in source


def normalize_url(path: str) -> str:
    """Resolve any of the shorthands connecto sees to a plain https:// URL."""
    _, rest = split_protocol(path)
    # `middleauth+` tells a *viewer* to run CAVE's login flow. It is not part of
    # the address, and leaving it in produces a DNS failure.
    rest = rest.replace("middleauth+", "")

    if rest.startswith("gs://"):
        return "https://storage.googleapis.com/" + rest[5:]
    if rest.startswith("s3://"):
        bucket, _, key = rest[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    if rest.startswith(("http://", "https://")):
        return rest
    if rest.startswith("file://"):
        return rest
    raise ValueError(f"Cannot resolve {path!r} to a URL.")


def decompress(data: bytes, encoding: str | None) -> bytes:
    """Undo a sharded value's ``data_encoding``/``minishard_index_encoding``."""
    if not data or encoding in (None, "", "raw"):
        return data
    if encoding == "gzip":
        return gzip.decompress(data)
    if encoding == "br":
        import brotli  # pragma: no cover - only if a bucket ever uses it

        return brotli.decompress(data)
    raise ValueError(f"Unsupported encoding {encoding!r}.")


@functools.cache
def _default_session() -> requests.Session:
    """The anonymous session every public bucket read shares.

    ``pool_maxsize`` is a ceiling on live connections to one host, and it is derived
    from the mesh fan-out rather than picked - see :mod:`connecto.precomputed.limits`
    for what happens when it is the smaller of the two.
    """
    sess = requests.Session()
    adapter = HTTPAdapter(
        max_retries=_RETRY, pool_maxsize=POOL_MAXSIZE, pool_connections=16
    )
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


class Store:
    """Read-only access to one precomputed prefix."""

    def __init__(self, path: str, session: requests.Session | None = None, timeout: float = 120):
        self.url = normalize_url(path).rstrip("/")
        self.timeout = timeout
        self._session = session

    @property
    def session(self) -> requests.Session:
        return self._session if self._session is not None else _default_session()

    def at(self, *parts: str) -> Store:
        """A store rooted at a subdirectory, sharing this one's session."""
        return Store(posixpath.join(self.url, *parts), self._session, self.timeout)

    # ------------------------------------------------------------------ reading

    def _get(self, key: str = "", headers: dict | None = None):
        """GET one object, leaving only a 404 for the caller to interpret.

        Everything else - 5xx, DNS, TLS, timeouts - goes through connecto's own
        translation layer, so an object store that is down reads the same way as a
        CAVE or neuPrint server that is down, rather than as a bare traceback.
        """
        url = posixpath.join(self.url, key) if key else self.url
        with server_errors("precomputed", server=urlparse(url).netloc, resource=url):
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            if response.status_code != 404:
                response.raise_for_status()
        return url, response

    def get(self, key: str = "", *, missing_ok: bool = True) -> bytes | None:
        """A whole object. ``None`` if it does not exist and `missing_ok`."""
        url, response = self._get(key)
        if response.status_code == 404:
            if missing_ok:
                return None
            raise FileNotFoundError(f"No object at {url}.")
        return response.content

    def get_range(self, key: str, start: int, end: int, *, missing_ok: bool = True) -> bytes | None:
        """Bytes ``[start, end)`` of an object."""
        start, end = int(start), int(end)
        if end <= start:
            return b""
        url, response = self._get(key, headers={"Range": f"bytes={start}-{end - 1}"})
        if response.status_code == 404:
            if missing_ok:
                return None
            raise FileNotFoundError(f"No object at {url}.")
        # A server that ignores Range answers 200 with the whole object. Rare, but
        # silently returning the wrong bytes would corrupt a shard index.
        if response.status_code == 200 and len(response.content) > (end - start):
            return response.content[start:end]
        return response.content

    def get_json(self, key: str = "info", *, missing_ok: bool = False) -> dict | None:
        raw = self.get(key, missing_ok=missing_ok)
        return None if raw is None else json.loads(raw)

    def __repr__(self) -> str:
        return f"Store({self.url!r})"
