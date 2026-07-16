"""On-disk cache.

Layout::

    ~/.connecto/cache/<dataset>/<version>/<kind>_<hash>.feather
    ~/.connecto/cache/_http/<sha1(url)>.<ext>

Staleness is handled by what goes into the key, and that differs by what the source
is (see :func:`connecto.sources.freshness` and ``Annotations._table``):

* A **materialization** is frozen, so a CAVE annotation table and a neuPrint version
  string are content-stable at a given version - the version in the key is enough,
  and a frozen dataset caches forever.
* The **live** sources are not: FlyTable (SeaTable) is edited daily and is
  independent of any materialization, and a neuPrint instance can be re-curated under
  the same version tag. For those a *freshness token* joins the key - SeaTable's
  ``COUNT(*)``/``MAX(_mtime)``, neuPrint's ``lastDatabaseEdit`` - so it turns over
  exactly when the source does. ``CacheEntry.supersede`` then drops the file it
  replaced, so a live source keeps one entry, not one per edit.

Frames are written as Feather, falling back to pickle for the ones Arrow cannot
hold - a SeaTable table with a mixed ``int``/``str`` object column, say. Without
that fallback such a frame *silently never cached*: the write raised, the entry was
deleted, and the next call re-downloaded the lot (aedes' FlyTable annotations,
17k rows, on every ``.annotations.get()``). Pickle round-trips the frame exactly,
so a cache hit is byte-for-byte what a miss would have returned.

We cache annotation tables, ROI hierarchies and HTTP downloads - things that are
large, slow, and fetched over and over. We deliberately do **not** cache
connectivity or synapse queries by default: the key space is unbounded and it
would quietly fill the user's disk. Pass ``cache=True`` to opt in.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

__all__ = ["cache_dir", "clear", "size", "enabled", "CacheEntry", "download"]

CACHE_ROOT = Path(os.environ.get("CONNECTO_CACHE_DIR", "~/.connecto/cache")).expanduser()


def enabled() -> bool:
    return not os.environ.get("CONNECTO_NO_CACHE")


_UNSAFE = re.compile(r'[<>:"/\\|?*\s]+')


def _safe(part) -> str:
    """neuPrint versions look like "banc:v888" - fine on POSIX, illegal on Windows."""
    return _UNSAFE.sub("_", str(part)).strip("_") or "_"


def cache_dir(*parts) -> Path:
    d = CACHE_ROOT.joinpath(*[_safe(p) for p in parts])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:12]


class CacheEntry:
    """A single cached DataFrame.

    Stored as Feather where possible and pickle otherwise, under the same stem.
    ``read``/``exists`` prefer whichever is present (Feather first).
    """

    def __init__(self, dataset: str, version, kind: str, *key_parts):
        d = cache_dir(dataset, str(version))
        stem = d / f"{kind}_{_hash(*key_parts)}"
        self.path = Path(f"{stem}.feather")
        self._pickle = Path(f"{stem}.pkl")
        self._family = (d, f"{kind}_")

    def supersede(self) -> None:
        """Delete other entries in this ``(dataset, version, kind)`` family.

        For the annotation cache only, where a *freshness token* is part of the key:
        when a live source is edited the token turns over and a new file is written,
        so without this the old snapshots would pile up, one per edit. Kept opt-in on
        purpose - the connectivity cache deliberately holds many entries under the one
        ``edges`` kind, and sweeping those would throw away good caches.
        """
        d, prefix = self._family
        keep = {self.path.name, self._pickle.name}
        for f in d.glob(f"{prefix}*"):
            if f.name not in keep:
                f.unlink(missing_ok=True)

    def superseding(self) -> bool:
        """True if writing this entry would replace a *different* one in its family.

        For user feedback only: it distinguishes "the source changed, so we are
        refreshing a stale cache" from "nothing was cached, so we are downloading
        fresh". A live source whose freshness token turned over leaves an older file
        behind that :meth:`supersede` will then remove.
        """
        d, prefix = self._family
        keep = {self.path.name, self._pickle.name}
        return any(f.name not in keep for f in d.glob(f"{prefix}*"))

    def _hit(self) -> Path | None:
        """The stored file for this key, Feather preferred, or None."""
        if not enabled():
            return None
        if self.path.exists():
            return self.path
        if self._pickle.exists():
            return self._pickle
        return None

    def exists(self) -> bool:
        return self._hit() is not None

    def read(self) -> pd.DataFrame:
        hit = self._hit()
        return pd.read_pickle(hit) if hit.suffix == ".pkl" else pd.read_feather(hit)

    def write(self, df: pd.DataFrame) -> pd.DataFrame:
        if not enabled():
            return df
        # Feather can't round-trip a non-default index; reset it for both formats.
        df = df.reset_index(drop=True)
        try:
            df.to_feather(self.path)
            self._pickle.unlink(missing_ok=True)  # one canonical copy per key
        except Exception:
            # Feather can't hold this frame - an object column of mixed type, say.
            # Fall back to pickle rather than never caching it (which just silently
            # re-downloads every call). A failure to cache must never fail the query.
            self.path.unlink(missing_ok=True)
            try:
                df.to_pickle(self._pickle)
            except Exception:
                self._pickle.unlink(missing_ok=True)
        return df


def download(url: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Download a URL into the cache and return the local path.

    The filename is derived from a hash of the *whole* URL, not its basename, so
    two sources that both serve ``annotations.tsv`` don't collide.
    """
    suffix = "".join(Path(urlparse(url).path).suffixes[-1:]) or ".dat"
    fp = cache_dir("_http") / f"{_hash(url)}{suffix}"

    if fp.exists() and not force and enabled():
        return fp

    if verbose:
        print(f"Caching {Path(urlparse(url).path).name} from {urlparse(url).netloc}...",
              end=" ", flush=True)

    with requests.get(url, allow_redirects=True, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = fp.with_suffix(fp.suffix + ".part")
        with open(tmp, "wb") as f:  # always binary: text mode corrupts anything else
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(fp)

    if verbose:
        print("Done.")
    return fp


def clear(dataset: str | None = None):
    """Delete cached data - for one dataset, or all of it."""
    target = CACHE_ROOT / dataset if dataset else CACHE_ROOT
    if target.exists():
        shutil.rmtree(target)


def size() -> int:
    """Total size of the cache in bytes."""
    if not CACHE_ROOT.exists():
        return 0
    return sum(f.stat().st_size for f in CACHE_ROOT.rglob("*") if f.is_file())
