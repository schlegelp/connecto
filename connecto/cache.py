"""On-disk cache.

Layout::

    ~/.connecto/cache/<dataset>/<version>/<kind>_<hash>.feather
    ~/.connecto/cache/_http/<sha1(url)>.<ext>

The cache key includes a *content hash*, so a stale entry can never be served:

* CAVE tables hash ``get_table_metadata(table)["last_modified"]`` - if the table
  is re-ingested the key changes and we re-download.
* neuPrint datasets are immutable, so the version string *is* the content hash and
  annotations can be cached forever.

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
    """A single cached DataFrame."""

    def __init__(self, dataset: str, version, kind: str, *key_parts):
        self.path = cache_dir(dataset, str(version)) / f"{kind}_{_hash(*key_parts)}.feather"

    def exists(self) -> bool:
        return enabled() and self.path.exists()

    def read(self) -> pd.DataFrame:
        return pd.read_feather(self.path)

    def write(self, df: pd.DataFrame) -> pd.DataFrame:
        if not enabled():
            return df
        try:
            # Feather can't round-trip a non-default index or object columns of
            # mixed type; a failure to cache must never fail the query.
            df.reset_index(drop=True).to_feather(self.path)
        except Exception:
            self.path.unlink(missing_ok=True)
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
