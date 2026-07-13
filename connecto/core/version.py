"""A version of a dataset.

CAVE and neuPrint have incompatible notions of "version":

* CAVE has integer *materializations* (783, 888, ...) - snapshots in which every
  annotation carries the root ID that was valid at that timestamp. Root IDs
  change as neurons are edited, so a set of IDs is only valid in *some* versions.
* neuPrint has immutable, semantically-versioned dataset strings
  (``hemibrain:v1.2.1``) - and sometimes no version at all (``fish2``, ``cns``).

:class:`Version` is a value object that carries whichever of these applies plus a
timestamp, so error messages and ``df.attrs`` can be precise. It compares equal to
the bare int/str it wraps, so users never need to know it exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

__all__ = ["Version", "semantic_key", "sort_versions"]


@dataclass(frozen=True)
class Version:
    """A resolved dataset version."""

    value: int | str
    backend: str
    timestamp: datetime | None = None
    expires: datetime | None = None

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Version({self.value!r}, backend={self.backend!r})"

    def __eq__(self, other):
        # Compare equal to the bare value, so `ds.version == 783` works.
        if isinstance(other, Version):
            return (self.value, self.backend) == (other.value, other.backend)
        return self.value == other

    def __hash__(self):
        return hash((self.value, self.backend))

    @property
    def is_live(self) -> bool:
        return self.value == "live"

    @property
    def expired(self) -> bool:
        if self.expires is None:
            return False
        return datetime.now(tz=self.expires.tzinfo) >= self.expires


def semantic_key(version: str) -> tuple:
    """Sort key for a (possibly) semantic version string.

    A plain string sort gets this wrong in the way that actually bites:
    ``"v0.9" > "v0.13"``. Anything non-numeric sorts before anything numeric, so
    unversioned datasets (``fish2``, ``cns``) don't blow up here.

    >>> sorted(["v0.9", "v0.13", "v1.2.1"], key=semantic_key)
    ['v0.9', 'v0.13', 'v1.2.1']
    """
    # Strip a leading dataset name ("male-cns:v0.9" -> "v0.9").
    version = str(version).split(":")[-1]
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)([a-z]*)", version.strip())
    if not m:
        return (0, (), version)
    numbers = tuple(int(p) for p in m.group(1).split("."))
    return (1, numbers, m.group(2))


def sort_versions(versions):
    """Sort version strings semantically, oldest first."""
    return sorted(versions, key=semantic_key)
