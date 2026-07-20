"""Sparse volumes from DVID, with the server and node discovered at runtime.

DVID keeps a live per-body index - body to blocks to runs - so one neuron's voxels
are a single cheap request. That is the thing a chunkedgraph cannot do (see
:mod:`connecto.voxels.pcg`), and it is why the Janelia datasets are the fast path.

**Nothing here is written down.** The FlyEM DVID servers are not public knowledge,
and connecto does not pin their URLs or node UUIDs in source. Both are parsed live
out of metadata connecto is already entitled to read:

* neuPrint's ``Meta`` advertises hemibrain's host outright, in ``neuroglancerInfo``.
* neuPrint's neuroglancer-layer endpoint carries ``dvid://server/node/instance``
  sources, which is how fish2 resolves.
* clio's ``/v2/datasets`` carries a ``dvid`` server per dataset, which is the only
  route for maleCNS and MANC - neuPrint returns ``None`` for every layer field on
  both.

A URL pinned in source would be both a leak and, by the next release, wrong.

The node comes from neuPrint's ``Meta.uuid`` wherever possible, because that is the
UUID of the *published snapshot* the rest of connecto is talking to. clio's own
``uuid`` is sometimes a moving tag (``"v1.0"``) or a six-character prefix, so it is
used to identify the right server, not to address the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .rle import decode_sparsevol

__all__ = ["DvidTarget", "resolve", "fetch_sparsevol", "segmentation_info"]

# `dvid://https://server/node/instance`, optionally with a query string. The node
# may be a UUID, a UUID prefix, or a branch alias like `634ee:master`.
_DVID_URL = re.compile(r"dvid://(?P<server>https?://[^/]+)/(?P<node>[^/?#]+)/(?P<instance>[^/?#]+)")

# Discovery is a few HTTP calls; the answer does not change within a session.
_CACHE: dict = {}


@dataclass(frozen=True)
class DvidTarget:
    """Where one dataset's segmentation actually lives, and how we found out."""

    server: str
    node: str
    instance: str = "segmentation"
    origin: str = ""  # human-readable provenance, surfaced in errors and verbose

    def url(self, *parts: str) -> str:
        return "/".join([self.server.rstrip("/"), "api", "node", self.node, self.instance, *parts])

    def __str__(self) -> str:
        return f"{self.server}/{self.node}/{self.instance}"


def _session():
    """One pooled session per process.

    Fetching a hundred neurons is a hundred requests to the same host; a session
    keeps the TLS connection open across them instead of renegotiating each time.
    """
    import requests

    if "session" not in _CACHE:
        _CACHE["session"] = requests.Session()
    return _CACHE["session"]


# ------------------------------------------------------------------- discovery


def _https(host: str) -> str:
    """Normalise a host to an absolute https URL.

    ``neuroglancerInfo.host`` is the one route that hands back a bare field rather
    than something parsed out of a URL, so it is the one that can arrive without a
    scheme. requests raises `MissingSchema` on those, which says nothing about DVID.
    """
    host = host.rstrip("/")
    return host if "://" in host else f"https://{host}"


def _from_neuprint_meta(meta) -> tuple[str, str, str] | None:
    """hemibrain: ``Meta.neuroglancerInfo`` names the host and the node outright."""
    info = (meta.get("neuroglancerInfo") or {}).get("segmentation") or {}
    host, uuid = info.get("host"), info.get("uuid")
    if not (host and uuid):
        return None
    # `dataInstance` is the DVID instance name; `dataType` is the *layer* type, which
    # on hemibrain happens to also read "segmentation". Preferring dataInstance means
    # a server that names its instance anything else still resolves, instead of
    # quietly asking for an instance called "segmentation" that may not exist.
    instance = info.get("dataInstance") or "segmentation"
    return _https(host), uuid, instance


def _from_neuprint_layers(meta) -> tuple[str, str, str] | None:
    """``neuroglancerMeta`` layers, when the segmentation one is a dvid:// source."""
    for layer in meta.get("neuroglancerMeta") or []:
        if layer.get("dataType") != "segmentation":
            continue
        m = _DVID_URL.search(str(layer.get("source") or ""))
        if m:
            return m["server"], m["node"], m["instance"]
    return None


def _from_nglayers(server: str, dataset: str) -> tuple[str, str, str] | None:
    """The neuroglancer-scene endpoint: fish2's segmentation is a dvid:// source.

    Populated for datasets whose ``Meta.neuroglancerMeta`` is empty, so it is worth
    asking even when the ``Meta`` route came up dry.
    """
    url = f"https://{server}/api/npexplorer/nglayers/{dataset}.json"
    try:
        r = _session().get(url, timeout=30)
        if r.status_code != 200:
            return None
        scene = r.json()
    except Exception:
        return None

    for layer in scene.get("layers", []):
        if layer.get("type") != "segmentation":
            continue
        # `source` is a str, a {"url": ...} dict, or a list of either.
        src = layer.get("source")
        for candidate in src if isinstance(src, list) else [src]:
            if isinstance(candidate, dict):
                candidate = candidate.get("url")
            m = _DVID_URL.search(str(candidate or ""))
            if m:
                return m["server"], m["node"], m["instance"]
    return None


def _clio_candidates(dataset: str, name: str):
    """Clio dataset names to try, best first.

    clio's keys are largely connecto's neuPrint version strings (``male-cns:v1.0``,
    ``fish2``), so the version string usually works outright. MANC is the exception:
    neuPrint serves ``manc:v1.2.3`` while clio stops at ``manc:v1.2``, hence the
    trimmed forms. Any of these getting us in the door is enough - the *server* is
    then chosen by UUID, not by whichever name happened to connect.
    """
    seen, out = set(), []
    base = dataset.split(":")[0]
    trimmed = dataset.rsplit(".", 1)[0] if "." in dataset else None
    for cand in (dataset, trimmed, base, name, base.upper(), name.upper()):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _from_clio(uuid: str, dataset: str, name: str) -> tuple[str, str] | None:
    """The clio dataset whose node *is* this one; returns (server, clio name).

    Matched on UUID rather than on name. clio records the same node as a prefix
    (``7b5e8f`` for neuPrint's ``7b5e8f7f805c4314bee37b75b4ff9292``), so a prefix
    match in either direction is the identity test - and it is a far stronger one
    than a name map, which would silently rot when a dataset is re-released.
    """
    try:
        import clio
    except ModuleNotFoundError:
        return None

    datasets = None
    for cand in _clio_candidates(dataset, name):
        try:
            datasets = clio.Client(dataset=cand).fetch_datasets()
            break
        except Exception:
            continue
    if not datasets:
        return None

    for clio_name, entry in datasets.items():
        server = entry.get("dvid")
        theirs = str(entry.get("uuid") or "")
        if not server or not theirs:
            continue
        if uuid.startswith(theirs) or theirs.startswith(uuid):
            return server.rstrip("/"), clio_name
    return None


def resolve(ds, *, verbose: bool = False) -> DvidTarget:
    """Find the DVID server, node and instance backing ``ds``.

    Raises rather than guessing. A wrong node returns another release's voxels,
    which look entirely reasonable and are wrong, so every route here is one that
    names the node explicitly.
    """
    from ..exceptions import CapabilityError

    server, name = ds._server_and_name
    dataset = str(ds.version)
    key = (server, dataset)
    if key in _CACHE:
        return _CACHE[key]

    meta = ds.client.meta
    uuid = str(meta.get("uuid") or "")

    found = _from_neuprint_meta(meta)
    origin = "neuPrint Meta.neuroglancerInfo"
    if not found:
        found = _from_neuprint_layers(meta)
        origin = "neuPrint Meta.neuroglancerMeta"
    if not found:
        found = _from_nglayers(server, dataset)
        origin = "neuPrint nglayers"

    if found:
        host, node, instance = found
        # The routes above are trusted for the *server* and the *instance*, but the
        # node is taken from neuPrint's Meta.uuid whenever there is one, because
        # that is the snapshot every other part of this handle is reading. The two
        # genuinely disagree: `fish2:v0.6` reports Meta.uuid `0b3b0be3...` while its
        # own neuroglancer scene points at `40dee3b0...`, and other scenes carry
        # branch aliases like `634ee:master` that follow HEAD. Taking voxels from
        # one node and annotations from another is the kind of mismatch that
        # produces a plausible answer about the wrong neuron.
        if uuid and uuid != node:
            origin += f" (server); node from neuPrint Meta.uuid, overriding {node!r}"
            node = uuid
    else:
        if not uuid:
            raise CapabilityError(
                f"{ds.label}: cannot locate a DVID server. neuPrint reports no "
                f"neuroglancer layers and no Meta.uuid for {dataset!r}, so there is "
                f"nothing to resolve a sparse volume against."
            )
        clio_hit = _from_clio(uuid, dataset, name)
        if not clio_hit:
            raise CapabilityError(
                f"{ds.label}: cannot locate a DVID server for {dataset!r}.\n"
                f"neuPrint advertises no dvid:// layer for it, so the server has to "
                f"come from clio, and clio could not be reached or has no dataset on "
                f"node {uuid[:8]}...\n"
                f"Fix: `pip install connecto[clio]` and make sure you have a clio "
                f"token (`clio.login()`). connecto deliberately does not hard-code "
                f"FlyEM DVID URLs."
            )
        host, clio_name = clio_hit
        node, instance = uuid, "segmentation"
        origin = f"clio dataset {clio_name!r} (matched on node {uuid[:8]}...)"

    target = DvidTarget(host, node, instance, origin)
    if verbose:
        print(f"{ds.label}: DVID resolved to {target} via {origin}")
    _CACHE[key] = target
    return target


# ----------------------------------------------------------------------- fetch


def segmentation_info(target: DvidTarget) -> dict:
    """The labelmap instance's ``Extended`` info: block size, voxel size, scales."""
    key = ("info", str(target))
    if key not in _CACHE:
        r = _session().get(target.url("info"), timeout=60)
        r.raise_for_status()
        _CACHE[key] = r.json()["Extended"]
    return _CACHE[key]


def max_scale(target: DvidTarget) -> int:
    """The coarsest scale this instance stores."""
    return int(segmentation_info(target).get("MaxDownresLevel", 0))


def resolution(target: DvidTarget, scale: int) -> tuple[float, float, float]:
    """nm per voxel at ``scale``.

    DVID's labelmap pyramid halves all three axes per level, so the scale-0
    ``VoxelSize`` the server reports is all that is needed. Read from the server
    rather than from the dataset spec, so the two cannot drift.
    """
    base = segmentation_info(target)["VoxelSize"]
    return tuple(float(v) * 2**int(scale) for v in base)


def fetch_sparsevol(target: DvidTarget, body: int, scale: int, *, timeout: int = 600) -> np.ndarray:
    """One body's sparse volume at ``scale``, as ``(M, 4)`` runs."""
    from ..exceptions import ConnectoError, NoSuchBodyError

    body = int(body)
    top = max_scale(target)
    if not 0 <= int(scale) <= top:
        raise ValueError(
            f"scale must be between 0 and {top} for {target.instance} on this "
            f"server, got {scale}."
        )

    url = f"{target.url('sparsevol', str(body))}?scale={int(scale)}"
    r = _session().get(url, timeout=timeout)

    if r.status_code == 404:
        # DVID answers 404 both for "no such body" and "body has no voxels at this
        # scale". Raising beats returning an empty array: a silently-empty volume for
        # a typo'd body ID is indistinguishable from a real neuron that happens to
        # vanish at a coarse scale, and the caller can only tell those apart if we
        # say which body and which node we asked about.
        raise NoSuchBodyError(
            f"DVID has no sparse volume for body {body} at scale {scale} "
            f"({target}). The body may not exist in this node, or it may be too "
            f"small to survive downsampling - try a finer `scale=`."
        )
    if r.status_code != 200:
        raise ConnectoError(
            f"DVID returned HTTP {r.status_code} for body {body} at scale {scale} "
            f"({target}): {r.text[:200]}"
        )

    return decode_sparsevol(r.content)
