"""Server outages, and how to say something useful about them.

CAVE deployments answer 503 while they build a new materialization version. That
is not a blip: it runs for **hours**. caveclient retries such a request three times
over about 0.7 seconds (``status_forcelist=(502, 503, 504)``, ``backoff_factor=0.1``)
and then raises - so what reaches the user is a ``requests.HTTPError`` whose message
carries a page of nginx HTML, and whose only real content is the number 503.

Two questions matter at that moment, and the traceback answers neither.

**Is it just this service, or is everything down?** A materialization takes out the
materialize service and leaves the chunkedgraph, skeleton and info services
answering normally. A dead VPN takes out all four. Those have completely different
fixes, and one cheap probe of each service tells you which you are looking at. That
distinction is the main thing this module exists to provide, and connecto is well
placed to provide it: it already knows every service its backends depend on.

**How long?** Nobody can say - but *hours* is the right order of magnitude, and
someone who knows that will go and do something else rather than hammer the server
in a ``while True``. So connecto does **not** silently retry 503s for you: no honest
backoff outlasts a materialization, and a library that sleeps for hours pretending
to work is worse than one that tells you to come back later. What it offers instead
is :func:`wait_until_available`, which does the waiting *explicitly*, and says so.

One thing deliberately not attempted: guessing which of *your* calls would still
succeed. It is tempting to print "segmentation still works!", but it is not reliably
true - ``segmentation.update_ids`` needs a materialization timestamp, and every
version lookup goes through the materialize service, so even a pinned
``ds.at(783)`` hits it. Claiming otherwise would send people down a path that dead-ends
one call later. This module reports which *services* answer, and what each one backs;
the mapping from that to your code is left honest rather than clever.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import requests

from .auth import CAVE_GLOBAL_SERVER, auth_errors, get_token
from .exceptions import ConnectoError, ConnectoServerError

__all__ = [
    "server_status",
    "wait_until_available",
    "server_errors",
    "upstream_errors",
]

# Short: this runs on a path that has already failed, and probes go out in parallel.
PROBE_TIMEOUT = 8

# 502/504 are the gateway in front of a service that is not answering; 503 is the
# service itself saying "not now". For our purposes they mean the same thing.
UNAVAILABLE_CODES = (502, 503, 504)

OK = "OK"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"
UNREACHABLE = "UNREACHABLE"
NO_AUTH = "NO AUTH"


@dataclass(frozen=True)
class _Service:
    name: str
    url: str  # a real, cheap endpoint on that service - not a synthetic health check
    backs: str  # what, in connecto's terms, stops working when this is down


# The probe URLs are the *real query surface* wherever there is a cheap one. For
# materialize that means the version list, which is the same endpoint every query
# goes through to resolve a version - not a generic /version health ping that could
# answer 200 from a service whose query path is down.
def _cave_services(datastack: str, client) -> list[_Service]:
    if client is None:
        # Without the info service we cannot even learn the datastack's server
        # addresses - handing those out is precisely what the info service does. The
        # global server is then the whole of what we can honestly probe, and finding
        # it down is itself the answer.
        return [
            _Service(
                "info",
                f"{CAVE_GLOBAL_SERVER}/info/version",
                "client setup, datastack metadata",
            )
        ]

    def addr(name: str) -> str | None:
        try:
            a = getattr(getattr(client, name), "_server_address", None)
        except Exception:  # noqa: BLE001 - a datastack need not have every service
            return None
        return a.rstrip("/") if a else None

    specs = [
        ("info", "{}/info/version", "client setup, datastack metadata"),
        (
            "materialize",
            "{}/materialize/api/v2/datastack/" + datastack + "/versions",
            "annotations, connectivity, synapses, somas, ROIs, version lookup",
        ),
        ("chunkedgraph", "{}/segmentation/api/version", "segmentation, edit history"),
        ("skeleton", "{}/skeletoncache/api/version", "skeletons"),
        # The l2cache goes down on its own - FANC's does it regularly - and when it
        # does, it takes skeletons and mesh/skeleton radii with it. Leaving it out of
        # the probe meant the health table could report every service OK while the
        # call that just failed was an l2cache 503, which is the one thing this
        # message must never do.
        ("l2cache", "{}/l2cache/api/v1/table_mapping", "skeletons (L2 fallback), L2 chunk data"),
    ]

    out = []
    for name, tpl, backs in specs:
        a = addr(name)
        if a:  # a service the datastack does not have is not a service that is down
            out.append(_Service(name, tpl.format(a), backs))
    return out


def _neuprint_services(server: str) -> list[_Service]:
    server = server.replace("https://", "").replace("http://", "").rstrip("/")
    return [
        _Service(
            "neuprint",
            f"https://{server}/api/dbmeta/datasets",
            "everything (neuPrint is a single service)",
        )
    ]


def _cached_cave_client(datastack: str):
    """The CAVEclient we already built, or None.

    Deliberately does not *construct* one. This is called while handling an
    exception that may itself have come from constructing a client, and building a
    fresh one there would recurse - and would hang on the very outage we are trying
    to describe.
    """
    from .backends.cave.dataset import _CLIENTS

    return _CLIENTS.get(datastack)


def _services_for(service: str, *, dataset=None, server=None, client=None) -> list[_Service]:
    if service == "cave":
        datastack = getattr(dataset, "source", None) or ""
        if client is None and datastack:
            client = _cached_cave_client(datastack)
        return _cave_services(datastack, client)
    if service == "neuprint":
        host = server or getattr(dataset, "_auth_server", None)
        return _neuprint_services(host) if host else []
    return []


def _probe(svc: _Service, token: str | None, timeout: int = PROBE_TIMEOUT) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    t0 = time.perf_counter()
    try:
        r = requests.get(svc.url, headers=headers, timeout=timeout)
    except requests.RequestException:
        return _row(svc, UNREACHABLE, time.perf_counter() - t0, None)

    if r.status_code in UNAVAILABLE_CODES:
        status = UNAVAILABLE
    elif r.status_code >= 500:
        status = ERROR
    elif r.status_code in (401, 403):
        # The service answered - it is up. Your token is the problem, and
        # auth_status() is the function that untangles that one.
        status = NO_AUTH
    else:
        # Any other answer, 404 included, means something is alive and serving.
        status = OK
    return _row(svc, status, time.perf_counter() - t0, r.status_code)


def _row(svc: _Service, status: str, elapsed: float, code: int | None) -> dict:
    host = svc.url.split("//", 1)[-1].split("/", 1)[0]
    return {
        "service": svc.name,
        "server": host,
        "status": status,
        "http": code,
        "seconds": round(elapsed, 2),
        "backs": svc.backs,
    }


def _probe_all(
    services: list[_Service], token: str | None, timeout: int = PROBE_TIMEOUT
) -> list[dict]:
    if not services:
        return []
    # Concurrently, because a total outage means every probe burns the full timeout
    # and doing that four times in series would add half a minute to an error.
    with ThreadPoolExecutor(max_workers=len(services)) as pool:
        return list(pool.map(lambda s: _probe(s, token, timeout), services))


# ---------------------------------------------------------------------- status


def server_status(dataset, *, timeout: int = PROBE_TIMEOUT):
    """Which servers behind ``dataset`` are answering, right now.

    Probes each service that the dataset's backend depends on and reports what it
    said. The point is the *shape* of the result rather than any single row: one
    service down means that service is having a bad day (on CAVE, almost always a
    materialization in progress); every service down means the deployment or your
    own network is gone, which is a different problem with a different fix.

    Parameters
    ----------
    dataset :   str | Dataset
                A registered dataset name (``"flywire"``) or a dataset object.
    timeout :   int
                Seconds to wait for each probe.

    Returns
    -------
    pandas.DataFrame
                One row per service: ``service, server, status, http, seconds, backs``.
                ``status`` is ``OK``, ``UNAVAILABLE``, ``ERROR``, ``UNREACHABLE`` or
                ``NO AUTH`` - never a bare boolean, because "answered" and "answered
                usefully" are different claims.

    Examples
    --------
    >>> co.server_status("flywire")                             # doctest: +SKIP
       service              server  status  http  seconds  backs
    0     info  global.daf-apis.com     OK   200     0.31  client setup, ...
    1  materialize  prod.flywire-daf.com  UNAVAILABLE  503  0.44  annotations, ...
    """
    import pandas as pd

    from .core.registry import get_dataset

    ds = get_dataset(dataset) if isinstance(dataset, str) else dataset
    service = ds.backend_kind

    client = None
    if service == "cave":
        # Unlike the error path, a cold status check may construct a client - but it
        # must not explode when the thing it is diagnosing is exactly that failing.
        try:
            client = ds.client
        except Exception:  # noqa: BLE001
            client = None

    token = get_token(service, server=ds._auth_server).token
    services = _services_for(service, dataset=ds, client=client)
    return pd.DataFrame(_probe_all(services, token, timeout))


def wait_until_available(
    dataset,
    *,
    service: str | None = None,
    interval: int = 300,
    timeout: int = 6 * 3600,
    verbose: bool = True,
):
    """Block until ``dataset``'s servers are serving again.

    This is the loop you would otherwise write yourself around a materialization,
    with a poll interval that will not get you rate-limited. It is a separate,
    explicit call rather than an automatic retry precisely because it can take
    hours: a library that quietly sleeps that long, pretending to work, is worse
    than one that raises and lets you decide.

    Parameters
    ----------
    dataset :   str | Dataset
    service :   str, optional
                Wait only for this service (e.g. ``"materialize"``). By default,
                waits until *every* service the dataset needs is answering.
    interval :  int
                Seconds between polls. Floored at 30 - a materialization does not
                finish in less time than that, and hammering a struggling server is
                not neighbourly.
    timeout :   int
                Give up after this many seconds. Default 6 hours.

    Returns
    -------
    pandas.DataFrame
                The final (healthy) status table.

    Raises
    ------
    TimeoutError
                If the servers are still down when ``timeout`` expires.
    """
    interval = max(30, int(interval))
    deadline = time.monotonic() + timeout
    n = 0

    while True:
        df = server_status(dataset)

        rows = df
        if service:
            rows = df[df["service"] == service]
            if rows.empty:
                # Otherwise a typo filters the table down to nothing, nothing is
                # down, and we cheerfully report "healthy" and return - having
                # waited for a service that does not exist.
                raise ValueError(
                    f"{_name(dataset)} has no service {service!r}. "
                    f"Known: {', '.join(df['service'])}."
                )

        # NO AUTH means the service is up and your token is the problem; waiting will
        # not fix that, so it does not count as "still down".
        down = rows[~rows["status"].isin([OK, NO_AUTH])]

        if down.empty:
            if verbose and n:
                print(f"\n{_name(dataset)} is back ({_hms(n * interval)} waited).")
            return df

        n += 1
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(
                f"{_name(dataset)}: still {', '.join(sorted(set(down['status'])))} "
                f"after {_hms(timeout)} "
                f"({', '.join(down['service'])}). Not waiting any longer."
            )

        if verbose:
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{', '.join(f'{r.service}={r.status}' for r in down.itertuples())} "
                f"- retrying in {interval}s "
                f"(giving up in {_hms(left)})",
                flush=True,
            )
        time.sleep(min(interval, max(1, left)))


def _name(dataset) -> str:
    return dataset if isinstance(dataset, str) else getattr(dataset, "label", str(dataset))


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m" if m else f"{seconds}s"


# ----------------------------------------------------------- error translation


SERVICE_NAMES = {"cave": "CAVE", "neuprint": "neuPrint"}


def _which_service(url: str | None, service: str) -> str | None:
    """Which CAVE service a failing URL belongs to.

    Read off the URL rather than probed or guessed: the path segment *is* the
    service, so this is ground truth and costs nothing.
    """
    if not url or service != "cave":
        return None
    for token, name in (
        ("/materialize/", "materialize"),
        ("/segmentation/", "chunkedgraph"),
        ("/skeletoncache/", "skeleton"),
        ("/l2cache/", "l2cache"),
        ("/annotation/", "annotation"),
        ("/info/", "info"),
        ("/schema/", "schema"),
        ("/nglstate/", "state"),
    ):
        if token in url:
            return name
    return None


def _table(rows: list[dict]) -> list[str]:
    w = max(len(r["service"]) for r in rows)
    s = max(len(r["status"]) for r in rows)
    return [
        f"  {r['service']:<{w}}  {r['status']:<{s}}  {r['backs']}" for r in rows
    ]


MATERIALIZATION = (
    "CAVE serves 503 while it builds a new materialization version. That runs for",
    "*hours*, not seconds, so a retry loop will not outlast it - and caveclient has",
    "already retried this three times.",
)


def _diagnose(kind, service, rows: list[dict], failed: str | None) -> list[str]:
    """The paragraph that turns the table into an answer.

    Every explanation offered here is *conditional on what the probe just found*.
    Saying "this is a materialization, go and have lunch" when the servers are in
    fact answering would be worse than saying nothing: it is a confident, plausible,
    checkable claim that happens to be false, and the person reading it has no easy
    way to tell.
    """
    cave = service == "cave"

    if not rows:
        # No probe (no cached client, so we could not learn the server addresses).
        if kind == ConnectoServerError.UNAVAILABLE:
            return list(MATERIALIZATION) if cave else [
                "The server is up but refusing work - usually an update or a restart.",
            ]
        if kind == ConnectoServerError.ERROR:
            return [
                "A server-side failure, not a bad request. Retrying rarely helps; if it",
                "persists it is worth reporting to whoever runs the server.",
            ]
        return []

    bad = [r for r in rows if r["status"] in (UNAVAILABLE, ERROR, UNREACHABLE)]

    if not bad:
        return [
            "Every service is answering again, so whatever this was, it was brief.",
            "Retry the call.",
        ]

    if all(r["status"] == UNREACHABLE for r in bad) and len(bad) == len(rows):
        return [
            "Nothing is reachable at all. That points at your own network - VPN, proxy,",
            "or simply offline - rather than at the server.",
        ]

    if len(bad) == len(rows):
        what = SERVICE_NAMES.get(service, service)
        return [
            f"Every {what} service is down, so this is the whole deployment rather than",
            "one service having a bad day. Waiting is still the only option; it may be a",
            "longer one.",
        ]

    if cave and failed and [r["service"] for r in bad] == [failed] == ["materialize"]:
        return [
            "Only materialize is down; the rest of CAVE is fine. That is exactly what a",
            "materialization in progress looks like from outside.",
            "",
            *MATERIALIZATION,
        ]

    return [
        f"Down: {', '.join(r['service'] for r in bad)}. The other services are fine, so",
        "this is one service rather than the whole deployment.",
    ]


def _message(kind, service, *, status, server, resource, url, dataset) -> str:
    what = SERVICE_NAMES.get(service, service)
    failed = _which_service(url, service)

    if kind == ConnectoServerError.UNREACHABLE:
        head = [f"Could not reach {what} - no response at all.", ""]
    else:
        reason = {502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}
        head = [
            f"{what} returned {status}"
            + (f" ({reason[status]})" if status in reason else "")
            + ".",
            "",
        ]

    if resource:
        head.append(f"  dataset:   {resource}")
    if failed:
        head.append(f"  service:   {failed}")
    if url:
        head.append(f"  endpoint:  {_short(url)}")

    # Probe *before* explaining, so that the explanation can be conditional on what
    # is actually true right now. Never let the probe mask the error it exists to
    # explain: a failure here just means a shorter message.
    rows = []
    try:
        rows = _probe_all(
            _services_for(service, dataset=dataset, server=server),
            get_token(service, server=server).token,
        )
    except Exception:  # noqa: BLE001
        pass

    body = []
    if rows:
        body += ["", f"{what} services, checked just now:", *_table(rows)]

    diag = _diagnose(kind, service, rows, failed)
    if diag:
        body += ["", *diag]

    return "\n".join(head + body + _next_steps(kind, rows, getattr(dataset, "name", None)))


def _next_steps(kind, rows: list[dict], label: str | None) -> list[str]:
    """What to actually do - which is not the same thing in every case.

    Telling someone whose VPN is down to ``wait_until_available()`` would have them
    watch a spinner for six hours over a problem they could fix in ten seconds.
    """
    bad = [r for r in rows if r["status"] in (UNAVAILABLE, ERROR, UNREACHABLE)]

    if rows and not bad:
        return ["", "Next:", "  Retry the call - the servers are answering again."]

    if kind == ConnectoServerError.UNREACHABLE:
        steps = ["", "Next:", "  Check your own connection first (VPN, proxy, offline?)."]
        if label:
            steps.append(f'  co.server_status("{label}")   # then re-check from here')
        return steps

    steps = ["", "Next:"]
    if label:
        steps += [
            f'  co.wait_until_available("{label}")   # block until it is back',
            f'  co.server_status("{label}")          # re-check the table above',
        ]
    steps += ["  Anything already in the cache still works (co.cache)."]
    return steps


def _short(url: str, keep: int = 96) -> str:
    return url if len(url) <= keep else url[: keep - 1] + "…"


@contextmanager
def server_errors(service: str, *, server=None, resource=None, dataset=None):
    """Translate a 5xx (or an unreachable host) into something worth reading.

    Sits *outside* :func:`~connecto.auth.auth_errors`, which handles 401/403. The two
    are strictly disjoint - a credential problem and an outage want completely
    different things said about them - so between them they cover every way an
    upstream HTTP call can fail without either one having to guess.
    """
    try:
        yield
    except ConnectoError:
        raise  # already ours; do not re-wrap
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status is None or status < 500:
            raise  # 4xx is auth_errors' business, or a genuine bad request
        kind = (
            ConnectoServerError.UNAVAILABLE
            if status in UNAVAILABLE_CODES
            else ConnectoServerError.ERROR
        )
        url = getattr(e.response, "url", None)
        raise ConnectoServerError(
            _message(kind, service, status=status, server=server, resource=resource,
                     url=url, dataset=dataset),
            kind=kind, service=service, server=server, status=status,
        ) from e
    except (requests.ConnectionError, requests.Timeout) as e:
        url = getattr(getattr(e, "request", None), "url", None)
        raise ConnectoServerError(
            _message(ConnectoServerError.UNREACHABLE, service, status=None,
                     server=server, resource=resource, url=url, dataset=dataset),
            kind=ConnectoServerError.UNREACHABLE, service=service, server=server,
        ) from e


@contextmanager
def upstream_errors(service: str, *, server=None, resource=None, dataset=None):
    """Everything that can go wrong between here and the server.

    The single translation layer applied at connecto's choke points: credentials
    (:func:`~connecto.auth.auth_errors`) and outages (:func:`server_errors`). Order
    matters only in that auth must see the 401/403 first; it re-raises everything
    else, so the 5xx falls through to the outer handler untouched.
    """
    with server_errors(service, server=server, resource=resource, dataset=dataset):
        with auth_errors(service, server=server, resource=resource):
            yield
