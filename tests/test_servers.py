"""Outage handling.

The thing under test is not really the exception type - it is the *message*. A 503
that says "503" is what we already had; the point of the feature is that the message
distinguishes a materialization (wait it out) from a dead deployment (wait longer)
from a dead VPN (fix your laptop), because those three want three different responses
from the person reading it.

So most of these tests assert on what the message says, and - just as importantly -
on what it does *not* say. A confidently wrong hint is worse than no hint.
"""

from __future__ import annotations

import pytest
import requests

import connecto as co
from connecto import servers
from connecto.exceptions import ConnectoAuthError, ConnectoServerError
from connecto.servers import (
    OK,
    UNAVAILABLE,
    UNREACHABLE,
    server_errors,
    upstream_errors,
)

MAT_URL = (
    "https://prod.flywire-daf.com/materialize/api/v2/datastack/"
    "flywire_fafb_public/version/783/table/valid_connection_v2/query"
)
CG_URL = "https://prod.flywire-daf.com/segmentation/api/v1/table/x/roots"

CAVE_SERVICES = ["info", "materialize", "chunkedgraph", "skeleton"]
CODE = {OK: 200, UNAVAILABLE: 503, UNREACHABLE: None}


@pytest.fixture
def probe(monkeypatch):
    """Stub the health probe. Nothing here touches the network."""

    def _set(**down):
        def fake(svc, token, timeout=servers.PROBE_TIMEOUT):
            status = down.get(svc.name, OK)
            return servers._row(svc, status, 0.1, CODE[status])

        monkeypatch.setattr(servers, "_probe", fake)

    return _set


class FakeDataset:
    """Enough of a Dataset for the message builder, with no client and no network."""

    name = "flywire"
    label = "FlyWire"
    source = "flywire_fafb_public"
    backend_kind = "cave"
    _auth_server = None


@pytest.fixture
def ds(monkeypatch):
    # _services_for() reaches for a *cached* CAVEclient to learn the per-service
    # server addresses. Hand it a stub so the probe list is built without a network.
    class Sub:
        def __init__(self, addr):
            self._server_address = addr

    class Client:
        server_address = "https://global.daf-apis.com"
        info = Sub("https://global.daf-apis.com")
        materialize = Sub("https://prod.flywire-daf.com")
        chunkedgraph = Sub("https://prod.flywire-daf.com")
        skeleton = Sub("https://prod.flywire-daf.com")

    monkeypatch.setattr(servers, "_cached_cave_client", lambda datastack: Client())
    return FakeDataset()


def raises_http(status, url):
    r = requests.Response()
    r.status_code = status
    r.url = url
    r.reason = "Service Unavailable"
    raise requests.HTTPError(f"{status} Server Error: ... content:<html>…</html>", response=r)


def caught(ds, fn):
    with pytest.raises(ConnectoServerError) as exc:
        with server_errors("cave", resource=ds.source, dataset=ds):
            fn()
    return exc.value


# ------------------------------------------------------------------- the kinds


def test_503_is_unavailable_not_a_bare_httperror(ds, probe):
    probe(materialize=UNAVAILABLE)
    err = caught(ds, lambda: raises_http(503, MAT_URL))

    assert err.kind == ConnectoServerError.UNAVAILABLE
    assert err.status == 503
    assert err.service == "cave"


def test_500_is_an_error_not_an_outage(ds, probe):
    probe()
    err = caught(ds, lambda: raises_http(500, MAT_URL))

    assert err.kind == ConnectoServerError.ERROR
    # A 500 is a bug, not a materialization. Do not send people off for lunch.
    assert "materialization" not in str(err).lower()


def test_connection_error_is_unreachable(ds, probe):
    probe(**dict.fromkeys(CAVE_SERVICES, UNREACHABLE))

    def boom():
        raise requests.ConnectionError("nodename nor servname provided")

    err = caught(ds, boom)
    assert err.kind == ConnectoServerError.UNREACHABLE


def test_4xx_is_not_ours(ds, probe):
    """A 404 is a bad request, not an outage. It must pass straight through."""
    probe()
    with pytest.raises(requests.HTTPError):
        with server_errors("cave", resource=ds.source, dataset=ds):
            raises_http(404, MAT_URL)


# ------------------------------------------------------- the diagnosis is earned


def test_only_materialize_down_reads_as_a_materialization(ds, probe):
    probe(materialize=UNAVAILABLE)
    msg = str(caught(ds, lambda: raises_http(503, MAT_URL)))

    assert "Only materialize is down" in msg
    assert "materialization in progress" in msg
    assert "hours" in msg  # the one fact that changes what you do next
    assert "wait_until_available" in msg


def test_everything_down_is_not_blamed_on_a_materialization(ds, probe):
    """The whole deployment being down is a different story, and gets told as one."""
    probe(**dict.fromkeys(CAVE_SERVICES, UNAVAILABLE))
    msg = str(caught(ds, lambda: raises_http(503, MAT_URL)))

    assert "whole deployment" in msg
    assert "Only materialize is down" not in msg


def test_nothing_reachable_blames_the_network_not_the_server(ds, probe):
    probe(**dict.fromkeys(CAVE_SERVICES, UNREACHABLE))

    def boom():
        raise requests.ConnectionError("dns go boom")

    msg = str(caught(ds, boom))
    assert "your own network" in msg
    assert "VPN" in msg
    # Telling someone whose VPN is down to block for six hours is malpractice.
    assert "wait_until_available" not in msg


def test_transient_503_says_just_retry(ds, probe):
    """Servers already back by the time we look: say so, do not lecture about hours."""
    probe()  # everything OK
    msg = str(caught(ds, lambda: raises_http(503, MAT_URL)))

    assert "answering again" in msg
    assert "Retry the call" in msg
    assert "hours" not in msg


def test_service_is_read_off_the_failing_url(ds, probe):
    """Ground truth, not a guess: the path segment *is* the service."""
    probe(chunkedgraph=UNAVAILABLE)
    msg = str(caught(ds, lambda: raises_http(503, CG_URL)))

    assert "service:   chunkedgraph" in msg
    # ...and so a chunkedgraph outage is not narrated as a materialization.
    assert "materialization in progress" not in msg


# --------------------------------------------------------------- interoperation


def test_auth_errors_still_win(ds, probe):
    """401 is auth's business. The outage layer must not swallow it."""
    probe()
    with pytest.raises(ConnectoAuthError):
        with upstream_errors("cave", resource=ds.source, dataset=ds):
            raises_http(401, MAT_URL)


def test_a_dead_probe_cannot_mask_the_real_error(ds, monkeypatch):
    """If the probe itself explodes we still raise the outage, just with less detail."""

    def explode(*a, **kw):
        raise RuntimeError("probe is broken")

    monkeypatch.setattr(servers, "_probe_all", explode)

    err = caught(ds, lambda: raises_http(503, MAT_URL))
    assert err.kind == ConnectoServerError.UNAVAILABLE
    assert "503" in str(err)


def test_connecto_errors_pass_through_unwrapped(ds, probe):
    probe()
    sentinel = co.CapabilityError("no meshes here")
    with pytest.raises(co.CapabilityError):
        with server_errors("cave", resource=ds.source, dataset=ds):
            raise sentinel


def test_exported():
    assert co.ConnectoServerError is ConnectoServerError
    assert callable(co.server_status)
    assert callable(co.wait_until_available)


# ------------------------------------------------------------ wait_until_available


def test_wait_returns_once_healthy(ds, monkeypatch):
    import pandas as pd

    calls = []

    def status(dataset, **kw):
        calls.append(1)
        st = UNAVAILABLE if len(calls) < 3 else OK
        return pd.DataFrame([{"service": "materialize", "status": st}])

    monkeypatch.setattr(servers, "server_status", status)
    monkeypatch.setattr(servers.time, "sleep", lambda s: None)

    df = servers.wait_until_available("flywire", interval=30, verbose=False)
    assert len(calls) == 3
    assert (df["status"] == OK).all()


def test_wait_gives_up(ds, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        servers,
        "server_status",
        lambda dataset, **kw: pd.DataFrame(
            [{"service": "materialize", "status": UNAVAILABLE}]
        ),
    )
    monkeypatch.setattr(servers.time, "sleep", lambda s: None)

    with pytest.raises(TimeoutError, match="materialize"):
        servers.wait_until_available("flywire", interval=30, timeout=0, verbose=False)


def test_wait_does_not_hammer_the_server(ds):
    """A materialization does not finish in five seconds; do not poll as if it might."""
    import inspect

    sig = inspect.signature(servers.wait_until_available)
    assert sig.parameters["interval"].default >= 60


def test_wait_on_an_unknown_service_is_an_error_not_a_silent_pass(ds, monkeypatch):
    """A typo in service= must not filter the table to nothing and read as healthy."""
    import pandas as pd

    monkeypatch.setattr(
        servers,
        "server_status",
        lambda dataset, **kw: pd.DataFrame(
            [{"service": "materialize", "status": UNAVAILABLE}]
        ),
    )

    with pytest.raises(ValueError, match="no service 'materialise'"):
        servers.wait_until_available("flywire", service="materialise", verbose=False)
