"""Credential handling.

The offline half tests the translation layer with fabricated HTTP responses. The
network half reproduces the four failure modes that motivated all this, in a
sandboxed HOME so real secrets can't leak in and make a broken test pass.
"""

from __future__ import annotations

import pytest
import requests

import connecto as co
from connecto import auth
from connecto.exceptions import ConnectoAuthError


def _http_error(status: int) -> requests.HTTPError:
    r = requests.Response()
    r.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=r)


@pytest.fixture
def no_tokens(monkeypatch):
    for var in (
        "CONNECTO_CAVE_TOKEN", "CONNECTO_NEUPRINT_TOKEN",
        "NEUPRINT_APPLICATION_CREDENTIALS", "SEATABLE_TOKEN", "CLIO_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    # Don't let a real cave-secret.json on the dev box leak in.
    monkeypatch.setattr(auth, "_cave_token", lambda server=None: (None, "not found"))


@pytest.fixture
def fake_token(monkeypatch):
    monkeypatch.setenv("CONNECTO_CAVE_TOKEN", "not-a-real-token")
    monkeypatch.setenv("CONNECTO_NEUPRINT_TOKEN", "not-a-real-token")


# ------------------------------------------------------------------ translation

def test_401_with_a_token_is_INVALID_and_names_the_source(fake_token):
    with pytest.raises(ConnectoAuthError) as exc:
        with auth.auth_errors("cave", resource="flywire_fafb_public"):
            raise _http_error(401)

    e = exc.value
    assert e.kind == ConnectoAuthError.INVALID
    # The single most useful fact in the message: *which* of the five possible
    # sources produced the token the server just rejected.
    assert "$CONNECTO_CAVE_TOKEN" in str(e)
    assert "flywire_fafb_public" in str(e)
    assert "invalid or expired" in str(e)


def test_401_with_no_token_is_MISSING_not_INVALID(no_tokens):
    # A 401 because you have no token is a different problem from a 401 because
    # your token is wrong, even though the server returns the same code.
    with pytest.raises(ConnectoAuthError) as exc:
        with auth.auth_errors("neuprint", server="neuprint.janelia.org"):
            raise _http_error(401)

    assert exc.value.kind == ConnectoAuthError.MISSING
    assert "No neuPrint credentials found" in str(exc.value)


def test_403_is_FORBIDDEN_and_does_not_send_you_for_a_new_token(fake_token, monkeypatch):
    # Building the message probes for identity; don't hit the network in a unit test.
    monkeypatch.setattr(
        auth, "validate_token",
        lambda *a, **k: auth.Identity(True, email="you@lab.org", access="flywire_train"),
    )

    with pytest.raises(ConnectoAuthError) as exc:
        with auth.auth_errors("cave", resource="flywire_fafb_production"):
            raise _http_error(403)

    e = exc.value
    assert e.kind == ConnectoAuthError.FORBIDDEN
    assert "you@lab.org" in str(e) and "flywire_train" in str(e)
    # The whole point of splitting 403 from 401: a new token would NOT help, and
    # telling someone to get one sends them down the wrong path entirely.
    assert "A new token will not help" in str(e)
    assert "create_token" not in str(e)


def test_caveclient_AuthException_becomes_MISSING(no_tokens):
    class AuthException(Exception):  # duck-typed by name, as caveclient's is
        pass

    with pytest.raises(ConnectoAuthError) as exc:
        with auth.auth_errors("cave"):
            raise AuthException("You have not setup a token")

    assert exc.value.kind == ConnectoAuthError.MISSING


def test_other_http_errors_pass_through_untouched():
    # A 500 is not a credential problem and must not be dressed up as one.
    with pytest.raises(requests.HTTPError):
        with auth.auth_errors("cave"):
            raise _http_error(500)


def test_connecto_auth_errors_are_not_re_wrapped(fake_token):
    original = ConnectoAuthError("boom", kind="invalid", service="cave")
    with pytest.raises(ConnectoAuthError) as exc:
        with auth.auth_errors("cave"):
            raise original
    assert exc.value is original


# ---------------------------------------------------------------------- status

def test_auth_status_offline_reports_missing(no_tokens):
    df = co.auth_status(validate=False)
    assert set(df.columns) == {"service", "server", "status", "source", "identity", "access"}
    assert (df[df.service == "cave"]["status"] == "MISSING").all()


def test_auth_status_never_says_OK_without_asking(monkeypatch, fake_token):
    # A token nobody validated is FOUND, not OK. Conflating "present" with "works"
    # is the bug this whole module exists to fix - a status function that says OK
    # for an expired token confirms exactly the wrong hypothesis.
    df = co.auth_status(validate=False)
    assert "OK" not in set(df["status"])
    assert "FOUND" in set(df["status"])


def test_auth_status_probes_every_neuprint_server():
    # fish2 is a separate deployment; a token good for one server is not
    # guaranteed good for another, so all of them get checked.
    servers = auth._neuprint_servers()
    assert "neuprint.janelia.org" in servers
    assert "neuprint-fish2.janelia.org" in servers


# --------------------------------------------------------------------- network

@pytest.mark.network
def test_garbage_token_is_reported_INVALID_not_found(monkeypatch):
    """The regression test for the actual bug.

    Before this change, auth_status() reported `found: True` for a garbage token -
    so the one function a confused user reaches for confidently told them their
    credentials were fine while every query 401'd.
    """
    monkeypatch.setenv("CONNECTO_CAVE_TOKEN", "deadbeefdeadbeefdeadbeef")
    monkeypatch.setenv("CONNECTO_NEUPRINT_TOKEN", "deadbeefdeadbeefdeadbeef")

    df = co.auth_status()
    assert (df[df.service == "cave"]["status"] == "INVALID").all()
    assert (df[df.service == "neuprint"]["status"] == "INVALID").all()


@pytest.mark.network
def test_real_credentials_validate_and_identify():
    df = co.auth_status()
    cave = df[df.service == "cave"].iloc[0]
    assert cave["status"] == "OK"
    assert "@" in cave["identity"]      # authenticated as somebody
    assert cave["access"] != "-"        # and in at least one CAVE group

    np_row = df[(df.service == "neuprint") & (df.server == "neuprint.janelia.org")].iloc[0]
    assert np_row["status"] == "OK"
    assert np_row["access"] in ("readwrite", "readonly")


@pytest.mark.network
def test_bad_token_raises_ConnectoAuthError_from_both_backends(monkeypatch):
    monkeypatch.setenv("CONNECTO_CAVE_TOKEN", "deadbeefdeadbeefdeadbeef")
    monkeypatch.setenv("CONNECTO_NEUPRINT_TOKEN", "deadbeefdeadbeefdeadbeef")
    # Clients are cached per datastack/server; a stale good one would mask this.
    from connecto.backends.cave import dataset as cave_ds
    from connecto.backends.neuprint import dataset as np_ds
    from connecto.backends.neuprint import versions as np_versions

    cave_ds._CLIENTS.clear()
    np_ds._CLIENTS.clear()
    np_versions.server_datasets.cache_clear()

    for build in (co.FlyWire, co.Hemibrain):
        with pytest.raises(ConnectoAuthError) as exc:
            build().connectivity.edges("DA1_lPN")
        assert exc.value.kind == ConnectoAuthError.INVALID

    cave_ds._CLIENTS.clear()
    np_ds._CLIENTS.clear()
    np_versions.server_datasets.cache_clear()
