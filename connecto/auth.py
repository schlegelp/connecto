"""Credentials.

A facade, not a competing secret store. connecto reads tokens from wherever they
already live and never breaks an existing caveclient or neuprint-python setup.

Resolution order, per service:

1. an explicit ``token=`` argument
2. ``$CONNECTO_<SERVICE>_TOKEN``
3. the service's own environment variable (``$NEUPRINT_APPLICATION_CREDENTIALS``)
4. ``~/.connecto/config.toml``
5. the service's native secret store (for CAVE, whatever ``caveclient`` itself resolves)

Never the reverse - a token you already have working must keep working.

Two things beyond mere lookup live here, and they matter more than the lookup does:

* :func:`auth_errors` turns the 401s and 403s that the upstream libraries throw into
  a :class:`~connecto.exceptions.ConnectoAuthError` that names the service, the
  server, *which source the token came from*, and the actual fix.
* :func:`auth_status` **validates** - it asks each service who you are, rather than
  merely noting that some string exists. "Present" and "works" are different claims,
  and a status function that conflates them will cheerfully confirm the wrong
  hypothesis for someone whose token expired last week.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import requests

from .exceptions import ConnectoAuthError

__all__ = [
    "get_token",
    "set_token",
    "auth_status",
    "auth_errors",
    "validate_token",
    "TokenInfo",
]

CONFIG_PATH = Path(
    os.environ.get("CONNECTO_CONFIG", "~/.connecto/config.toml")
).expanduser()

CAVE_SECRETS = Path("~/.cloudvolume/secrets").expanduser()

CAVE_GLOBAL_SERVER = "https://global.daf-apis.com"

# Cheap authenticated endpoints that answer "who am I, and what may I touch?".
CAVE_IDENTITY = "{server}/sticky_auth/api/v1/user/cache"
NEUPRINT_IDENTITY = "https://{server}/profile"

TIMEOUT = 20


@dataclass
class TokenInfo:
    service: str
    token: str | None
    source: str

    @property
    def ok(self) -> bool:
        return bool(self.token)


# --------------------------------------------------------------------- lookup


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    import tomllib

    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _cave_source(server: str | None = None) -> str:
    """Which secret file caveclient will have read the token out of.

    Reporting only. We deliberately do *not* re-implement caveclient's lookup to
    obtain the token itself (see :func:`_cave_token`) - a private copy of someone
    else's resolution order is how you end up with two secret stores that silently
    disagree about which token is current, which is the exact failure this library
    exists to avoid.
    """
    candidates = []
    if server:
        host = server.replace("https://", "").replace("http://", "").rstrip("/")
        candidates.append(CAVE_SECRETS / f"{host}-cave-secret.json")
    candidates += [
        CAVE_SECRETS / "cave-secret.json",
        CAVE_SECRETS / "chunkedgraph-secret.json",  # deprecated but still out there
    ]
    for fp in candidates:
        if fp.exists():
            try:
                if json.loads(fp.read_text()).get("token"):
                    return str(fp)
            except json.JSONDecodeError:
                continue
    return "not found"


def _cave_token(server: str | None = None) -> tuple[str | None, str]:
    """Ask caveclient what token it resolves, so we can never disagree with it."""
    try:
        from caveclient import CAVEclient

        token = CAVEclient(global_only=True, server_address=server or CAVE_GLOBAL_SERVER).auth.token
    except Exception:  # noqa: BLE001 - no token is a fact, not an error
        token = None

    return (token, _cave_source(server)) if token else (None, "not found")


def get_token(
    service: str, *, token: str | None = None, server: str | None = None
) -> TokenInfo:
    """Resolve a token for ``service`` ("cave", "neuprint", "clio", "seatable")."""
    service = service.lower()

    if token:
        return TokenInfo(service, token, "argument")

    env = f"CONNECTO_{service.upper()}_TOKEN"
    if os.environ.get(env):
        return TokenInfo(service, os.environ[env], f"${env}")

    native_env = {
        "neuprint": "NEUPRINT_APPLICATION_CREDENTIALS",
        "clio": "CLIO_TOKEN",
        "seatable": "SEATABLE_TOKEN",
    }.get(service)
    if native_env and os.environ.get(native_env):
        raw = os.environ[native_env]
        # neuprint accepts either the bare token or the whole JSON doc.
        if raw.strip().startswith("{"):
            try:
                raw = json.loads(raw)["token"]
            except (json.JSONDecodeError, KeyError):
                pass
        return TokenInfo(service, raw, f"${native_env}")

    cfg = _read_config().get("auth", {})
    if server and cfg.get(server):
        return TokenInfo(service, cfg[server], f"{CONFIG_PATH.name} [{server}]")
    if cfg.get(service):
        return TokenInfo(service, cfg[service], CONFIG_PATH.name)

    if service == "clio":
        # clio piggybacks on the Google-auth token clio-py manages itself.
        return TokenInfo(service, None, "clio-py (managed)")

    if service == "cave":
        tok, src = _cave_token(server)
        return TokenInfo(service, tok, src)

    return TokenInfo(service, None, "not found")


def set_token(service: str, token: str, *, write_native: bool = True):
    """Store a token.

    With ``write_native=True`` (the default) the token is written to *both*
    connecto's config and the service's native location, so that caveclient,
    cloud-volume and connecto can never disagree about which token is current -
    which is the failure mode fafbseg papers over by reading from cloud-volume and
    writing via caveclient.
    """
    service = service.lower()

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = _read_config()
    cfg.setdefault("auth", {})[service] = token

    lines = ["[auth]"]
    lines += [f'{k} = "{v}"' for k, v in cfg["auth"].items()]
    for section, values in cfg.items():
        if section == "auth":
            continue
        lines.append(f"\n[{section}]")
        lines += [f"{k} = {v!r}" for k, v in values.items()]
    CONFIG_PATH.write_text("\n".join(lines) + "\n")

    if write_native and service == "cave":
        from caveclient import CAVEclient

        CAVEclient().auth.save_token(token, overwrite=True)


# ---------------------------------------------------------------- validation


SERVICE_NAMES = {
    "cave": "CAVE",
    "neuprint": "neuPrint",
    "clio": "clio",
    "seatable": "SeaTable",
}


@dataclass
class Identity:
    """Who a token says you are, and what it lets you near."""

    valid: bool
    email: str | None = None
    access: str | None = None  # CAVE groups, or a neuPrint auth level
    detail: str = ""
    # Whether we actually asked the server. clio and SeaTable have no cheap probe,
    # and reporting "OK" for a token nobody checked would be the very conflation of
    # "present" with "works" that this whole module exists to stop.
    validated: bool = True


def _token_url(server: str | None = None) -> str:
    """The page where you get a CAVE token. Ask caveclient rather than hardcoding."""
    try:
        from caveclient.auth import auth_endpoints_v1

        return auth_endpoints_v1["create_token"].format(
            auth_server_address=server or CAVE_GLOBAL_SERVER
        )
    except Exception:  # noqa: BLE001
        return f"{server or CAVE_GLOBAL_SERVER}/auth/api/v1/create_token"


def validate_token(service: str, *, token=None, server=None) -> Identity:
    """Ask the service whether a token actually works, and who it belongs to.

    One cheap request (~0.4s). This is the difference between "a string exists"
    and "you are authenticated", and only the second one is worth reporting.
    """
    service = service.lower()
    info = get_token(service, token=token, server=server)

    if not info.ok:
        return Identity(False, detail="no token found")

    headers = {"Authorization": f"Bearer {info.token}"}

    if service == "cave":
        url = CAVE_IDENTITY.format(server=(server or CAVE_GLOBAL_SERVER).rstrip("/"))
    elif service == "neuprint":
        url = NEUPRINT_IDENTITY.format(server=(server or "neuprint.janelia.org"))
    else:
        # No cheap probe for clio/seatable. Say so rather than implying one ran.
        return Identity(True, detail="present, not validated", validated=False)

    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        return Identity(False, detail=f"could not reach server ({type(e).__name__})")

    if r.status_code in (401, 403):
        return Identity(False, detail="server rejected this token")
    if not r.ok:
        return Identity(False, detail=f"HTTP {r.status_code}")

    try:
        data = r.json()
    except ValueError:
        return Identity(True, detail="ok")

    if service == "cave":
        groups = data.get("groups") or []
        return Identity(
            True,
            email=data.get("email"),
            access=", ".join(groups) if groups else None,
        )

    return Identity(True, email=data.get("Email"), access=data.get("AuthLevel"))


# ----------------------------------------------------------- error translation


def _message(kind: str, service: str, info: TokenInfo, server, resource) -> str:
    what = SERVICE_NAMES.get(service, service)
    where = f"\n  {server}" if server else ""
    where += f"\n  [{resource}]" if resource else ""

    if kind == ConnectoAuthError.MISSING:
        lines = [f"No {what} credentials found.{where}", ""]
        if service == "cave":
            lines += [
                f"Get a token at: {_token_url(server)}",
                'Then:  co.set_token("cave", "<token>")',
            ]
        elif service == "neuprint":
            lines += [
                "Get a token at: https://neuprint.janelia.org/account",
                'Then:  co.set_token("neuprint", "<token>")',
                "  (or set $NEUPRINT_APPLICATION_CREDENTIALS)",
            ]
        else:
            lines += [f'Set one with:  co.set_token("{service}", "<token>")']

    elif kind == ConnectoAuthError.INVALID:
        lines = [
            f"{what} rejected your token (HTTP 401).{where}",
            "",
            f"The token came from: {info.source}",
            "It is present, but the server says it is invalid or expired.",
            "",
            "Fix:",
        ]
        if service == "cave":
            lines += [f"  1. Get a new token: {_token_url(server)}"]
        else:
            lines += ["  1. Get a new token: https://neuprint.janelia.org/account"]
        lines += [f'  2. co.set_token("{service}", "<token>")']

    else:  # FORBIDDEN
        ident = validate_token(service, server=server)
        lines = [
            f"Your {what} token is valid, but you do not have permission "
            f"for this resource (HTTP 403).{where}",
            "",
        ]
        if ident.email:
            lines.append(f"Authenticated as: {ident.email}")
        if ident.access:
            lines.append(f"Your access:      {ident.access}")
        lines += [
            "",
            "This needs a permission (usually a group membership) that your account "
            "does not have.",
            "A new token will not help - you need to be granted access.",
        ]

    lines += ["", "Run co.auth_status() to see every credential connecto can find."]
    return "\n".join(lines)


@contextmanager
def auth_errors(service: str, *, server=None, resource=None):
    """Translate upstream 401/403s into something the user can act on.

    caveclient raises its own ``AuthException`` when no token is configured, and a
    bare ``requests.HTTPError`` when the server rejects one. neuprint-python (and
    connecto's own ``server_datasets``) raise a bare 401 either way, so a missing
    token and a wrong token are indistinguishable. Both become a
    :class:`ConnectoAuthError` here, and the three kinds stay distinct.
    """
    try:
        yield
    except ConnectoAuthError:
        raise
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code not in (401, 403):
            raise
        kind = (
            ConnectoAuthError.FORBIDDEN
            if code == 403
            else ConnectoAuthError.INVALID
        )
        info = get_token(service, server=server)
        # A 401 with no token at all is a *missing* token, not an invalid one.
        if kind == ConnectoAuthError.INVALID and not info.ok:
            kind = ConnectoAuthError.MISSING
        raise ConnectoAuthError(
            _message(kind, service, info, server, resource),
            kind=kind, service=service, server=server,
        ) from e
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != "AuthException":  # caveclient's "no token" path
            raise
        info = get_token(service, server=server)
        raise ConnectoAuthError(
            _message(ConnectoAuthError.MISSING, service, info, server, resource),
            kind=ConnectoAuthError.MISSING, service=service, server=server,
        ) from e


def missing_token(service: str, *, server=None, resource=None) -> ConnectoAuthError:
    """Build a MISSING error directly (for sources that check up front)."""
    info = get_token(service, server=server)
    return ConnectoAuthError(
        _message(ConnectoAuthError.MISSING, service, info, server, resource),
        kind=ConnectoAuthError.MISSING, service=service, server=server,
    )


# --------------------------------------------------------------------- status


def _neuprint_servers() -> list[str]:
    """Every neuPrint server connecto knows about.

    Not just one: `neuprint-fish2.janelia.org` is a separate deployment from
    `neuprint.janelia.org`, so a token good for one is not guaranteed good for the
    other. Checking them all is three cheap requests and something only connecto is
    in a position to do.
    """
    from .backends.neuprint.versions import parse_source
    from .core.registry import REGISTRY

    servers = {
        parse_source(b.source)[0]
        for spec in REGISTRY.values()
        for b in spec.backends
        if b.kind == "neuprint"
    }
    return sorted(servers) or ["neuprint.janelia.org"]


def auth_status(validate: bool = True):
    """Which credentials connecto can see, whether they *work*, and who you are.

    ``validate=True`` (the default) asks each service. It costs about a second, and
    it is the whole point: a presence-only check reports ``found`` for a token that
    expired last week, which is precisely the wrong answer for the one person who
    ever runs this function.
    """
    import pandas as pd

    rows = []

    def add(service, server, info, ident=None):
        if not info.ok:
            status = "MISSING"
        elif ident is None or not ident.validated:
            # Present, but nobody asked the server. Not the same as OK.
            status = "FOUND"
        elif ident.valid:
            status = "OK"
        else:
            status = "INVALID"
        rows.append(
            {
                "service": service,
                "server": server or "-",
                "status": status,
                "source": info.source,
                "identity": (ident.email if ident else None) or "-",
                "access": (ident.access if ident else None)
                or (ident.detail if ident else "")
                or "-",
            }
        )

    cave = get_token("cave")
    add("cave", "global.daf-apis.com", cave,
        validate_token("cave") if (validate and cave.ok) else None)

    for server in _neuprint_servers():
        info = get_token("neuprint", server=server)
        add("neuprint", server, info,
            validate_token("neuprint", server=server) if (validate and info.ok) else None)

    for service in ("clio", "seatable"):
        info = get_token(service)
        add(service, None, info,
            validate_token(service) if (validate and info.ok) else None)

    df = pd.DataFrame(rows)

    # If connecto and caveclient have resolved *different* CAVE tokens, say so.
    # Two secret stores that quietly disagree is the fafbseg bug; it should never
    # be something you have to discover for yourself.
    native, _ = _cave_token()
    if cave.ok and native and native != cave.token:
        df.attrs["warning"] = (
            f"connecto is using the CAVE token from {cave.source}, but caveclient "
            f"resolves a different one from ~/.cloudvolume/secrets. They disagree; "
            f'run co.set_token("cave", ...) to make them match.'
        )
        print("!! " + df.attrs["warning"])

    return df
