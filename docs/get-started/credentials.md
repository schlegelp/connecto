# Credentials

connecto is a **facade over your existing setup**, not a competing secret store. It
reads tokens from wherever they already live — `~/.cloudvolume/secrets/`,
`$NEUPRINT_APPLICATION_CREDENTIALS`, and so on — so a token that works with
`caveclient` or `neuprint-python` today keeps working.

If you have used FlyWire or hemibrain from Python before, you are probably already
set up. Run this to find out:

```python
import connecto as cn

cn.auth_status()
```

```
    service                      server   status                                          source           identity                              access
0      cave         global.daf-apis.com       OK   ~/.cloudvolume/secrets/cave-secret.json        you@lab.org   flywire_train, FANC_edit, flywire
1  neuprint    neuprint-cns.janelia.org       OK         $NEUPRINT_APPLICATION_CREDENTIALS        you@lab.org                          readwrite
2  neuprint  neuprint-fish2.janelia.org       OK         $NEUPRINT_APPLICATION_CREDENTIALS        you@lab.org                          readwrite
3  neuprint        neuprint.janelia.org       OK         $NEUPRINT_APPLICATION_CREDENTIALS        you@lab.org                          readwrite
4      clio                           -  MISSING                        clio-py (managed)                  -                                   -
5  seatable                           -    FOUND                          $SEATABLE_TOKEN                  -             present, not validated
```

## `auth_status()` actually asks

It does not merely note that some string exists — it hits each service's identity
endpoint and reports who you are and what you may access. Takes about a second.

This distinction is the entire point. "Present" and "works" are different claims,
and a status function that says `OK` for an expired token confirms exactly the
wrong hypothesis for the one person who most needs the truth. So there are four
statuses, not a boolean:

| status | meaning |
|---|---|
| `OK` | validated against the server — this token works |
| `INVALID` | a token was found, and the server rejected it |
| `MISSING` | no token found anywhere |
| `FOUND` | a token is present, but this service has no cheap way to check it |

`FOUND` is not a softer `OK`. clio and SeaTable have no identity endpoint worth
calling, so connecto says so rather than implying a check ran.

Every neuPrint server gets probed, not just one — `neuprint-fish2` is a separate
deployment from `neuprint.janelia.org`, and a token good for one is not guaranteed
good for the other.

```python
cn.auth_status(validate=False)   # instant, presence-only
```

## Getting tokens

=== "CAVE"

    FlyWire, BANC, MICrONS. One token for all of them.

    1. Go to <https://global.daf-apis.com/auth/api/v1/create_token>
    2. ```python
       cn.set_token("cave", "<token>")
       ```

    `set_token` writes to **both** connecto's config and cloud-volume's secret
    file, so the two can never drift apart and disagree about which token is
    current.

    Public FlyWire and MICrONS need only a valid token. The *production* FlyWire
    stack additionally needs your account to be in the right group — see
    [403s](#403-is-a-different-problem) below.

=== "neuPrint"

    hemibrain, maleCNS, MANC, fish2 — and, by default, FlyWire and BANC.

    1. Log in at <https://neuprint.janelia.org/> and copy your token from the
       account menu.
    2. ```python
       cn.set_token("neuprint", "<token>")
       ```

    Or set `NEUPRINT_APPLICATION_CREDENTIALS` in your environment, which is what
    `neuprint-python` itself reads.

Where connecto looks, in order:

1. the `token=` argument, if you passed one
2. `$CONNECTO_<SERVICE>_TOKEN`
3. the service's own native variable (e.g. `$NEUPRINT_APPLICATION_CREDENTIALS`)
4. `~/.connecto/config.toml`
5. the service's native secret store (for CAVE, whatever `caveclient` resolves)

## When it goes wrong

Every credential failure surfaces as one exception — `ConnectoAuthError` — raised
at first use, wherever the 401 actually happened (client construction, version
resolution, or a private table you are not allowed to read).

```
ConnectoAuthError: CAVE rejected your token (HTTP 401).
  [flywire_fafb_public]

The token came from: $CONNECTO_CAVE_TOKEN
It is present, but the server says it is invalid or expired.

Fix:
  1. Get a new token: https://global.daf-apis.com/auth/api/v1/create_token
  2. cn.set_token("cave", "<token>")
```

The line that matters is **`The token came from:`**. With five possible sources, a
401 tells you nothing on its own; knowing *which* of the five produced the rejected
token tells you exactly what to go and fix. A stale `$CONNECTO_CAVE_TOKEN` shadowing
a perfectly good `cave-secret.json` is otherwise a genuinely nasty afternoon.

### 403 is a different problem

A 403 means your token is fine and you are simply not in the required group — so it
is reported as a different *kind* of error, with a different fix:

```
ConnectoAuthError: Your CAVE token is valid, but you do not have permission for
  datastack `flywire_fafb_production` (HTTP 403).

Authenticated as: you@lab.org
Your groups:      flywire_train, default

This datastack needs a group you are not in. Ask a FlyWire admin for access -
a new token will not help.
```

Telling someone to fetch a new token here would send them down a road that cannot
possibly work, which is why 401 and 403 are never collapsed into "auth failed".

### Catching it

```python
from connecto import ConnectoAuthError

try:
    fw.connectivity.edges("DA1_lPN")
except ConnectoAuthError as e:
    print(e.kind)      # "missing" | "invalid" | "forbidden"
    print(e.service)   # "cave"
    print(e.server)    # "global.daf-apis.com"
```

## Public datasets

FlyWire's public release and MICrONS are public, but CAVE still requires a valid
token to talk to it at all — "public" means *anyone may get a token*, not *no token
needed*. There is no anonymous path.
