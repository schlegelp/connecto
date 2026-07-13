# Auth and cache

## Credentials

connecto reads tokens from wherever they already live. See
[Credentials](../get-started/credentials.md) for the guided version, including what to do
when something 401s.

```python
cn.auth_status()                  # validates against every service
cn.auth_status(validate=False)    # instant, presence-only
cn.set_token("cave", "<token>")
```

::: connecto.auth
    options:
      members:
        - auth_status
        - get_token
        - set_token
        - validate_token
        - auth_errors

## Servers

CAVE answers `503` for hours at a time while it builds a new materialization. These are
the two functions that make that survivable — and
[When the server is down](../guides/outages.md) is the guide.

```python
cn.server_status("flywire")          # which services are answering, right now
cn.wait_until_available("flywire")   # block until they are (explicitly — never automatic)
```

::: connecto.servers
    options:
      members:
        - server_status
        - wait_until_available

## Cache

Keyed on `(dataset, backend, version, query)` — so a cached frame can never be served for
a different materialization. See [Caching](../guides/caching.md).

::: connecto.cache
    options:
      members:
        - cache_dir
        - clear
        - size
        - enabled
        - download

## Exceptions

::: connecto.exceptions
