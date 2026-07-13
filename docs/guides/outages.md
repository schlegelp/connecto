# When the server is down

CAVE goes down. Not rarely, and not briefly: every time a datastack builds a new
materialization version, the materialize service answers `503` for the duration —
which is **hours**, not seconds.

That is a normal, expected, scheduled part of running a CAVE deployment. It should
not look like a crash.

## What you used to get

```pytb
requests.exceptions.HTTPError: 503 Server Error: Service Unavailable for url:
https://prod.flywire-daf.com/materialize/api/v2/datastack/flywire_fafb_public/version/783/table/valid_connection_v2/query
content:b'<html>\r\n<head><title>503 Service Temporarily Unavailable</title></head>\r\n<body>\r\n
<center><h1>503 Service Temporarily Unavailable</h1></center>\r\n<hr><center>nginx</center>\r\n
</body>\r\n</html>\r\n'
```

Forty lines of stack, a page of nginx boilerplate, and one useful character: `503`.
It tells you nothing about *why*, nothing about *how long*, and — worst of all —
nothing about whether the problem is at their end or yours.

## What you get now

```pycon
>>> fw.connectivity.edges("DA1_lPN")
```

```
ConnectoServerError: CAVE returned 503 (Service Unavailable).

  dataset:   flywire_fafb_public
  service:   materialize
  endpoint:  https://prod.flywire-daf.com/materialize/api/v2/datastack/flywire_fafb…

CAVE services, checked just now:
  info          OK           client setup, datastack metadata
  materialize   UNAVAILABLE  annotations, connectivity, synapses, somas, ROIs, version lookup
  chunkedgraph  OK           segmentation, edit history
  skeleton      OK           skeletons

Only materialize is down; the rest of CAVE is fine. That is exactly what a
materialization in progress looks like from outside.

CAVE serves 503 while it builds a new materialization version. That runs for
*hours*, not seconds, so a retry loop will not outlast it - and caveclient has
already retried this three times.

Next:
  co.wait_until_available("flywire")   # block until it is back
  co.server_status("flywire")          # re-check the table above
  Anything already in the cache still works (co.cache).
```

## The question the table answers

The single most useful thing to know when CAVE 503s is **whether it is only CAVE**.

| what you see | what it means | what to do |
|---|---|---|
| one service down, rest OK | a materialization is running | wait — hours |
| every service `UNAVAILABLE` | the whole deployment is out | wait — possibly longer |
| every service `UNREACHABLE` | **your** network: VPN, proxy, offline | fix your laptop |

Those three have completely different fixes, and a bare `503` cannot tell them apart.
So when a query fails, connecto probes each service the dataset depends on — four
cheap requests, run concurrently, about 0.4 s — and reports what they actually said.
The wording of the message follows from the evidence rather than preceding it: if the
servers are answering again by the time it looks, it says *"whatever this was, it was
brief — retry the call"*, and does not lecture you about materializations.

You can ask the same question directly, without waiting for something to fail:

```pycon
>>> co.server_status("flywire")
```

```
     service               server status  http  seconds
        info  global.daf-apis.com     OK   200     0.33
 materialize prod.flywire-daf.com     OK   200     0.35
chunkedgraph prod.flywire-daf.com     OK   200     0.38
    skeleton prod.flywire-daf.com     OK   200     0.32
```

## Waiting it out

connecto does **not** silently retry a 503 for you. That is deliberate. No honest
backoff outlasts a multi-hour materialization, and a library that quietly sleeps for
three hours while pretending to work is worse than one that raises and lets you
decide.

(caveclient does retry, three times over about 0.7 seconds. By the time you see a
`ConnectoServerError`, that has already happened and failed.)

If you *do* want to wait — an overnight script, a long batch job — say so explicitly:

```python
co.wait_until_available("flywire")            # blocks; polls every 5 min, 6 h cap
co.wait_until_available("flywire", timeout=12 * 3600, interval=600)
```

```
[14:02:11] materialize=UNAVAILABLE - retrying in 300s (giving up in 5h58m)
[14:07:12] materialize=UNAVAILABLE - retrying in 300s (giving up in 5h53m)
...
flywire is back (1h35m waited).
```

It raises `TimeoutError` if it gives up. The poll interval is floored at 30 s —
a materialization does not finish in less time than that, and hammering a server
that is already struggling is not neighbourly.

## What still works during a materialization

Whatever is already **cached**. The cache key includes the materialization version,
so a cached frame can never be served for the wrong one — which means serving it
during an outage is safe. See [Caching](caching.md).

Beyond that, connecto deliberately does **not** print a list of "things that still
work". It is tempting — the chunkedgraph is up, so surely `ds.segmentation` is fine?
— but it is not reliably true:

- **every version lookup goes through the materialize service.** Even an explicitly
  pinned `ds.at(783)` calls `get_versions_metadata()` to check that 783 exists.
- `ds.segmentation.update_ids()` needs a materialization *timestamp*, so it goes
  there too.

So the honest answer is that most of the library needs materialize, and a cheerful
"segmentation still works!" would send you down a path that dead-ends one call later.
connecto reports which **services** are answering and what each one backs; the leap
from that to your particular call is left to you, because that is the part it cannot
make safely.

## Catching it

```python
import connecto as co

try:
    edges = fw.connectivity.edges("DA1_lPN")
except co.ConnectoServerError as e:
    if e.kind == e.UNAVAILABLE:
        co.wait_until_available("flywire")
        edges = fw.connectivity.edges("DA1_lPN")
    else:
        raise
```

`ConnectoServerError` has three kinds, because they want three different responses:

| kind | HTTP | means |
|---|---|---|
| `UNAVAILABLE` | 502, 503, 504 | up, but refusing work — a materialization, or an update |
| `ERROR` | 500 | a genuine server-side failure; worth reporting upstream |
| `UNREACHABLE` | *(none)* | no response at all — usually your own network |

It carries `.kind`, `.service`, `.server` and `.status`, so you can branch on the
cause rather than grepping the message.

Credential failures stay separate: a 401 or 403 raises
[`ConnectoAuthError`](../get-started/credentials.md), never `ConnectoServerError`.
"Your token is wrong" and "the server is down" are different problems with different
fixes, and conflating them is how people spend an afternoon regenerating a token that
was fine all along.
