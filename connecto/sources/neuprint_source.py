"""Annotations straight from the neuPrint :Neuron nodes.

Like :mod:`connecto.sources.cave_table`, the client here comes from the *spec's*
neuPrint backend, not from whichever backend is currently answering queries -
which is what makes ``FlyWire(backend="cave", annotations="neuprint")`` a sentence
rather than a crash. Annotation source and query backend are a cross-product; this
module is one axis of it and knows nothing about the other.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["fetch_neuprint", "neuprint_context", "neuprint_freshness"]


def neuprint_context(ds):
    """A neuPrint client for this dataset, whichever backend is answering queries."""
    from . import borrow

    return borrow(ds, "neuprint").client


def neuprint_freshness(source, ds) -> str | None:
    """neuPrint's own record of when the database was last edited, or None.

    A published dataset (hemibrain) never changes it, so the token is stable and the
    cache is permanent. The live-curated ones (maleCNS, fish2) bump it on every
    segment-property update *without* changing the version tag - which the version key
    alone would miss. Free: ``meta`` is already on the client.
    """
    try:
        return str(neuprint_context(ds).meta.get("lastDatabaseEdit") or "") or None
    except Exception:
        return None


def fetch_neuprint(source, ds, version) -> pd.DataFrame:
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_neurons

    client = neuprint_context(ds)

    # fetch_neurons returns (neurons, roi_counts), but collapses to a single frame
    # when omit_rois=True.
    result = fetch_neurons(NC(client=client), client=client, omit_rois=True)
    neurons = result[0] if isinstance(result, tuple) else result
    neurons = neurons.copy()

    # Positions come back as [x, y, z] lists - or, on this unfiltered query, as the
    # raw Neo4j point: {"coordinates": [x, y, z], "crs": {...}, "type": "Point"}.
    # Both shapes, because which one you get depends on how neuprint-python fetched
    # it: `fetch_neurons` on a bodyId list converts them, on the whole dataset it
    # does not. Reading only the list form left `soma_x/y/z` silently all-null on
    # every neuPrint dataset - a column that says "we have somas and they're empty"
    # while `ds.somas` returned them perfectly well.
    def _coords(v):
        if isinstance(v, dict):
            v = v.get("coordinates")
        return tuple(v) if isinstance(v, (list, tuple)) and len(v) == 3 else (None,) * 3

    # Split them so they survive the trip through feather and so `fields` can point
    # at them.
    for col in ("somaLocation", "tosomaLocation", "rootLocation"):
        if col in neurons.columns:
            base = col.replace("Location", "")
            xyz = neurons[col].apply(_coords)
            for i, axis in enumerate("xyz"):
                neurons[f"{base}_{axis}"] = [
                    v[i] if v[i] is not None else pd.NA for v in xyz
                ]
            neurons = neurons.drop(columns=[col])

    # Anything still holding a list won't survive feather; stringify it. Only an
    # object column can hold one, and a generator stops at the first hit rather
    # than building a boolean Series per column - which matters now that two 175k-
    # row datasets read this by default (0.9s -> 0.3s on a 175k x 50 frame).
    for col in neurons.columns[neurons.dtypes == "object"]:
        if any(isinstance(v, (list, dict)) for v in neurons[col].to_numpy()):
            neurons[col] = neurons[col].map(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            )

    return neurons.reset_index(drop=True)
