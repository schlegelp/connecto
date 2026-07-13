"""Annotations straight from the neuPrint :Neuron nodes."""

from __future__ import annotations

import pandas as pd

__all__ = ["fetch_neuprint"]


def fetch_neuprint(source, ds, version) -> pd.DataFrame:
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_neurons

    # fetch_neurons returns (neurons, roi_counts), but collapses to a single frame
    # when omit_rois=True.
    result = fetch_neurons(NC(client=ds.client), client=ds.client, omit_rois=True)
    neurons = result[0] if isinstance(result, tuple) else result
    neurons = neurons.copy()

    # Positions come back as [x, y, z] lists; split them so they survive the trip
    # through feather and so `fields` can point at them.
    for col in ("somaLocation", "tosomaLocation", "rootLocation"):
        if col in neurons.columns:
            base = col.replace("Location", "")
            xyz = neurons[col].apply(
                lambda v: v if isinstance(v, (list, tuple)) else (None, None, None)
            )
            for i, axis in enumerate("xyz"):
                neurons[f"{base}_{axis}"] = [
                    v[i] if v[i] is not None else pd.NA for v in xyz
                ]
            neurons = neurons.drop(columns=[col])

    # Anything still holding a list won't survive feather; stringify it.
    for col in neurons.columns:
        if neurons[col].map(lambda v: isinstance(v, (list, dict))).any():
            neurons[col] = neurons[col].map(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            )

    return neurons.reset_index(drop=True)
