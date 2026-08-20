"""Per-synapse transmitters on neuPrint.

This exists because ``neuprint-python`` cannot ask for them. Its
``fetch_synapse_connections`` builds a Cypher query that already pulls every
remaining Synapse property over the wire::

    apoc.map.removeKeys(ns, ['location', 'confidence', 'type']) as info_pre

...and then keeps only the keys that happen to be ROI names, discarding
``ntGabaProb`` and its siblings on the way out. There is no argument that turns
that off, so the properties are fetched and thrown away, every time.

Rather than fetch them twice and join on coordinates, connecto asks for them
directly. The query below is deliberately close to neuprint's own - same match
pattern, same ``WITH DISTINCT``, and the *same* ``SynapseCriteria`` rendering the
same WHERE clauses - so that the frame it returns is the frame
``fetch_synapse_connections`` would have returned, plus the columns it drops.
Reusing the criteria object rather than hand-rolling its conditions is what keeps
that true: it carries the dataset's default confidence threshold and the
``type``/ROI filters, so ``transmitters=True`` cannot quietly select a different
*set* of synapses than ``transmitters=False``.

Only used when ``transmitters=True``; the ordinary path stays on neuprint-python.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

__all__ = ["fetch_synapses_with_transmitters"]

# One query per this many bodies. neuprint batches by body *and* by partner count;
# this is the simpler half of that, and it is the half that matters - the failure
# mode being avoided is a single query over thousands of bodies timing out.
BATCH_SIZE = 25


def fetch_synapses_with_transmitters(
    ds, pre, post, *, nt_columns: dict[str, str], rois=None, progress: bool = True
) -> pd.DataFrame:
    """``fetch_synapse_connections``' frame, plus the transmitter columns.

    ``nt_columns`` names the Synapse properties to fetch - see
    :attr:`BackendSpec.nt_columns`. They come back raw, under their neuPrint names;
    :func:`connecto.core.schemas.add_transmitters` makes the call.
    """
    from neuprint import SynapseCriteria as SC
    from tqdm.auto import tqdm

    client = ds.client

    # Read the transmitter probabilities off the *presynapse*. A transmitter is a
    # property of the neuron releasing it, and neuPrint stores the prediction on
    # both ends of the connection; taking the post side would attribute the
    # upstream neuron's transmitter to the downstream one at every synapse.
    returns = ",\n                   ".join(f"ns.{p} AS {p}" for p in nt_columns)

    # The synapse-side filters, rendered by neuprint's own criteria object. ROIs go
    # on the postsynaptic side only, which is neuprint's convention: it is how
    # neuPrint assigns an ROI to a `ConnectsTo` edge, so filtering the other side
    # would make `synapses(rois=...)` disagree with `edges(rois=...)` for synapses
    # that straddle a border.
    src_crit = SC(matchvar="ns", client=client)
    src_crit.type = "pre"
    tgt_crit = SC(matchvar="ms", rois=rois, client=client)
    tgt_crit.type = "post"
    conditions = "\n".join(
        crit.condition("n", "m", "ns", "ms", prefix=12, comments=False)
        for crit in (src_crit, tgt_crit)
    )

    # Resolve the ROI in the query rather than shipping every synapse's ROI-flag map
    # back to be reduced here. Primary ROIs do not overlap, so `head` is the one ROI
    # or null - which is what neuprint's `primary_only=True` default returns too.
    primary = json.dumps(sorted(client.primary_rois))

    # `:Segment`, not `:Neuron`, on *both* sides. Taken off `_criteria` rather than
    # written out, so the label decision (and its long justification) lives in one
    # place and this path cannot drift from the ordinary one.
    label = ds._criteria().label

    # Which side carries the ids we were given. `Connectivity.synapses` calls this
    # once per direction, so exactly one of them is ever set.
    var, ids = ("n", pre) if pre is not None else ("m", post)
    ids = np.unique(np.asarray(ids, dtype="int64"))
    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    frames = []
    for batch in tqdm(
        batches, desc="Synapses", disable=not progress or len(batches) < 2, leave=False
    ):
        cypher = f"""\
            MATCH (n:{label})-[:Contains]->(nss:SynapseSet)
                  -[:ConnectsTo]->(mss:SynapseSet)<-[:Contains]-(m:{label}),
                  (nss)-[:Contains]->(ns:Synapse)
                  -[:SynapsesTo]->(ms:Synapse)<-[:Contains]-(mss)
            WHERE {var}.bodyId IN {json.dumps(batch.tolist())}
            WITH DISTINCT n, m, ns, ms
            {conditions}
            RETURN n.bodyId AS bodyId_pre,
                   m.bodyId AS bodyId_post,
                   ns.location.x AS x_pre,
                   ns.location.y AS y_pre,
                   ns.location.z AS z_pre,
                   ms.location.x AS x_post,
                   ms.location.y AS y_post,
                   ms.location.z AS z_post,
                   ns.confidence AS confidence_pre,
                   ms.confidence AS confidence_post,
                   {returns},
                   head([k IN keys(ns) WHERE k IN {primary}]) AS roi_pre,
                   head([k IN keys(ms) WHERE k IN {primary}]) AS roi_post
        """
        frames.append(client.fetch_custom(cypher))

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
