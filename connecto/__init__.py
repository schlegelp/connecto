"""connecto - a unified interface for querying connectomes.

    import connecto as co

    fw = co.FlyWire()                    # CAVE
    hb = co.Hemibrain()                  # neuPrint

    fw.connectivity.edges("DA1_lPN")     # pre, post, weight
    hb.connectivity.edges("DA1_lPN")     # the same frame, a different backend

Datasets are immutable handles. Backends are ten `_fetch_*` hooks. Everything the
user sees - column names, dtypes, units, capability errors - is normalised in one
place, so the same code works on FlyWire, hemibrain, BANC, MICrONS and zebrafish.

Nothing degrades silently. If a dataset cannot do what you asked, it says so:

    fw.connectivity.synapses(x, min_score=100)    # fine
    mic.connectivity.synapses(x, min_score=100)   # CapabilityError, not a shrug
"""

from __future__ import annotations

import logging

__version__ = "0.1.1"

logger = logging.getLogger("connecto")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def _configure_sessions():
    """Retry/pool defaults must be set before any CAVEclient is constructed."""
    try:
        from caveclient import set_session_defaults

        set_session_defaults(max_retries=3, pool_maxsize=20, backoff_factor=0.2)
    except Exception:  # noqa: BLE001 - never fail at import time
        pass


_configure_sessions()

from . import cache, viz  # noqa: E402
from .auth import auth_status, get_token, set_token  # noqa: E402
from .core import (  # noqa: E402
    Cap,
    Dataset,
    DatasetSpec,
    NeuronCriteria,
    Version,
    capability_matrix,
    get_dataset,
    get_spec,
    list_datasets,
    register,
)
from .core.spec import AnnotationSource, BackendSpec, Publication  # noqa: E402
from .datasets import (  # noqa: E402
    BANC,
    CAVE,
    FANC,
    MANC,
    Aedes,
    Fish2,
    FlyWire,
    Hemibrain,
    MaleCNS,
    MICrONS,
    NeuPrint,
)
from .exceptions import (  # noqa: E402
    AmbiguousVersionError,
    CapabilityError,
    ConnectoAuthError,
    ConnectoError,
    ConnectoServerError,
    MissingDependencyError,
    NoSuchDatasetError,
    NoSuchVersionError,
)
from .servers import server_status, wait_until_available  # noqa: E402

__all__ = [
    # datasets
    "Aedes", "BANC", "CAVE", "FANC", "Fish2", "FlyWire", "Hemibrain", "MANC",
    "MICrONS", "MaleCNS", "NeuPrint",
    # registry
    "capability_matrix", "get_dataset", "get_spec", "list_datasets", "register",
    # building blocks
    "AnnotationSource", "BackendSpec", "Cap", "Dataset", "DatasetSpec",
    "NeuronCriteria", "Publication", "Version",
    # auth, cache & viz
    "auth_status", "cache", "get_token", "set_token", "viz",
    # servers
    "server_status", "wait_until_available",
    # errors
    "AmbiguousVersionError", "CapabilityError", "ConnectoAuthError", "ConnectoError",
    "ConnectoServerError", "MissingDependencyError", "NoSuchDatasetError",
    "NoSuchVersionError",
    "__version__",
]
