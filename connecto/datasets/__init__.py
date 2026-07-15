"""The named datasets.

Each is a thin factory over a :class:`~connecto.core.spec.DatasetSpec` in the
registry. Importing this module registers them all.
"""

from .aedes import AEDES_SPEC, Aedes
from .banc import BANC, BANC_SPEC
from .fanc import FANC, FANC_SPEC
from .flywire import FLYWIRE, FLYWIRE_PRODUCTION, FlyWire
from .generic import CAVE, NeuPrint, probe_cave, probe_neuprint
from .janelia import (
    FISH2,
    HEMIBRAIN,
    MALECNS,
    MANC,
    MANC_SPEC,
    Fish2,
    Hemibrain,
    MaleCNS,
)
from .microns import MICRONS_SPEC, MICrONS

__all__ = [
    "AEDES_SPEC", "BANC_SPEC", "FANC_SPEC", "MANC_SPEC", "MICRONS_SPEC",
    "FISH2", "FLYWIRE", "FLYWIRE_PRODUCTION", "HEMIBRAIN", "MALECNS",
    "CAVE", "NeuPrint",
    "Aedes", "BANC", "FANC", "Fish2", "FlyWire", "Hemibrain", "MANC", "MICrONS",
    "MaleCNS",
    "probe_cave", "probe_neuprint",
]
