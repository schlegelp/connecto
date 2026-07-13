"""connecto's backend-independent core."""

from .criteria import NeuronCriteria, parse_ids
from .dataset import UNSET, Dataset, namespace, requires
from .registry import (
    REGISTRY,
    capability_matrix,
    get_dataset,
    get_spec,
    list_datasets,
    register,
)
from .spec import AnnotationSource, BackendSpec, Cap, DatasetSpec
from .version import Version

__all__ = [
    "AnnotationSource",
    "BackendSpec",
    "Cap",
    "Dataset",
    "DatasetSpec",
    "NeuronCriteria",
    "REGISTRY",
    "UNSET",
    "Version",
    "capability_matrix",
    "get_dataset",
    "get_spec",
    "list_datasets",
    "namespace",
    "parse_ids",
    "register",
    "requires",
]
