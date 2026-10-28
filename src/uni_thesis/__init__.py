"""uni-thesis-engine: topic-agnostic thesis automation engine."""

from .schema import (
    DataConfig,
    ExperimentResultBundle,
    IntegrityError,
    LiteratureQuery,
    LiteratureRecord,
    MetricItem,
    TopicProfile,
)

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "ExperimentResultBundle",
    "IntegrityError",
    "LiteratureQuery",
    "LiteratureRecord",
    "MetricItem",
    "TopicProfile",
    "__version__",
]
