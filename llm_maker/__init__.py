"""LLM Maker -- plan a language model training run on Hugging Face datasets.

LLM Maker is *not* a trainer.  It is the step before the trainer: it browses the
Hugging Face datasets list, turns a selection into a concrete plan (method,
batches, steps, memory, wall clock) and emits a runnable project.

    from llm_maker import DatasetCatalog, SearchFilters, planner

    catalog = DatasetCatalog()
    results = catalog.search(SearchFilters(search="code instructions"), limit=5)
    ...
"""

from __future__ import annotations

__version__ = "0.1.0"

from .catalog import DatasetCatalog, SearchFilters, SearchResult
from .estimator import estimate
from .hub import DatasetInfo, HubClient, HubError, HubUnavailable
from .models import (
    DatasetRef,
    HardwareSpec,
    HyperParams,
    ModelSpec,
    PlanRequest,
    TrainingPlan,
)
from .planner import plan
from .scaffold import write_project

__all__ = [
    "__version__",
    "DatasetCatalog",
    "DatasetRef",
    "DatasetInfo",
    "HardwareSpec",
    "HubClient",
    "HubError",
    "HubUnavailable",
    "HyperParams",
    "ModelSpec",
    "PlanRequest",
    "SearchFilters",
    "SearchResult",
    "TrainingPlan",
    "estimate",
    "plan",
    "write_project",
]
