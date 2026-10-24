"""Shared test fixtures.

The suite runs entirely offline: ``LLM_MAKER_OFFLINE=1`` is set before the
package is imported, so the catalog falls back to the bundled snapshot and no
test ever depends on huggingface.co being reachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LLM_MAKER_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def catalog(tmp_path: Path):
    from llm_maker.catalog import DatasetCatalog
    from llm_maker.hub import HubClient

    client = HubClient(endpoint="http://127.0.0.1:9", datasets_server="http://127.0.0.1:9", timeout=0.2)
    return DatasetCatalog(client, cache_dir=tmp_path / "cache", allow_network=False)


@pytest.fixture()
def project(tmp_path: Path):
    """A generated SFT project on disk, returned as (directory, plan)."""
    from llm_maker.models import DatasetRef, HyperParams, PlanRequest
    from llm_maker.planner import plan
    from llm_maker.recipes import model_from_preset
    from llm_maker.scaffold import write_project

    request = PlanRequest(
        goal="sft",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="tatsu-lab/alpaca", task="sft")],
        hyper=HyperParams(epochs=1.0, seq_len=512, micro_batch_size=2, grad_accum=4),
        project_name="pytest-project",
        output_root="runs",
    )
    built = plan(request, resolve_datasets=False)
    result = write_project(built, tmp_path, overwrite=True)
    return result.project_dir, built
