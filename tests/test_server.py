"""HTTP API surface, exercised through the real ASGI app."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_maker import __version__
from llm_maker.server import app


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    import llm_maker.server as server

    server.OUTPUT_ROOT = tmp_path_factory.mktemp("generated")
    with TestClient(app) as test_client:
        yield test_client


def test_index_serves_the_ui(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM" in response.text and "Maker" in response.text
    assert "text/html" in response.headers["content-type"]
    assert "optimized Trainer" in response.text
    assert "torch-compile" in response.text
    # The UI must not hardcode a sandbox-local host: it talks to its own origin.
    assert "localhost" not in response.text
    assert "127.0.0.1" not in response.text


def test_health_reports_catalog_state(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["version"] == __version__
    assert "hub" in payload and "online" in payload["hub"]
    assert payload["catalog"]["local_entries"] > 50
    assert payload["offline_mode"] is True


def test_presets_drive_the_ui_dropdowns(client) -> None:
    payload = client.get("/api/presets").json()
    goals = {goal["value"] for goal in payload["goals"]}
    assert goals == {"sft", "dpo", "cpt", "pretrain"}
    assert any(model["value"] == "7b" for model in payload["models"])
    assert any(gpu["value"] == "a100-80gb" for gpu in payload["gpus"])
    assert payload["references"]
    for goal in payload["goals"]:
        assert goal["blurb"] and goal["need_columns"]


def test_catalog_facets_are_offered(client) -> None:
    payload = client.get("/api/catalog").json()
    assert payload["count"] > 50
    assert payload["mode"] in ("hub", "snapshot")
    assert payload["facets"]["tasks"]
    assert payload["facets"]["languages"]


def test_dataset_search_works_offline(client) -> None:
    payload = client.get("/api/datasets", params={"q": "alpaca", "limit": 5}).json()
    assert payload["items"]
    assert payload["mode"] in ("snapshot", "cache")
    assert payload["items"][0]["id"] == "tatsu-lab/alpaca"
    assert payload["note"]


def test_dataset_search_filters(client) -> None:
    payload = client.get("/api/datasets", params={"task": "preference", "limit": 20}).json()
    assert payload["items"]
    assert all("preference" in item["task_categories"] for item in payload["items"])


def test_dataset_detail_and_ref(client) -> None:
    detail = client.get("/api/datasets/tatsu-lab/alpaca").json()
    assert detail["id"] == "tatsu-lab/alpaca"
    assert detail["est_tokens"] > 0

    ref = client.get("/api/datasets/tatsu-lab/alpaca/ref").json()
    assert ref["repo_id"] == "tatsu-lab/alpaca"
    assert ref["task"] in ("sft", "text", "preference", "unknown")
    assert ref["completion_field"] == "output"


def test_nested_repo_ids_route_correctly(client) -> None:
    """`owner/name` must reach the detail handler, not be swallowed by the list route."""
    response = client.get("/api/datasets/HuggingFaceH4/ultrachat_200k")
    assert response.status_code == 200
    assert response.json()["id"] == "HuggingFaceH4/ultrachat_200k"


def test_unknown_dataset_is_a_404_not_a_crash(client) -> None:
    response = client.get("/api/datasets/nobody/nothing")
    assert response.status_code == 404
    assert "not in the catalog" in response.json()["detail"]


def test_preview_says_why_it_is_unavailable_offline(client) -> None:
    payload = client.get("/api/datasets/tatsu-lab/alpaca/preview").json()
    assert payload["available"] is False
    assert payload["reason"]
    assert payload["rows"] == []


def test_plan_endpoint_returns_estimates(client) -> None:
    body = {
        "goal": "sft",
        "model": {"preset": "7b"},
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "task": "sft", "est_tokens": 15_000_000}],
        "hardware": {"gpu": "a100-80gb", "count": 1},
    }
    payload = client.post("/api/plan", json=body).json()
    assert payload["method"] in ("full", "lora", "qlora")
    assert payload["total_steps"] > 0
    assert payload["resource"]["vram_per_gpu_gb"] > 0
    assert payload["resource"]["basis"]
    assert payload["mixture"][0]["repo_id"] == "tatsu-lab/alpaca"
    assert payload["goal_label"]
    assert payload["references"]


def test_plan_endpoint_accepts_the_optimized_trainer_profile(client) -> None:
    body = {
        "goal": "sft",
        "model": {"preset": "0.5b"},
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "task": "sft", "est_tokens": 15_000_000}],
        "hardware": {"gpu": "a100-80gb", "count": 1},
        "hyper": {"trainer": "optimized", "torch_compile": True, "compile_mode": "max-autotune"},
    }
    payload = client.post("/api/plan", json=body).json()
    hyper = payload["request"]["hyper"]
    assert hyper["trainer"] == "optimized"
    assert hyper["torch_compile"] is True
    assert hyper["compile_mode"] == "max-autotune"


def test_plan_without_datasets_is_a_400(client) -> None:
    response = client.post("/api/plan", json={"goal": "sft", "datasets": []})
    assert response.status_code == 400
    assert "dataset" in response.json()["detail"].lower()


def test_recipe_generation_writes_a_project(client) -> None:
    body = {
        "goal": "sft",
        "model": {"preset": "0.5b"},
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "task": "sft", "est_tokens": 15_000_000}],
        "hardware": {"gpu": "a100-80gb", "count": 1},
        "project_name": "api-test",
    }
    payload = client.post("/api/recipes", json=body).json()
    assert payload["slug"] == "api-test"
    names = {entry["name"] for entry in payload["files"]}
    assert {"train.py", "config.yaml", "README.md", "data_audit.py"} <= names
    assert payload["commands"]["train"]
    assert payload["project"]["written"]

    listing = client.get("/api/recipes/api-test").json()
    assert listing["project_dir"]
    assert {entry["name"] for entry in listing["files"]} >= names

    text = client.get("/api/recipes/api-test/file/train.py")
    assert text.status_code == 200
    assert "load_dataset" in text.text

    archive = client.get("/api/recipes/api-test/download")
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert archive.content[:2] == b"PK"


def test_recipe_file_paths_cannot_escape_the_project(client) -> None:
    body = {
        "goal": "sft",
        "model": {"preset": "0.5b"},
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "est_tokens": 1_000_000}],
        "hardware": {"gpu": "a100-80gb", "count": 1},
        "project_name": "escape-test",
    }
    client.post("/api/recipes", json=body)
    response = client.get("/api/recipes/escape-test/file/../../etc/passwd")
    assert response.status_code in (400, 404)


def test_unknown_recipe_is_a_404(client) -> None:
    assert client.get("/api/recipes/never-generated").status_code == 404


def test_delete_removes_the_generated_project(client, tmp_path) -> None:
    body = {
        "goal": "sft",
        "model": {"preset": "0.5b"},
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "est_tokens": 1_000_000}],
        "hardware": {"gpu": "a100-80gb", "count": 1},
        "project_name": "delete-me",
    }
    created = client.post("/api/recipes", json=body).json()
    directory = created["project"]["project_dir"]
    assert client.delete("/api/recipes/delete-me").json()["removed"] is True
    import pathlib

    assert not pathlib.Path(directory).exists()


def test_openapi_schema_is_valid(client) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "LLM Maker"
    assert "/api/datasets" in schema["paths"]
    assert "/api/recipes" in schema["paths"]


def test_project_listing_starts_empty_shape(client) -> None:
    payload = client.get("/api/projects").json()
    assert "output_root" in payload
    assert isinstance(payload["projects"], list)
    for project in payload["projects"]:
        assert "plan" not in project, "the listing should stay small"
        json.dumps(project)
