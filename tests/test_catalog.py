"""Catalog, hub parsing and offline fallback."""

from __future__ import annotations

import json

import pytest

from llm_maker.hub import DatasetInfo, HubClient, HubUnavailable, estimate_tokens
from llm_maker.catalog import DatasetCatalog, SearchFilters


def test_snapshot_loads_and_is_unique(catalog: DatasetCatalog) -> None:
    entries = catalog.local_entries()
    assert len(entries) > 50
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate repo ids in the bundled snapshot"


def test_search_ranks_exact_matches_first(catalog: DatasetCatalog) -> None:
    result = catalog.search(SearchFilters(search="alpaca"), limit=5)
    assert result.mode in ("snapshot", "cache")
    assert result.items
    assert result.items[0].id == "tatsu-lab/alpaca"


def test_search_respects_task_filter(catalog: DatasetCatalog) -> None:
    result = catalog.search(SearchFilters(tasks=["preference"]), limit=50)
    assert result.items
    for item in result.items:
        assert "preference" in item.task_categories


def test_search_min_downloads_filters(catalog: DatasetCatalog) -> None:
    result = catalog.search(SearchFilters(min_downloads=500_000), limit=100)
    assert result.items
    assert all((item.downloads or 0) >= 500_000 for item in result.items)


def test_gated_datasets_can_be_excluded(catalog: DatasetCatalog) -> None:
    excluded = catalog.search(SearchFilters(search="stack", include_gated=False), limit=20)
    assert all(not item.gated for item in excluded.items)


def test_unknown_query_returns_nothing(catalog: DatasetCatalog) -> None:
    result = catalog.search(SearchFilters(search="zzzzz-not-a-real-dataset-zzzzz"), limit=5)
    assert result.items == []


def test_facets_cover_snapshot(catalog: DatasetCatalog) -> None:
    facets = catalog.facets()
    assert facets["count"] == len(catalog.local_entries())
    assert any(f["value"] == "pretrain" for f in facets["tasks"])
    assert any(f["value"] == "en" for f in facets["languages"])
    assert any(f["value"] == "hi" for f in facets["languages"]), "Indic datasets should show up"


def test_offline_search_reports_snapshot_mode(catalog: DatasetCatalog) -> None:
    result = catalog.search(SearchFilters(search="code"), limit=3)
    assert result.mode == "snapshot"
    assert result.note and "offline" in result.note.lower()


def test_hub_client_reports_unreachable_endpoint() -> None:
    client = HubClient(endpoint="http://127.0.0.1:9", datasets_server="http://127.0.0.1:9", timeout=0.2)
    status = client.probe()
    assert status["online"] is False
    assert "error" in status


def test_list_datasets_raises_hub_unavailable() -> None:
    client = HubClient(endpoint="http://127.0.0.1:9", timeout=0.2)
    with pytest.raises(HubUnavailable):
        client.list_datasets(limit=1)


def test_cursor_parsing_from_link_header() -> None:
    from llm_maker.hub import _cursor_from_url

    link = "https://huggingface.co/api/datasets?limit=30&cursor=abc123%3D%3D"
    assert _cursor_from_url(link) == "abc123=="


def test_dataset_info_parses_hub_payload() -> None:
    payload = {
        "id": "acme/widgets",
        "author": "acme",
        "downloads": 1234,
        "likes": 56,
        "lastModified": "2024-05-01T10:00:00.000Z",
        "tags": ["task_categories:instruction", "language:en", "license:mit", "size_categories:10K<n<100K"],
        "cardData": {
            "language": ["en"],
            "license": ["mit"],
            "pretty_name": "Widget instructions",
            "dataset_info": {
                "config_name": "default",
                "splits": [{"name": "train", "num_examples": 12_345, "num_bytes": 4_000_000}],
            },
        },
        "gated": False,
    }
    info = DatasetInfo.from_hub(payload)
    assert info.id == "acme/widgets"
    assert info.task_categories == ["instruction"]
    assert info.languages == ["en"]
    assert info.licenses == ["mit"]
    assert info.num_examples == 12_345
    assert info.num_bytes == 4_000_000
    assert info.best_size_category() == "10K<n<100K"
    assert info.description == "Widget instructions"
    assert info.url.endswith("/datasets/acme/widgets")
    # 4MB of bytes is preferred over the row count: it is the tighter signal.
    assert estimate_tokens(info) == int(4_000_000 / 3.6)


def test_token_estimate_uses_row_count_when_no_bytes() -> None:
    payload = {
        "id": "acme/chat",
        "tags": ["task_categories:chat"],
        "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 200_000}]}},
    }
    info = DatasetInfo.from_hub(payload)
    assert info.num_examples == 200_000
    # chat rows are long: 200k rows x ~450 tokens/row
    assert 80_000_000 < estimate_tokens(info) < 100_000_000


def test_token_estimate_falls_back_to_size_bucket() -> None:
    info = DatasetInfo.from_hub({"id": "acme/thing", "tags": ["size_categories:10K<n<100K"]})
    estimate = estimate_tokens(info)
    assert estimate and 1_000_000 < estimate < 100_000_000


def test_snapshot_round_trip_is_json_safe(catalog: DatasetCatalog) -> None:
    payload = catalog.local_entries()[0].to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_cache_round_trip(tmp_path) -> None:
    client = HubClient(endpoint="http://127.0.0.1:9", timeout=0.2)
    catalog = DatasetCatalog(client, cache_dir=tmp_path, allow_network=False)
    info = DatasetInfo.from_hub({"id": "acme/cached", "downloads": 5, "tags": ["language:en"]})
    catalog.remember([info], source="hub")

    fresh = DatasetCatalog(client, cache_dir=tmp_path, allow_network=False)
    ids = [e.id for e in fresh.cached_entries()]
    assert "acme/cached" in ids
    assert any(e.id == "acme/cached" for e in fresh.local_entries())


def test_detail_falls_back_to_snapshot(catalog: DatasetCatalog) -> None:
    payload = catalog.detail("tatsu-lab/alpaca", allow_network=False)
    assert payload["id"] == "tatsu-lab/alpaca"
    assert payload["est_tokens"]
    assert payload["mode"] in ("snapshot", "cache")


def test_detail_unknown_repo_raises(catalog: DatasetCatalog) -> None:
    with pytest.raises(KeyError):
        catalog.detail("nobody/nothing", allow_network=False)


def test_preview_is_unavailable_offline(catalog: DatasetCatalog) -> None:
    preview = catalog.preview("tatsu-lab/alpaca")
    assert preview["available"] is False
    assert preview["rows"] == []
