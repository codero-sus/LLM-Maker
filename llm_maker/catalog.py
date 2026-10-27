"""Dataset catalog: live Hub list + on-disk cache + bundled offline snapshot.

The rest of the app (server, CLI, estimators) only ever talks to
:class:`DatasetCatalog`.  It decides whether a query can be answered by the
live Hub, by a previously cached Hub response, or by the bundled snapshot, and
it always reports which of the three it used in ``mode``.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from .hub import (
    SIZE_CATEGORIES,
    DatasetInfo,
    HubClient,
    HubError,
    HubUnavailable,
    estimate_tokens,
)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("LLM_MAKER_CACHE", Path.home() / ".cache" / "llm_maker")
)
MAX_CACHED_ENTRIES = 4000
PROBE_TTL_SECONDS = 60.0

SORT_KEYS = {
    "trending": ("downloads", -1),
    "downloads": ("downloads", -1),
    "likes": ("likes", -1),
    "lastModified": ("lastModified", -1),
    "createdAt": ("createdAt", -1),
    "relevance": ("downloads", -1),
}


@dataclass
class SearchFilters:
    search: str | None = None
    author: str | None = None
    languages: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    min_downloads: int | None = None
    include_gated: bool = True

    def to_hub_filters(self) -> list[str]:
        out: list[str] = []
        out += [f"language:{v}" for v in self.languages]
        out += [f"task_categories:{v}" for v in self.tasks]
        out += [f"size_categories:{v}" for v in self.sizes]
        out += [f"license:{v}" for v in self.licenses]
        return out


@dataclass
class SearchResult:
    items: list[DatasetInfo]
    mode: str                       # "hub" | "cache" | "snapshot"
    total_matched: int
    next_cursor: str | None = None
    note: str | None = None
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "mode": self.mode,
            "total_matched": self.total_matched,
            "next_cursor": self.next_cursor,
            "note": self.note,
            "elapsed_ms": self.elapsed_ms,
        }


class DatasetCatalog:
    """Searchable view over Hugging Face datasets."""

    def __init__(
        self,
        client: HubClient | None = None,
        *,
        cache_dir: Path | str | None = None,
        cache_ttl: float = 900.0,
        allow_network: bool | None = None,
    ) -> None:
        self.client = client or HubClient()
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_ttl = cache_ttl
        self.allow_network = (
            allow_network
            if allow_network is not None
            else os.environ.get("LLM_MAKER_OFFLINE", "") not in {"1", "true", "yes"}
        )
        self._snapshot: list[DatasetInfo] | None = None
        self._cache: dict[str, dict[str, Any]] | None = None
        self._probe: tuple[float, dict[str, Any]] = (0.0, {})

    # ------------------------------------------------------------- persistence
    @property
    def snapshot_path(self) -> Path:
        return self.cache_dir / "catalog_cache.json"

    def load_snapshot(self) -> list[DatasetInfo]:
        """The bundled catalog shipped inside the package."""
        if self._snapshot is None:
            raw = (
                resources.files("llm_maker.data")
                .joinpath("hf_datasets_snapshot.json")
                .read_text(encoding="utf-8")
            )
            payload = json.loads(raw)
            self._snapshot = [
                DatasetInfo.from_snapshot(entry) for entry in payload.get("datasets", [])
            ]
            for entry, source_payload in zip(self._snapshot, payload.get("datasets", [])):
                entry.description = source_payload.get("description")
        return self._snapshot

    def load_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        entries: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_entries = payload.get("entries")
                if isinstance(raw_entries, dict):
                    for repo_id, record in raw_entries.items():
                        if isinstance(record, dict) and isinstance(record.get("data"), dict):
                            entries[repo_id] = record
        except (OSError, ValueError):
            entries = {}
        self._cache = entries
        return entries

    def save_cache(self) -> None:
        if not self._cache:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {"updated_at": time.time(), "entries": self._cache}
            self.snapshot_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # caching is best-effort

    def remember(self, items: Iterable[DatasetInfo], *, source: str = "hub") -> None:
        cache = self.load_cache()
        now = time.time()
        for item in items:
            cache[item.id] = {
                "fetched_at": now,
                "source": source,
                "data": {**item.to_dict(), "source": source},
            }
        if len(cache) > MAX_CACHED_ENTRIES:
            # keep the most recently fetched slice
            ordered = sorted(
                cache.items(), key=lambda kv: kv[1].get("fetched_at", 0), reverse=True
            )[:MAX_CACHED_ENTRIES]
            self._cache = dict(ordered)
        self.save_cache()

    def cached_entries(self) -> list[DatasetInfo]:
        out = []
        for record in self.load_cache().values():
            try:
                out.append(DatasetInfo.from_snapshot(record["data"]))
            except (KeyError, TypeError):
                continue
        return out

    def local_entries(self) -> list[DatasetInfo]:
        """Cache ∪ snapshot, cache winning on id collisions."""
        merged: dict[str, DatasetInfo] = {}
        for item in self.load_snapshot():
            merged[item.id] = item
        for item in self.cached_entries():
            item.source = "cache"
            item.metrics_approximate = bool(item.metrics_approximate and item.downloads is None)
            merged[item.id] = item
        return list(merged.values())

    # ----------------------------------------------------------------- network
    def hub_status(self, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        cached_at, cached = self._probe
        if not force and cached and now - cached_at < PROBE_TTL_SECONDS:
            return cached
        if not self.allow_network:
            status: dict[str, Any] = {
                "online": False,
                "endpoint": self.client.endpoint,
                "reason": "network disabled (LLM_MAKER_OFFLINE=1)",
            }
        else:
            status = self.client.probe()
        self._probe = (now, status)
        return status

    # -------------------------------------------------------------- searching
    def search(
        self,
        filters: SearchFilters | None = None,
        *,
        sort: str = "downloads",
        limit: int = 24,
        cursor: str | None = None,
        offset: int = 0,
        use_hub: bool | None = None,
    ) -> SearchResult:
        filters = filters or SearchFilters()
        started = time.perf_counter()
        hub_ok = (self.allow_network if use_hub is None else use_hub) and bool(
            self.hub_status().get("online")
        )

        if hub_ok:
            hub_sort, direction = SORT_KEYS.get(sort, ("downloads", -1))
            if sort == "relevance" and not filters.search:
                hub_sort, direction = "downloads", -1
            try:
                items, next_cursor = self.client.list_datasets(
                    search=filters.search,
                    author=filters.author,
                    filters=filters.to_hub_filters(),
                    sort=hub_sort,
                    direction=direction,
                    limit=limit,
                    cursor=cursor,
                    full=True,
                )
                items = [i for i in items if self._post_filter(i, filters)]
                self.remember(items, source="hub")
                return SearchResult(
                    items=items,
                    mode="hub",
                    total_matched=len(items),
                    next_cursor=next_cursor,
                    note=f"live from {self.client.endpoint}/api/datasets",
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            except (HubUnavailable, HubError) as exc:
                note = f"Hub query failed ({exc}); fell back to the local catalog."
                hub_ok = False
        else:
            note = None

        local = self.local_entries()
        scored = []
        for entry in local:
            if not self._post_filter(entry, filters):
                continue
            score = _score(entry, filters.search)
            if filters.search and score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda pair: (-pair[0], -(pair[1].downloads or 0), pair[1].id))
        total = len(scored)
        window = [entry for _, entry in scored[offset : offset + limit]]

        modes = {e.source for e in local}
        mode = "cache" if "cache" in modes else "snapshot"
        if note is None:
            note = (
                "offline catalog: previously cached Hub entries plus the bundled snapshot "
                "(metrics are approximate)"
                if mode == "cache"
                else "offline catalog: the bundled snapshot of Hugging Face datasets "
                "(metrics are approximate)"
            )
        return SearchResult(
            items=window,
            mode=mode,
            total_matched=total,
            next_cursor=None,
            note=note,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _post_filter(self, entry: DatasetInfo, f: SearchFilters) -> bool:
        if entry.private or entry.disabled:
            return False
        if not f.include_gated and entry.gated:
            return False
        if f.author and (entry.author or "").lower() != f.author.lower():
            return False
        if f.languages and not (set(map(str.lower, entry.languages)) & set(map(str.lower, f.languages))):
            return False
        if f.tasks and not (set(map(str.lower, entry.task_categories)) & set(map(str.lower, f.tasks))):
            return False
        if f.sizes and not (set(entry.size_categories) & set(f.sizes)):
            return False
        if f.licenses and not (set(map(str.lower, entry.licenses)) & set(map(str.lower, f.licenses))):
            return False
        if f.modalities and not (set(map(str.lower, entry.modalities)) & set(map(str.lower, f.modalities))):
            return False
        if f.min_downloads is not None and (entry.downloads or 0) < f.min_downloads:
            return False
        return True

    # --------------------------------------------------------------- details
    def detail(self, repo_id: str, *, allow_network: bool | None = None) -> dict[str, Any]:
        """Full record for one dataset, live if possible, local otherwise."""
        network = (self.allow_network if allow_network is None else allow_network) and bool(
            self.hub_status().get("online")
        )
        if network:
            try:
                info = self.client.get_dataset(repo_id, full=True)
                try:
                    splits = self.client.splits(repo_id)
                except (HubUnavailable, HubError):
                    splits = []
                if splits:
                    info.splits = [
                        {
                            "config": s.get("config"),
                            "split": s.get("split"),
                            "num_examples": s.get("num_examples"),
                            "num_bytes": s.get("num_bytes"),
                        }
                        for s in splits
                    ]
                    info.num_examples = sum(
                        s["num_examples"] or 0 for s in info.splits
                    ) or info.num_examples
                    info.num_bytes = sum(s["num_bytes"] or 0 for s in info.splits) or info.num_bytes
                    info.configs = sorted({s["config"] for s in info.splits if s["config"]})
                self.remember([info], source="hub")
                payload = info.to_dict()
                payload["mode"] = "hub"
                payload["est_tokens"] = estimate_tokens(info)
                return payload
            except (HubUnavailable, HubError) as exc:
                note = f"Hub lookup failed ({exc}); served from the local catalog."
            else:
                note = None
        else:
            note = "offline: served from the local catalog."

        # local fallback
        for entry in self.local_entries():
            if entry.id.lower() == repo_id.lower():
                payload = entry.to_dict()
                payload["mode"] = entry.source
                payload["note"] = note
                return payload
        raise KeyError(repo_id)

    def preview(
        self,
        repo_id: str,
        *,
        config: str | None = None,
        split: str | None = None,
        offset: int = 0,
        length: int = 5,
    ) -> dict[str, Any]:
        """Preview raw rows via the Hub dataset viewer (live only)."""
        if not self.allow_network or not self.hub_status().get("online"):
            return {
                "available": False,
                "reason": "The Hugging Face Hub is not reachable from this machine, "
                "so row previews are unavailable.",
                "rows": [],
            }
        try:
            payload = self.client.first_rows(
                repo_id, config=config, split=split, offset=offset, length=length
            )
        except (HubUnavailable, HubError) as exc:
            return {"available": False, "reason": str(exc), "rows": []}
        features = payload.get("features") or []
        columns = [f.get("name") for f in features if isinstance(f, dict)]
        rows = []
        for row in payload.get("rows") or []:
            values = row.get("row") if isinstance(row, dict) else None
            rows.append(values if isinstance(values, dict) else {"value": values})
        return {
            "available": True,
            "columns": columns,
            "rows": rows,
            "config": (payload.get("config") or config),
            "split": payload.get("split") or split,
            "truncated": payload.get("truncated"),
            "num_rows_total": payload.get("num_rows_total"),
        }

    # ---------------------------------------------------------------- facets
    def facets(self) -> dict[str, Any]:
        """Filter chips for the UI, derived from the local catalog + cache."""
        entries = self.local_entries()
        langs: dict[str, int] = {}
        tasks: dict[str, int] = {}
        sizes: dict[str, int] = {}
        licenses: dict[str, int] = {}
        modalities: dict[str, int] = {}
        authors: dict[str, int] = {}
        for e in entries:
            for lang in e.languages or ["unknown"]:
                langs[lang] = langs.get(lang, 0) + 1
            for task in e.task_categories or ["uncategorised"]:
                tasks[task] = tasks.get(task, 0) + 1
            for size in e.size_categories:
                sizes[size] = sizes.get(size, 0) + 1
            for lic in e.licenses or ["unspecified"]:
                licenses[lic] = licenses.get(lic, 0) + 1
            for mod in e.modalities:
                modalities[mod] = modalities.get(mod, 0) + 1
            if e.author:
                authors[e.author] = authors.get(e.author, 0) + 1

        def top(d: dict[str, int], n: int = 40) -> list[dict[str, Any]]:
            return [
                {"value": k, "count": v}
                for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
            ]

        return {
            "count": len(entries),
            "languages": top(langs, 30),
            "tasks": top(tasks, 30),
            "sizes": [
                {"value": s, "count": sizes.get(s, 0)}
                for s in SIZE_CATEGORIES
                if sizes.get(s)
            ],
            "licenses": top(licenses, 20),
            "modalities": top(modalities, 12),
            "authors": top(authors, 25),
        }


def _score(entry: DatasetInfo, query: str | None) -> float:
    popularity = math.log10(max(entry.downloads or 0, 0) + 1) * 6.0
    likes = math.log10(max(entry.likes or 0, 0) + 1) * 2.0
    if not query:
        return popularity + likes

    q = query.strip().lower()
    repo_id = entry.id.lower()
    description = (entry.description or "").lower()
    tags = " ".join(entry.tags).lower()
    tasks = " ".join(entry.task_categories).lower()

    score = popularity + likes
    if q == repo_id:
        score += 1000
    if repo_id == q or repo_id.split("/")[-1] == q:
        score += 400
    if q in repo_id:
        score += 250
    if q in tasks:
        score += 80
    if q in description:
        score += 60
    if q in tags:
        score += 25

    tokens = [t for t in re.split(r"[\s/,]+", q) if t]
    for token in tokens:
        if token in repo_id:
            score += 90
        if token in tasks:
            score += 30
        if token in description:
            score += 18
        if token in tags:
            score += 10

    matched_tokens = sum(
        1
        for token in tokens
        if token in repo_id or token in description or token in tags or token in tasks
    )
    if tokens and matched_tokens == 0:
        return 0.0
    if tokens:
        score += 40 * (matched_tokens / len(tokens))
    return score
