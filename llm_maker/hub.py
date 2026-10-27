"""Hugging Face Hub client used to browse the datasets list.

Everything the app knows about datasets comes from one of two places:

1. **The live Hub** -- ``https://huggingface.co/api/datasets`` (paged with
   ``Link: rel="next"`` cursors) plus the datasets-server for row previews.
2. **A bundled/cached snapshot** -- used automatically when the Hub cannot be
   reached, so the UI never turns into a blank page on an offline machine.

Nothing here requires ``datasets``/``huggingface_hub`` to be installed; the
client only needs ``httpx`` and talks plain HTTP.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal

import httpx

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DEFAULT_DATASETS_SERVER = os.environ.get(
    "HF_DATASETS_SERVER", "https://datasets-server.huggingface.co"
).rstrip("/")

# Official Hub size buckets, ordered smallest -> largest.
SIZE_CATEGORIES: tuple[str, ...] = (
    "n<1K",
    "1K<n<10K",
    "10K<n<100K",
    "100K<n<1M",
    "1M<n<10M",
    "10M<n<100M",
    "100M<n<1B",
    "1B<n<10B",
    "10B<n<100B",
    "100B<n<1T",
    "n>1T",
)

MODE = Literal["hub", "cache", "snapshot"]
LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


class HubError(RuntimeError):
    """The Hub answered, but with something we could not use."""


class HubUnavailable(HubError):
    """The Hub could not be reached (offline, DNS failure, timeout, 5xx)."""


@dataclass
class DatasetInfo:
    """A normalised view over one entry of the Hub datasets list."""

    id: str
    author: str | None = None
    description: str | None = None
    downloads: int | None = None   # monthly, as reported by the Hub
    likes: int | None = None
    last_modified: str | None = None
    tags: list[str] = field(default_factory=list)
    task_categories: list[str] = field(default_factory=list)
    size_categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    gated: bool | str = False
    private: bool = False
    disabled: bool = False
    url: str = ""
    source: str = "hub"           # "hub" | "cache" | "snapshot"
    metrics_approximate: bool = False
    splits: list[dict[str, Any]] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    num_examples: int | None = None
    num_bytes: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_hub(cls, payload: dict[str, Any], *, source: str = "hub") -> "DatasetInfo":
        card = payload.get("cardData") or {}
        dataset_info = card.get("dataset_info") or {}
        if isinstance(dataset_info, list):
            info_list = [i for i in dataset_info if isinstance(i, dict)]
        elif isinstance(dataset_info, dict):
            info_list = [dataset_info]
        else:
            info_list = []

        splits: list[dict[str, Any]] = []
        configs: list[str] = []
        num_examples = 0
        num_bytes = 0
        for info in info_list:
            cfg_name = info.get("config_name") or info.get("config") or None
            if cfg_name:
                configs.append(cfg_name)
            for split in info.get("splits") or []:
                if not isinstance(split, dict):
                    continue
                splits.append(
                    {
                        "config": cfg_name or split.get("config"),
                        "split": split.get("name") or split.get("split"),
                        "num_examples": split.get("num_examples"),
                        "num_bytes": split.get("num_bytes"),
                    }
                )
                if isinstance(split.get("num_examples"), int):
                    num_examples += split["num_examples"]
                if isinstance(split.get("num_bytes"), int):
                    num_bytes += split["num_bytes"]

        tags = [str(t) for t in payload.get("tags") or []]
        card_language = _as_list(card.get("language"))
        card_license = _as_list(card.get("license"))
        card_tasks = _as_list(card.get("task_categories"))
        card_sizes = _as_list(card.get("size_categories"))
        card_modality = _as_list(card.get("modalities")) + _as_list(card.get("modality"))

        repo_id = payload.get("id") or ""
        author = payload.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None)

        return cls(
            id=repo_id,
            author=author,
            description=payload.get("description") or card.get("pretty_name") or None,
            downloads=payload.get("downloads"),
            likes=payload.get("likes"),
            last_modified=payload.get("lastModified") or payload.get("last_modified"),
            tags=tags,
            task_categories=_dedupe(card_tasks + _from_tags(tags, "task_categories")),
            size_categories=_dedupe(card_sizes + _from_tags(tags, "size_categories")),
            languages=_dedupe(card_language + _from_tags(tags, "language")),
            licenses=_dedupe(card_license + _from_tags(tags, "license")),
            modalities=_dedupe(card_modality),
            gated=payload.get("gated", False) or False,
            private=bool(payload.get("private")),
            disabled=bool(payload.get("disabled")),
            url=f"{DEFAULT_ENDPOINT}/datasets/{repo_id}",
            source=source,
            splits=splits,
            configs=_dedupe(configs),
            num_examples=num_examples or None,
            num_bytes=num_bytes or None,
            raw={
                "configs": [c for c in (card.get("configs") or []) if isinstance(c, (str, dict))][:20],
                "downloadsAllTime": payload.get("downloadsAllTime"),
            },
        )

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "DatasetInfo":
        info = cls.from_hub(payload, source="snapshot")
        info.metrics_approximate = True
        info.url = f"{DEFAULT_ENDPOINT}/datasets/{info.id}"
        return info

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "description": self.description,
            "downloads": self.downloads,
            "likes": self.likes,
            "last_modified": self.last_modified,
            "tags": self.tags,
            "task_categories": self.task_categories,
            "size_categories": self.size_categories,
            "languages": self.languages[:12],
            "license": self.licenses[0] if self.licenses else None,
            "modalities": self.modalities,
            "gated": bool(self.gated),
            "private": self.private,
            "url": self.url,
            "source": self.source,
            "metrics_approximate": self.metrics_approximate,
            "configs": self.configs[:12],
            "splits": self.splits[:24],
            "num_examples": self.num_examples,
            "num_bytes": self.num_bytes,
            "size_bucket": self.best_size_category(),
            "est_tokens": estimate_tokens(self),
        }

    # ------------------------------------------------------------------ helpers
    def best_size_category(self) -> str | None:
        """Smallest bucket that contains every declared bucket (best-case hint)."""
        buckets = [b for b in self.size_categories if b in SIZE_CATEGORIES]
        if not buckets:
            return None
        return min(buckets, key=SIZE_CATEGORIES.index)

    def searchable_text(self) -> str:
        return " ".join(
            filter(
                None,
                [
                    self.id,
                    self.author or "",
                    (self.description or "")[:400],
                    " ".join(self.task_categories),
                    " ".join(self.languages[:20]),
                    " ".join(self.licenses),
                ],
            )
        ).lower()


# Token estimates, used only for planning numbers in the UI. The generated
# project always measures the real count with the tokenizer (`data_audit.py`),
# because these are coarse: Hub metadata gives row counts far more often than it
# gives byte sizes.
#
# ``_BUCKET_SHAPE`` maps a size category to (assumed rows, tokens per row) and is
# the last resort when the Hub reports neither `num_examples` nor `num_bytes`.
_BUCKET_SHAPE: dict[str, tuple[float, float]] = {
    "n<1K": (700, 400),
    "1K<n<10K": (5_000, 300),
    "10K<n<100K": (50_000, 300),
    "100K<n<1M": (400_000, 300),
    "1M<n<10M": (4_000_000, 300),
    "10M<n<100M": (40_000_000, 300),
    "100M<n<1B": (400_000_000, 300),
    "1B<n<10B": (4_000_000_000, 300),
    "10B<n<100B": (40_000_000_000, 300),
    "100B<n<1T": (300_000_000_000, 250),
    "n>1T": (2_000_000_000_000, 250),
}

# Typical tokens per row by dataset role. Multi-turn chat rows are long,
# classification rows are short, and a preference pair counts both responses.
TOKENS_PER_ROW_BY_ROLE: dict[str, float] = {
    "instruction": 140.0,
    "chat": 450.0,
    "preference": 520.0,
    "math": 220.0,
    "code": 380.0,
    "qa": 120.0,
    "summarization": 450.0,
    "translation": 90.0,
    "classification": 70.0,
    "pretrain": 900.0,
    "domain": 250.0,
    "audio": 120.0,
    "vision": 150.0,
    "default": 300.0,
}

# `datasets` reports roughly 4 UTF-8 bytes per token for English prose; 3.6 is a
# safer divisor once code, markup and non-Latin scripts are in the mix.
BYTES_PER_TOKEN = 3.6


def tokens_per_row(info: "DatasetInfo") -> float:
    """Assumed tokens per row for a dataset, from its declared tasks."""
    tasks = [t.lower() for t in info.task_categories]
    tags = " ".join(info.tags).lower()
    repo = info.id.lower()

    if any(t in tasks for t in ("preference", "reward-modeling")):
        return TOKENS_PER_ROW_BY_ROLE["preference"]
    if any(k in repo for k in ("fineweb", "c4", "pajama", "corpus", "wikipedia", "webtext", "pile", "oscar", "culturax", "dolma")):
        return TOKENS_PER_ROW_BY_ROLE["pretrain"]
    for key in ("chat", "instruction", "code", "math", "summarization", "translation", "classification", "qa"):
        if key in tasks or key in tags:
            return TOKENS_PER_ROW_BY_ROLE[key]
    if "pretrain" in tags or "monolingual" in tags:
        return TOKENS_PER_ROW_BY_ROLE["pretrain"]
    return TOKENS_PER_ROW_BY_ROLE["default"]


def estimate_tokens(info: "DatasetInfo") -> int | None:
    """Rough token count for a dataset, preferring real Hub metadata."""
    if info.num_bytes:
        return int(info.num_bytes / BYTES_PER_TOKEN)
    if info.num_examples:
        return int(info.num_examples * tokens_per_row(info))
    bucket = info.best_size_category()
    if bucket is None:
        return None
    samples, per_sample = _BUCKET_SHAPE[bucket]
    role_scale = tokens_per_row(info) / TOKENS_PER_ROW_BY_ROLE["default"]
    return int(samples * per_sample * min(max(role_scale, 0.4), 2.0))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                out.append(item["name"])
        return out
    return []


def _from_tags(tags: Iterable[str], prefix: str) -> list[str]:
    needle = f"{prefix}:"
    return [t.split(":", 1)[1] for t in tags if t.startswith(needle)]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        if v and v not in seen:
            seen[v] = None
    return list(seen)


class HubClient:
    """Thin, dependency-light client for the Hub datasets list."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        datasets_server: str = DEFAULT_DATASETS_SERVER,
        token: str | None = None,
        timeout: float = 8.0,
        user_agent: str = "llm-maker/0.1 (+https://github.com/codero-sus/LLM-Maker)",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.datasets_server = datasets_server.rstrip("/")
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.timeout = timeout
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HubClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        try:
            resp = self._client.get(url, params=params)
        except httpx.HTTPError as exc:  # DNS, TLS, connect, timeout
            raise HubUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code in (401, 403):
            raise HubError(
                f"Hub returned {resp.status_code} for {url}. "
                "Private or gated resources need HF_TOKEN set."
            )
        if resp.status_code == 404:
            raise HubError(f"Not found on the Hub: {url}")
        if resp.status_code >= 500:
            raise HubUnavailable(f"Hub returned {resp.status_code} for {url}")
        if resp.status_code >= 400:
            raise HubError(f"Hub returned {resp.status_code} for {url}")
        return resp

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self._get(url, params).json()

    def probe(self) -> dict[str, Any]:
        """Cheap reachability check; never raises."""
        try:
            resp = self._get(f"{self.endpoint}/api/datasets", {"limit": 1})
        except HubError as exc:
            return {"online": False, "endpoint": self.endpoint, "error": str(exc)}
        items = resp.json()
        sample = items[0]["id"] if isinstance(items, list) and items else None
        return {
            "online": True,
            "endpoint": self.endpoint,
            "authenticated": bool(self.token),
            "sample": sample,
        }

    # -------------------------------------------------------------------- reads
    def list_datasets(
        self,
        *,
        search: str | None = None,
        author: str | None = None,
        filters: Iterable[str] | None = None,
        sort: str = "downloads",
        direction: int = -1,
        limit: int = 30,
        cursor: str | None = None,
        full: bool = False,
    ) -> tuple[list[DatasetInfo], str | None]:
        """One page of the Hub datasets list.

        Returns ``(datasets, next_cursor)`` where ``next_cursor`` comes straight
        from the response's ``Link: rel="next"`` header (``None`` at the end).
        """
        params: list[tuple[str, str]] = [
            ("sort", sort),
            ("direction", str(direction)),
            ("limit", str(min(max(limit, 1), 100))),
        ]
        if search:
            params.append(("search", search))
        if author:
            params.append(("author", author))
        for f in filters or []:
            if f:
                params.append(("filter", f))
        if full:
            params.append(("full", "true"))
        if cursor:
            params.append(("cursor", cursor))

        resp = self._get(f"{self.endpoint}/api/datasets", params)
        payload = resp.json()
        if not isinstance(payload, list):
            raise HubError(f"Unexpected datasets list payload: {type(payload).__name__}")

        items = [DatasetInfo.from_hub(p, source="hub") for p in payload if isinstance(p, dict)]
        next_cursor = None
        link = resp.headers.get("link") or resp.headers.get("Link")
        if link:
            match = LINK_NEXT_RE.search(link)
            if match:
                next_cursor = _cursor_from_url(match.group(1))
        return items, next_cursor

    def iter_datasets(self, *, max_pages: int = 3, **kwargs: Any) -> Iterator[DatasetInfo]:
        cursor = kwargs.pop("cursor", None)
        for _ in range(max_pages):
            items, cursor = self.list_datasets(cursor=cursor, **kwargs)
            yield from items
            if not cursor:
                return

    def get_dataset(self, repo_id: str, *, full: bool = True) -> DatasetInfo:
        params = {"full": "true"} if full else None
        payload = self._get_json(f"{self.endpoint}/api/datasets/{repo_id}", params)
        if not isinstance(payload, dict):
            raise HubError(f"Unexpected dataset payload for {repo_id}")
        payload.setdefault("id", repo_id)
        return DatasetInfo.from_hub(payload, source="hub")

    def dataset_files(self, repo_id: str) -> list[dict[str, Any]]:
        """Files in a dataset repo (needed to spot parquet/csv/jsonl data)."""
        payload = self._get_json(f"{self.endpoint}/api/datasets/{repo_id}/tree/main")
        if not isinstance(payload, list):
            return []
        return [p for p in payload if isinstance(p, dict)]

    def splits(self, repo_id: str) -> list[dict[str, Any]]:
        """Split metadata from the datasets-server (num_examples per config/split)."""
        payload = self._get_json(f"{self.datasets_server}/splits", {"dataset": repo_id})
        splits = payload.get("splits") if isinstance(payload, dict) else None
        return [s for s in (splits or []) if isinstance(s, dict)]

    def first_rows(
        self,
        repo_id: str,
        *,
        config: str | None = None,
        split: str | None = None,
        offset: int = 0,
        length: int = 5,
    ) -> dict[str, Any]:
        """Preview rows straight from the Hub's dataset viewer API."""
        params: dict[str, Any] = {"dataset": repo_id, "offset": offset, "length": length}
        if config:
            params["config"] = config
        if split:
            params["split"] = split
        payload = self._get_json(f"{self.datasets_server}/first-rows", params)
        if not isinstance(payload, dict):
            raise HubError(f"Unexpected first-rows payload for {repo_id}")
        if payload.get("error"):
            raise HubError(str(payload["error"]))
        return payload


def _cursor_from_url(url: str) -> str | None:
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(url).query)
    cursor = (query.get("cursor") or [None])[0]
    return cursor
