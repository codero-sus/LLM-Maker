"""HTTP API + web UI.

Run it with ``llm-maker serve`` or ``python -m llm_maker.server``.  Every response
that depends on the Hub reports which of the three catalog modes it used
(``hub`` / ``cache`` / ``snapshot``) so an offline session is never mistaken for
live data.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import __version__
from .catalog import DatasetCatalog, SearchFilters
from .codegen import render_project, safe_dirname
from .models import PlanRequest, TrainingPlan
from .planner import dataset_ref_from_catalog
from .planner import plan as build_plan
from .recipes import GOAL_PRESETS, GPU_PRESETS, MODEL_PRESETS
from .references import all_references
from .scaffold import clean_project, write_project

WEB_DIR = Path(__file__).parent / "web"
OUTPUT_ROOT = Path(os.environ.get("LLM_MAKER_OUT", "generated"))

app = FastAPI(
    title="LLM Maker",
    version=__version__,
    description=(
        "Browse the Hugging Face datasets list, plan a training run and generate a "
        "runnable `datasets` + `transformers` project. LLM Maker is a planner, not a trainer."
    ),
)

_catalog = DatasetCatalog()
_LAST_INDEX: dict[str, dict[str, Any]] = {}


def catalog() -> DatasetCatalog:
    return _catalog


# --------------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/favicon.svg")
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="7" fill="#0f172a"/>'
        '<path d="M8 22V10h3v9h5v3H8zm10 0V10h3v12h-3z" fill="#ffd166"/>'
        "</svg>"
    )
    return Response(svg, media_type="image/svg+xml")


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health() -> dict[str, Any]:
    status = catalog().hub_status()
    return {
        "status": "ok",
        "version": __version__,
        "hub": status,
        "catalog": {
            "local_entries": len(catalog().local_entries()),
            "cache_dir": str(catalog().cache_dir),
            "cache_file": str(catalog().snapshot_path),
        },
        "output_root": str(OUTPUT_ROOT.resolve()),
        "offline_mode": not catalog().allow_network,
    }


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    """Everything the UI needs to render its dropdowns."""
    return {
        "goals": [
            {"value": key, **value} for key, value in GOAL_PRESETS.items()
        ],
        "models": [
            {"value": key, **{k: v for k, v in shape.items()}}
            for key, shape in MODEL_PRESETS.items()
        ],
        "gpus": [
            {"value": key, **shape} for key, shape in GPU_PRESETS.items()
        ],
        "references": all_references(),
        "sort_options": [
            {"value": "downloads", "label": "Most downloads"},
            {"value": "likes", "label": "Most likes"},
            {"value": "trending", "label": "Trending"},
            {"value": "lastModified", "label": "Recently updated"},
        ],
    }


# ---------------------------------------------------------------- datasets
@app.get("/api/catalog")
def catalog_overview(refresh: bool = False) -> dict[str, Any]:
    """Facets for the filter chips, plus the state of the Hub connection."""
    hub = catalog().hub_status(force=refresh)
    facets = catalog().facets()
    return {
        "hub": hub,
        "mode": "hub" if hub.get("online") else "snapshot",
        "count": facets["count"],
        "facets": {k: v for k, v in facets.items() if k != "count"},
        "note": (
            "Filter counts come from the local catalog (bundled snapshot plus anything "
            "already cached from live Hub queries)."
        ),
    }


@app.get("/api/datasets")
def search_datasets(
    q: str | None = Query(None, description="free-text search passed to the Hub"),
    author: str | None = None,
    language: list[str] | None = Query(None),
    task: list[str] | None = Query(None),
    size: list[str] | None = Query(None),
    license: list[str] | None = Query(None),
    min_downloads: int | None = None,
    include_gated: bool = True,
    sort: str = "downloads",
    limit: int = Query(24, ge=1, le=100),
    cursor: str | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Search the Hub datasets list; falls back to the local catalog when offline."""
    filters = SearchFilters(
        search=q,
        author=author,
        languages=list(language or []),
        tasks=list(task or []),
        sizes=list(size or []),
        licenses=list(license or []),
        min_downloads=min_downloads,
        include_gated=include_gated,
    )
    result = catalog().search(
        filters, sort=sort, limit=limit, cursor=cursor, use_hub=False if offline else None
    )
    payload = result.to_dict()
    payload["hub"] = catalog().hub_status()
    return payload


@app.get("/api/datasets/{repo_id:path}/preview")
def dataset_preview(
    repo_id: str,
    config: str | None = None,
    split: str | None = None,
    offset: int = Query(0, ge=0),
    length: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    return catalog().preview(repo_id, config=config, split=split, offset=offset, length=length)


@app.get("/api/datasets/{repo_id:path}/ref")
def dataset_ref(repo_id: str) -> dict[str, Any]:
    """A plan-ready dataset reference with the field mapping pre-filled."""
    try:
        ref = dataset_ref_from_catalog(catalog(), repo_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{repo_id} is not in the catalog or cache")
    return ref.to_dict()


@app.get("/api/datasets/{repo_id:path}")
def dataset_detail(repo_id: str) -> dict[str, Any]:
    try:
        payload = catalog().detail(repo_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{repo_id} is not in the catalog or cache")
    payload["references"] = []
    return payload



# -------------------------------------------------------------------- plan
@app.post("/api/plan")
def create_plan(request: PlanRequest = Body(...)) -> dict[str, Any]:
    """Build a plan without writing anything to disk."""
    try:
        built = build_plan(request, catalog=catalog())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return built.to_dict()


@app.post("/api/recipes")
def create_recipe(
    request: PlanRequest = Body(...),
    write: bool = Query(True, description="write the project to disk"),
) -> dict[str, Any]:
    """Build a plan and generate the project files."""
    try:
        built = build_plan(request, catalog=catalog())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    slug = safe_dirname(request.project_name or "llm-project")
    payload: dict[str, Any] = {"plan": built.to_dict(), "slug": slug}

    if write:
        result = write_project(built, OUTPUT_ROOT, name=slug, overwrite=True)
        _LAST_INDEX[slug] = {
            "project_dir": str(result.project_dir),
            "written": result.written,
            "plan": built.to_dict(),
            "created_at": time.time(),
        }
        payload["project"] = result.to_dict()
    else:
        files = render_project(built)
        payload["files"] = [
            {"name": name, "bytes": len(content.encode("utf-8"))}
            for name, content in files.items()
        ]

    payload["files"] = [
        {"name": name, "bytes": len(content.encode("utf-8"))}
        for name, content in render_project(built).items()
    ]
    payload["commands"] = _commands_for(built, slug)
    return payload


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    return {
        "output_root": str(OUTPUT_ROOT.resolve()),
        "projects": [
            {"slug": slug, **{k: v for k, v in record.items() if k != "plan"}}
            for slug, record in sorted(_LAST_INDEX.items(), key=lambda kv: -kv[1]["created_at"])
        ],
    }


@app.get("/api/recipes/{slug}")
def get_recipe(slug: str) -> dict[str, Any]:
    record = _LAST_INDEX.get(safe_dirname(slug))
    if not record:
        raise HTTPException(status_code=404, detail=f"No generated project named {slug!r} this session")
    directory = Path(record["project_dir"])
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files.append(
                {"name": str(path.relative_to(directory)), "bytes": path.stat().st_size}
            )
    return {"slug": slug, "project_dir": str(directory), "files": files, "plan": record["plan"]}


@app.get("/api/recipes/{slug}/file/{name:path}")
def get_recipe_file(slug: str, name: str) -> Response:
    record = _LAST_INDEX.get(safe_dirname(slug))
    if not record:
        raise HTTPException(status_code=404, detail=f"No generated project named {slug!r} this session")
    directory = Path(record["project_dir"]).resolve()
    target = (directory / name).resolve()
    if directory not in target.parents and target != directory:
        raise HTTPException(status_code=400, detail="path escapes the project directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return Response(target.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@app.get("/api/recipes/{slug}/download")
def download_recipe(slug: str) -> Response:
    record = _LAST_INDEX.get(safe_dirname(slug))
    if not record:
        raise HTTPException(status_code=404, detail=f"No generated project named {slug!r} this session")
    directory = Path(record["project_dir"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_dirname(slug)}.zip"'},
    )


@app.delete("/api/recipes/{slug}")
def delete_recipe(slug: str) -> dict[str, Any]:
    record = _LAST_INDEX.pop(safe_dirname(slug), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"No generated project named {slug!r} this session")
    removed = clean_project(record["project_dir"])
    return {"removed": removed, "project_dir": record["project_dir"]}


# ------------------------------------------------------------------ helpers
def _commands_for(plan: TrainingPlan, slug: str) -> dict[str, str]:
    directory = OUTPUT_ROOT / slug
    return {
        "inspect": f"cd {directory} && python data_audit.py",
        "dry_run": f"cd {directory} && python train.py --dry-run",
        "smoke": f"cd {directory} && python train.py --smoke-test",
        "train": f"cd {directory} && bash run.sh",
        "merge": f"cd {directory} && python merge_adapter.py --adapter runs/{slug} --out merged/{slug}",
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}", "path": request.url.path},
    )


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(
        "llm_maker.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
