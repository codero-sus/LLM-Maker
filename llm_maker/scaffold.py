"""Writes a generated project to disk."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .codegen import render_project, safe_dirname
from .models import TrainingPlan


@dataclass
class ScaffoldResult:
    project_dir: Path
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "project_dir": str(self.project_dir),
            "written": self.written,
            "skipped": self.skipped,
            "total_bytes": self.total_bytes,
        }


def write_project(
    plan: TrainingPlan,
    out_root: str | Path,
    *,
    name: str | None = None,
    prompts: list[str] | None = None,
    overwrite: bool = False,
) -> ScaffoldResult:
    """Write every generated file under ``out_root/name``.

    Existing files are left alone unless ``overwrite`` is set, so re-generating a
    plan never silently destroys edits you made to a previous project.
    """
    files = render_project(plan, prompts=prompts)
    directory = Path(out_root) / safe_dirname(name or plan.request.project_name)

    if directory.exists() and not overwrite:
        keep = any(directory.iterdir())
        if keep:
            directory = _unique_directory(directory)

    directory.mkdir(parents=True, exist_ok=True)
    result = ScaffoldResult(project_dir=directory)

    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            result.skipped.append(relative)
            continue
        path.write_text(content, encoding="utf-8")
        result.written.append(relative)
        result.total_bytes += len(content.encode("utf-8"))

    # make the shell script executable
    run_script = directory / "run.sh"
    if run_script.exists():
        run_script.chmod(0o755)

    plan.written_to = str(directory)
    plan.generated_files = sorted(result.written)
    return result


def _unique_directory(directory: Path) -> Path:
    for suffix in range(2, 1000):
        candidate = directory.with_name(f"{directory.name}-{suffix}")
        if not candidate.exists():
            return candidate
    return directory


def clean_project(directory: str | Path) -> bool:
    """Remove a generated project directory."""
    path = Path(directory)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        return True
    return False
