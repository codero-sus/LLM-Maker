"""Command line interface.

    llm-maker search "code instructions" --task code --limit 5
    llm-maker show tatsu-lab/alpaca
    llm-maker plan --goal sft --preset 7b --gpu a100-80gb --datasets tatsu-lab/alpaca
    llm-maker make --goal sft --datasets tatsu-lab/alpaca,databricks/datasets --out runs
    llm-maker serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .catalog import DatasetCatalog, SearchFilters
from .codegen import render_project
from .models import DatasetRef, PlanRequest
from .planner import plan as build_plan
from .recipes import GOAL_PRESETS, GPU_PRESETS, MODEL_PRESETS
from .scaffold import write_project


def _catalog(args: argparse.Namespace) -> DatasetCatalog:
    return DatasetCatalog(allow_network=not getattr(args, "offline", False))


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


# ------------------------------------------------------------------ commands
def cmd_search(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    filters = SearchFilters(
        search=args.query,
        author=args.author,
        languages=_csv(args.language),
        tasks=_csv(args.task),
        sizes=_csv(args.size),
        licenses=_csv(args.license),
        min_downloads=args.min_downloads,
        include_gated=not args.no_gated,
    )
    result = catalog.search(filters, sort=args.sort, limit=args.limit)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"{result.mode} · {result.total_matched} local match(es) · {result.elapsed_ms} ms")
    if result.note:
        print(f"note: {result.note}")
    print()
    for item in result.items:
        flags = []
        if item.gated:
            flags.append("gated")
        if item.source != "hub":
            flags.append(item.source)
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{item.id}{suffix}")
        print(
            f"    ↓ {item.downloads or 0:,}/mo   ♥ {item.likes or 0:,}   "
            f"size {item.best_size_category() or '?'}   "
            f"~{(item.to_dict().get('est_tokens') or 0) / 1e9:.2f}B tokens   "
            f"tasks: {', '.join(item.task_categories) or 'unlabelled'}"
        )
        if item.description:
            print(f"    {item.description[:150]}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    try:
        payload = catalog.detail(args.repo_id)
    except KeyError:
        print(f"{args.repo_id} is not in the catalog or the local cache.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{payload['id']}  ({payload.get('mode') or payload.get('source')})")
    print(f"  url         : {payload.get('url')}")
    print(f"  description : {(payload.get('description') or '')[:300]}")
    print(f"  downloads   : {payload.get('downloads'):,} / month")
    print(f"  likes       : {payload.get('likes'):,}")
    print(f"  licence     : {payload.get('license')}   gated: {payload.get('gated')}")
    print(f"  tasks       : {', '.join(payload.get('task_categories') or []) or 'unlabelled'}")
    print(f"  languages   : {', '.join(payload.get('languages') or []) or 'unspecified'}")
    print(f"  size bucket : {payload.get('size_bucket')}")
    print(f"  est tokens  : {(payload.get('est_tokens') or 0) / 1e9:.3f}B")
    if payload.get("num_examples"):
        print(f"  rows        : {payload['num_examples']:,}")
    if payload.get("configs"):
        print(f"  configs     : {', '.join(payload['configs'][:12])}")
    if payload.get("splits"):
        print("  splits:")
        for split in payload["splits"][:12]:
            print(
                f"    {split.get('config') or '-'} / {split.get('split') or '-'}: "
                f"{split.get('num_examples') or 0:,} rows"
            )
    if args.preview:
        preview = catalog.preview(args.repo_id, length=args.preview)
        if preview.get("available"):
            print("\n  first rows:")
            for row in preview["rows"]:
                for key, value in row.items():
                    text = json.dumps(value)[:180] if not isinstance(value, str) else value[:180]
                    print(f"    {key}: {text}")
                print("    ---")
        else:
            print(f"\n  preview unavailable: {preview.get('reason')}")
    print(f"\n  plan with: llm-maker plan --datasets {payload['id']} --goal sft")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    request = args_to_request(args)
    plan = build_plan(request, catalog=catalog)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    print_plan(plan)
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    request = args_to_request(args)
    plan = build_plan(request, catalog=catalog)
    result = write_project(plan, args.out, name=request.project_name, overwrite=args.force)

    print_plan(plan)
    print(f"\nproject written to {result.project_dir}")
    for name in result.written:
        print(f"  + {name}")
    if result.skipped:
        print("  (kept existing: " + ", ".join(result.skipped) + ")")
    directory = result.project_dir
    print("\nnext steps:")
    print(f"  cd {directory}")
    print("  python data_audit.py        # inspect the real rows and token counts")
    print("  python train.py --dry-run   # verify the pipeline")
    print("  bash run.sh                 # train (accelerate launch)" if args.gpus > 1 else "  python train.py             # train")
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    """Print the generated files to stdout instead of writing them."""
    catalog = _catalog(args)
    request = args_to_request(args)
    plan = build_plan(request, catalog=catalog)
    files = render_project(plan)
    for name, content in files.items():
        print(f"{'=' * 78}\n# {name}\n{'=' * 78}")
        print(content)
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    payload = {
        "goals": {key: value for key, value in GOAL_PRESETS.items()},
        "models": {key: value for key, value in MODEL_PRESETS.items()},
        "gpus": {key: value for key, value in GPU_PRESETS.items()},
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("goals:")
    for key, value in GOAL_PRESETS.items():
        print(f"  {key:9s} {value['label']} -- {value['blurb']}")
        print(f"            needs {value['need_columns']}; default method {value['default_method']}")
    print("\nmodel presets:")
    for key, value in MODEL_PRESETS.items():
        print(f"  {key:6s} {value['label']}")
    print("\nGPUs (dense bf16 TFLOP/s, no sparsity):")
    for key, value in GPU_PRESETS.items():
        print(f"  {key:12s} {value['mem_gb']:>3}GB  {value['peak_tflops_bf16']:>6.0f} TFLOP/s  {value['note']}")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    status = catalog.hub_status(force=True)
    facets = catalog.facets()
    if args.json:
        print(json.dumps({"hub": status, "facets": facets}, indent=2))
        return 0
    print(f"endpoint : {status.get('endpoint')}")
    print(f"online   : {status.get('online')}" + (f"  ({status.get('error')})" if status.get("error") else ""))
    if status.get("authenticated"):
        print("auth     : HF_TOKEN is set (gated datasets are visible)")
    print(f"locally known datasets: {facets['count']}")
    print(f"cache file: {catalog.snapshot_path}")
    for key in ("tasks", "languages", "sizes", "licenses"):
        values = ", ".join(f"{v['value']} ({v['count']})" for v in facets[key][:12])
        print(f"  {key:10s}: {values}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("LLM_MAKER_OUT", str(args.out))
    if args.offline:
        os.environ["LLM_MAKER_OFFLINE"] = "1"
    import uvicorn

    print(f"LLM Maker {__version__} -- open http://localhost:{args.port} (docs at /docs)")
    uvicorn.run("llm_maker.server:app", host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ----------------------------------------------------------------- plumbing
def _zero_stage_of(plan: Any) -> int:
    return int(getattr(plan, "zero_stage", 0) or 0)

def args_to_request(args: argparse.Namespace) -> PlanRequest:
    goal = args.goal if args.goal in GOAL_PRESETS else "sft"
    datasets = [
        DatasetRef(repo_id=repo_id.strip())
        for repo_id in _csv(args.datasets)
    ]
    request = PlanRequest.from_dict(
        {
            "goal": goal,
            "method": getattr(args, "method", None),
            "model": {"preset": args.preset, "base_model": getattr(args, "base_model", None) or ""},
            "datasets": [d.to_dict() for d in datasets],
            "hardware": {"gpu": args.gpu, "count": args.gpus, "nodes": getattr(args, "nodes", 1)},
            "hyper": {},
            "project_name": args.name,
            # --out is where the *project* goes; checkpoints land inside it.
            "output_root": "runs",
            "mixture_strategy": getattr(args, "mixture", "weighted"),
            "track": getattr(args, "track", "none"),
        }
    )
    # 0 means "let the planner decide" for the numeric overrides.
    if getattr(args, "lr", 0):
        request.hyper.learning_rate = args.lr
    if getattr(args, "epochs", 0):
        request.hyper.epochs = args.epochs
    if getattr(args, "seq_len", 0):
        request.hyper.seq_len = args.seq_len
    if getattr(args, "micro_batch", 0):
        request.hyper.micro_batch_size = args.micro_batch
    if getattr(args, "grad_accum", 0):
        request.hyper.grad_accum = args.grad_accum
    if getattr(args, "lora_r", 0):
        request.hyper.lora_r = args.lora_r
        request.hyper.lora_alpha = args.lora_r * 2
    if getattr(args, "dpo_beta", 0):
        request.hyper.dpo_beta = args.dpo_beta
    if getattr(args, "mask_prompt", False):
        request.hyper.mask_prompt = True
    request.hyper.trainer = getattr(args, "trainer", "optimized")
    request.hyper.torch_compile = bool(getattr(args, "torch_compile", False))
    request.hyper.compile_mode = getattr(args, "compile_mode", "reduce-overhead")
    request.hyper.tf32 = not bool(getattr(args, "no_tf32", False))
    return request


def print_plan(plan: Any) -> None:
    request = plan.request
    print(f"goal        : {plan.goal_label} ({request.goal})")
    print(f"method      : {plan.method} -- {plan.method_reason}")
    print(f"base model  : {request.model.base_model or '(from scratch)'} "
          f"({request.model.params_b:.2f}B, preset {request.model.preset})")
    print(f"hardware    : {request.hardware.count}x {request.hardware.gpu} "
          f"({request.hardware.gpu_mem_gb:.0f}GB each)")
    print()
    print("datasets:")
    for row in plan.mixture:
        tokens = f"{row['est_tokens'] / 1e9:.3f}B" if row.get("est_tokens") else "unknown"
        print(f"  - {row['repo_id']}  split={row.get('split') or 'train'}  "
              f"weight={row['weight']:.2f}  ~{tokens} tokens  role={row.get('task') or 'auto'}")
        if row.get("fields"):
            print(f"      fields: {', '.join(f'{k}={v}' for k, v in row['fields'].items())}")
        if row.get("needs"):
            print(f"      needs at runtime: {', '.join(row['needs'])}")
    print()
    print(f"tokens      : {plan.dataset_tokens / 1e9:.3f}B available, "
          f"{plan.train_tokens / 1e9:.3f}B planned")
    print(f"zero stage  : ZeRO-{_zero_stage_of(plan)}")
    print(f"batch       : {plan.global_batch_size} sequences = "
          f"{plan.global_batch_tokens:,} tokens/step "
          f"(micro {request.hyper.micro_batch_size} x accum {request.hyper.grad_accum} "
          f"x {request.hardware.count * request.hardware.nodes} GPUs)")
    print(f"steps       : {plan.total_steps:,} ({plan.warmup_steps:,} warmup), "
          f"lr {request.hyper.learning_rate:g}, seq_len {request.hyper.seq_len:,}")
    resource = plan.resource
    print()
    print(f"VRAM per GPU: {resource.vram_per_gpu_gb:.1f} GiB "
          f"(model {resource.vram_breakdown_gb.get('model_state', 0):.1f} + "
          f"activations {resource.vram_breakdown_gb.get('activations', 0):.1f} + "
          f"overhead {resource.vram_breakdown_gb.get('framework_overhead', 0):.1f}) "
          f"-> {'fits' if resource.fits else 'DOES NOT FIT'} "
          f"(headroom {resource.headroom_gb:.1f} GiB)")
    print(f"throughput  : {resource.tokens_per_second_per_gpu:,.0f} tokens/s/GPU "
          f"at {resource.mfu_assumed * 100:.0f}% MFU")
    print(f"wall clock  : {resource.wall_clock_hours:.2f} h "
          f"({resource.total_gpu_hours:.2f} GPU-hours)")
    if plan.warnings:
        print("\nwarnings:")
        for warning in plan.warnings:
            print(f"  ! {warning}")
    if resource.suggestions:
        print("\nsuggested fixes:")
        for suggestion in resource.suggestions:
            print(f"  - {suggestion['action']}: {suggestion['detail']}  [{suggestion['flag']}]")
    print("\nestimator basis:")
    for line in resource.basis:
        print(f"  · {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-maker",
        description="Plan a language model training run on Hugging Face datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  llm-maker search 'code instructions' --task code --limit 5\n"
            "  llm-maker plan --goal sft --preset 7b --gpu a100-80gb --datasets tatsu-lab/alpaca\n"
            "  llm-maker make --goal sft --datasets tatsu-lab/alpaca,HuggingFaceH4/ultrachat_200k --out ./runs\n"
            "  llm-maker serve --port 8000\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"llm-maker {__version__}")
    parser.add_argument("--offline", action="store_true", help="never touch the network; use the bundled snapshot")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search the Hub datasets list")
    search.add_argument("query", nargs="?", default=None)
    search.add_argument("--author")
    search.add_argument("--language", help="comma separated, e.g. en,hi")
    search.add_argument("--task", help="comma separated task tags, e.g. instruction,code")
    search.add_argument(
        "--size",
        help="size categories, comma separated, e.g. 1K<n<10K,100K<n<1M "
             "(see `llm-maker catalog` for the exact strings)",
    )
    search.add_argument("--license")
    search.add_argument("--min-downloads", type=int)
    search.add_argument("--no-gated", action="store_true", help="hide gated datasets")
    search.add_argument("--sort", default="downloads", choices=["downloads", "likes", "lastModified", "trending"])
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    show = sub.add_parser("show", help="show one dataset in detail")
    show.add_argument("repo_id")
    show.add_argument("--preview", type=int, default=0, help="also print N rows from the dataset viewer")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    def add_plan_args(sub_parser: argparse.ArgumentParser) -> None:
        sub_parser.add_argument("--goal", default="sft", choices=list(GOAL_PRESETS))
        sub_parser.add_argument("--method", default=None, choices=["lora", "qlora", "full", "from-scratch"])
        sub_parser.add_argument("--datasets", required=True, help="comma separated Hub repo ids")
        sub_parser.add_argument("--preset", default="7b", choices=list(MODEL_PRESETS))
        sub_parser.add_argument("--base-model", default=None, help="override the base checkpoint")
        sub_parser.add_argument("--gpu", default="a100-80gb", choices=list(GPU_PRESETS))
        sub_parser.add_argument("--gpus", type=int, default=1)
        sub_parser.add_argument("--nodes", type=int, default=1)
        sub_parser.add_argument("--name", default="my-llm")
        sub_parser.add_argument("--out", default="runs")
        sub_parser.add_argument("--mixture", default="weighted", choices=["weighted", "concat"])
        sub_parser.add_argument("--track", default="none", choices=["none", "tensorboard", "wandb", "mlflow"])
        sub_parser.add_argument("--lr", type=float, default=0.0, help="0 = planner default")
        sub_parser.add_argument("--epochs", type=float, default=0.0)
        sub_parser.add_argument("--seq-len", type=int, default=0)
        sub_parser.add_argument("--micro-batch", type=int, default=0)
        sub_parser.add_argument("--grad-accum", type=int, default=0)
        sub_parser.add_argument("--lora-r", type=int, default=0)
        sub_parser.add_argument("--dpo-beta", type=float, default=0.0)
        sub_parser.add_argument("--mask-prompt", action="store_true", help="train only on completions")
        sub_parser.add_argument(
            "--trainer", default="optimized", choices=["optimized", "standard"],
            help="generated runtime profile: optimized uses chunked loss and safe CUDA fast paths",
        )
        sub_parser.add_argument(
            "--torch-compile", action="store_true",
            help="opt in to torch.compile in the optimized trainer (single-GPU full/LoRA SFT/CPT)",
        )
        sub_parser.add_argument(
            "--compile-mode", default="reduce-overhead",
            choices=["default", "reduce-overhead", "max-autotune"],
            help="torch.compile mode when --torch-compile is set",
        )
        sub_parser.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul acceleration")
        sub_parser.add_argument("--json", action="store_true")

    plan_parser = sub.add_parser("plan", help="print a training plan and its estimates")
    add_plan_args(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)

    make = sub.add_parser("make", help="generate a project on disk")
    add_plan_args(make)
    make.add_argument("--force", action="store_true", help="overwrite existing files")
    make.set_defaults(func=cmd_make)

    recipes = sub.add_parser("recipes", help="print generated files to stdout")
    add_plan_args(recipes)
    recipes.set_defaults(func=cmd_recipes)

    presets = sub.add_parser("presets", help="list goals, model shapes and GPUs")
    presets.add_argument("--json", action="store_true")
    presets.set_defaults(func=cmd_presets)

    catalog = sub.add_parser("catalog", help="show Hub connectivity and local catalog stats")
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(func=cmd_catalog)

    serve = sub.add_parser("serve", help="run the web UI and JSON API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--out", default="generated", help="where generated projects are written")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should print, not traceback
        if getattr(args, "json", False):
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def console_entry() -> None:  # pragma: no cover - packaging shim
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
