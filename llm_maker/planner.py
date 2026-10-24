"""Turns a :class:`PlanRequest` into an executable :class:`TrainingPlan`.

The planner is the only place where decisions are made: which method to use,
how big the batches can be, how many steps that implies, and which warnings to
raise.  It never talks to the network itself -- it goes through the catalog.
"""

from __future__ import annotations

import math
from typing import Any

from .catalog import DatasetCatalog
from .estimator import estimate, max_micro_batch
from .models import (
    DatasetRef,
    Goal,
    HardwareSpec,
    HyperParams,
    ModelSpec,
    PlanRequest,
    TrainingPlan,
)
from .recipes import (
    GOAL_PRESETS,
    TARGET_GLOBAL_TOKENS,
    choose_method,
    dataset_ref_from_info,
    default_hyperparams,
    merge_hyperparams,
    gpu_from_preset,
    missing_columns,
    model_from_preset,
)
from .references import references_for

# Cap on gradient accumulation. Beyond this the step time is dominated by the
# sequential micro-batches and the plan should use more GPUs instead.
MAX_GRAD_ACCUM = 64

# Sensible upper bounds on the token budget for each goal, applied only when the
# caller left `epochs` and `max_steps` unset. Beyond these sizes the goal itself
# is usually the wrong one -- a 240B-token web corpus is not an instruction set --
# so the planner clamps, warns, and says how to override.
TOKEN_BUDGET_CAP: dict[str, int] = {
    "sft": 1_000_000_000,
    "dpo": 200_000_000,
    "cpt": 100_000_000_000,
}


def plan(
    request: PlanRequest,
    *,
    catalog: DatasetCatalog | None = None,
    resolve_datasets: bool = True,
    zero_stage: int | None = None,
    sequence_parallel: bool = False,
) -> TrainingPlan:
    """Build a training plan, resolving datasets from the catalog when asked."""
    catalog = catalog or DatasetCatalog()
    warnings: list[str] = []
    goal: Goal = request.goal if request.goal in GOAL_PRESETS else "sft"
    zero_stage_override = zero_stage

    # ------------------------------------------------------------- model shape
    model = request.model or ModelSpec()
    base_model = (model.base_model or "").strip()
    if model.est_params() <= 0:
        # No usable architecture was given: take the preset shape, keeping any
        # checkpoint the caller named explicitly.
        model = model_from_preset(model.preset or "7b", base_model=base_model or None)
    elif not base_model:
        # Architecture is known but the checkpoint is not: use the preset's.
        model = model_from_preset(model.preset or "7b")
    request.model = model  # so the reported plan shows what was actually planned

    gpu = gpu_from_preset(request.hardware.gpu)
    gpu_count = max(1, request.hardware.count)
    nodes = max(1, request.hardware.nodes)
    gpu_mem = request.hardware.gpu_mem_gb or float(gpu["mem_gb"])
    peak = request.hardware.peak_tflops_bf16 or float(gpu["peak_tflops_bf16"])
    # Resolve the placeholder zeros so downstream consumers see real numbers.
    request.hardware.gpu_mem_gb = gpu_mem
    request.hardware.peak_tflops_bf16 = peak
    request.hardware.gpu = gpu["key"]

    # Which training method fits is decided by the same estimator that reports
    # the final VRAM number, so the two can never disagree.
    method, zero_stage, method_reason = decide_method(
        goal=goal,
        model=model,
        hyper=request.hyper,
        gpu_mem_gb=gpu_mem,
        gpu_count=gpu_count * nodes,
        peak_tflops_bf16=peak,
        requested_method=request.method,
        requested_zero_stage=zero_stage_override,
        gpu_label=gpu["label"],
    )

    # --------------------------------------------------------------- datasets
    datasets = list(request.datasets)
    if resolve_datasets:
        datasets = [_resolve(ref, catalog, warnings) for ref in datasets]

    if not datasets:
        raise ValueError(
            "At least one dataset is required. Search the Hugging Face datasets list "
            "with `llm-maker search <query>` or pick one from the catalog endpoint."
        )

    _check_dataset_roles(goal, datasets, warnings)

    dataset_tokens = sum(int(ref.est_tokens or 0) for ref in datasets)
    if dataset_tokens <= 0:
        dataset_tokens = _fallback_tokens(datasets, goal)
        warnings.append(
            "The Hub did not report row counts for every dataset, so the token budget is a "
            "rough estimate. The generated script prints the real number at launch."
        )

    # ------------------------------------------------------------ hyperparams
    # Remember what the caller actually asked for before defaults are merged in,
    # so the planner can be opinionated about the things nobody specified.
    requested_epochs = request.hyper.epochs if request.hyper else 0.0
    requested_steps = request.hyper.max_steps if request.hyper else None
    defaults = default_hyperparams(goal, method, model)
    hyper = merge_hyperparams(request.hyper, defaults)
    request.hyper = hyper
    hyper.seq_len = min(hyper.seq_len, model.max_seq_len)
    for field_name in ("micro_batch_size", "grad_accum", "seq_len", "epochs"):
        if getattr(hyper, field_name) < 0:
            raise ValueError(f"hyper.{field_name} cannot be negative")
    if hyper.seq_len < 1:
        raise ValueError("hyper.seq_len must be at least 1")
    if hyper.trainer not in ("optimized", "standard"):
        raise ValueError("hyper.trainer must be 'optimized' or 'standard'")
    if hyper.compile_mode not in ("default", "reduce-overhead", "max-autotune"):
        raise ValueError("hyper.compile_mode must be default, reduce-overhead or max-autotune")
    if hyper.torch_compile and hyper.trainer != "optimized":
        warnings.append("torch.compile is an optimized-trainer feature; it will be ignored by the standard trainer.")
    request.hyper = hyper  # keep the resolved values on the plan's request

    if goal == "dpo":
        if hyper.dpo_beta <= 0:
            hyper.dpo_beta = float(GOAL_PRESETS["dpo"].get("beta", 0.1))
    if method in ("lora", "qlora") and hyper.learning_rate < 1e-5:
        warnings.append(
            f"Learning rate {hyper.learning_rate:g} is very low for {method.upper()}; adapters "
            "usually want 1e-4 to 2e-4 while full fine-tuning wants 1e-5 to 2e-5."
        )
    if goal == "dpo" and method in ("lora", "qlora") and hyper.learning_rate > 1e-4:
        warnings.append(
            f"Learning rate {hyper.learning_rate:g} is high for DPO: 5e-7 to 5e-6 is the usual "
            "range, because the reference policy amplifies large steps."
        )

    # ----------------------------------------------------- batch / step sizing
    world_size = gpu_count * nodes

    _size_micro_batch(
        goal=goal, method=method, model=model, hyper=hyper, gpu_count=world_size,
        gpu_mem=gpu_mem, peak=peak, zero_stage=zero_stage,
        sequence_parallel=sequence_parallel, gpu_label=gpu["label"],
    )
    _size_grad_accum(goal=goal, hyper=hyper, world_size=world_size, warnings=warnings)

    fits = _size_batches(
        goal=goal, method=method, model=model, hyper=hyper, gpu_count=world_size,
        gpu_mem=gpu_mem, peak=peak, zero_stage=zero_stage,
        sequence_parallel=sequence_parallel, warnings=warnings, gpu_label=gpu["label"],
    )

    global_batch = hyper.micro_batch_size * hyper.grad_accum * world_size
    global_batch_tokens = global_batch * hyper.seq_len

    # ------------------------------------------------------------------ steps
    if requested_steps:
        total_steps = int(requested_steps)
        train_tokens = total_steps * global_batch_tokens
        if train_tokens > dataset_tokens and dataset_tokens:
            warnings.append(
                f"{total_steps:,} steps x {global_batch_tokens:,} tokens/step is "
                f"{train_tokens / 1e9:.2f}B tokens but the selected datasets only hold about "
                f"{dataset_tokens / 1e9:.2f}B: the run will cycle over the data more than once."
            )
    else:
        train_tokens, budget_note = _train_token_budget(
            goal=goal,
            dataset_tokens=dataset_tokens,
            epochs=hyper.epochs,
            params=model.est_params(),
            plan_epochs_explicit=bool(requested_epochs),
        )
        if budget_note:
            warnings.append(budget_note)
        total_steps = max(1, math.ceil(train_tokens / max(global_batch_tokens, 1)))
        # Keep the requested epochs when nothing clamped the budget.
        if not budget_note and dataset_tokens:
            hyper.epochs = min(hyper.epochs, max(train_tokens / dataset_tokens, 1e-6))

    # The generated config pins max_steps, so a run always executes exactly the
    # step count this plan was estimated for.
    hyper.max_steps = total_steps
    hyper.warmup_steps = max(1, int(total_steps * hyper.warmup_ratio)) if total_steps > 1 else 0    # ------------------------------------------------------------ final check
    resource = estimate(
        goal=goal, method=method, model=model, hyper=hyper,
        gpu_count=world_size, gpu_mem_gb=gpu_mem, peak_tflops_bf16=peak,
        train_tokens=max(train_tokens, 1), zero_stage=zero_stage,
        cpu_offload=request.hardware.cpu_offload, sequence_parallel=sequence_parallel,
        gpu_label=gpu["label"],
    )
    warnings.extend(resource.warnings)
    resource.warnings = warnings
    resource.basis = resource.basis + [
        f"steps: {train_tokens:,} tokens / {global_batch_tokens:,} tokens per step",
        f"assumed GPU: {gpu['label']} ({peak:.0f} TFLOP/s bf16 dense)",
    ]

    plan_obj = TrainingPlan(
        request=request,
        method=method,
        dataset_tokens=dataset_tokens,
        train_tokens=train_tokens,
        total_steps=total_steps,
        warmup_steps=hyper.warmup_steps,
        global_batch_size=global_batch,
        global_batch_tokens=global_batch_tokens,
        resource=resource,
        mixture=_mixture_rows(datasets, dataset_tokens, goal),
        warnings=warnings,
        references=references_for(goal),
        catalog_mode=request.notes or "hub",
        goal_label=GOAL_PRESETS[goal]["label"],
        method_reason=method_reason,
        zero_stage=zero_stage,
    )
    if not fits:
        plan_obj.warnings.append("Batch configuration does not fit; see suggestions.")
    return plan_obj


# ----------------------------------------------------------------- the pieces
def _check_dataset_roles(goal: Goal, datasets: list[DatasetRef], warnings: list[str]) -> None:
    """Warn when the selected datasets do not match the goal.

    Nothing is blocked: training an instruction format on a web corpus is a real
    technique (continued pretraining on a chat corpus, for instance) but it is
    usually a mistake, and it is the kind of mistake that only shows up hours into
    a run.
    """
    roles: dict[str, int] = {}
    for ref in datasets:
        roles[(ref.task or "unknown").lower()] = roles.get((ref.task or "unknown").lower(), 0) + 1

    if goal == "sft" and roles.get("text"):
        if not (roles.get("sft") or roles.get("preference")):
            warnings.append(
                f"{roles['text']} of {len(datasets)} dataset(s) look like raw pretraining corpora. "
                "SFT trains on instruction/answer pairs, so either pick datasets tagged "
                "`instruction` or `chat`, or switch the goal to `cpt` to keep training the base "
                "model on raw text."
            )
    if goal == "dpo" and not roles.get("preference"):
        warnings.append(
            "DPO needs a dataset with chosen/rejected pairs; none of the selected datasets look "
            "like a preference set. Try `Anthropic/hh-rlhf`, `HuggingFaceH4/ultrafeedback_binarized` "
            "or `argilla/dpo-mix-7k`."
        )
    if goal == "sft" and roles.get("preference") and not roles.get("sft"):
        warnings.append(
            "These datasets hold preference pairs, which SFT can only use as plain text. "
            "If you want to align on preferred answers, plan a `dpo` run instead."
        )
    if goal == "pretrain" and (roles.get("sft") or roles.get("preference")):
        warnings.append(
            "Instruction or preference data is an unusual choice for pretraining: it will be "
            "trained as raw text with no prompt/answer structure."
        )


def _resolve(ref: DatasetRef, catalog: DatasetCatalog, warnings: list[str]) -> DatasetRef:
    """Fill in missing dataset metadata from the catalog."""
    try:
        payload = catalog.detail(ref.repo_id, allow_network=True)
    except KeyError:
        warnings.append(
            f"{ref.repo_id} is not in the Hub catalog and not in the local cache; it will be "
            "passed straight to `datasets.load_dataset` (typos surface there)."
        )
        ref.hub_url = ref.hub_url or f"https://huggingface.co/datasets/{ref.repo_id}"
        return ref

    if not ref.task:
        ref.task = _role_from_payload(payload)
    if not ref.est_tokens:
        ref.est_tokens = payload.get("est_tokens")
    if not ref.hub_url:
        ref.hub_url = payload.get("url") or ""
    if payload.get("configs") and not ref.config:
        # Prefer a config name that matches a language hint if there are many.
        ref.config = payload["configs"][0] if len(payload["configs"]) == 1 else None
    if payload.get("splits") and not ref.split:
        ref.split = payload["splits"][0].get("split") or "train"
    if payload.get("metrics_approximate") and payload.get("downloads"):
        ref.notes = ref.notes or "popularity figures in this plan come from the bundled snapshot"
    return ref


def _role_from_payload(payload: dict[str, Any]) -> str:
    tasks = {t.lower() for t in payload.get("task_categories", [])}
    if tasks & {"preference", "reward-modeling"}:
        return "preference"
    if tasks & {"instruction", "chat"}:
        return "sft"
    return "text"


def _fallback_tokens(datasets: list[DatasetRef], goal: Goal) -> int:
    """Very rough token budget when the Hub gave us nothing."""
    per_dataset = 2_000_000 if goal in ("sft", "dpo") else 20_000_000
    return per_dataset * max(len(datasets), 1)


METHOD_ORDER: tuple[Any, ...] = ("full", "lora", "qlora")
ZERO_STAGE_LABELS = {
    0: "no sharding (everything replicated)",
    1: "ZeRO-1 (optimizer state sharded)",
    2: "ZeRO-2 (optimizer state and gradients sharded)",
    3: "ZeRO-3 (weights, gradients and optimizer state sharded)",
}


def decide_method(
    *,
    goal: Goal,
    model: ModelSpec,
    hyper: HyperParams,
    gpu_mem_gb: float,
    gpu_count: int,
    peak_tflops_bf16: float,
    requested_method: str | None = None,
    requested_zero_stage: int | None = None,
    gpu_label: str = "GPU",
) -> tuple[Any, int, str]:
    """Choose the training method *and* the ZeRO stage by simulating each option.

    The estimator is the referee: the first (method, stage) pair that fits is the
    one that gets used, preferring cheaper methods (full > LoRA > QLoRA) and less
    sharding, in that order.  Returns ``(method, zero_stage, reason)``.
    """
    if goal == "pretrain":
        return (
            "from-scratch",
            3 if gpu_count > 1 else 0,
            "pretraining starts from random weights, so no adapter is involved",
        )

    probe = HyperParams(**{**hyper.__dict__})
    probe.micro_batch_size = 1
    if probe.seq_len <= 0:
        probe.seq_len = 2048

    def fits(method: str, stage: int) -> tuple[bool, float]:
        result = estimate(
            goal=goal, method=method, model=model, hyper=probe,
            gpu_count=gpu_count, gpu_mem_gb=gpu_mem_gb, peak_tflops_bf16=peak_tflops_bf16,
            train_tokens=1_000_000, zero_stage=stage, gpu_label=gpu_label,
        )
        return result.fits, result.vram_per_gpu_gb

    labels = {"full": "full fine-tuning", "lora": "bf16 LoRA", "qlora": "4-bit QLoRA"}

    # Sharding is only ever an option with more than one GPU.
    if requested_zero_stage is not None:
        stages = [requested_zero_stage]
    else:
        stages = [0, 1, 2, 3] if gpu_count > 1 else [0]

    if requested_method in labels:
        # An explicit request is honoured even if it does not fit: the user gets the
        # numbers and the suggestions instead of a silent substitution.
        results = [(stage, *fits(requested_method, stage)) for stage in stages]
        fitting = [r for r in results if r[1]]
        if fitting:
            stage, _, vram = fitting[0]
            return (
                requested_method,
                stage,
                f"you asked for {labels[requested_method]}: fits at {vram:.1f} GiB per GPU "
                f"with {ZERO_STAGE_LABELS[stage]} (limit {gpu_mem_gb:.0f} GiB of {gpu_label})",
            )
        stage, _, vram = min(results, key=lambda r: r[2])
        return (
            requested_method,
            stage,
            f"you asked for {labels[requested_method]}: it needs {vram:.1f} GiB per GPU even with "
            f"{ZERO_STAGE_LABELS[stage]}, more than the {gpu_mem_gb:.0f} GiB of {gpu_label}, so this "
            "configuration will not fit -- the suggestions below list the cheapest ways out",
        )

    # Nothing the planner is willing to pick fits. Take the cheapest combination
    # and let the estimator's suggestions explain the shortfall.
    first_failure: str | None = None
    best = None
    for method in METHOD_ORDER:
        for stage in stages:
            ok, vram = fits(method, stage)
            if ok:
                return (
                    method,
                    stage,
                    f"{labels[method]} fits: {vram:.1f} GiB per GPU with {ZERO_STAGE_LABELS[stage]}, "
                    f"inside the {gpu_mem_gb:.0f} GiB of {gpu_label}.",
                )
            if first_failure is None or vram < best[2]:
                first_failure = f"{labels[method]} at {ZERO_STAGE_LABELS[stage]} needs {vram:.1f} GiB"
                best = (method, stage, vram)

    method, stage, _ = best or ("qlora", 0, 0.0)
    return (
        method,
        stage,
        f"nothing fits as configured (the cheapest option, {first_failure}, is still over "
        f"{gpu_mem_gb:.0f} GiB per GPU) -- the suggestions below are the way out",
    )


def _size_micro_batch(
    *,
    goal: Goal,
    method: str,
    model: Any,
    hyper: Any,
    gpu_count: int,
    gpu_mem: float,
    peak: float,
    zero_stage: int,
    sequence_parallel: bool,
    gpu_label: str,
) -> None:
    """Pick a micro batch when the caller left it unset (``0``).

    The largest batch that fits is not always the right answer -- very large
    micro batches slow the step down and change the effective learning rate -- so
    it is capped at 8 for instruction tuning and 16 for corpus training.
    """
    if hyper.micro_batch_size > 0:
        return
    cap = 16 if goal in ("cpt", "pretrain") else 8
    probe = HyperParams(**{**hyper.__dict__, "micro_batch_size": 1})
    best = max_micro_batch(
        goal=goal, method=method, model=model, hyper=probe,
        gpu_count=gpu_count, gpu_mem_gb=gpu_mem, peak_tflops_bf16=peak,
    )
    hyper.micro_batch_size = max(1, min(best, cap))


def _size_grad_accum(*, goal: Goal, hyper: Any, world_size: int, warnings: list[str]) -> None:
    """Derive gradient accumulation from a target global batch in tokens."""
    if hyper.grad_accum > 0:
        return
    target_tokens = TARGET_GLOBAL_TOKENS.get(goal, 32_768)
    target_sequences = max(1, target_tokens // max(hyper.seq_len, 1))
    per_step = hyper.micro_batch_size * world_size
    wanted = max(1, round(target_sequences / max(per_step, 1)))
    hyper.grad_accum = min(wanted, MAX_GRAD_ACCUM)
    if wanted > MAX_GRAD_ACCUM:
        achieved = hyper.grad_accum * per_step * hyper.seq_len
        warnings.append(
            f"A global batch of {target_tokens:,} tokens would need gradient accumulation "
            f"{wanted}; it is capped at {MAX_GRAD_ACCUM}, so this plan uses "
            f"{achieved:,} tokens/step. Raise the micro batch, add GPUs, or use "
            "`--grad-accum` to accept the longer step."
        )


def _size_batches(
    *,
    goal: Goal,
    method: str,
    model: Any,
    hyper: Any,
    gpu_count: int,
    gpu_mem: float,
    peak: float,
    zero_stage: int,
    sequence_parallel: bool,
    warnings: list[str],
    gpu_label: str,
) -> bool:
    """Shrink the micro batch until the configuration fits, then re-balance."""
    kwargs = dict(
        goal=goal, method=method, model=model, gpu_count=gpu_count,
        gpu_mem_gb=gpu_mem, peak_tflops_bf16=peak,
    )
    est = estimate(hyper=hyper, train_tokens=1, zero_stage=zero_stage,
                   sequence_parallel=sequence_parallel, gpu_label=gpu_label, **kwargs)
    if est.fits:
        return True

    original = hyper.micro_batch_size
    best = max_micro_batch(hyper=hyper, **kwargs)
    if best < original:
        factor = math.ceil(original / max(best, 1))
        hyper.micro_batch_size = best
        hyper.grad_accum = max(hyper.grad_accum * factor, 1)
        warnings.append(
            f"Micro batch reduced {original} -> {best} to fit {gpu_mem:.0f}GB per GPU; gradient "
            f"accumulation raised to {hyper.grad_accum} so the global batch is unchanged."
        )
    est = estimate(hyper=hyper, train_tokens=1, zero_stage=zero_stage,
                   sequence_parallel=sequence_parallel, gpu_label=gpu_label, **kwargs)
    if not est.fits and method != "qlora":
        warnings.append(
            f"Even at micro batch 1 this does not fit in {gpu_mem:.0f}GB on {gpu_label}. "
            "Switching to QLoRA or adding GPUs is the realistic fix: "
            f"needs about {est.vram_per_gpu_gb:.1f}GB per GPU."
        )
    return est.fits


def _train_token_budget(
    *,
    goal: Goal,
    dataset_tokens: int,
    epochs: float,
    params: float,
    plan_epochs_explicit: bool,
) -> tuple[int, str | None]:
    """How many tokens this run should actually train on.

    Returns ``(tokens, warning)``. The budget is ``epochs x dataset_tokens``
    unless that exceeds what the goal sensibly supports, in which case it is
    clamped and the caller is told how to override it.
    """
    if dataset_tokens <= 0:
        return int(max(epochs, 1.0) * 2_000_000), None

    wanted = int(dataset_tokens * max(epochs, 1e-6))
    cap = TOKEN_BUDGET_CAP.get(goal)
    if goal == "pretrain":
        # Chinchilla compute-optimal budget: ~20 tokens per parameter.
        cap = int(20 * params)
    if cap is None or wanted <= cap:
        return wanted, None

    if plan_epochs_explicit:
        return wanted, (
            f"Training on {wanted / 1e9:.1f}B tokens, which is far more than a typical "
            f"{goal} run ({cap / 1e9:.1f}B). You set the epochs explicitly, so this is what "
            "was planned -- double-check that the datasets match the goal."
        )
    return cap, (
        f"Token budget clamped from {wanted / 1e9:.1f}B to {cap / 1e9:.1f}B "
        f"({GOAL_PRESETS[goal]['label']}{' is compute-optimal at ~20 tokens/parameter' if goal == 'pretrain' else ''}). "
        "Set `--epochs` or `--max-steps` explicitly to train longer."
    )


def _mixture_rows(
    datasets: list[DatasetRef], total_tokens: int, goal: Goal
) -> list[dict[str, Any]]:
    weight_sum = sum(max(d.weight, 0.0) for d in datasets) or 1.0
    rows = []
    for ref in datasets:
        share = max(ref.weight, 0.0) / weight_sum
        rows.append(
            {
                "repo_id": ref.repo_id,
                "split": ref.split,
                "config": ref.config,
                "task": ref.task,
                "weight": round(share, 4),
                "est_tokens": ref.est_tokens,
                "planned_tokens": int(total_tokens * share) if total_tokens else None,
                "hub_url": ref.hub_url or f"https://huggingface.co/datasets/{ref.repo_id}",
                "fields": {
                    k: getattr(ref, k)
                    for k in (
                        "text_field", "prompt_field", "completion_field",
                        "messages_field", "chosen_field", "rejected_field",
                    )
                    if getattr(ref, k)
                },
                "needs": missing_columns(ref, goal),
            }
        )
    return rows


def default_request(
    *,
    goal: Goal = "sft",
    preset: str = "7b",
    gpu: str = "a100-80gb",
    gpu_count: int = 1,
    datasets: list[str] | None = None,
) -> PlanRequest:
    """A sensible request, useful for the CLI and for tests."""
    model = model_from_preset(preset)
    gpu_info = gpu_from_preset(gpu)
    method, _ = choose_method(goal, model.params_b, gpu_info["mem_gb"], gpu_count, None)
    return PlanRequest(
        goal=goal,
        model=model,
        datasets=[DatasetRef(repo_id=d) for d in (datasets or [])],
        hardware=HardwareSpec(
            gpu=gpu_info["key"],
            count=gpu_count,
            gpu_mem_gb=gpu_info["mem_gb"],
            peak_tflops_bf16=gpu_info["peak_tflops_bf16"],
        ),
        hyper=default_hyperparams(goal, method, model),
    )


def dataset_ref_from_catalog(catalog: DatasetCatalog, repo_id: str) -> DatasetRef:
    """Convenience wrapper used by the API when a user picks a catalog entry."""
    payload = catalog.detail(repo_id)
    from .hub import DatasetInfo

    info = DatasetInfo.from_hub(payload, source=payload.get("mode", "hub"))
    if not info.description:
        info.description = payload.get("description")
    ref = dataset_ref_from_info(info)
    if payload.get("est_tokens"):
        ref.est_tokens = payload["est_tokens"]
    return ref
