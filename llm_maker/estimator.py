"""Analytic memory and throughput estimation.

The point of this module is to tell you *before* you rent the GPUs whether a
configuration fits, and roughly how long it will run.  The formulas are the
published ones, not curve-fits:

* **Model state** -- bf16 weights + optional fp32 master copy + gradients +
  Adam ``m``/``v``, following the byte accounting used by ZeRO
  (Rajbhandari et al., 2020, "ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models") and the "Rethinking" 2022 follow-up for
  per-parameter byte counts.
* **Activations** -- the per-layer formula from Korthikanti et al., 2022
  ("Reducing Activation Recomputation in Large Transformer Models"):
  ``s*b*h*(34 + 5*a*s/h)`` bytes per layer, which collapses to ``2*s*b*h``
  when every layer is checkpointed.
* **Quantised base weights** -- 4-bit NF4 doubles-up with double quantisation
  to about 0.5 bytes/param, per the QLoRA paper (Dettmers et al., 2023).
* **Compute** -- ``6*N`` FLOPs per token for forward+backward
  (Kaplan et al., 2020; Hoffmann et al., 2022) plus the ``12*L*d*s``
  attention term, divided by an assumed model-FLOPs-utilisation.

Every assumption is reported back to the caller in ``basis`` so the numbers can
be audited rather than trusted.
"""

from __future__ import annotations

import math
from typing import Any

from .models import Goal, HyperParams, Method, ModelSpec, ResourceEstimate

# ------------------------------------------------------------------ constants
BYTES = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "int8": 1.0, "nf4": 0.5}

# Per-parameter optimizer state bytes. Adam keeps two fp32 moments (8 bytes);
# its 8-bit variant keeps two int8 moments plus fp32 scale/offset (~2 bytes);
# Adafactor keeps only row and column statistics (~0.5 bytes).
OPTIMIZER_BYTES = {
    "adamw_torch": 8.0,
    "adamw_torch_fused": 8.0,
    "adamw_8bit": 2.0,
    "paged_adamw_8bit": 2.0,
    "paged_adamw_32bit": 8.0,
    "adafactor": 0.5,
    "sgd": 0.0,
}

# CUDA context, cuBLAS workspaces, NCCL buffers and fragmentation.
FIXED_OVERHEAD_GB = 0.9
NCCL_OVERHEAD_GB = 0.35
FRAGMENTATION = 0.08
PYTORCH_CUDNN_RESERVE_GB = 0.15

# Assumed achievable fraction of *dense* peak, by training mode. These sit at the
# cautious end of what well-tuned runs report, so the estimate errs towards
# "slower than reality". Treat the result as a range, not a promise: throughput
# swings with batch size, kernel availability and interconnect.
MFU_BY_METHOD = {
    "full": 0.40,
    "lora": 0.35,
    "qlora": 0.28,         # 4-bit dequantisation traffic on top of the step
    "from-scratch": 0.38,
}
MFU_NO_FLASH = 0.55     # multiplier when the attention kernel is not fused
DATALOADER_OVERHEAD = 1.15
RECOMPUTE_FLOP_FACTOR = 4.0 / 3.0  # forward is run twice when checkpointing

# Backward FLOPs per token, in units of N (parameters).
#   full fine-tuning   : 2N forward + 4N backward            = 6N
#   LoRA / QLoRA       : 2N forward + 2N backward for dL/dx   = 4N, plus the
#                        adapters, which are negligible next to N.
# This is the same accounting used by the LoRA paper's efficiency analysis.
FLOPS_PER_TOKEN_BY_METHOD = {"full": 6.0, "lora": 4.0, "qlora": 4.0, "from-scratch": 6.0}


def _state_bytes_per_param(
    method: Method, optimizer: str, *, master_weights: bool = True, fp8: bool = False
) -> dict[str, float]:
    """Bytes per trainable parameter for each memory bucket."""
    optim = OPTIMIZER_BYTES.get(optimizer, 8.0)
    if method in ("lora", "qlora"):
        # Base weights are frozen; only the adapters carry optimizer state.
        return {
            "base_weights": BYTES["nf4"] if method == "qlora" else BYTES["bf16"],
            "adapter_weights": BYTES["bf16"],
            "adapter_grads": BYTES["bf16"],
            "adapter_optimizer": optim,
        }
    weight_bytes = BYTES["bf16"]
    return {
        "weights": weight_bytes,
        "master_weights": BYTES["fp32"] if master_weights and not fp8 else 0.0,
        "grads": BYTES["bf16"],
        "optimizer": optim,
    }


def activation_bytes(
    model: ModelSpec, *, micro_batch: int, seq_len: int, checkpointing: bool
) -> tuple[float, str]:
    """Peak activation memory in bytes, plus a one-line explanation."""
    b, s = micro_batch, seq_len
    h, a, layers = model.d_model, max(model.n_head, 1), model.n_layer
    if checkpointing:
        per_layer = 2.0 * s * b * h
        note = (
            f"checkpointed activations: {layers} layers x 2*s*b*h "
            f"(=2x{seq_len}x{micro_batch}x{h})"
        )
    else:
        # Korthikanti et al. 2022, tensor-parallel degree 1.
        per_layer = s * b * h * (34.0 + 5.0 * a * s / h)
        note = (
            f"live activations: {layers} layers x s*b*h*(34 + 5*a*s/h) "
            f"(=s{b} b{micro_batch} h{h} and {a} heads)"
        )
    return per_layer * layers, note


def estimate(
    *,
    goal: Goal,
    method: Method,
    model: ModelSpec,
    hyper: HyperParams,
    gpu_count: int,
    gpu_mem_gb: float,
    peak_tflops_bf16: float,
    train_tokens: int,
    zero_stage: int = 2,
    cpu_offload: bool = False,
    sequence_parallel: bool = False,
    chunked_loss: bool = True,
    gpu_label: str = "GPU",
) -> ResourceEstimate:
    """Estimate VRAM per GPU and wall-clock time for a configuration."""
    n_gpus = max(1, gpu_count)
    params = model.est_params()
    adapter_params = _adapter_params(model, hyper) if method in ("lora", "qlora") else 0.0
    per_param = _state_bytes_per_param(method, hyper.optimizer, master_weights=True)
    warnings: list[str] = []
    basis: list[str] = []

    # ------------------------------------------------------------ model state
    if method in ("lora", "qlora"):
        shard = n_gpus if zero_stage >= 3 else 1
        base = per_param["base_weights"] * params / shard
        adapters = adapter_params * (
            per_param["adapter_weights"] + per_param["adapter_grads"] + per_param["adapter_optimizer"]
        )
        adapters /= n_gpus if zero_stage >= 2 else 1
        grad_bytes = 0.0  # base weights are frozen: no full-size gradients
        model_state = base + adapters
        basis.append(
            f"frozen base: {per_param['base_weights']:.1f} B/param x {params / 1e9:.2f}B params"
            + (" (4-bit NF4 base, double-quantised)" if method == "qlora" else " (bf16 base)")
        )
        basis.append(
            f"trainable adapters: {adapter_params / 1e6:.1f}M params x "
            f"{per_param['adapter_weights'] + per_param['adapter_grads'] + per_param['adapter_optimizer']:.0f} B/param"
        )
    else:
        weights = per_param["weights"] * params
        masters = per_param.get("master_weights", 0.0) * params
        grad_bytes = per_param["grads"] * params
        optim_bytes = per_param["optimizer"] * params
        shard_weights = n_gpus if zero_stage >= 3 else 1
        shard_grads = n_gpus if zero_stage >= 2 else 1
        shard_optim = n_gpus if zero_stage >= 1 else 1
        base = weights / shard_weights + masters / shard_weights
        grad_bytes /= shard_grads
        optim_bytes /= shard_optim
        model_state = base + grad_bytes + optim_bytes
        basis.append(
            f"model state: bf16 weights + fp32 master + grads + {'8-bit' if OPTIMIZER_BYTES.get(hyper.optimizer, 8) <= 2 else 'fp32'} "
            f"Adam, sharded with ZeRO-{zero_stage} across {n_gpus} GPU(s)"
        )
        if cpu_offload:
            model_state *= 0.5
            warnings.append(
                "CPU offload is enabled: the estimate assumes half the optimizer state lives in "
                "host RAM, which costs a lot of step time."
            )

    if sequence_parallel and n_gpus == 1:
        warnings.append("Sequence parallelism needs more than one GPU; ignoring it.")

    # ------------------------------------------------------------ activations
    act_bytes, act_note = activation_bytes(
        model,
        micro_batch=hyper.micro_batch_size,
        seq_len=hyper.seq_len,
        checkpointing=hyper.gradient_checkpointing,
    )
    if sequence_parallel and n_gpus > 1:
        act_bytes /= n_gpus
        act_note += f", divided across {n_gpus} GPUs with sequence parallelism"
    basis.append(act_note)

    # Logits are b*s*vocab wide: materialising them is a real cliff on long
    # sequences, so the generated script defaults to chunked cross-entropy.
    logits_bytes = 2.0 * hyper.micro_batch_size * hyper.seq_len * model.vocab_size
    if not chunked_loss:
        act_bytes += logits_bytes
        warnings.append(
            f"Without chunked loss the logits tensor alone is {logits_bytes / 1e9:.2f}GB at "
            f"seq_len={hyper.seq_len}; the generated script enables it by default."
        )

    overhead = (
        FIXED_OVERHEAD_GB
        + (NCCL_OVERHEAD_GB if n_gpus > 1 else 0.0)
        + PYTORCH_CUDNN_RESERVE_GB
    )
    subtotal = model_state + act_bytes
    total_bytes = subtotal * (1.0 + FRAGMENTATION) + overhead * (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    breakdown = {
        "model_state": round(model_state / 1024 ** 3, 2),
        "activations": round(act_bytes / 1024 ** 3, 2),
        "framework_overhead": round(overhead + subtotal * FRAGMENTATION / 1024 ** 3, 2),
    }

    # -------------------------------------------------------------- throughput
    mfu = MFU_BY_METHOD.get(method, 0.32)
    if not hyper.flash_attention:
        mfu *= MFU_NO_FLASH
        warnings.append("No fused attention kernel: expect roughly 45% slower steps.")
    if model.est_params() < 1e9:
        mfu *= 0.7  # small models leave the tensor cores idle
    if n_gpus > 1 and gpu_count <= 8:
        mfu *= 0.95

    flops_per_token = (
        FLOPS_PER_TOKEN_BY_METHOD.get(method, 6.0) * params
        + 12.0 * model.n_layer * model.d_model * hyper.seq_len
    )
    if hyper.gradient_checkpointing:
        flops_per_token *= RECOMPUTE_FLOP_FACTOR
    peak = max(peak_tflops_bf16, 1.0) * 1e12 * n_gpus * mfu
    tokens_per_second = peak / max(flops_per_token, 1.0)
    seconds = train_tokens / max(tokens_per_second, 1e-9) * DATALOADER_OVERHEAD
    gpu_hours = seconds * n_gpus / 3600.0

    flop_factor = FLOPS_PER_TOKEN_BY_METHOD.get(method, 6.0)
    basis.append(
        f"compute: {flops_per_token / 1e9:.2f} GFLOP/token = {flop_factor:.0f}N "
        f"({params / 1e9:.2f}B params){' + attention at seq_len ' + str(hyper.seq_len) if model.n_layer else ''}"
        f"{' x1.33 recompute' if hyper.gradient_checkpointing else ''}"
    )
    basis.append(
        f"throughput: {mfu * 100:.0f}% MFU of {peak_tflops_bf16:.0f} TFLOP/s dense bf16 "
        f"x {n_gpus} GPU(s) = {tokens_per_second:,.0f} tokens/s"
    )
    basis.append(
        "MFU is an assumption, not a measurement: expect the real wall clock to land "
        "within roughly half to twice this number."
    )

    fits = total_gb <= gpu_mem_gb
    suggestions: list[dict[str, str]] = []

    if not fits:
        suggestions.extend(_suggest_fixes(model, hyper, method, total_gb, gpu_mem_gb, n_gpus, gpu_label))
        warnings.append(
            f"Does not fit: {total_gb:.1f}GB needed per GPU on {gpu_label} with "
            f"{gpu_mem_gb:.0f}GB. Apply one of the fixes below."
        )

    if goal == "pretrain":
        chinchilla = 20 * params
        if train_tokens < chinchilla * 0.5:
            warnings.append(
                f"Compute-optimal pretraining would want about {chinchilla / 1e9:.0f}B tokens for "
                f"{params / 1e9:.1f}B parameters (20x, Hoffmann et al. 2022); this plan trains on "
                f"{train_tokens / 1e9:.2f}B."
            )

    if seconds > 24 * 3600:
        warnings.append(
            f"Wall clock is about {seconds / 3600:.0f} hours ({seconds / 86400:.1f} days). "
            "Consider more GPUs, a smaller model, or fewer epochs."
        )

    return ResourceEstimate(
        vram_per_gpu_gb=round(total_gb, 2),
        vram_breakdown_gb=breakdown,
        fits=fits,
        headroom_gb=round(gpu_mem_gb - total_gb, 2),
        total_gpu_hours=round(gpu_hours, 2),
        wall_clock_hours=round(seconds / 3600.0, 2),
        tokens_per_second_per_gpu=round(tokens_per_second / n_gpus, 1),
        mfu_assumed=round(mfu, 3),
        warnings=warnings,
        suggestions=suggestions,
        basis=basis,
    )


def _adapter_params(model: ModelSpec, hyper: HyperParams) -> float:
    """LoRA parameter count for the configured target modules."""
    r = max(hyper.lora_r, 1)
    d = model.d_model
    kv = model.n_kv_head / max(model.n_head, 1)
    targets = set(hyper.lora_targets)
    qkvo = 0.0
    if "q_proj" in targets:
        qkvo += 2 * r * d
    if "k_proj" in targets:
        qkvo += r * d * (1 + kv)
    if "v_proj" in targets:
        qkvo += r * d * (1 + kv)
    if "o_proj" in targets:
        qkvo += 2 * r * d
    mlp = 0.0
    ffn = (8.0 / 3.0) * d
    if "gate_proj" in targets:
        mlp += r * (d + ffn)
    if "up_proj" in targets:
        mlp += r * (d + ffn)
    if "down_proj" in targets:
        mlp += r * (ffn + d)
    return model.n_layer * (qkvo + mlp)


def _suggest_fixes(
    model: ModelSpec,
    hyper: HyperParams,
    method: Method,
    need_gb: float,
    mem_gb: float,
    n_gpus: int,
    gpu_label: str,
) -> list[dict[str, str]]:
    """Concrete, ordered remedies when a configuration does not fit."""
    out: list[dict[str, str]] = []
    ratio = need_gb / max(mem_gb, 1e-9)

    if not hyper.gradient_checkpointing:
        out.append({
            "action": "Enable gradient checkpointing",
            "detail": "Recompute activations in the backward pass: saves up to ~70% of activation "
                      "memory for about 33% more compute.",
            "flag": "--gradient-checkpointing",
        })
    if hyper.packing:
        out.append({
            "action": "Turn off sequence packing",
            "detail": "Packing maximises tokens per step but every packed sample is a full seq_len "
                      "row; disabling it lets the collator trim padding.",
            "flag": "--no-packing",
        })
    if hyper.micro_batch_size > 1:
        smaller = max(1, int(hyper.micro_batch_size / max(ratio, 1.05)))
        out.append({
            "action": f"Micro batch {hyper.micro_batch_size} -> {smaller}",
            "detail": f"Keep the global batch by raising gradient accumulation to "
                      f"{math.ceil(hyper.micro_batch_size / smaller) * hyper.grad_accum}.",
            "flag": f"--micro-batch {smaller} --grad-accum {math.ceil(hyper.micro_batch_size / smaller) * hyper.grad_accum}",
        })
    if method == "full":
        out.append({
            "action": "Switch to 8-bit AdamW",
            "detail": "Optimizer moments drop from 8 to ~2 bytes/param, the single biggest saving in "
                      "full fine-tuning.",
            "flag": "--optimizer adamw_8bit",
        })
        out.append({
            "action": "Use QLoRA instead of full fine-tuning",
            "detail": "Freeze 4-bit base weights and train adapters: ~8x less memory for slightly "
                      "lower final quality.",
            "flag": "--method qlora",
        })
    if hyper.seq_len > 1024:
        shorter = 1024 if hyper.seq_len > 2048 else hyper.seq_len // 2
        out.append({
            "action": f"Sequence length {hyper.seq_len} -> {shorter}",
            "detail": "Attention memory is quadratic in sequence length; shorter sequences also make "
                      "the step budget go further.",
            "flag": f"--seq-len {shorter}",
        })
    if n_gpus == 1:
        out.append({
            "action": "Add GPUs with ZeRO-2/3",
            "detail": "Sharding grads/optimizer (and optionally weights) splits the model state and "
                      "activations across devices.",
            "flag": "--gpus 4 --zero-stage 3",
        })
    out.append({
        "action": f"Bigger card than {gpu_label}",
        "detail": f"This plan needs about {need_gb:.1f}GB of VRAM per GPU; the current card has "
                  f"{mem_gb:.0f}GB.",
        "flag": "--gpu h100-80gb",
    })
    return out


def max_micro_batch(
    *,
    goal: Goal,
    method: Method,
    model: ModelSpec,
    hyper: HyperParams,
    gpu_count: int,
    gpu_mem_gb: float,
    peak_tflops_bf16: float,
) -> int:
    """Largest micro batch that still fits, by bisection over the estimator."""
    best = 0
    for candidate in (1, 2, 4, 8, 16, 32, 64, 128):
        trial = HyperParams(**{**hyper.__dict__, "micro_batch_size": candidate})
        est = estimate(
            goal=goal, method=method, model=model, hyper=trial,
            gpu_count=gpu_count, gpu_mem_gb=gpu_mem_gb,
            peak_tflops_bf16=peak_tflops_bf16, train_tokens=1_000_000,
        )
        if est.fits:
            best = candidate
        else:
            break
    return max(best, 1)


def plan_summary_lines(est: ResourceEstimate) -> list[str]:
    """Small helper for the CLI table output."""
    return [
        f"VRAM per GPU: {est.vram_per_gpu_gb:.1f} GiB "
        f"(model {est.vram_breakdown_gb.get('model_state', 0):.1f} + "
        f"activations {est.vram_breakdown_gb.get('activations', 0):.1f} + "
        f"overhead {est.vram_breakdown_gb.get('framework_overhead', 0):.1f})",
        f"Fits: {'yes' if est.fits else 'NO'} (headroom {est.headroom_gb:.1f} GiB)",
        f"Throughput: {est.tokens_per_second_per_gpu:,.0f} tok/s/GPU at {est.mfu_assumed * 100:.0f}% MFU",
        f"Wall clock: {est.wall_clock_hours:.2f} h ({est.total_gpu_hours:.2f} GPU-hours)",
    ]


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"
