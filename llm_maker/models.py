"""Data models for an LLM Maker training plan.

These dataclasses are the contract between the server, the CLI, the estimators
and the code generator.  Everything is plain-data so it round-trips through
JSON/YAML without custom encoders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Goal = Literal["sft", "dpo", "cpt", "pretrain"]
Method = Literal["lora", "qlora", "full", "from-scratch"]


@dataclass
class ModelSpec:
    """Architecture choice for the model being made.

    For ``sft``/``dpo``/``cpt`` these describe the *base* checkpoint to load;
    for ``pretrain`` they describe the network to build from scratch.
    """

    preset: str = "7b"
    base_model: str = "Qwen/Qwen2.5-7B"
    tokenizer: str = ""
    params_b: float = 7.6
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    d_model: int = 4096
    ffn_dim: int = 14336
    vocab_size: int = 128256
    max_seq_len: int = 8192
    tie_embeddings: bool = False
    rope_theta: float = 1000000.0

    def est_params(self) -> float:
        """Parameter count estimate from the architectural shape.

        Uses the standard transformer accounting: attention projections
        (``2 + 2 * n_kv_head / n_head`` d²), a SwiGLU MLP of ``8 d²``, and the
        embedding matrix (counted once when tied, twice when untied).
        """
        kv_ratio = self.n_kv_head / max(self.n_head, 1)
        d2 = self.d_model * self.d_model
        attn = (2 + 2 * kv_ratio) * self.n_layer * d2          # q, o + GQA-scaled k, v
        ffn = self.ffn_dim or (8 / 3) * self.d_model
        mlp = 3 * self.n_layer * self.d_model * ffn            # SwiGLU: gate, up, down
        embed = self.vocab_size * self.d_model * (1 if self.tie_embeddings else 2)
        return attn + mlp + embed

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["est_params_b"] = round(self.est_params() / 1e9, 3)
        return data


@dataclass
class DatasetRef:
    """One dataset in a plan, optionally weighted inside a mixture."""

    repo_id: str
    weight: float = 1.0
    split: str = "train"
    config: str | None = None
    text_field: str | None = None
    prompt_field: str | None = None
    completion_field: str | None = None
    messages_field: str | None = None
    chosen_field: str | None = None
    rejected_field: str | None = None
    task: str | None = None
    est_tokens: int | None = None
    notes: str | None = None
    hub_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareSpec:
    """Target hardware.

    ``gpu_mem_gb`` and ``peak_tflops_bf16`` default to 0.0, which means "take them
    from the named GPU preset" -- explicit non-zero values override the preset, so
    an unfamiliar accelerator can still be described by hand.
    """

    gpu: str = "a100-80gb"
    count: int = 1
    nodes: int = 1
    gpu_mem_gb: float = 0.0
    peak_tflops_bf16: float = 0.0
    cpu_offload: bool = False
    interconnects: str = "single-node"


@dataclass
class HyperParams:
    """Training knobs.

    ``0`` (and ``""`` for the optimizer) means *unset*: the planner fills those
    from the goal presets, so a caller only has to specify what it cares about.
    The planner resolves every field before planning, and the generated
    ``config.yaml`` therefore always contains concrete numbers.

    ``trainer`` selects the generated project's execution profile. ``optimized``
    uses Hugging Face's public ``Trainer`` lifecycle with a safe performance
    path; ``standard`` is the debugging/baseline escape hatch.
    """

    micro_batch_size: int = 0
    grad_accum: int = 0
    seq_len: int = 0
    epochs: float = 0.0
    max_steps: int | None = None
    learning_rate: float = 0.0
    lr_scheduler: str = ""
    warmup_ratio: float = -1.0
    weight_decay: float = -1.0
    optimizer: str = ""
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    packing: bool = True
    bf16: bool = True
    # Execution profile for the generated script. The optimizer/compile settings
    # below are runtime choices; they do not change the planner's VRAM estimate.
    trainer: Literal["optimized", "standard"] = "optimized"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    tf32: bool = True
    mask_prompt: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    dpo_beta: float = 0.0
    seed: int = 42


@dataclass
class PlanRequest:
    """What the user asked for, before any planning logic runs."""

    goal: Goal = "sft"
    model: ModelSpec = field(default_factory=ModelSpec)
    datasets: list[DatasetRef] = field(default_factory=list)
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    hyper: HyperParams = field(default_factory=HyperParams)
    method: Method | None = None
    mixture_strategy: str = "weighted"  # weighted | concat | per-dataset
    project_name: str = "my-llm"
    output_root: str = "runs"
    notes: str = ""
    track: str = "none"  # none | mlflow | wandb | tensorboard

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRequest":
        data = dict(data or {})
        dataset_items = []
        for item in data.get("datasets") or []:
            if isinstance(item, str):
                item = {"repo_id": item}
            if not isinstance(item, dict):
                continue
            repo_id = item.get("repo_id") or item.get("id")
            if not repo_id:
                continue
            allowed = {f for f in DatasetRef.__dataclass_fields__}
            dataset_items.append(DatasetRef(**{k: v for k, v in item.items() if k in allowed}))
        hyper = data.get("hyper") or {}
        hw = data.get("hardware") or {}
        model = data.get("model") or {}
        return cls(
            goal=data.get("goal", "sft"),
            model=ModelSpec(**_pick(model, ModelSpec)),
            datasets=dataset_items,
            hardware=HardwareSpec(**_pick(hw, HardwareSpec)),
            hyper=HyperParams(**_pick(hyper, HyperParams)),
            method=data.get("method"),
            mixture_strategy=data.get("mixture_strategy", "weighted"),
            project_name=data.get("project_name", "my-llm"),
            output_root=data.get("output_root", "runs"),
            notes=data.get("notes", ""),
            track=data.get("track", "none"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceEstimate:
    vram_per_gpu_gb: float = 0.0
    vram_breakdown_gb: dict[str, float] = field(default_factory=dict)
    fits: bool = True
    headroom_gb: float = 0.0
    total_gpu_hours: float = 0.0
    wall_clock_hours: float = 0.0
    tokens_per_second_per_gpu: float = 0.0
    mfu_assumed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    suggestions: list[dict[str, str]] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingPlan:
    request: PlanRequest
    method: Method
    dataset_tokens: int
    train_tokens: int
    total_steps: int
    warmup_steps: int
    global_batch_size: int
    global_batch_tokens: int
    resource: ResourceEstimate
    mixture: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    written_to: str | None = None
    catalog_mode: str = "hub"
    goal_label: str = ""
    method_reason: str = ""
    zero_stage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "method": self.method,
            "dataset_tokens": self.dataset_tokens,
            "train_tokens": self.train_tokens,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "global_batch_size": self.global_batch_size,
            "global_batch_tokens": self.global_batch_tokens,
            "resource": self.resource.to_dict(),
            "mixture": self.mixture,
            "warnings": self.warnings,
            "references": self.references,
            "generated_files": self.generated_files,
            "written_to": self.written_to,
            "catalog_mode": self.catalog_mode,
            "goal_label": self.goal_label,
            "method_reason": self.method_reason,
            "zero_stage": self.zero_stage,
        }


def _pick(data: dict[str, Any], cls: type) -> dict[str, Any]:
    allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return {k: v for k, v in (data or {}).items() if k in allowed}
