"""Presets and heuristics used by the planner.

Everything in here is intentionally data-over-code: model shapes, GPU specs,
per-goal hyper-parameter defaults and dataset-to-prompt field mappings live in
plain dicts so they can be edited without touching planning logic.
"""

from __future__ import annotations

from typing import Any

from .hub import DatasetInfo
from .models import DatasetRef, Goal, HyperParams, Method, ModelSpec

# --------------------------------------------------------------------- models
# Shapes follow the public Llama/Qwen/Mistral family configs so the generated
# scripts produce checkpoints that are directly comparable to released models.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "0.5b": dict(
        label="0.5B (Qwen2.5-0.5B shape)", params_b=0.49, n_layer=24, n_head=14, n_kv_head=2,
        d_model=896, ffn_dim=4864, vocab_size=151936, max_seq_len=32768, rope_theta=1000000.0,
        tie_embeddings=True, base_model="Qwen/Qwen2.5-0.5B",
    ),
    "1.5b": dict(
        label="1.5B (Qwen2.5-1.5B shape)", params_b=1.54, n_layer=28, n_head=12, n_kv_head=2,
        d_model=1536, ffn_dim=8960, vocab_size=151936, max_seq_len=32768, rope_theta=1000000.0,
        tie_embeddings=True, base_model="Qwen/Qwen2.5-1.5B",
    ),
    "3b": dict(
        label="3B (Qwen2.5-3B shape)", params_b=3.09, n_layer=36, n_head=16, n_kv_head=2,
        d_model=2048, ffn_dim=11008, vocab_size=151936, max_seq_len=32768, rope_theta=1000000.0,
        tie_embeddings=True, base_model="Qwen/Qwen2.5-3B",
    ),
    "7b": dict(
        label="7B (Mistral-7B-v0.3 shape)", params_b=7.24, n_layer=32, n_head=32, n_kv_head=8,
        d_model=4096, ffn_dim=14336, vocab_size=32768, max_seq_len=32768, rope_theta=1000000.0,
        tie_embeddings=False, base_model="mistralai/Mistral-7B-v0.3",
    ),
    "8b": dict(
        label="8B (Llama-3.1-8B shape)", params_b=8.03, n_layer=32, n_head=32, n_kv_head=8,
        d_model=4096, ffn_dim=14336, vocab_size=128256, max_seq_len=131072, rope_theta=500000.0,
        tie_embeddings=False, base_model="meta-llama/Llama-3.1-8B",
    ),
    "13b": dict(
        label="13B (Llama-2-13B shape)", params_b=13.0, n_layer=40, n_head=40, n_kv_head=40,
        d_model=5120, ffn_dim=13824, vocab_size=32000, max_seq_len=4096, rope_theta=10000.0,
        tie_embeddings=False, base_model="meta-llama/Llama-2-13b-hf",
    ),
    "70b": dict(
        label="70B (Llama-3.1-70B shape)", params_b=70.6, n_layer=80, n_head=64, n_kv_head=8,
        d_model=8192, ffn_dim=28672, vocab_size=128256, max_seq_len=131072, rope_theta=500000.0,
        tie_embeddings=False, base_model="meta-llama/Llama-3.1-70B",
    ),
}

DEFAULT_PRESET = "7b"


def model_from_preset(preset: str | None, *, base_model: str | None = None) -> ModelSpec:
    key = (preset or DEFAULT_PRESET).lower()
    if key not in MODEL_PRESETS:
        key = DEFAULT_PRESET
    shape = MODEL_PRESETS[key]
    return ModelSpec(
        preset=key,
        base_model=base_model or shape["base_model"],
        params_b=shape["params_b"],
        n_layer=shape["n_layer"],
        n_head=shape["n_head"],
        n_kv_head=shape["n_kv_head"],
        d_model=shape["d_model"],
        ffn_dim=shape["ffn_dim"],
        vocab_size=shape["vocab_size"],
        max_seq_len=shape["max_seq_len"],
        tie_embeddings=shape["tie_embeddings"],
        rope_theta=shape["rope_theta"],
    )


# ----------------------------------------------------------------------- GPUs
# Dense bf16 throughput (tensor cores, no sparsity). Several vendor datasheets
# headline the 2:4-sparsity number, which is twice these values -- using the dense
# figure keeps the estimate honest. Cards without bf16 tensor cores list their
# equivalent fp16 tensor throughput and say so in `note`.
GPU_PRESETS: dict[str, dict[str, Any]] = {
    "t4-16gb": dict(label="Tesla T4 16GB", mem_gb=16, peak_tflops_bf16=65.0,
                    note="Turing has no real bf16; the script switches to fp16 + GradScaler"),
    "v100-16gb": dict(label="Tesla V100 16GB", mem_gb=16, peak_tflops_bf16=125.0,
                      note="Volta has no bf16; the script switches to fp16 + GradScaler"),
    "v100-32gb": dict(label="Tesla V100 32GB", mem_gb=32, peak_tflops_bf16=125.0,
                      note="Volta has no bf16; the script switches to fp16 + GradScaler"),
    "a10g-24gb": dict(label="NVIDIA A10G 24GB", mem_gb=24, peak_tflops_bf16=62.5,
                      note="strong cost/perf for LoRA runs"),
    "l4-24gb": dict(label="NVIDIA L4 24GB", mem_gb=24, peak_tflops_bf16=60.5,
                    note="low-power Ada card, better at serving than training"),
    "rtx3090-24gb": dict(label="GeForce RTX 3090 24GB", mem_gb=24, peak_tflops_bf16=35.6,
                         note="Ampere consumer: no hardware bf16, use fp16"),
    "rtx4090-24gb": dict(label="GeForce RTX 4090 24GB", mem_gb=24, peak_tflops_bf16=82.6,
                         note="no NVLink; multi-GPU all-reduce runs over PCIe"),
    "a6000-48gb": dict(label="RTX A6000 48GB", mem_gb=48, peak_tflops_bf16=77.4,
                       note="no NVLink; use ZeRO-2/3 across PCIe"),
    "a100-40gb": dict(label="A100 40GB", mem_gb=40, peak_tflops_bf16=312.0,
                      note="solid datacentre baseline"),
    "a100-80gb": dict(label="A100 80GB", mem_gb=80, peak_tflops_bf16=312.0,
                      note="the reference card for full fine-tuning"),
    "l40s-48gb": dict(label="NVIDIA L40S 48GB", mem_gb=48, peak_tflops_bf16=181.0,
                      note="Ada datacentre card without NVLink"),
    "h100-80gb": dict(label="H100 80GB", mem_gb=80, peak_tflops_bf16=989.0,
                      note="needs FlashAttention-2 and fused optimisers to approach peak"),
    "mi300x-192gb": dict(label="AMD MI300X 192GB", mem_gb=192, peak_tflops_bf16=1307.5,
                         note="ROCm: verify kernel support for your attention implementation"),
    "cpu": dict(label="CPU (debug only)", mem_gb=64, peak_tflops_bf16=0.5,
                note="fine for dry-running the generated script, not for training"),
}

DEFAULT_GPU = "a100-80gb"


def gpu_from_preset(key: str | None) -> dict[str, Any]:
    gpu = GPU_PRESETS.get((key or DEFAULT_GPU).lower(), GPU_PRESETS[DEFAULT_GPU])
    return {"key": key if key in GPU_PRESETS else DEFAULT_GPU, **gpu}


# ------------------------------------------------------------------- goal spec
GOAL_PRESETS: dict[Goal, dict[str, Any]] = {
    "sft": dict(
        label="Supervised fine-tuning",
        blurb="Teach a base model to follow instructions and hold a conversation.",
        default_method="qlora",
        lr=2e-4,
        epochs=3.0,
        weight_decay=0.0,
        seq_len=2048,
        lr_scheduler="cosine",
        warmup_ratio=0.03,
        need_columns="prompt + completion, a messages list, or plain text",
    ),
    "dpo": dict(
        label="Preference alignment (DPO)",
        blurb="Align a fine-tuned model to preferred answers using chosen/rejected pairs.",
        default_method="qlora",
        lr=5e-6,
        epochs=1.0,
        weight_decay=0.0,
        seq_len=1024,
        lr_scheduler="cosine",
        warmup_ratio=0.1,
        beta=0.1,
        need_columns="prompt + chosen + rejected",
    ),
    "cpt": dict(
        label="Continued pretraining",
        blurb="Push raw domain text into an existing model before instruction tuning.",
        default_method="lora",
        lr=5e-5,
        epochs=1.0,
        weight_decay=0.01,
        seq_len=4096,
        lr_scheduler="cosine",
        warmup_ratio=0.02,
        need_columns="one text column, packed into blocks",
    ),
    "pretrain": dict(
        label="Pretraining from scratch",
        blurb="Train a new transformer on a text corpus. Costs orders of magnitude more than the others.",
        default_method="from-scratch",
        lr=3e-4,
        epochs=1.0,
        weight_decay=0.1,
        seq_len=4096,
        lr_scheduler="cosine",
        warmup_ratio=0.01,
        need_columns="raw text / web documents",
    ),
}


# --------------------------------------------------- dataset field heuristics
# Known repo -> column mapping used when building a plan from the catalog.
KNOWN_DATASET_FIELDS: dict[str, dict[str, str]] = {
    "tatsu-lab/alpaca": {"prompt": "instruction+input", "completion": "output", "text_field": "text"},
    "yahma/alpaca-cleaned": {"prompt": "instruction+input", "completion": "output"},
    "databricks/databricks-dolly-15k": {"prompt": "instruction+context", "completion": "response", "text_field": "text"},
    "timdettmers/openassistant-guanaco": {"text_field": "text"},
    "HuggingFaceH4/ultrachat_200k": {"messages": "messages"},
    "HuggingFaceH4/deita-10k-v0": {"messages": "messages"},
    "teknium/OpenHermes-2.5": {"messages": "conversations"},
    "Open-Orca/OpenOrca": {"messages": "messages", "prompt": "question", "completion": "response"},
    "microsoft/orca-math-word-problems-200k": {"prompt": "question", "completion": "answer"},
    "meta-math/MetaMathQA": {"prompt": "query", "completion": "response"},
    "openai/gsm8k": {"prompt": "question", "completion": "answer"},
    "sahil2801/CodeAlpaca-20k": {"prompt": "instruction+input", "completion": "output"},
    "ise-uiuc/Magicoder-OSS-Instruct-75K": {"prompt": "problem", "completion": "solution"},
    "m-a-p/CodeFeedback-Filtered-Instruction": {"prompt": "query", "completion": "answer"},
    "Anthropic/hh-rlhf": {"chosen": "chosen", "rejected": "rejected"},
    "Dahoas/full-hh-rlhf": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    "HuggingFaceH4/ultrafeedback_binarized": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    "Intel/orca_dpo_pairs": {"prompt": "question", "chosen": "chosen", "rejected": "rejected"},
    "argilla/dpo-mix-7k": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    "stanfordnlp/SHP": {"prompt": "history", "chosen": "human_ref_A", "rejected": "human_ref_B"},
    "HuggingFaceFW/fineweb": {"text_field": "text"},
    "HuggingFaceFW/fineweb-edu": {"text_field": "text"},
    "HuggingFaceTB/cosmopedia": {"text_field": "text"},
    "HuggingFaceTB/smollm-corpus": {"text_field": "text"},
    "allenai/c4": {"text_field": "text"},
    "roneneldan/TinyStories": {"text_field": "text"},
    "wikimedia/wikipedia": {"text_field": "text"},
    "cerebras/SlimPajama-627B": {"text_field": "text"},
    "DKYoon/SlimPajama-6B": {"text_field": "text"},
    "togethercomputer/RedPajama-Data-1T": {"text_field": "text"},
    "uonlp/CulturaX": {"text_field": "text"},
    "ai4bharat/IndicCorpV2": {"text_field": "text"},
    "common-pile/common-pile-v0.1": {"text_field": "text"},
    "open-web-math/open-web-math": {"text_field": "text"},
    "HuggingFaceTB/finemath": {"text_field": "text"},
    "hotpotqa/hotpot_qa": {"prompt": "question", "completion": "answer"},
    "rajpurkar/squad": {"prompt": "question", "completion": "answers"},
    "abisee/cnn_dailymail": {"prompt": "article", "completion": "highlights"},
    "EdinburghNLP/xsum": {"prompt": "document", "completion": "summary"},
    "samsum": {"prompt": "dialogue", "completion": "summary"},
    "qiaojin/PubMedQA": {"prompt": "question", "completion": "long_answer"},
    "epfl-llm/guidelines": {"prompt": "instruction", "completion": "response"},
    "liuhaotian/LLaVA-Instruct-150K": {"messages": "conversations"},
}

CHAT_TEMPLATE_HINT = (
    "Render every sample with the base model's own chat template "
    "(``tokenizer.apply_chat_template``) so the fine-tune stays compatible with the "
    "released inference stack."
)


def guess_dataset_role(info: DatasetInfo) -> str:
    """Best guess at what a dataset is for: ``sft`` / ``preference`` / ``text``."""
    tasks = {t.lower() for t in info.task_categories}
    tags = " ".join(info.tags).lower()
    repo = info.id.lower()
    if tasks & {"preference", "reward-modeling"}:
        return "preference"
    if any(k in repo for k in ("dpo", "rlhf", "preference")) or "preference" in tags:
        return "preference"
    known = KNOWN_DATASET_FIELDS.get(info.id)
    if known:
        if "chosen" in known:
            return "preference"
        if known.get("messages") or known.get("completion"):
            return "sft"
        if known.get("text_field"):
            return "text"
    if tasks & {"instruction", "chat"} or "instruct" in repo:
        return "sft"
    if any(k in repo for k in ("fineweb", "c4", "corpus", "slimpajama", "redpajama", "webtext", "wikipedia")):
        return "text"
    if tags.count("pretrain") or "dataset:pretrain" in tags:
        return "text"
    if tasks & {"text-generation", "text2text-generation", "text-classification"}:
        return "sft"
    if "raw" in tags or "monolingual" in tags:
        return "text"
    return "unknown"


def dataset_ref_from_info(info: DatasetInfo) -> DatasetRef:
    """Promote a catalog entry into a plan-ready dataset reference."""
    role = guess_dataset_role(info)
    known = KNOWN_DATASET_FIELDS.get(info.id, {})
    split = "train"
    config: str | None = None
    if info.splits:
        first = info.splits[0]
        config = first.get("config")
        split = first.get("split") or "train"
    elif info.configs:
        config = info.configs[0]

    # Preference datasets usually name their split "train_prefs"/"train_preference".
    if role == "preference" and info.splits:
        preferred = [s for s in info.splits if "pref" in str(s.get("split", ""))]
        pick = preferred[0] if preferred else next(
            (s for s in info.splits if s.get("split") == "train"), info.splits[0]
        )
        config = pick.get("config") or config
        split = pick.get("split") or split

    ref = DatasetRef(
        repo_id=info.id,
        split=split,
        config=config,
        task=role,
        hub_url=info.url,
    )
    if known:
        ref.text_field = known.get("text_field")
        ref.prompt_field = known.get("prompt")
        ref.completion_field = known.get("completion")
        ref.messages_field = known.get("messages")
        ref.chosen_field = known.get("chosen")
        ref.rejected_field = known.get("rejected")
    elif role == "preference":
        ref.chosen_field = ref.chosen_field or "chosen"
        ref.rejected_field = ref.rejected_field or "rejected"
        ref.notes = "the generated script lists this dataset's real column names at launch"
    else:
        ref.notes = f"column names are auto-detected at runtime ({role} heuristics)"

    payload = info.to_dict()
    ref.est_tokens = payload.get("est_tokens")
    return ref


def missing_columns(ref: DatasetRef, goal: Goal) -> list[str]:
    """Field mappings the generated script has to guess at runtime."""
    needs_preference = goal == "dpo" or ref.task == "preference"
    if needs_preference:
        return [] if (ref.chosen_field and ref.rejected_field) else ["chosen/rejected"]
    if goal in ("cpt", "pretrain"):
        return [] if (ref.text_field or ref.completion_field) else ["text field"]
    if ref.messages_field or ref.completion_field or ref.text_field or ref.prompt_field:
        return []
    return ["field mapping"]


# -------------------------------------------------------------------- methods
def choose_method(
    goal: Goal,
    params_b: float,
    gpu_mem_gb: float,
    gpu_count: int,
    requested: Method | None = None,
) -> tuple[Method, str]:
    """Pick a training method, returning ``(method, why)``."""
    if goal == "pretrain":
        return "from-scratch", "pretraining starts from random weights, so no adapter is involved"
    reasons = {
        "full": "you asked for full fine-tuning",
        "lora": "you asked for bf16 LoRA",
        "qlora": "you asked for QLoRA (4-bit base weights)",
    }
    if requested in reasons:
        return requested, reasons[requested]  # type: ignore[return-value]

    # Full fine-tuning costs ~16 bytes/param of model state (bf16 weights + fp32
    # master + grads + Adam moments) before activations; require ~20 to be safe.
    if params_b * 20 <= gpu_mem_gb * max(gpu_count, 1):
        return "full", (
            f"full fine-tuning needs roughly {params_b * 20:.0f}GB of model state and "
            f"{gpu_mem_gb * max(gpu_count, 1):.0f}GB of total VRAM is available"
        )
    if params_b * 1.2 <= gpu_mem_gb:
        return "lora", (
            f"{params_b * 2:.0f}GB of bf16 base weights fit in {gpu_mem_gb:.0f}GB, so bf16 "
            "LoRA is faster and slightly more accurate than 4-bit"
        )
    return "qlora", (
        f"{params_b:.1f}B parameters do not fit in {gpu_mem_gb:.0f}GB of bf16, so the base "
        "model is loaded in 4-bit and only adapters are trained"
    )


# Global-batch targets per goal, in tokens. These are the batch sizes the
# generated config aims for before hardware limits get in the way.
TARGET_GLOBAL_TOKENS: dict[str, int] = {
    "sft": 32_768,        # 16 sequences at seq_len 2048
    "dpo": 16_384,        # DPO is stable at smaller batches
    "cpt": 131_072,
    "pretrain": 1_048_576,
}


def default_hyperparams(goal: Goal, method: Method, model: ModelSpec) -> HyperParams:
    """Concrete hyper-parameters for a goal/method pair."""
    preset = GOAL_PRESETS[goal]
    hyper = HyperParams(
        learning_rate=float(preset["lr"]),
        epochs=float(preset["epochs"]),
        weight_decay=float(preset["weight_decay"]),
        seq_len=min(int(preset["seq_len"]), model.max_seq_len),
        lr_scheduler=str(preset["lr_scheduler"]),
        warmup_ratio=float(preset["warmup_ratio"]),
    )
    # Always resolved, even for goals that do not use it, so the generated
    # config.yaml is self-describing rather than half-empty.
    hyper.dpo_beta = float(preset.get("beta", 0.1))
    if hyper.weight_decay == -1.0:
        hyper.weight_decay = float(preset["weight_decay"])
    if method == "full":
        # Full fine-tuning wants roughly a 10x smaller LR than LoRA and no dropout.
        hyper.learning_rate = hyper.learning_rate / 10
        hyper.lora_dropout = 0.0
        hyper.optimizer = "adamw_torch_fused"
    if method == "qlora":
        hyper.optimizer = "paged_adamw_8bit"
        hyper.lora_r = 16
        hyper.lora_alpha = 32
        hyper.packing = False  # 4-bit + packing + long sequences is memory-hungry
    if goal == "pretrain":
        hyper.optimizer = "adamw_torch_fused"
        hyper.warmup_ratio = 0.01
        hyper.gradient_checkpointing = False
    return hyper


# -------------------------------------------------------------- dataset copy
def merge_hyperparams(requested: HyperParams, defaults: HyperParams) -> HyperParams:
    """Fill every unset field of ``requested`` from ``defaults``.

    Unset means 0, -1.0 or an empty string -- see :class:`~llm_maker.models.HyperParams`.
    Explicit values are always kept, so a caller can override any single knob.
    """
    merged = HyperParams(**requested.__dict__)
    for name in ("micro_batch_size", "grad_accum", "seq_len", "epochs", "learning_rate", "dpo_beta"):
        if getattr(merged, name) <= 0:
            setattr(merged, name, getattr(defaults, name))
    for name in ("warmup_ratio", "weight_decay"):
        if getattr(merged, name) < 0:
            setattr(merged, name, getattr(defaults, name))
    for name in ("lr_scheduler", "optimizer"):
        if not getattr(merged, name):
            setattr(merged, name, getattr(defaults, name))
    return merged


def prompt_template_for(goal: Goal, method: Method) -> str:
    """A short, human-readable description of what the generated script builds."""
    if goal == "dpo":
        return (
            "Each sample is rendered as prompt / chosen / rejected and the objective is the DPO "
            "log-sigmoid over the implicit reward difference."
        )
    if goal in ("cpt", "pretrain"):
        return (
            "Documents are concatenated into fixed-length blocks of `seq_len` tokens with labels "
            "copied from the input ids, so every token contributes to the loss."
        )
    if method in ("lora", "qlora"):
        return (
            "Instructions are rendered with the tokenizer's chat template; set "
            "`--mask-prompt` if you want only the completion to contribute to the loss."
        )
    return "Instructions are rendered with the tokenizer's chat template and trained end to end."


def size_category_label(bucket: str | None) -> str:
    """Human-readable size bucket, e.g. ``10K<n<100K`` -> ``10,000-100,000 rows``."""
    if not bucket:
        return "unknown size"
    pretty = bucket.replace("n<", "").replace("n>", "over ")
    lo, _, hi = pretty.partition("<")
    try:
        return f"{int(lo):,}-{int(hi):,} rows"
    except ValueError:
        return bucket
