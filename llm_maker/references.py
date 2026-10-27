"""Papers each part of a plan is derived from.

These are real, well-known references; the app shows them next to the numbers
so a user can audit an assumption instead of trusting it.  Links point at arXiv
(or the ACL Anthology where a paper lives there).
"""

from __future__ import annotations

from typing import Any

REFERENCES: dict[str, dict[str, str]] = {
    "lora": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al., 2021",
        "url": "https://arxiv.org/abs/2106.09685",
        "why": "Rank-decomposed adapter matrices; the basis of every LoRA/QLoRA plan here.",
    },
    "qlora": {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Dettmers et al., 2023",
        "url": "https://arxiv.org/abs/2305.14314",
        "why": "4-bit NF4 base weights with double quantisation and paged optimisers (~0.5 bytes/param).",
    },
    "zero": {
        "title": "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
        "authors": "Rajbhandari et al., 2020",
        "url": "https://arxiv.org/abs/1910.02054",
        "why": "How optimizer state, gradients and weights are sharded across GPUs in the memory estimate.",
    },
    "activation_memory": {
        "title": "Reducing Activation Recomputation in Large Transformer Models",
        "authors": "Korthikanti et al., 2022",
        "url": "https://arxiv.org/abs/2205.05198",
        "why": "The s*b*h*(34 + 5*a*s/h) per-layer activation formula and the checkpointing saving.",
    },
    "instructgpt": {
        "title": "Training language models to follow instructions with human feedback",
        "authors": "Ouyang et al., 2022",
        "url": "https://arxiv.org/abs/2203.02155",
        "why": "The three-stage SFT -> reward model -> RLHF pipeline the SFT goal follows.",
    },
    "flant5": {
        "title": "Scaling Instruction-Finetuned Language Models",
        "authors": "Chung et al., 2022",
        "url": "https://arxiv.org/abs/2210.11416",
        "why": "Evidence that instruction-tuning many tasks generalises better than tuning one.",
    },
    "alpaca": {
        "title": "Alpaca: A Strong, Replicable Instruction-Following Model",
        "authors": "Taori et al., 2023",
        "url": "https://crfm.stanford.edu/2023/03/13/alpaca.html",
        "why": "Self-instruct data generation, the recipe behind tatsu-lab/alpaca.",
    },
    "self_instruct": {
        "title": "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        "authors": "Wang et al., 2022",
        "url": "https://arxiv.org/abs/2212.10560",
        "why": "How most synthetic instruction corpora in the catalog were built.",
    },
    "dpo": {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": "Rafailov et al., 2023",
        "url": "https://arxiv.org/abs/2305.18290",
        "why": "The loss used by the DPO goal: no separate reward model needed.",
    },
    "ppo": {
        "title": "Proximal Policy Optimization Algorithms",
        "authors": "Schulman et al., 2017",
        "url": "https://arxiv.org/abs/1707.06347",
        "why": "The RLHF alternative to DPO, for when you want an explicit reward model.",
    },
    "chinchilla": {
        "title": "Training Compute-Optimal Large Language Models",
        "authors": "Hoffmann et al., 2022",
        "url": "https://arxiv.org/abs/2203.15556",
        "why": "The ~20 tokens-per-parameter rule used to sanity-check pretraining budgets.",
    },
    "kaplan": {
        "title": "Scaling Laws for Neural Language Models",
        "authors": "Kaplan et al., 2020",
        "url": "https://arxiv.org/abs/2001.08361",
        "why": "Where the 6N FLOPs-per-token approximation comes from.",
    },
    "gpt3": {
        "title": "Language Models are Few-Shot Learners",
        "authors": "Brown et al., 2020",
        "url": "https://arxiv.org/abs/2005.14165",
        "why": "The original model/data/optimiser scaling recipe for the pretraining goal.",
    },
    "falcon": {
        "title": "The RefinedWeb Dataset for Falcon LLM",
        "authors": "Penedo et al., 2023",
        "url": "https://arxiv.org/abs/2306.01116",
        "why": "Why web corpora are deduplicated and filtered the way fineweb/C4 entries describe.",
    },
    "dedup": {
        "title": "Deduplicating Training Data Makes Language Models Better",
        "authors": "Lee et al., 2021",
        "url": "https://arxiv.org/abs/2107.06499",
        "why": "Near-duplicate removal: the reason packing corpora are deduplicated before training.",
    },
    "flash_attention": {
        "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "authors": "Dao, 2023",
        "url": "https://arxiv.org/abs/2307.08691",
        "why": "The fused attention kernel the generated scripts request by default.",
    },
    "gqa": {
        "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
        "authors": "Ainslie et al., 2023",
        "url": "https://arxiv.org/abs/2305.13245",
        "why": "Grouped-query attention, which is why n_kv_head < n_head in the model presets.",
    },
    "glu_variants": {
        "title": "GLU Variants Improve Transformer",
        "authors": "Shazeer, 2020",
        "url": "https://arxiv.org/abs/2002.05202",
        "why": "The 8/3 d SwiGLU feed-forward width behind the parameter-count estimate.",
    },
    "peft": {
        "title": "PEFT: Parameter-Efficient Fine-Tuning (library documentation)",
        "authors": "Hugging Face",
        "url": "https://huggingface.co/docs/peft",
        "why": "The adapter library the generated LoRA/QLoRA scripts use.",
    },
    "trl_dpo": {
        "title": "TRL: DPOTrainer documentation",
        "authors": "Hugging Face",
        "url": "https://huggingface.co/docs/trl/dpo_trainer",
        "why": "Reference implementation of the DPO loss the generated script mirrors.",
    },
    "transformers_trainer": {
        "title": "Transformers: Trainer and the SFT recipe",
        "authors": "Hugging Face",
        "url": "https://huggingface.co/docs/transformers/main_classes/trainer",
        "why": "The training loop, scheduler wiring and checkpointing used by the generated scripts.",
    },
    "seq_packing": {
        "title": "Efficient Sequence Packing for Training (NeMo / TRL notes)",
        "authors": "NVIDIA / Hugging Face",
        "url": "https://huggingface.co/docs/trl/sft_trainer",
        "why": "How packing multiple short samples into one seq_len window avoids wasted padding.",
    },
    "flan": {
        "title": "Finetuned Language Models Are Zero-Shot Learners",
        "authors": "Wei et al., 2021",
        "url": "https://arxiv.org/abs/2109.01652",
        "why": "The paper that introduced the instruction-formatted (prompt / completion) training style "
               "the SFT data pipeline targets.",
    },
    "tulu3": {
        "title": "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
        "authors": "Lambert et al., 2024",
        "url": "https://arxiv.org/abs/2411.15124",
        "why": "A fully documented, reproducible post-training recipe for instruction mixtures.",
    },
    "olm": {
        "title": "The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs",
        "authors": "Parthasarathy et al., 2024",
        "url": "https://arxiv.org/abs/2408.13296",
        "why": "A practical survey used to sanity-check the per-goal hyper-parameter defaults.",
    },
    "lora_hyper": {
        "title": "Practical Tips for Finetuning LLMs Using LoRA",
        "authors": "Raschka, 2023",
        "url": "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms",
        "why": "Why LoRA defaults to r=16, alpha=32 and a 1e-4..2e-4 learning rate here.",
    },
    "deepspeed": {
        "title": "DeepSpeed ZeRO configuration",
        "authors": "Microsoft",
        "url": "https://www.deepspeed.ai/tutorials/zero/",
        "why": "The stage 1/2/3 semantics behind the sharding factors in the memory estimate.",
    },
}


def references_for(goal: str, method: str | None = None) -> list[dict[str, str]]:
    """Pick the references relevant to a plan, rather than dumping all of them."""
    keys = ["zero", "activation_memory", "kaplan", "flash_attention", "deepspeed", "transformers_trainer"]
    if goal == "sft":
        keys += ["instructgpt", "flant5", "flan", "self_instruct", "alpaca", "seq_packing", "tulu3"]
    elif goal == "dpo":
        keys += ["dpo", "ppo", "instructgpt"]
    elif goal == "cpt":
        keys += ["falcon", "dedup", "seq_packing"]
    elif goal == "pretrain":
        keys += ["chinchilla", "gpt3", "falcon", "dedup", "gqa", "glu_variants"]
    if method == "qlora":
        keys += ["qlora", "lora"]
    elif method in ("lora", "full"):
        keys += ["lora_hyper"] if method == "lora" else []
    if method in ("lora", "qlora"):
        keys.append("peft")
    if goal == "dpo":
        keys.append("trl_dpo")
    keys.append("olm")

    seen: dict[str, dict[str, str]] = {}
    for key in keys:
        ref = REFERENCES.get(key)
        if ref and ref["title"] not in seen:
            seen[ref["title"]] = dict(ref, key=key)
    return list(seen.values())


def all_references() -> list[dict[str, str]]:
    return [dict(ref, key=key) for key, ref in REFERENCES.items()]


def as_markdown(refs: list[dict[str, Any]]) -> str:
    """Render references as a markdown list for the generated README."""
    lines = []
    for ref in refs:
        lines.append(f"- **{ref['title']}** — {ref['authors']}. {ref['url']}\n  _{ref['why']}_")
    return "\n".join(lines)
