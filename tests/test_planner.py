"""The resource estimator and the planning decisions that depend on it."""

from __future__ import annotations

import pytest

from llm_maker.estimator import activation_bytes, estimate, max_micro_batch
from llm_maker.models import DatasetRef, HardwareSpec, HyperParams, PlanRequest
from llm_maker.planner import decide_method, plan
from llm_maker.recipes import GPU_PRESETS, model_from_preset


def build(goal="sft", preset="7b", gpu="a100-80gb", gpus=1, **hyper):
    request = PlanRequest(
        goal=goal,
        model=model_from_preset(preset),
        datasets=[DatasetRef(repo_id="tatsu-lab/alpaca")],
        hardware=HardwareSpec(gpu=gpu, count=gpus),
        hyper=HyperParams(**hyper),
    )
    return plan(request, resolve_datasets=False)


# ------------------------------------------------------------------ parameters
@pytest.mark.parametrize(
    "preset, expected_b",
    [("0.5b", 0.49), ("1.5b", 1.54), ("3b", 3.09), ("7b", 7.24), ("8b", 8.03), ("13b", 13.0)],
)
def test_parameter_estimate_matches_published_sizes(preset: str, expected_b: float) -> None:
    model = model_from_preset(preset)
    computed = model.est_params() / 1e9
    assert computed == pytest.approx(expected_b, rel=0.02)


def test_gqa_reduces_parameter_count() -> None:
    """Fewer KV heads means fewer parameters -- the estimate has to notice."""
    wide = model_from_preset("7b")
    wide.n_kv_head = wide.n_head
    narrow = model_from_preset("7b")
    narrow.n_kv_head = 1
    assert narrow.est_params() < wide.est_params()


# ----------------------------------------------------------------- activations
def test_checkpointing_cuts_activation_memory() -> None:
    model = model_from_preset("7b")
    live, _ = activation_bytes(model, micro_batch=1, seq_len=2048, checkpointing=False)
    ckpt, _ = activation_bytes(model, micro_batch=1, seq_len=2048, checkpointing=True)
    assert ckpt < live / 5


def test_activation_memory_scales_with_batch() -> None:
    model = model_from_preset("7b")
    one, _ = activation_bytes(model, micro_batch=1, seq_len=1024, checkpointing=True)
    two, _ = activation_bytes(model, micro_batch=2, seq_len=1024, checkpointing=True)
    assert two == pytest.approx(2 * one)


# ------------------------------------------------------------------- memory
def test_lora_fits_where_full_finetuning_does_not() -> None:
    common = dict(
        goal="sft", model=model_from_preset("7b"), hyper=HyperParams(micro_batch_size=1, seq_len=2048),
        gpu_count=1, gpu_mem_gb=24.0, peak_tflops_bf16=82.6, train_tokens=1_000_000,
    )
    full = estimate(method="full", **common)
    qlora = estimate(method="qlora", **common)
    assert not full.fits
    assert qlora.fits
    assert qlora.vram_per_gpu_gb < full.vram_per_gpu_gb


def test_qlora_uses_less_memory_than_bf16_lora() -> None:
    common = dict(
        goal="sft", model=model_from_preset("13b"),
        hyper=HyperParams(micro_batch_size=1, seq_len=1024, lora_r=16),
        gpu_count=1, gpu_mem_gb=48.0, peak_tflops_bf16=155.0, train_tokens=1_000_000,
    )
    lora = estimate(method="lora", **common)
    qlora = estimate(method="qlora", **common)
    assert qlora.vram_per_gpu_gb < lora.vram_per_gpu_gb


def test_zero_stages_shard_model_state() -> None:
    common = dict(
        goal="sft", method="full", model=model_from_preset("7b"),
        hyper=HyperParams(micro_batch_size=1, seq_len=1024),
        gpu_count=8, gpu_mem_gb=80.0, peak_tflops_bf16=312.0, train_tokens=1_000_000,
    )
    stages = [estimate(zero_stage=stage, **common).vram_per_gpu_gb for stage in (0, 1, 2, 3)]
    assert stages == sorted(stages, reverse=True), "more sharding must mean less memory per GPU"
    assert stages[3] < stages[0] / 2


def test_multi_gpu_divides_the_work() -> None:
    base = dict(
        goal="sft", method="full", model=model_from_preset("3b"),
        hyper=HyperParams(micro_batch_size=1, seq_len=2048),
        gpu_mem_gb=80.0, peak_tflops_bf16=312.0, train_tokens=1_000_000_000, zero_stage=3,
    )
    one = estimate(gpu_count=1, **base)
    eight = estimate(gpu_count=8, **base)
    assert eight.wall_clock_hours == pytest.approx(one.wall_clock_hours / 8, rel=0.35)
    assert eight.total_gpu_hours == pytest.approx(one.total_gpu_hours, rel=0.4)


def test_chunked_loss_avoids_the_logits_cliff() -> None:
    common = dict(
        goal="cpt", method="lora", model=model_from_preset("7b"),
        hyper=HyperParams(micro_batch_size=1, seq_len=8192),
        gpu_count=1, gpu_mem_gb=80.0, peak_tflops_bf16=312.0, train_tokens=1_000_000,
    )
    chunked = estimate(chunked_loss=True, **common)
    plain = estimate(chunked_loss=False, **common)
    assert plain.vram_per_gpu_gb > chunked.vram_per_gpu_gb
    assert any("chunked" in warning.lower() for warning in plain.warnings)


def test_non_flash_attention_is_slower() -> None:
    common = dict(
        goal="sft", method="lora", model=model_from_preset("7b"),
        hyper=HyperParams(micro_batch_size=1, seq_len=2048),
        gpu_count=1, gpu_mem_gb=80.0, peak_tflops_bf16=312.0, train_tokens=1_000_000,
    )
    fast = estimate(**common)
    slow = estimate(hyper=HyperParams(micro_batch_size=1, seq_len=2048, flash_attention=False), **{k: v for k, v in common.items() if k != "hyper"})
    assert slow.wall_clock_hours > fast.wall_clock_hours


def test_estimates_include_a_basis() -> None:
    result = build().resource
    assert len(result.basis) >= 4
    assert any("MFU" in line for line in result.basis)
    assert any("activation" in line.lower() for line in result.basis)


def test_max_micro_batch_is_monotonic() -> None:
    model = model_from_preset("7b")
    small = max_micro_batch(
        goal="sft", method="qlora", model=model, hyper=HyperParams(seq_len=2048),
        gpu_count=1, gpu_mem_gb=24.0, peak_tflops_bf16=82.6,
    )
    big = max_micro_batch(
        goal="sft", method="qlora", model=model, hyper=HyperParams(seq_len=2048),
        gpu_count=1, gpu_mem_gb=80.0, peak_tflops_bf16=312.0,
    )
    assert 1 <= small <= big


# ----------------------------------------------------------- method decisions
def test_small_model_on_a_big_gpu_gets_full_finetuning() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb")
    assert pb.method == "full"
    assert pb.zero_stage == 0, "a single GPU needs no sharding"


def test_seven_b_can_still_use_bf16_lora_on_a_24gb_card() -> None:
    """bf16 LoRA of a 7B fits in 24GB once the weights are the only resident thing."""
    pb = build(preset="7b", gpu="rtx4090-24gb", seq_len=2048)
    assert pb.method == "lora"
    assert pb.resource.fits
    assert pb.request.hyper.micro_batch_size <= 8


def test_thirteen_b_on_a_consumer_gpu_falls_back_to_qlora() -> None:
    pb = build(preset="13b", gpu="rtx4090-24gb", seq_len=2048)
    assert pb.method == "qlora"
    assert pb.resource.fits


def test_full_finetuning_shards_only_as_much_as_it_must() -> None:
    """Less sharding means less communication, so the planner prefers the smallest stage."""
    many = build(preset="7b", gpu="a100-80gb", gpus=8, seq_len=2048)
    few = build(preset="7b", gpu="a100-80gb", gpus=2, seq_len=2048)
    assert many.method == "full" and few.method == "full"
    assert many.zero_stage <= few.zero_stage
    assert few.zero_stage > 0, "2 GPUs cannot hold a 7B full fine-tune without sharding"


def test_no_sharding_when_one_gpu_is_enough() -> None:
    pb = build(preset="0.5b", gpu="a100-80gb", gpus=4)
    assert pb.zero_stage == 0


def test_requested_method_is_respected() -> None:
    request = PlanRequest(
        goal="sft",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="tatsu-lab/alpaca")],
        hardware=HardwareSpec(gpu="a100-80gb", count=1),
        method="lora",
    )
    pb = plan(request, resolve_datasets=False)
    assert pb.method == "lora"
    assert "you asked" in pb.method_reason
    assert "fits" in pb.method_reason


def test_decide_method_reports_numbers_in_the_reason() -> None:
    method, stage, reason = decide_method(
        goal="sft", model=model_from_preset("7b"), hyper=HyperParams(seq_len=2048),
        gpu_mem_gb=80.0, gpu_count=1, peak_tflops_bf16=312.0, gpu_label="A100 80GB",
    )
    assert method in ("full", "lora", "qlora")
    assert "GiB" in reason


def test_pretrain_is_always_from_scratch() -> None:
    pb = build(goal="pretrain", preset="0.5b", gpu="a100-80gb")
    assert pb.method == "from-scratch"


# ----------------------------------------------------------------- batch math
def test_global_batch_is_the_product_of_its_parts() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb")
    hyper = pb.request.hyper
    assert pb.global_batch_size == hyper.micro_batch_size * hyper.grad_accum
    assert pb.global_batch_tokens == pb.global_batch_size * hyper.seq_len


def test_auto_batch_sizing_respects_the_gpu_budget() -> None:
    """Nobody set a batch size, so the planner had to pick one that fits."""
    pb = build(preset="7b", gpu="rtx4090-24gb")
    assert pb.request.hyper.micro_batch_size >= 1
    assert pb.resource.fits
    assert pb.global_batch_tokens >= 1024


def test_explicit_micro_batch_is_not_second_guessed() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb", micro_batch_size=3, grad_accum=5)
    assert pb.request.hyper.micro_batch_size == 3
    assert pb.request.hyper.grad_accum == 5
    assert pb.global_batch_size == 15


def test_sequence_length_is_capped_by_the_model() -> None:
    pb = build(preset="7b", seq_len=999_999)
    assert pb.request.hyper.seq_len <= pb.request.model.max_seq_len


def test_steps_follow_from_tokens_and_batch() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb", epochs=1.0, micro_batch_size=2, grad_accum=4)
    expected = -(-pb.train_tokens // pb.global_batch_tokens)  # ceil
    assert pb.total_steps == expected
    assert pb.request.hyper.max_steps == pb.total_steps, "the config must pin max_steps"


def test_warmup_is_a_fraction_of_steps() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb", epochs=1.0, micro_batch_size=1, grad_accum=1)
    expected = max(1, int(pb.total_steps * pb.request.hyper.warmup_ratio))
    assert pb.warmup_steps == expected
    assert 0 < pb.warmup_steps < pb.total_steps


def test_explicit_max_steps_wins() -> None:
    pb = build(preset="1.5b", gpu="a100-80gb", max_steps=100)
    assert pb.total_steps == 100


# --------------------------------------------------------- goal sanity checks
def test_sft_on_a_raw_corpus_warns_and_clamps() -> None:
    request = PlanRequest(
        goal="sft",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="HuggingFaceFW/fineweb-edu", task="text", est_tokens=2_400_000_000_000)],
        hardware=HardwareSpec(gpu="a100-80gb", count=1),
    )
    pb = plan(request, resolve_datasets=False)
    assert pb.train_tokens <= 1_000_000_000
    assert any("pretraining corpora" in w for w in pb.warnings)
    assert any("clamped" in w for w in pb.warnings)


def test_pretraining_budget_uses_chinchilla() -> None:
    request = PlanRequest(
        goal="pretrain",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="HuggingFaceFW/fineweb-edu", task="text", est_tokens=2_400_000_000_000)],
        hardware=HardwareSpec(gpu="a100-80gb", count=8),
    )
    pb = plan(request, resolve_datasets=False)
    chinchilla = 20 * pb.request.model.est_params()
    assert pb.train_tokens == pytest.approx(chinchilla, rel=0.05)


def test_dpo_without_preference_data_warns() -> None:
    request = PlanRequest(
        goal="dpo",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="tatsu-lab/alpaca", task="sft", est_tokens=10_000_000)],
        hardware=HardwareSpec(gpu="a100-80gb", count=1),
    )
    pb = plan(request, resolve_datasets=False)
    assert any("preference" in w for w in pb.warnings)


def test_empty_dataset_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="dataset"):
        plan(PlanRequest(goal="sft", datasets=[]), resolve_datasets=False)


def test_goal_defaults_survive_a_bare_request() -> None:
    """A request that only names a goal and datasets still gets sane hyper-params."""
    request = PlanRequest(
        goal="dpo",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="Anthropic/hh-rlhf", task="preference")],
        hardware=HardwareSpec(gpu="a100-80gb", count=1),
        method="lora",
    )
    pb = plan(request, resolve_datasets=False)
    hyper = pb.request.hyper
    assert hyper.learning_rate == pytest.approx(5e-6)
    assert hyper.epochs == pytest.approx(1.0)
    assert hyper.seq_len == 1024
    assert hyper.dpo_beta == pytest.approx(0.1)


def test_full_finetuning_dpo_uses_a_smaller_step() -> None:
    """DPO on full weights is 10x more sensitive than the LoRA equivalent."""
    lo = plan(
        PlanRequest(
            goal="dpo", model=model_from_preset("1.5b"),
            datasets=[DatasetRef(repo_id="Anthropic/hh-rlhf", task="preference")],
            hardware=HardwareSpec(gpu="a100-80gb", count=1), method="lora",
        ),
        resolve_datasets=False,
    )
    full = plan(
        PlanRequest(
            goal="dpo", model=model_from_preset("1.5b"),
            datasets=[DatasetRef(repo_id="Anthropic/hh-rlhf", task="preference")],
            hardware=HardwareSpec(gpu="a100-80gb", count=1), method="full",
        ),
        resolve_datasets=False,
    )
    assert full.request.hyper.learning_rate < lo.request.hyper.learning_rate


def test_lora_learning_rate_is_higher_than_full_finetuning() -> None:
    lora = build(preset="1.5b", gpu="rtx4090-24gb", seq_len=1024).request.hyper.learning_rate
    full = build(preset="0.5b", gpu="a100-80gb").request.hyper.learning_rate
    assert lora > full


def test_high_dpo_learning_rate_is_flagged() -> None:
    request = PlanRequest(
        goal="dpo",
        model=model_from_preset("1.5b"),
        datasets=[DatasetRef(repo_id="Anthropic/hh-rlhf", task="preference", est_tokens=10_000_000)],
        hardware=HardwareSpec(gpu="rtx4090-24gb", count=1),
        hyper=HyperParams(learning_rate=2e-4),
    )
    pb = plan(request, resolve_datasets=False)
    assert any("high for DPO" in w for w in pb.warnings)


def test_cpu_preset_is_flagged_slow() -> None:
    pb = build(preset="0.5b", gpu="cpu")
    assert pb.resource.wall_clock_hours > 1
    assert GPU_PRESETS["cpu"]["peak_tflops_bf16"] < 1


def test_references_are_attached_to_every_plan() -> None:
    for goal in ("sft", "dpo", "cpt", "pretrain"):
        pb = build(goal=goal, preset="0.5b", gpu="a100-80gb")
        titles = " ".join(ref["title"] for ref in pb.references)
        assert "ZeRO" in titles
        assert all(ref["url"].startswith("http") for ref in pb.references)
