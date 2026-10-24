"""The generated project: file set, config correctness and template validity."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from llm_maker.codegen import DEFAULT_PROMPTS, render_project, safe_dirname
from llm_maker.models import DatasetRef, HardwareSpec, HyperParams, PlanRequest
from llm_maker.planner import plan
from llm_maker.recipes import MODEL_PRESETS, model_from_preset

REQUIRED_FILES = {
    "train.py", "data_audit.py", "merge_adapter.py", "requirements.txt",
    "config.yaml", "config.json", "plan.json", "run.sh", "README.md",
    "prompts.jsonl", ".gitignore",
}


def make_plan(goal="sft", preset="1.5b", gpu="a100-80gb", gpus=1, datasets=None, **hyper):
    return plan(
        PlanRequest(
            goal=goal,
            model=model_from_preset(preset),
            datasets=datasets
            or [DatasetRef(repo_id="tatsu-lab/alpaca", task="sft", est_tokens=15_000_000)],
            hardware=HardwareSpec(gpu=gpu, count=gpus),
            hyper=HyperParams(**hyper),
            project_name="unit-test",
        ),
        resolve_datasets=False,
    )


@pytest.mark.parametrize("goal", ["sft", "dpo", "cpt", "pretrain"])
def test_every_goal_renders_the_full_file_set(goal: str) -> None:
    files = render_project(make_plan(goal=goal))
    assert REQUIRED_FILES <= set(files)
    for name, content in files.items():
        assert content.strip(), f"{name} is empty"


def test_deepspeed_configs_appear_whenever_the_plan_assumes_sharding() -> None:
    """A plan whose VRAM number assumed ZeRO must ship the config that does it."""
    single = render_project(make_plan(gpu="a100-80gb", gpus=1))
    assert not [name for name in single if name.startswith("deepspeed")]

    eight = render_project(make_plan(preset="7b", gpu="a100-80gb", gpus=8))
    assert f"deepspeed_zero{eight_zero_stage(eight)}.json" in eight
    # only the stages at or above the chosen one are included
    chosen = eight_zero_stage(eight)
    for stage in range(4):
        name = f"deepspeed_zero{stage}.json"
        assert (name in eight) == (stage >= chosen and chosen >= 1)


def eight_zero_stage(files: dict) -> int:
    stages = [
        int(name[len("deepspeed_zero"):-len(".json")])
        for name in files
        if name.startswith("deepspeed_zero")
    ]
    return min(stages) if stages else 0


def test_run_script_uses_deepspeed_when_sharding_is_assumed() -> None:
    script = render_project(make_plan(preset="7b", gpu="a100-80gb", gpus=8))["run.sh"]
    assert "config_file deepspeed_zero" in script
    assert "pip install deepspeed" in script


def test_run_script_stays_plain_ddp_when_nothing_is_sharded() -> None:
    script = render_project(make_plan(preset="0.5b", gpu="a100-80gb", gpus=4))["run.sh"]
    assert "config_file" not in script
    assert "ZeRO-0" in script


def test_train_py_is_valid_python() -> None:
    files = render_project(make_plan())
    ast.parse(files["train.py"])
    ast.parse(files["data_audit.py"])
    ast.parse(files["merge_adapter.py"])


def test_train_py_uses_the_hugging_face_datasets_api() -> None:
    source = render_project(make_plan())["train.py"]
    # The whole point of the project is that it is a plain `datasets` pipeline.
    assert "from datasets import" in source
    assert "load_dataset" in source
    assert "interleave_datasets" in source
    assert "concatenate_datasets" in source
    assert "apply_chat_template" in source
    assert "Trainer" in source


def test_generated_config_is_loadable_yaml_and_json(project) -> None:
    directory, _ = project
    yaml_config = yaml.safe_load((directory / "config.yaml").read_text())
    json_config = json.loads((directory / "config.json").read_text())
    assert yaml_config == json_config
    assert yaml_config["goal"] == "sft"
    assert yaml_config["datasets"][0]["repo_id"] == "tatsu-lab/alpaca"
    assert yaml_config["hyperparameters"]["max_steps"] > 0


def test_config_echoes_the_plan_numbers(project) -> None:
    directory, plan_obj = project
    cfg = yaml.safe_load((directory / "config.yaml").read_text())
    assert cfg["expectations"]["total_steps"] == plan_obj.total_steps
    assert cfg["hyperparameters"]["seq_len"] == plan_obj.request.hyper.seq_len
    assert cfg["hyperparameters"]["micro_batch_size"] == plan_obj.request.hyper.micro_batch_size
    assert cfg["hyperparameters"]["grad_accum"] == plan_obj.request.hyper.grad_accum
    assert cfg["expectations"]["est_vram_per_gpu_gb"] == plan_obj.resource.vram_per_gpu_gb


def test_config_has_no_placeholder_zeros() -> None:
    """Unset hyper-parameters must be resolved before they reach the config."""
    cfg = render_project(make_plan())["config.yaml"]
    data = yaml.safe_load(cfg)
    for key, value in data["hyperparameters"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert value != 0 or key in {"weight_decay", "lora_dropout", "mask_prompt"}, key


def test_output_dir_is_relative_to_the_project() -> None:
    cfg = yaml.safe_load(render_project(make_plan())["config.yaml"])
    assert not Path(cfg["output_dir"]).is_absolute()
    assert cfg["output_dir"].startswith("runs/")


def test_model_shapes_are_copied_into_the_config() -> None:
    cfg = yaml.safe_load(render_project(make_plan(preset="7b"))["config.yaml"])
    shape = MODEL_PRESETS["7b"]
    assert cfg["model"]["n_layer"] == shape["n_layer"]
    assert cfg["model"]["d_model"] == shape["d_model"]
    assert cfg["model"]["ffn_dim"] == shape["ffn_dim"] if "ffn_dim" in cfg["model"] else True


def test_pretrain_config_targets_a_fresh_model() -> None:
    files = render_project(make_plan(goal="pretrain", preset="0.5b"))
    cfg = yaml.safe_load(files["config.yaml"])
    assert cfg["method"] == "from-scratch"
    assert cfg["goal"] == "pretrain"
    assert "AutoConfig" in files["train.py"]


def test_dpo_config_carries_beta_and_preference_fields() -> None:
    plan_obj = make_plan(
        goal="dpo",
        datasets=[DatasetRef(repo_id="Anthropic/hh-rlhf", task="preference", est_tokens=40_000_000)],
    )
    cfg = yaml.safe_load(render_project(plan_obj)["config.yaml"])
    assert cfg["hyperparameters"]["dpo_beta"] == pytest.approx(0.1)
    assert cfg["goal"] == "dpo"
    source = render_project(plan_obj)["train.py"]
    assert "chosen_attention_mask" in source
    assert "rejected_attention_mask" in source


def test_optimized_trainer_profile_is_emitted_and_is_optional() -> None:
    optimized = yaml.safe_load(render_project(make_plan())["config.yaml"])
    assert optimized["trainer"] == {
        "backend": "optimized",
        "torch_compile": False,
        "compile_mode": "reduce-overhead",
        "tf32": True,
    }
    assert optimized["chunked_loss"] is True
    source = render_project(make_plan())["train.py"]
    assert "class OptimizedTrainer(Trainer)" in source
    assert "torch.compile" in source
    assert "preferred_optimizer" in source

    standard = yaml.safe_load(render_project(make_plan(trainer="standard"))["config.yaml"])
    assert standard["trainer"]["backend"] == "standard"
    assert standard["chunked_loss"] is False


def test_compiler_preferences_round_trip_to_the_generated_config() -> None:
    cfg = yaml.safe_load(
        render_project(make_plan(torch_compile=True, compile_mode="max-autotune", tf32=False))["config.yaml"]
    )
    assert cfg["trainer"] == {
        "backend": "optimized",
        "torch_compile": True,
        "compile_mode": "max-autotune",
        "tf32": False,
    }


def test_mixture_weights_are_normalised() -> None:
    plan_obj = make_plan(
        datasets=[
            DatasetRef(repo_id="tatsu-lab/alpaca", weight=3.0, task="sft", est_tokens=10_000_000),
            DatasetRef(repo_id="yahma/alpaca-cleaned", weight=1.0, task="sft", est_tokens=10_000_000),
        ]
    )
    cfg = yaml.safe_load(render_project(plan_obj)["config.yaml"])
    weights = [d["weight"] for d in cfg["datasets"]]
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(0.25)


def test_readme_documents_the_plan(project) -> None:
    directory, plan_obj = project
    readme = (directory / "README.md").read_text()
    assert plan_obj.request.model.base_model in readme
    assert "tatsu-lab/alpaca" in readme
    assert "datasets" in readme
    assert "https://arxiv.org" in readme, "references must be real links"
    assert "data_audit.py" in readme


def test_readme_includes_estimates_and_basis(project) -> None:
    directory, plan_obj = project
    readme = (directory / "README.md").read_text()
    assert f"{plan_obj.resource.vram_per_gpu_gb:.1f}" in readme
    assert "MFU" in readme


def test_run_script_is_a_shell_script(project) -> None:
    directory, _ = project
    script = (directory / "run.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "train.py" in script


def test_multi_gpu_run_script_launches_accelerate() -> None:
    script = render_project(make_plan(preset="7b", gpu="a100-80gb", gpus=8))["run.sh"]
    assert "accelerate launch" in script
    assert "NUM_GPUS=8" in script
    assert "--config_file deepspeed_zero" in script


def test_requirements_cover_the_imports() -> None:
    requirements = render_project(make_plan())["requirements.txt"]
    for package in ("torch", "transformers", "datasets", "accelerate", "peft", "pyyaml"):
        assert package in requirements


def test_prompts_file_is_jsonl() -> None:
    files = render_project(make_plan())
    lines = [line for line in files["prompts.jsonl"].splitlines() if line.strip()]
    assert len(lines) == len(DEFAULT_PROMPTS)
    for line in lines:
        assert json.loads(line)["prompt"]


def test_plan_json_is_complete(project) -> None:
    directory, plan_obj = project
    payload = json.loads((directory / "plan.json").read_text())
    assert payload["total_steps"] == plan_obj.total_steps
    assert payload["warnings"] == plan_obj.warnings
    assert payload["resource"]["basis"]
    assert payload["zero_stage"] == plan_obj.zero_stage


def test_safe_dirname_strips_traversal() -> None:
    assert safe_dirname("../../etc/passwd") == "etc-passwd"
    assert safe_dirname("my cool model!") == "my-cool-model"
    assert safe_dirname("") == "llm-project"


def test_qwen_presets_tie_embeddings() -> None:
    """Qwen2.5 checkpoints tie their embeddings -- the parameter math depends on it."""
    for preset in ("0.5b", "1.5b", "3b"):
        assert MODEL_PRESETS[preset]["tie_embeddings"] is True
