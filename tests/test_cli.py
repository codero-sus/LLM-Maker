"""Command line interface."""

from __future__ import annotations

import json

import pytest

from llm_maker.cli import build_parser, main


def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_search_prints_ranked_results(capsys) -> None:
    code, out, _ = run(["--offline", "search", "alpaca", "--limit", "3"], capsys)
    assert code == 0
    assert "tatsu-lab/alpaca" in out
    assert "snapshot" in out


def test_search_json_output_is_parseable(capsys) -> None:
    code, out, _ = run(["--offline", "search", "code", "--json", "--limit", "2"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["items"]
    assert payload["mode"] in ("snapshot", "cache")
    for item in payload["items"]:
        assert "id" in item and "est_tokens" in item


def test_search_with_filters(capsys) -> None:
    code, out, _ = run(["--offline", "search", "--task", "preference", "--limit", "5"], capsys)
    assert code == 0
    assert "hh-rlhf" in out or "dpo" in out or "preference" in out.lower()


def test_show_prints_dataset_details(capsys) -> None:
    code, out, _ = run(["--offline", "show", "HuggingFaceH4/ultrachat_200k"], capsys)
    assert code == 0
    assert "ultrachat_200k" in out
    assert "tasks" in out
    assert "est tokens" in out


def test_show_json(capsys) -> None:
    code, out, _ = run(["--offline", "show", "openai/gsm8k", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["id"] == "openai/gsm8k"
    assert payload["est_tokens"]


def test_show_unknown_dataset_exits_nonzero(capsys) -> None:
    code, _, err = run(["--offline", "show", "nobody/nothing"], capsys)
    assert code == 1
    assert "not in the catalog" in err


def test_plan_prints_the_estimate_table(capsys) -> None:
    code, out, _ = run(
        ["--offline", "plan", "--goal", "sft", "--preset", "1.5b", "--gpu", "a100-80gb",
         "--datasets", "tatsu-lab/alpaca"],
        capsys,
    )
    assert code == 0
    for fragment in ("goal", "method", "VRAM per GPU", "throughput", "wall clock", "estimator basis"):
        assert fragment in out
    assert "tatsu-lab/alpaca" in out


def test_plan_json_includes_the_whole_plan(capsys) -> None:
    code, out, _ = run(
        ["--offline", "plan", "--goal", "dpo", "--preset", "7b", "--gpu", "rtx4090-24gb",
         "--datasets", "Anthropic/hh-rlhf", "--json"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["goal_label"] == "Preference alignment (DPO)"
    assert payload["method"] in ("lora", "qlora", "full")
    assert payload["request"]["hyper"]["learning_rate"] > 0
    assert payload["zero_stage"] in (0, 1, 2, 3)


def test_plan_needs_datasets(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--offline", "plan", "--goal", "sft"])


def test_make_writes_and_reports_a_project(tmp_path, capsys) -> None:
    code, out, _ = run(
        ["--offline", "make", "--goal", "sft", "--preset", "0.5b", "--gpu", "a100-80gb",
         "--datasets", "tatsu-lab/alpaca", "--out", str(tmp_path), "--name", "cli-test"],
        capsys,
    )
    assert code == 0
    directory = tmp_path / "cli-test"
    assert (directory / "train.py").exists()
    assert (directory / "config.yaml").exists()
    assert (directory / "README.md").exists()
    assert "python train.py --dry-run" in out
    # the generated script must be syntactically valid Python
    import ast

    ast.parse((directory / "train.py").read_text())


def test_make_does_not_clobber_hand_edits_by_default(tmp_path, capsys) -> None:
    argv = ["--offline", "make", "--goal", "sft", "--preset", "0.5b", "--gpu", "a100-80gb",
            "--datasets", "tatsu-lab/alpaca", "--out", str(tmp_path), "--name", "keep-me"]
    run(argv, capsys)
    edited = tmp_path / "keep-me" / "train.py"
    edited.write_text("# my careful edit\n")
    run(argv, capsys)
    assert edited.read_text() == "# my careful edit\n"
    # the second run went to a fresh directory instead
    assert (tmp_path / "keep-me-2" / "train.py").exists()


def test_make_force_overwrites(tmp_path, capsys) -> None:
    argv = ["--offline", "make", "--goal", "sft", "--preset", "0.5b", "--gpu", "a100-80gb",
            "--datasets", "tatsu-lab/alpaca", "--out", str(tmp_path), "--name", "forced"]
    run(argv, capsys)
    (tmp_path / "forced" / "train.py").write_text("# old\n")
    run(argv + ["--force"], capsys)
    assert "load_dataset" in (tmp_path / "forced" / "train.py").read_text()


def test_recipes_prints_files_to_stdout(capsys) -> None:
    code, out, _ = run(
        ["--offline", "recipes", "--goal", "cpt", "--preset", "0.5b", "--gpu", "a100-80gb",
         "--datasets", "HuggingFaceFW/fineweb-edu"],
        capsys,
    )
    assert code == 0
    assert "# train.py" in out
    assert "# config.yaml" in out
    assert "interleave_datasets" in out


def test_presets_lists_everything(capsys) -> None:
    code, out, _ = run(["presets"], capsys)
    assert code == 0
    for fragment in ("sft", "dpo", "cpt", "pretrain", "7b", "a100-80gb", "TFLOP/s"):
        assert fragment in out


def test_catalog_reports_connectivity(capsys) -> None:
    code, out, _ = run(["--offline", "catalog"], capsys)
    assert code == 0
    assert "online   : False" in out
    assert "locally known datasets" in out


def test_cli_overrides_reach_the_plan(capsys, tmp_path) -> None:
    code, out, _ = run(
        ["--offline", "plan", "--goal", "sft", "--preset", "1.5b", "--gpu", "a100-80gb",
         "--datasets", "tatsu-lab/alpaca", "--lr", "3e-5", "--epochs", "2", "--seq-len", "1024",
         "--micro-batch", "4", "--grad-accum", "2", "--json"],
        capsys,
    )
    assert code == 0
    hyper = json.loads(out)["request"]["hyper"]
    assert hyper["learning_rate"] == pytest.approx(3e-5)
    assert hyper["epochs"] == pytest.approx(2.0)
    assert hyper["seq_len"] == 1024
    assert hyper["micro_batch_size"] == 4
    assert hyper["grad_accum"] == 2


def test_trainer_flags_reach_the_generated_plan(capsys) -> None:
    code, out, _ = run(
        ["--offline", "plan", "--goal", "sft", "--preset", "0.5b", "--gpu", "a100-80gb",
         "--datasets", "tatsu-lab/alpaca", "--trainer", "optimized", "--torch-compile",
         "--compile-mode", "max-autotune", "--no-tf32", "--json"],
        capsys,
    )
    assert code == 0
    hyper = json.loads(out)["request"]["hyper"]
    assert hyper["trainer"] == "optimized"
    assert hyper["torch_compile"] is True
    assert hyper["compile_mode"] == "max-autotune"
    assert hyper["tf32"] is False


def test_method_override_is_honoured(capsys) -> None:
    code, out, _ = run(
        ["--offline", "plan", "--goal", "sft", "--method", "qlora", "--preset", "7b",
         "--gpu", "rtx4090-24gb", "--datasets", "tatsu-lab/alpaca", "--json"],
        capsys,
    )
    assert code == 0
    assert json.loads(out)["method"] == "qlora"


def test_bad_goal_is_rejected_by_the_parser(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--offline", "plan", "--goal", "nonsense", "--datasets", "x/y"])


def test_unknown_dataset_in_a_plan_still_produces_a_plan(capsys) -> None:
    """A typo should surface when `datasets` loads the repo, with a warning here."""
    code, out, _ = run(
        ["--offline", "plan", "--goal", "sft", "--preset", "0.5b", "--gpu", "a100-80gb",
         "--datasets", "somebody/typo-ed-dataset", "--json"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert any("not in the Hub catalog" in warning for warning in payload["warnings"])
    assert payload["mixture"][0]["repo_id"] == "somebody/typo-ed-dataset"
