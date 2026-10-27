"""Execute the generated project's data pipeline against real datasets.

These tests import the ``train.py`` that LLM Maker writes and run its pure-Python
half for real: a hand-built tokenizer with a chat template, real
``datasets.Dataset`` objects and real rows, through field detection, SFT
rendering, prompt masking, DPO pair splitting and sequence packing.

Torch-dependent pieces (the collators, the trainers) are skipped when PyTorch is
not installed -- see ``tests/generated_loader.py`` for the shim that makes the
import possible.
"""

from __future__ import annotations

import json

import pytest

from generated_loader import load_generated_train, torch_available
from llm_maker.models import DatasetRef, HardwareSpec, HyperParams, PlanRequest
from llm_maker.planner import plan
from llm_maker.recipes import model_from_preset
from llm_maker.scaffold import write_project

DATASETS_AVAILABLE = True
try:
    from datasets import Dataset
except ImportError:  # pragma: no cover
    DATASETS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DATASETS_AVAILABLE, reason="the `datasets` package is needed to run the generated pipeline"
)

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ '<|' + m['role'] + '|>' + m['content'] }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


@pytest.fixture(scope="module")
def train_module(tmp_path_factory):
    """A generated SFT project, imported as a module."""
    directory = tmp_path_factory.mktemp("generated")
    built = plan(
        PlanRequest(
            goal="sft",
            model=model_from_preset("0.5b"),
            datasets=[DatasetRef(repo_id="tatsu-lab/alpaca", task="sft")],
            hardware=HardwareSpec(gpu="a100-80gb", count=1),
            hyper=HyperParams(seq_len=64, epochs=1.0),
            project_name="pipeline-test",
        ),
        resolve_datasets=False,
    )
    write_project(built, directory, overwrite=True)
    return load_generated_train(directory / "pipeline-test")


@pytest.fixture(scope="module")
def tokenizer():
    """A real, tiny tokenizer with a chat template -- built offline."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = ["<unk>", "<s>", "</s>", "<pad>"]
    words = [
        "What", "is", "the", "capital", "of", "France", "Paris", "Explain", "recursion",
        "in", "one", "sentence", "a", "function", "that", "calls", "itself", "Human:",
        "Assistant:", "How", "are", "you", "I", "am", "fine", "thanks", "Write", "code",
        "Always", "test", "your", "work", "Why", "sky", "blue", "Answer", "briefly",
        "good", "bad", "Refuse", "politely", "Sure", "here", "it", "is",
    ]
    vocab += sorted({word for text in words for word in text.split()})
    vocab = list(dict.fromkeys(vocab))
    model = WordLevel(vocab={token: index for index, token in enumerate(vocab)}, unk_token="<unk>")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


# ------------------------------------------------------------ field detection
def test_finds_columns_in_a_prompt_completion_dataset(train_module) -> None:
    row = {"instruction": "Explain recursion", "input": "", "output": "A function that calls itself"}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    assert fields["prompt"] == "instruction"
    assert fields["completion"] == "output"


def test_finds_sharegpt_conversation_columns(train_module) -> None:
    row = {"id": "x", "conversations": [{"from": "human", "value": "How are you"}]}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    assert fields["messages"] == "conversations"


def test_finds_preference_columns(train_module) -> None:
    row = {"prompt": "Why is the sky blue", "chosen": "Answer briefly", "rejected": "Refuse politely"}
    fields = train_module.detect_fields(row, {"fields": {}}, "dpo")
    assert (fields["chosen"], fields["rejected"], fields["prompt"]) == ("chosen", "rejected", "prompt")


def test_config_field_mapping_wins_over_heuristics(train_module) -> None:
    row = {"question": "a", "answer": "b", "extra": "c"}
    entry = {"fields": {"prompt_field": "extra", "completion_field": "answer"}}
    fields = train_module.detect_fields(row, entry, "sft")
    assert fields["prompt"] == "extra"
    assert fields["completion"] == "answer"


def test_unknown_columns_yield_nothing_rather_than_a_wrong_guess(train_module) -> None:
    fields = train_module.detect_fields({"alpha": 1, "beta": 2}, {"fields": {}}, "sft")
    assert fields["prompt"] is None
    assert fields["completion"] is None


# --------------------------------------------------------------- SFT rendering
def test_chat_template_is_applied_to_prompt_completion_rows(train_module, tokenizer) -> None:
    row = {"instruction": "Explain recursion", "input": "", "output": "A function that calls itself"}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    text, messages = train_module.build_text_and_messages(row, fields, tokenizer, "sft")
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    assert "<|user|>" in text and "<|assistant|>" in text


def test_input_column_is_appended_to_the_prompt(train_module, tokenizer) -> None:
    row = {"instruction": "What is", "input": "the capital of France", "output": "Paris"}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    _, messages = train_module.build_text_and_messages(row, fields, tokenizer, "sft")
    assert "the capital of France" in messages[0]["content"]
    assert messages[1]["content"] == "Paris"


def test_sharegpt_turns_are_normalised(train_module, tokenizer) -> None:
    row = {"conversations": [
        {"from": "human", "value": "How are you"},
        {"from": "gpt", "value": "I am fine thanks"},
    ]}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    _, messages = train_module.build_text_and_messages(row, fields, tokenizer, "sft")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "I am fine thanks"


def test_tokeniser_masks_the_prompt_when_asked(train_module, tokenizer) -> None:
    row = {"instruction": "Explain recursion", "output": "A function that calls itself"}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    cfg = {"goal": "sft", "hyperparameters": {"mask_prompt": False}}

    unmasked = train_module.make_sft_tokeniser(cfg, fields, tokenizer, 64)(row)
    assert all(label != -100 for label in unmasked["labels"])

    cfg["hyperparameters"]["mask_prompt"] = True
    masked = train_module.make_sft_tokeniser(cfg, fields, tokenizer, 64)(row)
    assert masked["labels"][0] == -100
    assert masked["labels"][0] != masked["input_ids"][0]
    supervised = sum(1 for label in masked["labels"] if label != -100)
    assert 0 < supervised < len(masked["labels"])


def test_long_rows_are_truncated_to_seq_len(train_module, tokenizer) -> None:
    row = {"instruction": " ".join(["Explain recursion"] * 200), "output": "A function that calls itself"}
    fields = train_module.detect_fields(row, {"fields": {}}, "sft")
    cfg = {"goal": "sft", "hyperparameters": {}}
    out = train_module.make_sft_tokeniser(cfg, fields, tokenizer, 16)(row)
    assert len(out["input_ids"]) <= 16
    assert len(out["labels"]) == len(out["input_ids"])


def test_plain_text_rows_pass_through_for_corpus_training(train_module, tokenizer) -> None:
    row = {"text": "Explain recursion in one sentence"}
    fields = train_module.detect_fields(row, {"fields": {}}, "cpt")
    cfg = {"goal": "cpt", "hyperparameters": {}}
    out = train_module.make_sft_tokeniser(cfg, fields, tokenizer, 64)(row)
    assert out["input_ids"]
    assert out["labels"] == out["input_ids"], "corpus training supervises every token"


# ------------------------------------------------------------- DPO rendering
def test_dpo_pairs_are_split_on_the_prompt(train_module, tokenizer) -> None:
    row = {"prompt": "Why is the sky blue", "chosen": "Answer briefly", "rejected": "Refuse politely"}
    fields = train_module.detect_fields(row, {"fields": {}}, "dpo")
    out = train_module.tokenise_dpo(row, fields, tokenizer, 128)
    assert out["prompt_ids"], "the shared prompt must be kept as context"
    assert out["chosen_ids"] and out["rejected_ids"]
    assert out["chosen_ids"] != out["rejected_ids"]
    assert out["chosen_ids"][-1] == tokenizer.eos_token_id, "completions end with EOS"


def test_dpo_recovers_a_prompt_from_full_transcripts(train_module, tokenizer) -> None:
    """HH-RLHF-style rows have no prompt column: only the shared prefix."""
    shared = "Human: Why is the sky blue\nAssistant:"
    row = {"chosen": shared + " Answer briefly", "rejected": shared + " Refuse politely"}
    fields = train_module.detect_fields(row, {"fields": {}}, "dpo")
    out = train_module.tokenise_dpo(row, fields, tokenizer, 128)
    assert out["prompt_ids"], "the common transcript prefix should become the prompt"
    assert "Answer briefly" not in tokenizer.decode(out["prompt_ids"])


def test_dpo_without_any_shared_text_still_produces_a_pair(train_module, tokenizer) -> None:
    row = {"chosen": "Answer briefly", "rejected": "Refuse politely"}
    fields = train_module.detect_fields(row, {"fields": {}}, "dpo")
    out = train_module.tokenise_dpo(row, fields, tokenizer, 128)
    assert out["prompt_ids"] == []
    assert len(out["chosen_ids"]) > 0 and len(out["rejected_ids"]) > 0


def test_longest_common_prefix_helper(train_module) -> None:
    # trims back to the last sentence/line boundary so the prompt reads naturally
    assert train_module.longest_common_prefix("abc. def", "abc. xyz") == "abc."
    assert train_module.longest_common_prefix("totally", "different") == ""


def test_chat_ids_normalises_the_return_type(train_module, tokenizer) -> None:
    """transformers 4.x returns a list, 5.x a BatchEncoding dict -- both must work."""
    ids = train_module.chat_ids(tokenizer, [{"role": "user", "content": "Explain recursion"}])
    assert ids and all(isinstance(token, int) for token in ids)
    assert train_module.chat_ids(tokenizer, []) == [], "an empty conversation is empty, not a crash"


# ------------------------------------------------------------------ packing
def test_packing_produces_full_length_blocks(train_module, tokenizer) -> None:
    rows = {"input_ids": [list(range(1, 6))] * 10, "labels": [list(range(1, 6))] * 10}
    dataset = Dataset.from_dict(rows)
    packed = train_module.pack_sequences(dataset, tokenizer, 8, tokenizer.eos_token_id)
    assert len(packed) > 0
    lengths = {len(example) for example in packed["input_ids"]}
    assert lengths == {8}, "every packed block should be exactly seq_len"
    assert packed.column_names == ["input_ids", "labels", "attention_mask"]
    for labels, ids in zip(packed["labels"], packed["input_ids"]):
        assert labels == ids, "packing supervises every token, so labels mirror the inputs"


def test_packing_respects_the_eos_boundary(train_module, tokenizer) -> None:
    rows = {"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]}
    dataset = Dataset.from_dict(rows)
    packed = train_module.pack_sequences(dataset, tokenizer, 4, 99)
    assert packed["input_ids"][0] == [1, 2, 3, 99], "EOS is inserted between documents"


# ------------------------------------------------------- dataset mixing
def test_mixing_concatenates_when_asked(train_module) -> None:
    left = Dataset.from_dict({"a": [1, 2]})
    right = Dataset.from_dict({"a": [3]})
    cfg = {"datasets": [{"weight": 1}, {"weight": 1}], "mixture": {"strategy": "concat"}}
    assert len(train_module.mix_datasets(cfg, [left, right])) == 3


def test_mixing_single_dataset_is_a_passthrough(train_module) -> None:
    only = Dataset.from_dict({"a": [1, 2, 3]})
    cfg = {"datasets": [{"weight": 1}], "mixture": {"strategy": "weighted"}}
    assert train_module.mix_datasets(cfg, [only]) is only


def test_weighted_mixing_keeps_both_sources(train_module) -> None:
    left = Dataset.from_dict({"source": ["left"] * 20})
    right = Dataset.from_dict({"source": ["right"] * 20})
    cfg = {
        "datasets": [{"weight": 3}, {"weight": 1}],
        "mixture": {"strategy": "weighted"},
        "hyperparameters": {"seed": 0},
    }
    mixed = train_module.mix_datasets(cfg, [left, right])
    assert set(mixed["source"]) == {"left", "right"}
    assert len(mixed) >= 40, "all_exhausted should consume both streams"


# --------------------------------------------------- loading from local files
def test_local_dataset_directory_loads(train_module, tmp_path) -> None:
    """`load_dataset_entry` must work for any path `datasets` accepts."""
    directory = tmp_path / "tiny-dataset"
    directory.mkdir()
    rows = [
        {"instruction": "Explain recursion", "output": "A function that calls itself"},
        {"instruction": "Why is the sky blue", "output": "Answer briefly"},
    ]
    (directory / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    entry = {"repo_id": str(directory), "split": "train"}
    loaded = train_module.load_dataset_entry(entry, None)
    assert len(loaded) == 2
    fields = train_module.detect_fields(loaded[0], entry, "sft")
    assert fields["prompt"] == "instruction"


def test_load_respects_a_row_cap(train_module, tmp_path) -> None:
    directory = tmp_path / "capped"
    directory.mkdir()
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(50)]
    (directory / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = {"datasets": [{"repo_id": str(directory), "split": "train"}], "limits": {"max_train_samples": 5}}
    parts, _ = train_module.load_raw_parts(cfg, None)
    assert len(parts[0][1]) == 5


def test_load_splits_off_a_validation_set(train_module, tmp_path) -> None:
    directory = tmp_path / "with-val"
    directory.mkdir()
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
    (directory / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = {
        "datasets": [{"repo_id": str(directory), "split": "train", "val_ratio": 0.2}],
        "limits": {},
    }
    parts, validations = train_module.load_raw_parts(cfg, None)
    assert len(validations) == 1
    assert len(parts[0][1]) + len(validations[0]) == 100
    assert len(validations[0]) == 20


# ------------------------------------------------------------------ collators
@pytest.mark.skipif(not torch_available(), reason="needs a real PyTorch for tensors")
def test_sft_collator_pads_and_ignores_padding(train_module) -> None:
    import torch

    collator = train_module.SFTDataCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[1, 2, 3], [4, 5, -100]]
    assert batch["labels"].dtype == torch.long


@pytest.mark.skipif(not torch_available(), reason="needs a real PyTorch for tensors")
def test_dpo_collator_masks_the_prompt_tokens(train_module) -> None:
    collator = train_module.DPOCollator(pad_token_id=0)
    batch = collator([
        {"prompt_ids": [7, 7], "chosen_ids": [1, 2], "rejected_ids": [3, 4, 5]},
    ])
    assert batch["chosen_ids"].tolist()[0] == [7, 7, 1, 2]
    assert batch["chosen_mask"].tolist()[0] == [0, 0, 1, 1], "only completions count for the loss"
    assert batch["rejected_mask"].tolist()[0] == [0, 0, 1, 1, 1]


@pytest.mark.skipif(not torch_available(), reason="needs a real PyTorch for tensors")
def test_dpo_collator_truncates_to_the_longest_pair(train_module) -> None:
    collator = train_module.DPOCollator(pad_token_id=0)
    batch = collator([
        {"prompt_ids": [1], "chosen_ids": [2, 3], "rejected_ids": [4]},
        {"prompt_ids": [1], "chosen_ids": [5], "rejected_ids": [6, 7, 8]},
    ])
    assert batch["chosen_ids"].shape[1] == 3
    assert batch["rejected_ids"].shape[1] == 4


@pytest.mark.skipif(not torch_available(), reason="needs a real PyTorch for tensors")
def test_dpo_collator_keeps_prompt_context_in_attention_mask(train_module) -> None:
    """DPO scores answers only, but model forward passes must see the prompt."""
    collator = train_module.DPOCollator(pad_token_id=0, max_length=4)
    batch = collator([{"prompt_ids": [7, 8, 9], "chosen_ids": [1, 2], "rejected_ids": [3]}])
    # The context is retained (up to max_length), while only the completion
    # token contributes to DPO's log-probability.
    assert batch["chosen_ids"].tolist()[0] == [7, 8, 9, 1]
    assert batch["chosen_mask"].tolist()[0] == [0, 0, 0, 1]
    assert batch["chosen_attention_mask"].tolist()[0] == [1, 1, 1, 1]
    assert batch["rejected_attention_mask"].tolist()[0] == [1, 1, 1, 1]
