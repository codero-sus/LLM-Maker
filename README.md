# LLM Maker

**A planner for language models — not a trainer.**

LLM Maker browses the [Hugging Face datasets](https://huggingface.co/datasets) list, turns a
selection into a concrete training plan, and generates a runnable
`datasets` + `transformers` project. It answers the questions that come *before* you
rent GPUs:

- which datasets on the Hub actually fit what I'm trying to build?
- will this run fit on the hardware I have (VRAM, per GPU, per method)?
- how many optimizer steps, how many tokens, how many GPU-hours?
- what does a correct training script for *this* goal and *this* data actually look like?

And it stops there. It does not train, upload, or host anything for you: the output is a
normal project you read, edit and launch yourself.

```
   Hugging Face Hub  ──►  catalog search  ──►  plan + estimates  ──►  runnable project
     /api/datasets        filters & facets       VRAM / steps / time    train.py, config.yaml
```

---

## Quick start

```bash
pip install -e .                 # the planner itself: httpx, fastapi, uvicorn, pyyaml
llm-maker serve                  # web UI on http://localhost:8000  (API docs at /docs)
```

Or drive it entirely from the terminal:

```bash
# 1. find datasets
llm-maker search "hindi instruction" --task instruction --limit 10
llm-maker show HuggingFaceH4/ultrachat_200k --preview 2

# 2. plan a run (prints the estimate table, writes nothing)
llm-maker plan --goal sft --preset 7b --gpu a100-80gb --gpus 2 \
  --datasets tatsu-lab/alpaca,databricks/databricks-dolly-15k

# 3. generate the project
llm-maker make --goal sft --preset 7b --gpu a100-80gb --gpus 2 \
  --datasets tatsu-lab/alpaca,databricks/databricks-dolly-15k \
  --out ./runs --name my-assistant

cd runs/my-assistant
python data_audit.py        # what the data really looks like, and how many tokens it holds
python train.py --dry-run   # build the pipeline, print one batch
bash run.sh                 # train
```

The planner itself needs no PyTorch. The **generated** project needs the normal stack
(`torch transformers datasets accelerate peft bitsandbytes`), pinned in the
`requirements.txt` it ships with.

---

## What a plan contains

```
$ llm-maker plan --goal dpo --preset 7b --gpu rtx4090-24gb --datasets Anthropic/hh-rlhf
goal        : Preference alignment (DPO) (dpo)
method      : qlora -- 4-bit QLoRA fits: 5.1 GiB per GPU with no sharding, inside the 24 GiB of GeForce RTX 4090 24GB
base model  : mistralai/Mistral-7B-v0.3 (7.24B, preset 7b)
hardware    : 1x rtx4090-24gb (24GB each)

VRAM per GPU: 6.4 GiB (model 4.1 + activations 0.7 + overhead 1.6) -> fits (headroom 17.6 GiB)
throughput  : 1,842 tokens/s/GPU at 28% MFU
wall clock  : 4.31 h (4.31 GPU-hours)
```

Every number is explained. The estimator prints its own basis:

```
· frozen base: 0.5 B/param x 7.25B params (4-bit NF4 base, double-quantised)
· trainable adapters: 36.7M params x 12 B/param
· checkpointed activations: 32 layers x 2*s*b*h (=2x1024x1x4096)
· compute: 25.19 GFLOP/token = 4N (7.25B params) + attention at seq_len 1024 x1.33 recompute
· throughput: 28% MFU of 83 TFLOP/s dense bf16 x 1 GPU(s) = 1,842 tokens/s
· MFU is an assumption, not a measurement: expect the real wall clock to land within
  roughly half to twice this number.
```

The formulas are the published ones, not curve fits:

| Quantity | Source |
|---|---|
| Optimizer/gradient/weight bytes, ZeRO sharding | Rajbhandari et al., *ZeRO* — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) |
| Activation memory `s·b·h·(34 + 5·a·s/h)` | Korthikanti et al., *Reducing Activation Recomputation* — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) |
| 4-bit NF4 base weights ≈ 0.5 B/param | Dettmers et al., *QLoRA* — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| `6N` (full) / `4N` (adapters) FLOPs per token | Kaplan et al. — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361); Hu et al., *LoRA* — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| ~20 tokens/parameter pretraining budget | Hoffmann et al., *Chinchilla* — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |

Each generated project also writes out the papers its defaults came from, with links.

---

## Goals

| Goal | What it does | Typical data |
|---|---|---|
| `sft` | Supervised instruction tuning | instruction/answer pairs, chat transcripts |
| `dpo` | Preference alignment (Direct Preference Optimization) | chosen/rejected pairs |
| `cpt` | Continued pretraining on domain text | plain text corpora |
| `pretrain` | Train a new transformer from scratch | large web/document corpora |

Methods are chosen by the estimator, not by a hardcoded rule: it simulates
full fine-tuning / bf16 LoRA / 4-bit QLoRA at each ZeRO stage and picks the cheapest
combination that actually fits — preferring full fine-tuning over LoRA over QLoRA, and
less sharding over more. An explicit `--method` is always honoured, even when it does not
fit, in which case the plan says so and lists the fixes.

---

## What gets generated

For any plan, `llm-maker make` writes a project like this:

| File | What it is |
|---|---|
| `train.py` | The training script: dataset loading, chat templating, LoRA/QLoRA wiring, `Trainer`, chunked loss, sampling |
| `config.yaml` / `config.json` | Every knob: datasets and field mappings, mixture weights, hyper-parameters, hardware assumptions |
| `data_audit.py` | Prints real rows, detected columns, rendered samples and measured token counts per dataset |
| `merge_adapter.py` | Folds a LoRA adapter into the base weights so vLLM/TGI can serve it |
| `run.sh` | The `accelerate launch` (and DeepSpeed, when the plan assumes sharding) invocation |
| `plan.json` | The full plan and its estimates, machine-readable |
| `README.md` | The plan, the estimates, the warnings and the references |
| `prompts.jsonl` | Prompts used by `--smoke-test` to sanity-check generations |
| `deepspeed_zero*.json` | Only when the plan's VRAM estimate assumed ZeRO sharding |

Design choices worth knowing about:

- **`data_audit.py` first.** Token counts in the plan come from Hub metadata and are
  coarse. The audit streams a sample and measures the real number before you spend GPU
  time.
- **Conservative defaults.** Gradient checkpointing on, FlashAttention-2 requested when
  available, chunked cross-entropy on (the `[batch, seq, vocab]` logits tensor is
  gigabytes at long sequence lengths), reference model shared between policies in
  LoRA-DPO via `disable_adapter()`, and `-100` masking so padding never trains.
- **The config is the plan.** Re-running with a different dataset list is a config edit,
  not a regeneration.

---

## Optional optimized trainer

Generated projects can use either the regular Hugging Face `Trainer` or the
**optimized trainer**. The latter is still an ordinary `transformers.Trainer`
subclass — no opaque launcher — but it uses a few carefully bounded fast paths:

- **chunked causal-LM loss:** projects the final hidden states through `lm_head`
  in 512-token pieces rather than keeping a full `[batch, sequence, vocabulary]`
  logits tensor alive. This is particularly useful at long sequence lengths and
  for 128k-vocabulary models.
- **fused AdamW when available:** automatically selected only for the ordinary
  `adamw_torch` case on CUDA hardware that supports it. QLoRA's paged 8-bit
  optimizer and any explicit optimizer setting are left untouched.
- **TF32 on NVIDIA Ampere or newer:** enabled only for float32 matmuls; bf16/fp16
  weights and gradients are unchanged. Use `--no-tf32` to disable it.
- **optional `torch.compile`:** opt in with `--torch-compile`. The generated
  script compiles only the isolated transformer backbone on the first batch,
  preserving both chunked logits and clean checkpoint keys. It intentionally
  falls back to eager mode for QLoRA, DPO, multi-GPU and ZeRO runs, where graph
  compilation is often counterproductive or incompatible.

The profile is visible and editable in every generated `config.yaml`:

```yaml
trainer:
  backend: optimized       # optimized | standard
  torch_compile: false     # set true, or pass --torch-compile at launch
  compile_mode: reduce-overhead
  tf32: true
```

Choose it while planning:

```bash
# Recommended safe performance profile (the default)
llm-maker make --goal sft --datasets tatsu-lab/alpaca --trainer optimized --out runs

# Baseline/debugging path: plain Transformers Trainer with the same config/data
llm-maker make --goal sft --datasets tatsu-lab/alpaca --trainer standard --out runs

# Generate a single-GPU LoRA/full SFT/CPT project that requests torch.compile
llm-maker make --goal cpt --datasets HuggingFaceFW/fineweb-edu \
  --trainer optimized --torch-compile --compile-mode reduce-overhead --out runs
```

The DPO objective remains custom in both profiles because upstream
`transformers.Trainer` is a causal-LM trainer, not a preference trainer. Its
collator keeps prompt/context **attention masks** separate from completion-only
**loss masks**, so chosen/rejected answers can attend to their prompts while DPO
scores only their completions.

---

## Web UI

`llm-maker serve` starts a single-page app over the same API the CLI uses:

- search the Hub with live task/language/size facets, or fall back to the bundled
  offline snapshot when the Hub is unreachable (every response says which it used)
- add datasets to a mixture and set per-dataset weights
- pick a goal, model shape, GPU and count; the plan and its estimates update
- inspect a dataset's card, splits and first rows via the Hub dataset viewer
- generate the project, read any file in it, or download it as a zip

`GET /docs` is the OpenAPI UI for the JSON API.

---

## Offline behaviour

The catalog has three sources of truth and always tells you which one answered
(`mode`: `hub`, `cache` or `snapshot`):

1. **Live Hub** — `https://huggingface.co/api/datasets`, paged with the `Link: rel="next"`
   cursors the API returns, with row/split metadata from the datasets-server.
2. **Cache** — `~/.cache/llm_maker/catalog_cache.json`, written whenever the Hub answers.
3. **Bundled snapshot** — `llm_maker/data/hf_datasets_snapshot.json`, ~120 datasets across
   instruction, preference, math, code, pretraining, multilingual, Indic, audio and vision
   work. Regenerate with `python tools/build_snapshot.py`.

In offline mode popularity figures are approximate and rounded; they only ever affect
ranking, never training. Set `HF_TOKEN` to see private and gated repositories, and
`LLM_MAKER_OFFLINE=1` to force offline mode.

---

## Python API

```python
from llm_maker import DatasetCatalog, SearchFilters, PlanRequest, plan, write_project

catalog = DatasetCatalog()
results = catalog.search(SearchFilters(search="hindi pretraining", tasks=["pretrain"]), limit=5)
print([d.id for d in results.items], results.mode)

request = PlanRequest.from_dict({
    "goal": "sft",
    "model": {"preset": "7b"},
    "datasets": ["tatsu-lab/alpaca", {"repo_id": "HuggingFaceH4/ultrachat_200k", "weight": 2}],
    "hardware": {"gpu": "a100-80gb", "count": 2},
})
built = plan(request, catalog=catalog)
print(built.method, built.total_steps, built.resource.vram_per_gpu_gb, built.resource.fits)

print(write_project(built, "runs", overwrite=True).project_dir)
```

---

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The suite runs fully offline (no network, no GPU). It covers the estimator's formulas and
their monotonicity, the planner's decisions, the CLI and the HTTP API — and it *executes*
the generated project's data pipeline: it builds a real tokenizer and real `datasets`
objects, then runs field detection, chat-template rendering, prompt masking, preference-pair
splitting, sequence packing and dataset mixing through the generated code
(`tests/test_generated_pipeline.py`). That is how the `apply_chat_template` return-type
change in transformers 5.x was caught, which would otherwise have silently trained on the
string `"input_ids"`.

PyTorch is a multi-gigabyte install, so tests for the tensor-only pieces (the collators)
skip themselves when it is absent; `tests/generated_loader.py` documents the shim used to
import the generated script without it.

---

## Layout

```
llm_maker/
  hub.py          Hub API client (datasets list, detail, files, first rows, cursors)
  catalog.py      live + cached + bundled catalog, search, ranking, facets
  models.py       PlanRequest / TrainingPlan / estimates (plain dataclasses)
  recipes.py      model shapes, GPU table, goal presets, dataset field heuristics
  planner.py      decisions: method, ZeRO stage, batch sizes, step budget, warnings
  estimator.py    VRAM and throughput from published formulas
  codegen.py      the generated project (templates live in templates/)
  scaffold.py     writes a project to disk without clobbering edits
  references.py   the papers the defaults come from
  server.py       FastAPI app: JSON API + web UI
  cli.py          `llm-maker` command line
  templates/      train.py, data_audit.py and merge_adapter.py, as real Python files
  web/            the single-page UI
tools/build_snapshot.py   rebuilds the offline catalog snapshot
tests/                    offline test suite, including the generated pipeline
```

## Licence

MIT. Dataset licences are the dataset owners': every plan links to the Hub page for each
one it uses, and the generated README reminds you to check them before commercial use.
