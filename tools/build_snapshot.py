#!/usr/bin/env python3
"""Build the bundled Hugging Face datasets snapshot.

The app talks to the live Hugging Face Hub API (`/api/datasets`) whenever the
network allows it.  When the Hub is unreachable (air-gapped machine, corporate
proxy, offline demo) the catalog falls back to this snapshot so the UI is still
usable.

IMPORTANT: the ``downloads``/``likes`` numbers below are *approximate*,
hand-maintained figures captured for ranking purposes only.  They are rounded,
never presented as exact live metrics, and every consumer is told the entry came
from ``snapshot``.  Descriptions are paraphrased summaries, not Hub text.

Regenerate with:

    python tools/build_snapshot.py

which writes ``llm_maker/data/hf_datasets_snapshot.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

# (id, tasks, size_category, downloads~, likes~, languages, license, gated, description)
# tasks: coarse buckets used by the UI facets.
ENTRIES: list[tuple] = [
    # ---------------------------------------------------------------- instruction
    ("tatsu-lab/alpaca", ["instruction"], "10K<n<100K", 1_900_000, 4_100, ["en"], "cc-by-nc-4.0", False,
     "52k instruction-following examples generated from text-davinci-003. The canonical starter set for supervised fine-tuning."),
    ("yahma/alpaca-cleaned", ["instruction"], "10K<n<100K", 640_000, 1_500, ["en"], "apache-2.0", False,
     "Community-cleaned Alpaca: hallucinated, malformed and instruction-less rows removed. Same schema, better signal."),
    ("databricks/databricks-dolly-15k", ["instruction"], "10K<n<100K", 780_000, 1_300, ["en"], "cc-by-sa-3.0", False,
     "15k human-written instruction/response pairs with a free-form category field, covering brainstorming, closed QA, summarisation and more."),
    ("garage-bAInd/Open-Platypus", ["instruction"], "10K<n<100K", 210_000, 620, ["en"], "other", False,
     "25k curated STEM/reasoning instruction samples from eleven open datasets, deduplicated against common eval sets."),
    ("teknium/OpenHermes-2.5", ["instruction", "chat"], "100K<n<1M", 900_000, 2_300, ["en"], "other", False,
     "~1M mostly synthetic, heavily filtered instruction and chat samples. A workhorse mixture for modern open SFT runs."),
    ("HuggingFaceH4/ultrachat_200k", ["chat"], "100K<n<1M", 620_000, 1_100, ["en"], "mit", False,
     "200k multi-turn conversations distilled from UltraChat; ships ready-made SFT and DPO splits."),
    ("HuggingFaceH4/deita-10k-v0", ["instruction", "chat"], "10K<n<100K", 62_000, 260, ["en"], "apache-2.0", False,
     "High-quality multi-turn instruction data with explicit system turns, built for chat-model alignment."),
    ("allenai/tulu-3-sft-mixture", ["instruction", "chat"], "100K<n<1M", 210_000, 480, ["en"], "odc-by", False,
     "Tulu 3's SFT mixture: a large, carefully decontaminated blend of open instruction datasets."),
    ("Open-Orca/OpenOrca", ["instruction", "chat"], "1M<n<10M", 480_000, 700, ["en"], "mit", False,
     "3.5M GPT-4/GPT-3.5-augmented FLAN-style samples. Large, cheap and popular for scale-up SFT."),
    ("OpenAssistant/oasst1", ["chat"], "10K<n<100K", 320_000, 900, ["en", "es", "ru", "de"], "apache-2.0", False,
     "Human-written assistant conversations collected by volunteers in 35 languages, arranged as a message tree."),
    ("OpenAssistant/oasst2", ["chat"], "10K<n<100K", 180_000, 470, ["en", "es", "ru", "de", "zh"], "apache-2.0", False,
     "Extension of OASST1 with broader language coverage and additional human annotations."),
    ("timdettmers/openassistant-guanaco", ["instruction", "chat"], "1K<n<10K", 190_000, 520, ["en"], "apache-2.0", False,
     "The OASST1 subset used in the QLoRA paper: small, clean, and the default sanity-check for a new fine-tuning setup."),
    ("WizardLMTeam/WizardLM_evol_instruct_70k", ["instruction"], "10K<n<100K", 260_000, 560, ["en"], "other", False,
     "70k instructions produced by the Evol-Instruct loop, famous for its breadth of task types."),
    ("lmsys/lmsys-chat-1m", ["chat"], "1M<n<10M", 240_000, 570, ["en", "zh", "ru"], "other", False,
     "1M real-world conversations with 25 different models, harvested from the Chatbot Arena's public leaderboard traffic."),
    ("stingning/ultrachat", ["chat"], "100K<n<1M", 90_000, 200, ["en"], "other", False,
     "The original 1.4M-message UltraChat release that HuggingFaceH4/ultrachat_200k later cleaned up."),
    ("HuggingFaceH4/stack-exchange-preferences", ["instruction", "preference"], "100K<n<1M", 44_000, 110, ["en"], "cc-by-sa-4.0", False,
     "Questions and answers scraped from the Stack Exchange network with metadata for answer ranking."),
    ("mlabonne/FineTome-100k", ["instruction", "chat"], "100K<n<1M", 120_000, 330, ["en"], "other", False,
     "ShareGPT-formatted 100k mixture with a custom system prompt, commonly used for LoRA fine-tunes."),

    # ---------------------------------------------------------------- preference
    ("Anthropic/hh-rlhf", ["preference"], "10K<n<100K", 260_000, 840, ["en"], "mit", False,
     "Human preference data about helpfulness and harmlessness: chosen/rejected pairs for RLHF and DPO."),
    ("Dahoas/full-hh-rlhf", ["preference"], "10K<n<100K", 130_000, 300, ["en"], "mit", False,
     "Flat-packaged version of Anthropic's HH-RLHF with prompt and response columns ready for reward modelling."),
    ("HuggingFaceH4/ultrafeedback_binarized", ["preference"], "10K<n<100K", 150_000, 350, ["en"], "mit", False,
     "UltraFeedback cleaned and binarised into chosen/rejected pairs; the standard DPO dataset for open chat models."),
    ("allenai/ultrafeedback_binarized_cleaned", ["preference"], "10K<n<100K", 60_000, 150, ["en"], "other", False,
     "AllenAI's decontaminated UltraFeedback derivative, used to train the Tulu family."),
    ("stanfordnlp/SHP", ["preference"], "100K<n<1M", 70_000, 190, ["en"], "other", False,
     "385k Reddit-sourced preference comparisons split into SHP-1 and SHP-2, with train/val/test protocols."),
    ("Intel/orca_dpo_pairs", ["preference"], "1K<n<10K", 190_000, 340, ["en"], "other", False,
     "12k GPT-4-preferred versus Llama-2-rejected pairs; a common second stage after an Orca-style SFT."),
    ("argilla/dpo-mix-7k", ["preference"], "1K<n<10K", 80_000, 220, ["en"], "other", False,
     "A hand-picked 7k blend of high-signal preference pairs across several open DPO datasets."),
    ("lmsys/chatbot_arena_conversations", ["preference", "chat"], "10K<n<100K", 90_000, 300, ["en", "zh"], "cc-by-nc-4.0", False,
     "33k arena battles with human votes: useful for reward modelling and evaluating judge models."),
    ("openbmb/UltraFeedback", ["preference"], "10K<n<100K", 60_000, 180, ["en"], "mit", False,
     "64k prompts each scored by GPT-4 across four fine-grained aspects, with response annotations."),
    ("allenai/real-toxicity-prompts", ["preference", "safety"], "10K<n<100K", 40_000, 130, ["en"], "apache-2.0", False,
     "100k prompts labelled for toxicity by a classifier and by crowd workers; the base of the REALTOXICITYPROMPTS benchmark."),

    # ------------------------------------------------------------------- math
    ("meta-math/MetaMathQA", ["math"], "100K<n<1M", 150_000, 350, ["en"], "other", False,
     "395k augmented GSM8K/MATH questions with step-by-step rationales produced by a backward-evolution pipeline."),
    ("openai/gsm8k", ["math"], "1K<n<10K", 700_000, 1_600, ["en"], "mit", False,
     "8.5k grade-school word problems with calculator annotations. The reference benchmark for CoT reasoning."),
    ("microsoft/orca-math-word-problems-200k", ["math"], "100K<n<1M", 120_000, 290, ["en"], "mit", False,
     "200k GPT-4 generated word problems with detailed solutions, aimed at small-model maths ability."),
    ("nvidia/OpenMathInstruct-2", ["math"], "1M<n<10M", 70_000, 200, ["en"], "cc-by-4.0", False,
     "14M problem/solution pairs generated with Llama-3.1-405B, with tool-integrated and code-execution variants."),
    ("TIGER-Lab/MathInstruct", ["math"], "10K<n<100K", 60_000, 170, ["en"], "mit", False,
     "263k chain-of-thought maths examples merged from nine sources including Camel, Flan and OpenAI GPT-4 outputs."),
    ("EleutherAI/hendrycks_math", ["math"], "1K<n<10K", 90_000, 300, ["en"], "mit", False,
     "The MATH competition-problem suite (algebra, geometry, calculus, number theory) in Elementary-level splits."),
    ("deepmind/aqua_rat", ["math"], "n<1K", 30_000, 90, ["en"], "other", False,
     "97k multiple-choice algebra word problems with rationale fields, widely used for evaluating reasoning chains."),

    # ------------------------------------------------------------------- code
    ("bigcode/the-stack", ["code", "pretrain"], "10B<n<100B", 90_000, 260, ["en"], "other", True,
     "3TB of permissively licensed source code in 358 languages. The classic pretraining corpus for code LLMs."),
    ("bigcode/the-stack-v2", ["code", "pretrain"], "10B<n<100B", 40_000, 150, ["en"], "other", True,
     "The Stack v2 with software-heritage metadata and repository-level context, built for long-context code training."),
    ("bigcode/starcoderdata", ["code", "pretrain"], "1B<n<10B", 25_000, 90, ["en"], "other", True,
     "The curated subset of The Stack that StarCoder was trained on, including GitHub issues and Jupyter notebooks."),
    ("codeparrot/codeparrot-clean", ["code", "pretrain"], "1M<n<10M", 30_000, 110, ["en"], "other", False,
     "Cleaned Python source used in the CodeParrot project; small enough to pretrain a toy model on a single GPU."),
    ("sahil2801/CodeAlpaca-20k", ["code", "instruction"], "10K<n<100K", 220_000, 380, ["en"], "cc-by-4.0", False,
     "20k code-generation instructions created self-instruct style; a cheap first SFT for coding assistants."),
    ("m-a-p/CodeFeedback-Filtered-Instruction", ["code", "instruction"], "100K<n<1M", 70_000, 190, ["en"], "other", False,
     "~156k instructions filtered for correctness and clarity; strong gains for small code models."),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", ["code", "instruction"], "10K<n<100K", 90_000, 240, ["en"], "mit", False,
     "74k instruction/response pairs generated from open-source code snippets, used to train Magicoder."),
    ("codeparrot/apps", ["code", "eval"], "1K<n<10K", 90_000, 250, ["en"], "other", False,
     "10k competitive-programming problems with 131k Python solutions and unit-test-based judging."),
    ("DeepMind/code_contests", ["code", "eval"], "1K<n<10K", 60_000, 190, ["en"], "cc-by-4.0", False,
     "13k contest problems with official test cases; the source set behind CodeContests/MBPP style evaluations."),
    ("google-research-datasets/mbpp", ["code", "eval"], "1K<n<10K", 30_000, 90, ["en"], "cc-by-4.0", False,
     "Mostly Basic Python Problems: 974 few-shot tasks with functional-level test suites."),
    ("openai/openai_humaneval", ["code", "eval"], "n<1K", 90_000, 260, ["en"], "mit", False,
     "164 hand-written Python problems with unit tests; the most cited code-generation benchmark."),
    ("google-research-datasets/conala", ["code", "instruction"], "1K<n<10K", 25_000, 80, ["en"], "cc-by-4.0", False,
     "Natural-language to Python snippets mined from Stack Overflow, useful for code-intent translation."),

    # --------------------------------------------------------------- pretraining
    ("HuggingFaceFW/fineweb", ["pretrain"], "1B<n<10B", 260_000, 780, ["en"], "odc-by", False,
     "15T tokens of deduplicated CommonCrawl filtered with the FineWeb pipeline; the reference English web corpus."),
    ("HuggingFaceFW/fineweb-edu", ["pretrain"], "1B<n<10B", 220_000, 640, ["en"], "odc-by", False,
     "FineWeb filtered by an educational-quality classifier, with 350B, 100B and 10B sample tiers."),
    ("HuggingFaceFW/fineweb-2", ["pretrain", "multilingual"], "10B<n<100B", 70_000, 230, ["en", "hi", "es", "zh", "ar", "fr"], "odc-by", False,
     "Multilingual FineWeb covering 1000+ languages with per-language quality filters — the go-to non-English web corpus."),
    ("HuggingFaceTB/smollm-corpus", ["pretrain"], "1B<n<10B", 60_000, 220, ["en"], "odc-by", False,
     "2.2T tokens of FineWeb-Edu plus Cosmopedia, curated for the SmolLM family of small models."),
    ("HuggingFaceTB/cosmopedia", ["pretrain"], "10M<n<100M", 90_000, 280, ["en"], "apache-2.0", False,
     "30M synthetic textbooks, stories and exercises generated from web seeds, organised by age and format."),
    ("HuggingFaceTB/finemath", ["pretrain"], "100M<n<1B", 20_000, 90, ["en"], "odc-by", False,
     "FineMath: web pages filtered for mathematical content, with 10B/3B token variants for continued pretraining."),
    ("open-web-math/open-web-math", ["pretrain", "math"], "10M<n<100M", 30_000, 120, ["en"], "odc-by", False,
     "14.7B tokens of maths-focused web text plus 28B tokens of arXiv/Library of Congress material."),
    ("allenai/c4", ["pretrain"], "100M<n<1B", 300_000, 800, ["en"], "odc-by", False,
     "The cleaned C4 corpus with per-split shards and multilingual variants; a standard baseline pretraining set."),
    ("wikimedia/wikipedia", ["pretrain"], "10M<n<100M", 260_000, 500, ["en", "hi", "de", "fr", "ar"], "cc-by-sa-4.0", False,
     "Wikimedia's own Wikipedia dumps, one config per language and snapshot date, with clean article structure."),
    ("EleutherAI/pile", ["pretrain"], "100M<n<1B", 70_000, 260, ["en"], "mit", False,
     "The Pile: an 825GB diverse English corpus of 22 subsets, from PubMed to GitHub to the Bible."),
    ("EleutherAI/the_pile_deduplicated", ["pretrain"], "100M<n<1B", 25_000, 90, ["en"], "mit", False,
     "The Pile after overlap-aware deduplication, with both full and streaming-friendly layouts."),
    ("togethercomputer/RedPajama-Data-1T", ["pretrain"], "1B<n<10B", 60_000, 230, ["en"], "apache-2.0", False,
     "The 1.2T-token LLaMA-1 reproduction corpus: CommonCrawl, C4, books, arXiv, StackExchange and Wikipedia."),
    ("togethercomputer/RedPajama-Data-V2", ["pretrain"], "10B<n<100B", 25_000, 110, ["en", "de", "fr", "hi"], "apache-2.0", False,
     "30T raw / 20B deduped tokens of CommonCrawl with quality signals and 100+ language configs."),
    ("cerebras/SlimPajama-627B", ["pretrain"], "1B<n<10B", 40_000, 160, ["en"], "apache-2.0", False,
     "627B tokens of deduplicated RedPajama with globally-consistent near-duplicate removal."),
    ("DKYoon/SlimPajama-6B", ["pretrain"], "1M<n<10M", 40_000, 130, ["en"], "apache-2.0", False,
     "A 6B-token subset of SlimPajama that actually fits on a laptop-class train run; great for smoke tests."),
    ("roneneldan/TinyStories", ["pretrain"], "1M<n<10M", 120_000, 420, ["en"], "cdla-sharing-1.0", False,
     "2M short stories written with a restricted vocabulary, showing that small models can learn grammar and coherence."),
    ("skylion007/openwebtext", ["pretrain"], "1M<n<10M", 40_000, 140, ["en"], "cc0-1.0", False,
     "The re-created WebText: 38GB of Reddit-linked pages, packaged as easily downloadable shards."),
    ("stas/openwebtext-10k", ["pretrain"], "10K<n<1M", 25_000, 90, ["en"], "cc0-1.0", False,
     "A 10k-document slice of OpenWebText for pipeline debugging without downloading the full corpus."),
    ("tiiuae/falcon-refinedweb", ["pretrain"], "1B<n<10B", 60_000, 220, ["en"], "odc-by", False,
     "The filtered and deduplicated web corpus behind Falcon, with FuzzyDedupe applied at scale."),
    ("oscar-corpus/OSCAR-2301", ["pretrain", "multilingual"], "1B<n<10B", 40_000, 130, ["hi", "en", "fr", "es", "ar"], "cc-by-4.0", True,
     "OSCAR 23.01: language-classified CommonCrawl text for 150+ languages, one config per language."),
    ("uonlp/CulturaX", ["pretrain", "multilingual"], "10B<n<100B", 25_000, 110, ["hi", "en", "es", "zh", "ar", "ru"], "other", False,
     "6.3T tokens of cleaned multilingual web data for 167 languages, built from mC4 and OSCAR."),
    ("nvidia/Nemotron-CC", ["pretrain"], "10B<n<100B", 20_000, 90, ["en"], "other", False,
     "6.6T-token CommonCrawl re-extraction with synthetic rephrasing and high-quality subsets."),
    ("allenai/dolma", ["pretrain"], "1B<n<10B", 15_000, 70, ["en"], "odc-by", True,
     "The Dolma 3T-token pretraining corpus behind OLMo, with full provenance and mixing tooling."),
    ("common-pile/common-pile-v0.1", ["pretrain"], "100M<n<1B", 10_000, 60, ["en"], "odc-by", False,
     "Permissively licensed text drawn from books, code, forum and academic sources for fully-open models."),

    # -------------------------------------------------------------- multilingual
    ("cfilt/iitb-english-hindi", ["multilingual", "translation"], "1M<n<10M", 60_000, 190, ["en", "hi"], "cc-by-nc-4.0", False,
     "IIT Bombay English-Hindi parallel corpus: 1.6M sentence pairs used across Indic MT research."),
    ("ai4bharat/IndicCorpV2", ["pretrain", "multilingual"], "10B<n<100B", 25_000, 110, ["hi", "bn", "ta", "te", "mr", "kn"], "cc-by-nc-4.0", False,
     "The IndicCorp v2 monolingual corpus for 24 Indian languages, with 20B+ tokens in Hindi alone."),
    ("ai4bharat/IndicGLUE", ["multilingual", "classification"], "100K<n<1M", 20_000, 80, ["hi", "bn", "ta", "te", "gu", "mr"], "cc-by-nc-4.0", False,
     "Natural-language-understanding benchmark suite for Indic languages, modelled on GLUE."),
    ("facebook/flores", ["multilingual", "translation"], "1K<n<10K", 90_000, 260, ["hi", "en", "bn", "ta", "ar", "sw"], "cc-by-sa-4.0", False,
     "FLORES-200 evaluation sentences for 200 languages, the standard low-resource translation benchmark."),
    ("Helsinki-NLP/opus-100", ["multilingual", "translation"], "100M<n<1B", 40_000, 140, ["hi", "en", "fr", "de", "es", "ru"], "other", False,
     "1M parallel sentences per language pair mined from OPUS, across 100 languages and 2200 directions."),
    ("wmt16", ["multilingual", "translation"], "1M<n<10M", 60_000, 190, ["en", "de", "ro", "fi", "tr"], "other", False,
     "The classic WMT 2016 news translation task data, still the default for machine-translation baselines."),
    ("wmt/wmt19", ["multilingual", "translation"], "1M<n<10M", 25_000, 90, ["en", "de", "ru", "cs", "fi"], "other", False,
     "WMT 2019 news task with additional language pairs and news-commentary test sets."),

    # ------------------------------------------------------------ nlu / classic
    ("nyu-mll/glue", ["classification"], "100K<n<1M", 300_000, 650, ["en"], "other", False,
     "The GLUE benchmark bundle: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE and WNLI in one repo."),
    ("nyu-mll/super_glue", ["classification"], "100K<n<1M", 90_000, 260, ["en", "hi"], "other", False,
     "The harder SuperGLUE suite plus diagnostic sets, with configs for BoolQ, CB, COPA, MultiRC and more."),
    ("imdb", ["classification"], "10K<n<100K", 400_000, 1_100, ["en"], "other", False,
     "50k movie reviews labelled positive or negative; the standard binary sentiment warm-up."),
    ("fancyzhx/ag_news", ["classification"], "100K<n<1M", 180_000, 420, ["en"], "cc-by-4.0", False,
     "120k news articles in four topic classes, used in every text-classification tutorial ever written."),
    ("fancyzhx/amazon_polarity", ["classification"], "1M<n<10M", 90_000, 260, ["en"], "apache-2.0", False,
     "3.6M Amazon reviews with polarity labels plus titles; a solid scaling dataset for classifiers."),
    ("fancyzhx/yelp_review_full", ["classification"], "100K<n<1M", 60_000, 190, ["en"], "other", False,
     "700k Yelp reviews with five-star labels, often used for ordinal or multi-class sentiment work."),
    ("mteb/amazon_polarity", ["classification"], "1M<n<10M", 10_000, 60, ["en"], "apache-2.0", False,
     "MTEB-packaged Amazon polarity data with a fixed test split for reproducible embedding evaluation."),
    ("stanfordnlp/sentiment140", ["classification"], "1M<n<10M", 40_000, 140, ["en"], "other", False,
     "1.6M tweets labelled by emoticon, the classic noisy social-media sentiment corpus."),

    # ----------------------------------------------------------------- summarise
    ("abisee/cnn_dailymail", ["summarization"], "100K<n<1M", 260_000, 640, ["en"], "other", False,
     "300k news articles paired with human-written highlights; the canonical abstractive summarisation set."),
    ("EdinburghNLP/xsum", ["summarization"], "100K<n<1M", 220_000, 560, ["en"], "cc-by-4.0", False,
     "226k BBC articles with single-sentence summaries, deliberately difficult to copy from the source."),
    ("samsum", ["summarization"], "10K<n<100K", 190_000, 460, ["en"], "cc-by-nc-nd-4.0", False,
     "16k messenger-style dialogues with short human summaries; great for dialogue summarisation."),
    ("GEM/xlsum", ["summarization", "multilingual"], "1M<n<10M", 40_000, 130, ["en", "hi", "ar", "fr", "es"], "cc-by-4.0", False,
     "GEM's XL-Sum: BBC articles and summaries in 44 languages including 11 Indic ones."),
    ("ccdv/govreport-summarization", ["summarization"], "10K<n<100K", 30_000, 100, ["en"], "other", False,
     "Long US government reports with section summaries; useful for long-context summarisation."),
    ("billsum", ["summarization"], "10K<n<100K", 30_000, 100, ["en"], "cc0-1.0", False,
     "US Congressional bills paired with Congressional Research Service summaries, a legal-domain summarisation set."),

    # ------------------------------------------------------------------------ qa
    ("rajpurkar/squad", ["qa"], "10K<n<100K", 300_000, 800, ["en"], "cc-by-sa-4.0", False,
     "100k crowd-written questions over Wikipedia paragraphs; the reference extractive reading-comprehension set."),
    ("rajpurkar/squad_v2", ["qa"], "100K<n<1M", 120_000, 350, ["en"], "cc-by-sa-4.0", False,
     "SQuAD 2.0 adds 50k unanswerable questions so models must learn when to abstain."),
    ("natural_questions", ["qa", "rag"], "100K<n<1M", 70_000, 220, ["en"], "cc-by-sa-3.0", False,
     "Real Google search questions with Wikipedia answer annotations and long/short answer spans."),
    ("microsoft/ms_marco", ["qa", "rag"], "100K<n<1M", 40_000, 140, ["en"], "other", False,
     "1M passage-ranked queries from Bing; a standard retrieval and passage-re-ranking benchmark."),
    ("BeIR/scifact", ["rag", "eval"], "1K<n<10K", 25_000, 90, ["en"], "cc-by-4.0", False,
     "Scientific claim verification with abstracts as the corpus; one of the BEIR zero-shot retrieval tasks."),
    ("BeIR/nq", ["rag", "eval"], "100K<n<1M", 20_000, 80, ["en"], "cc-by-4.0", False,
     "Natural Questions re-packaged for retrieval evaluation, with a 2.6M-passage Wikipedia corpus."),
    ("hotpotqa/hotpot_qa", ["qa", "rag"], "100K<n<1M", 60_000, 190, ["en"], "cc-by-sa-4.0", False,
     "113k multi-hop questions that require reasoning across two or more Wikipedia articles."),
    ("truthful_qa", ["qa", "safety", "eval"], "n<1K", 90_000, 280, ["en"], "apache-2.0", False,
     "817 adversarial questions probing truthfulness, with correct and incorrect reference answers."),
    ("hails/mmlu_no_train", ["eval"], "10K<n<100K", 40_000, 140, ["en"], "mit", False,
     "MMLU under the canonical train-free format: 14k multiple-choice questions across 57 subjects."),
    ("cais/mmlu", ["eval"], "10K<n<100K", 90_000, 280, ["en"], "mit", False,
     "MMLU with auxiliary train/dev/test splits, handy when you want few-shot exemplars."),
    ("allenai/ai2_arc", ["eval"], "10K<n<100K", 60_000, 190, ["en"], "cc-by-sa-4.0", False,
     "7.8k science exam questions split into Easy and Challenge sets, with answer keys."),
    ("Rowan/hellaswag", ["eval"], "10K<n<100K", 60_000, 190, ["en"], "mit", False,
     "40k multiple-choice sentence-completion items derived from ActivityNet captions — a common-sense staple."),
    ("google/boolq", ["classification", "eval"], "1K<n<10K", 60_000, 190, ["en"], "cc-by-sa-3.0", False,
     "12k yes/no questions about Wikipedia passages, testing inference over a single document."),
    ("truthfulqa/truthful_qa", ["eval"], "n<1K", 30_000, 110, ["en"], "apache-2.0", False,
     "Earlier namespace of the TruthfulQA generation and multiple-choice tasks."),
    ("openai/grade-school-math", ["math", "eval"], "1K<n<10K", 30_000, 110, ["en"], "mit", False,
     "Older mirror of GSM8K with slight formatting differences, still referenced by early CoT papers."),

    # ----------------------------------------------------------- domain specific
    ("epfl-llm/guidelines", ["domain"], "1K<n<10K", 40_000, 140, ["en"], "apache-2.0", False,
     "Clinical practice guidelines with instruction-formatted QA, built for the Meditron project."),
    ("medalpaca/medical_meadow_medical_flashcards", ["domain"], "10K<n<100K", 40_000, 140, ["en"], "other", False,
     "33k medical flashcard question/answer pairs; a compact dataset for medical QA fine-tuning."),
    ("qiaojin/PubMedQA", ["domain", "qa"], "100K<n<1M", 40_000, 140, ["en"], "mit", False,
     "273k biomedical research questions with yes/no/maybe labels and long-context evidence passages."),
    ("pile-of-law/pile-of-law", ["domain", "pretrain"], "100B<n<1T", 10_000, 70, ["en"], "other", True,
     "256GB of legal and administrative text from US, EU, Canadian, Indian and other jurisdictions."),
    ("lex_glue", ["domain", "classification"], "10K<n<100K", 10_000, 60, ["en"], "cc-by-4.0", False,
     "A GLUE-style benchmark of eight legal text-classification tasks from EU and US case law."),
    ("financial_phrasebank", ["domain", "classification"], "1K<n<10K", 90_000, 260, ["en"], "cc-by-nc-sa-3.0", False,
     "4.8k financial news sentences annotated by experts for sentiment, with several agreement splits."),
    ("ai4bharat/IndicSentiment", ["domain", "multilingual"], "10K<n<100K", 10_000, 60, ["hi", "bn", "ta", "mr", "gu"], "cc-by-nc-4.0", False,
     "Sentiment-labelled reviews and social posts in 10 Indian languages, from the Bhasha-Abhijnaanam project."),
    ("covid19-radiography-database", ["domain", "vision"], "1K<n<10K", 10_000, 50, ["en"], "other", False,
     "COVID-19 chest X-ray images with metadata; included as a reminder that this catalog spans modalities."),
    ("mozilla-foundation/common_voice_16_1", ["audio"], "10M<n<100M", 60_000, 190, ["hi", "en", "ta", "fr", "de"], "cc0-1.0", True,
     "Crowdsourced read speech in 100+ languages with validated train/dev/test splits, for ASR and speech LLMs."),
    ("google/fleurs", ["audio", "multilingual"], "1K<n<10K", 70_000, 220, ["hi", "en", "bn", "ta", "sw"], "cc-by-4.0", False,
     "Few-shot learning evaluation of universal representations of speech: 102 languages, read Wikipedia sentences."),
    ("PolyAI/minds14", ["audio", "classification"], "1K<n<10K", 25_000, 80, ["en", "de", "fr", "hi"], "cc-by-4.0", False,
     "Multilingual spoken intent classification from banking call recordings."),
    ("liuhaotian/LLaVA-Instruct-150K", ["vision", "instruction"], "100K<n<1M", 90_000, 260, ["en"], "other", False,
     "158k GPT-generated conversation, detail and complex-reasoning instructions paired with COCO images."),
    ("HuggingFaceM4/VQAv2", ["vision", "qa"], "100K<n<1M", 25_000, 90, ["en"], "other", False,
     "1.1M open-ended visual questions about COCO images with ten human answers each."),
]


def build() -> dict:
    datasets = []
    for (repo_id, tasks, size, downloads, likes, langs, license_, gated, description) in ENTRIES:
        author = repo_id.split("/")[0] if "/" in repo_id else None
        tags = (
            [f"task_categories:{t}" for t in sorted(set(tasks))]
            + [f"language:{lang}" for lang in langs]
            + [f"license:{license_}", f"size_categories:{size}"]
            + ["llm-maker:catalog"]
        )
        datasets.append(
            {
                "id": repo_id,
                "author": author,
                "description": description,
                "downloads": downloads,
                "likes": likes,
                "tags": tags,
                "size_categories": [size],
                "task_categories": sorted(set(tasks)),
                "languages": langs,
                "license": license_,
                "gated": gated,
                "source": "snapshot",
            }
        )
    return {
        "schema": 1,
        "generated_by": "tools/build_snapshot.py",
        "note": (
            "Bundled offline catalog. Downloads/likes are approximate, rounded figures used only for "
            "ranking; descriptions are paraphrased. Live data comes from https://huggingface.co/api/datasets."
        ),
        "source": "bundled-snapshot",
        "datasets": datasets,
    }


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "llm_maker" / "data" / "hf_datasets_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(payload['datasets'])} datasets")


if __name__ == "__main__":
    main()
