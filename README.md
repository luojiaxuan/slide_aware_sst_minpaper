# Slide/Context-Aware Streaming ST: Minimal Paper Package

This package contains a minimal publishable-paper starter kit for
**slide/context-aware streaming speech translation (ST)**. The current paper
framing is not pure vision-aware SST. It studies latency-aware use of
slide-derived context: OCR terms, OCR-derived glossary, VLM slide summaries,
deck/topic metadata, distractors, and later speaker/topic context.

The central rule is faithfulness: visual evidence may help disambiguate spoken content, but the translator must not add visible objects, actions, labels, or facts that were not spoken.

## Source of Truth

This project uses Git and Hugging Face as the durable sources of truth.

- Git stores code, configs, paper notes, lightweight metadata, progress, and
  handoff state.
- Hugging Face stores reusable datasets, generated data artifacts, checkpoints,
  adapters, and revisions.
- Hyper00 `/data` paths are temporary staging and active-run storage only.
- Notion is not used for project source-of-truth state.

Current source-of-truth docs:

- [`docs/SOURCE_OF_TRUTH.md`](docs/SOURCE_OF_TRUTH.md)
- [`docs/PROGRESS.md`](docs/PROGRESS.md)
- [`docs/QWEN3_CONTEXT_QA_20260706.md`](docs/QWEN3_CONTEXT_QA_20260706.md)
- [`docs/QWEN3_REFERENCE_PILOT_20260706.md`](docs/QWEN3_REFERENCE_PILOT_20260706.md)
- [`docs/QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md`](docs/QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md)
- [`docs/DIAGNOSTIC_REVIEW_GUIDE.md`](docs/DIAGNOSTIC_REVIEW_GUIDE.md)
- [`code_plan/SLIDE_CONTEXT_AWARE_MVP.md`](code_plan/SLIDE_CONTEXT_AWARE_MVP.md)

## Directory layout

```text
paper/       LaTeX skeleton with section-level TODOs and BibTeX.
code_plan/   Agent-facing implementation plan, data schema, experiment matrix, tickets.
repo/        Minimal code scaffold and config files for the implementation agent.
```

## Eval and debug data (Hugging Face)

- [`gavinlaw/mtedx-v-eval`](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) —
  **mTEDx-V**: talk-level long-form X→En ST eval manifests (es/fr/it/ru/el → en,
  test+valid, 100 talks / ~18 h speech). `talk_id` is the real YouTube video ID
  (alive check 2026-07-16: 100/100), and segment timestamps sit on the original
  video timeline, so frames can be aligned to speech directly. Manifests only —
  no media is redistributed (mTEDx/TEDx are CC BY-NC-ND 4.0); obtain videos from
  the original sources and extract frames locally with
  `repo/scripts/extract_frames_by_manifest.py`. Built by
  `repo/scripts/build_mtedx_v_manifest.py`.
- [`gavinlaw/chinese-lips-longform-debug`](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) —
  three continuous zh long-speech streams (~97 min total) concatenated from
  same-video Chinese-LiPS test segments (silence removed by construction), with
  per-segment offsets, transcripts, and slide OCR/VL2 annotations. Debug/dev
  resource only, not a benchmark (CC BY-NC-SA 4.0). Built by
  `repo/scripts/build_chinese_lips_longform.py`.

## Recommended MVP scope

1. Dataset: Chinese lecture or VASR-like local subset with transcript, video/audio path, and optional OCR/visual metadata.
2. Translation direction: Chinese speech/transcript to English text.
3. Main benchmark: homophones, technical terms, visual deixis, object/action grounding, on-screen text, and mismatch cases.
4. Methods: no-context, text context, OCR-only, visual-caption-only, naive all-visual context, policy-based visual context, and oracle supporting evidence.
5. Metrics: BLEU/COMET as secondary; hard-label accuracy, visual grounded accuracy, visual hallucination rate, wrong visual adoption rate, evidence selection, and latency as primary.

## How to use

Give `code_plan/AGENT_START_HERE.md` and the `repo/` scaffold to a coding agent.  Use `paper/main.tex` as the paper outline and fill results after M2/M3 experiments.

No datasets, model weights, API keys, or proprietary slide materials are included.
