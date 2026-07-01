# Visual-Evidence-Aware Streaming ST: Minimal Paper Package

This package contains a minimal publishable-paper starter kit for **visual-evidence-aware streaming speech translation (ST)**.  The first version keeps the original slide-aware contextual SST path runnable, but generalizes context into unified evidence from slides, video OCR, scene/object/action descriptions, glossary entries, background documents, and history.

The central rule is faithfulness: visual evidence may help disambiguate spoken content, but the translator must not add visible objects, actions, labels, or facts that were not spoken.

## Directory layout

```text
paper/       LaTeX skeleton with section-level TODOs and BibTeX.
code_plan/   Agent-facing implementation plan, data schema, experiment matrix, tickets.
repo/        Minimal code scaffold and config files for the implementation agent.
```

## Recommended MVP scope

1. Dataset: Chinese lecture or VASR-like local subset with transcript, video/audio path, and optional OCR/visual metadata.
2. Translation direction: Chinese speech/transcript to English text.
3. Main benchmark: homophones, technical terms, visual deixis, object/action grounding, on-screen text, and mismatch cases.
4. Methods: no-context, text context, OCR-only, visual-caption-only, naive all-visual context, policy-based visual context, and oracle supporting evidence.
5. Metrics: BLEU/COMET as secondary; hard-label accuracy, visual grounded accuracy, visual hallucination rate, wrong visual adoption rate, evidence selection, and latency as primary.

## How to use

Give `code_plan/AGENT_START_HERE.md` and the `repo/` scaffold to a coding agent.  Use `paper/main.tex` as the paper outline and fill results after M2/M3 experiments.

No datasets, model weights, API keys, or proprietary slide materials are included.
