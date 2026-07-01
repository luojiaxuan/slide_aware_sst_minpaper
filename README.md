# Slide-Aware Contextual SST: Minimal Paper Package

This package contains a minimal publishable-paper starter kit for **slide-aware contextual simultaneous speech translation (SST)**.  The planned first version is intentionally small: build a Chinese lecture challenge set with slides/glossary/background context, evaluate whether slides help hard ambiguous cases, and show that naive context injection can hurt under slide-speech mismatch while a lightweight evidence policy helps.

## Directory layout

```text
paper/       LaTeX skeleton with section-level TODOs and BibTeX.
code_plan/   Agent-facing implementation plan, data schema, experiment matrix, tickets.
repo/        Minimal code scaffold and config files for the implementation agent.
```

## Recommended MVP scope

1. Dataset: Chinese lecture subset, preferably Chinese-LiPS if available locally; otherwise any lecture videos with slide frames, transcripts, and permission to use.
2. Translation direction: Chinese speech/transcript to English text.
3. Main benchmark: hard homophone / near-homophone / technical-term cases where slides or background documents should help.
4. Methods: no-context baseline, glossary-only baseline, slide-only baseline, naive all-context baseline, and lightweight evidence-aware context policy.
5. Metrics: BLEU/COMET as secondary; term accuracy, homophone disambiguation accuracy, context overuse rate, wrong-slide adoption rate, and latency as primary.

## How to use

Give `code_plan/AGENT_START_HERE.md` and the `repo/` scaffold to a coding agent.  Use `paper/main.tex` as the paper outline and fill results after M2/M3 experiments.

No datasets, model weights, API keys, or proprietary slide materials are included.
