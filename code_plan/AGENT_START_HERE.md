# Agent start-here plan

You are implementing the minimal publishable version of a paper on **slide-aware contextual simultaneous speech translation**.

## Core hypothesis

Slides/glossary/background context helps hard Chinese lecture translation cases such as homophones and technical terms, but naive context injection causes wrong-context hallucination when slides are stale, future, or loosely related. A lightweight evidence policy should keep most of the benefit while reducing context overuse under streaming latency constraints.

## MVP deliverables

1. A JSONL challenge set with 500-1000 manually verified Chinese-to-English hard examples.
2. A streaming simulation pipeline over Chinese lecture transcripts/audio.
3. Context indices from slides, glossary, and background docs.
4. Translation runs for: no context, glossary only, slide only, naive all-context, policy-based, oracle context.
5. Evaluation scripts for BLEU/COMET, homophone accuracy, term F1, context overuse rate, wrong-slide adoption rate, and latency.
6. CSV tables and plots ready to paste into `paper/sections/07_results.tex`.

## Constraints

- Do not assume data can be redistributed. Store IDs and local paths.
- Keep model providers pluggable. Do not hard-code private API keys.
- Make every run reproducible: config file + seed + output directory + exact model name.
- Implement transcript-oracle streaming first, then ASR-based streaming. The paper can report both.
- Prefer simple baselines over complicated fine-tuning in v0.

## First coding milestones

### M0: Repo bootstrap
- Implement the JSONL schema loader and validation.
- Implement config parsing from YAML.
- Add a tiny toy dataset under `repo/examples/` with 3 examples.
- Make `pytest` pass on schema and metrics.

### M1: Dataset builder
- Write an adapter for Chinese-LiPS-style metadata: transcript, audio path, slide video/frame path, slide OCR if available.
- Implement pinyin-based homophone mining.
- Generate candidate hard examples with matched and mismatched slide evidence.
- Export annotation sheet CSV for manual verification.

### M2: Baseline runs
- Implement no-context, glossary-only, slide-only, and naive all-context prompts.
- Run on 50 verified examples first.
- Save outputs as JSONL with timing, prompt, evidence packet, and model response.

### M3: Evidence policy
- Implement BM25 + pinyin + temporal-prior retrieval.
- Implement rule-based `use / ignore / delay` policy.
- Run policy vs naive context under matched and mismatched slide settings.

### M4: Evaluation and tables
- Implement term/homophone accuracy and context-overuse metrics.
- Generate `outputs/tables/main_results.csv` and `outputs/tables/ablation.csv`.
- Generate 3-5 case studies for the paper.

### M5: Robustness
- Add wrong-current-slide simulation.
- Add noisy OCR simulation.
- Add distractor glossary-size ablation.

## Stop condition for first paper draft

A minimal draft is ready when we have: (i) at least 500 verified hard examples, (ii) evidence that slides/glossary improve hard-case accuracy over no context, (iii) evidence that naive context increases overuse on mismatched context, and (iv) the policy reduces overuse while preserving most of the hard-case gain.
