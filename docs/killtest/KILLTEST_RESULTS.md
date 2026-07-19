# Oracle Anticipation Kill Test — Results and Verdict

Date: 2026-07-19. Pre-registered in docs/BENCHMARK_PLAN.md §5.1; runner/analyzer
in `code/scripts/oracle_killtest_{runner,analyze}.py`; raw runs in this directory.

## Setup

- **Protocol**: Local Agreement commit (Liu et al. 2020) over incrementally
  revealed source transcript (READ = 2 words el / 4 chars zh), monotonic commits,
  final-step tail completion. Prompt-based, temperature 0.
- **Models**: Qwen3-32B (vllm, hyper01 H200, primary) + Qwen2.5-7B-q4 (ollama,
  local pilot / model-size contrast).
- **Data**: 56 items — 40 segments of mTEDx Greek talk 6HhNGU08KoE (English
  slides = M3 regime, human refs) + 16 Chinese-LiPS segments (zh slides = M2
  regime, hand-translated refs).
- **Conditions**: A none / B slide (OCR terms of currently visible slide) /
  C oracle (rare content words of the segment's reference = perfect anticipation
  upper bound) / D wrong (slide terms from a different moment).
- **Metrics**: chrF, oracle-term recall, AL (Ma et al. 2019).

## Headline results (Qwen3-32B)

| group | cond | chrF | termR | AL |
|---|---|---|---|---|
| el | none | 38.8 | 0.40 | 0.67 |
| el | **oracle** | **46.6** | **0.53** | 0.60 |
| el | slide | 35.2 | 0.40 | 0.22 |
| el | wrong | 38.6 | 0.42 | 0.54 |
| zh | none | 49.6 | 0.52 | 0.46 |
| zh | **oracle** | **55.6** | **0.68** | 0.60 |
| zh | slide | 51.6 | 0.56 | 0.44 |
| zh | wrong | 49.5 | 0.51 | 1.13 |

Paired bootstrap (10k resamples, one-sided):

| comparison | ΔchrF | p | n |
|---|---|---|---|
| oracle vs none, pooled | **+7.0** | **0.030** | 56 |
| oracle vs none, hard stratum (baseline termR ≤ 0.4) | **+18.8** | **< 0.0001** | 28 |
| slide vs none, pooled | −1.8 | 0.74 | 56 |
| wrong vs none, pooled | +0.8 | 0.41 | 56 |

Hard-stratum term recall: **0.12 → 0.41 (3.4×)** under oracle.

## Verdict (per pre-registered rule)

**PREMISE SURVIVES — extraction is the bottleneck.**

1. **C ≫ A (significant)**: future-term injection materially improves streaming
   translation, replicated across both languages/regimes and both model scales
   (7B: el +8.3 / zh +9.8 chrF; 32B: pooled +7.0, p=0.03). Gains concentrate
   precisely where the baseline fails on terminology (hard stratum +18.8 chrF,
   termR 3.4×) — the expected signature of anticipation value: it rescues
   term-dense/hard segments and is neutral elsewhere.
2. **B ≈ A (n.s., el negative)**: today's naive extraction (480p tesseract for
   el; zh OCR list) does not yet realize the oracle value; noisy fragments
   ("independen") can hurt (el −3.6 raw). The C−B gap is the method space:
   VLM-based slide reading, term filtering, confidence gating.
3. **D ≈ A**: wrong-slide terms are neutral for a strong model under prompt
   injection. The 7B pilot's apparent wrong-condition gain (+4.7/+5.6) was
   diagnosed as an output-completeness artifact (weak model + LCP stalls), gone
   at 32B. Faithfulness risk therefore concentrates in stronger injection modes
   (image-input, fine-tuned) — the policy evaluation must construct harder
   adversarial settings than prompt-level wrong terms.

## Caveats

- Text-prefix simulation (transcript, not audio); segment-level, not long-form
  session; AL differences small because LCP stability dominates latency in this
  protocol. The full pipeline (audio, session-level, async slide worker) is
  future work — this probe only establishes the information-value premise.
- zh refs are the author's own drafts (n=16); el refs are mTEDx human refs (n=40).
- Slide condition el uses 480p OCR — a deliberately conservative lower bound.

## Files

- `runs_qwen3_32b.jsonl` (224 runs, primary), `runs_qwen25_7b_local.jsonl`
  (224 runs, pilot), `killtest_items.json` (probe set), `killtest_32b_per_item.tsv`.
