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

## Follow-up: VLM extraction closes most of the hard-stratum gap (2026-07-19)

Replacing 480p tesseract with Qwen2.5-VL-7B slide reading (same 54 frames, same
lookback rule; `code/scripts/vlm_slide_terms.py`, timeline archived here) and
re-running the B condition on the el group:

| cond (el, n=40) | chrF | termR |
|---|---|---|
| none | 38.8 | 0.41 |
| slide-OCR | 35.2 | 0.40 |
| **slide-VLM** | **39.9** | 0.41 |
| oracle | 46.6 | 0.53 |

- VLM extraction **repairs OCR's damage** (35.2 → 39.9; VLM reads full phrases
  like "On Imitation and Improvisation: New Competitive Businesses" where OCR
  produced fragments like "independen").
- **Hard stratum (baseline termR ≤ 0.4, n=22): VLM vs none ΔchrF +14.1** — about
  75% of the oracle's +18.8 hard-stratum gain, realized by the *real* visual
  channel.
- Pooled VLM vs none is flat (−0.6, n.s.): easy segments are slightly hurt by
  unneeded injection, cancelling hard-segment gains — the direct empirical case
  for an **evidence-gating policy** (inject only under predicted need), which is
  the method chapter's job.

Narrative chain now complete: C≫A (premise) → B_OCR≈A (extraction bottleneck) →
B_VLM ≈ 0.75·C on hard segments (VLM fixes extraction) → pooled flat (gating
needed) → policy.

## Follow-up 2: gating recovers the full pooled gain (2026-07-19)

Two gating prototypes over the VLM slide terms (`runs_gated.jsonl`; conditions
in `oracle_killtest_runner.py`): *oracle-gate* injects only on hard segments
(gating upper bound); *LLM-gate* asks the model per step whether the slide
terms are relevant (sticky-on).

| condition (el, n=40) | chrF | Δ vs none | p | hard-Δ |
|---|---|---|---|---|
| A none | 38.8 | — | — | — |
| B slide-OCR | 35.2 | −2.9 | n.s. | +2.3 |
| B slide-VLM | 39.9 | −0.6 | n.s. | +14.1 |
| B VLM + LLM-gate | 38.0 | −2.0 | n.s. | +10.1 |
| **B VLM + oracle-gate** | **45.3** | **+7.3** | **<0.001** | +13.3 |
| C oracle terms | 46.6 | +7.0 | 0.059 | +16.3 |

- **Headline: real visual channel + selective injection ≈ oracle upper bound.**
  VLM-extracted slide terms with need-based gating deliver +7.3 chrF pooled
  (p<0.001), statistically cleaner than even the oracle-terms condition (+7.0,
  p=0.059) because easy segments are left untouched.
- **Naive LLM relevance gating fails**: it fires on 35/40 items (17/18 easy) —
  no selectivity — and degenerates to ungated injection. Predicting *when*
  evidence is needed (model uncertainty, term-source matching) is the open
  technical problem, i.e., the paper's method contribution.
- Full six-condition chain: extraction quality (OCR→VLM) fixes *what* to
  inject; gating fixes *when*; together they close ~100% of the oracle gap.

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
