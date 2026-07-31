# Slide-Aware Simultaneous Speech Translation — investigation record

> **Status (2026-07-31).** With audio input and the slide supplied through a
> vision encoder (Qwen3-Omni-30B, 206 segments), a *correct* slide and a *wrong*
> slide produce statistically indistinguishable gains in quality
> (+2.63 vs +2.29 chrF) and latency (AL −0.202 vs −0.199) — so **segment-level
> slide specificity is not being exploited**. Important caveat: the wrong slide
> was drawn from the *same lecture*, so this does not yet rule out
> domain-level visual priming; an unrelated-domain control is the missing
> experiment. **Read [docs/FINDINGS.md](docs/FINDINGS.md) first** — it carries
> per-claim confidence levels, raw-evidence pointers, and what would overturn
> each conclusion.
>
> **Current research goal:** treat the current slide as an asynchronously
> updated, persistent semantic-evidence state. Encode it once on a slide-change
> event, reuse it across its full dwell window, and let causal speech state
> retrieve only relevant evidence. Lip video and slide+lip hybrid experiments
> are out of scope. See
> [docs/RESEARCH_GOAL_20260731.md](docs/RESEARCH_GOAL_20260731.md).

## Original hypothesis (for the record)

Slides in talks often precede the speech that discusses them and persist for
tens of seconds. The visual worker can therefore run off the audio critical path:
it pays a nonzero cost once per slide, then exposes a cached evidence state to
many speech chunks. Three mechanisms were posited — M1 anticipation, M2
recognition support, M3 target-form supply — with slide language as a
stratification variable.

What survived testing: vision does make the model commit earlier, and injected
context does shift quality — but neither effect depends on what is on the slide.
What remains untested: high-ambiguity speech (where headroom exists) and
relevance-selected injection (RASST-style retrieval over the visual channel).

Lineage and neighbours (survey retained in
[latex/sections/02_related_work.tex](latex/sections/02_related_work.tex)):
RASST (retrieval-augmented terminology for SST), Do-Slides-Help (EMNLP'25,
offline ASR), OmniFusion (En→X SimulST with synchronous vision), Caglayan'20
line (simultaneous *text* MT with caption images), MCIF.

## Directory layout

```text
code/    scripts, configs, tests (data prep, scoring, eval pipeline)
data/    DATA.md — pointers to HF datasets and local staging (no media in git)
docs/    stage-by-stage progress, plans (BENCHMARK_PLAN.md), planning/ archive
latex/   paper draft by sections, refs.bib, figures/ + plotting/ code
```

## Existing assets (built through 2026-07-17)

- **Direction decided**: **direction-general** policy on one unified
  multilingual base — X→En primary evidence (M3 + Wiki-verifiable terms),
  En→Zh control on ACL 60/60 (tagged terms), optional X→X generality stratum
  (mTEDx es→fr/it human refs, Pareto/copy-rate only). Mechanism decomposition
  (M1 anticipation / M2 recognition support / M3 target-form supply) and cost
  accounting in [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md).
- **Benchmark strata**:
  - S1 realistic-noisy: [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval)
    — 100 long-form talks es/fr/it/ru/el→en (~18 h), talk_id = live YouTube ID
    (100/100 alive), human refs, OCR visual-signal stratification included. DONE.
  - S2 clean-strong: [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug)
    — zh long-form rebuilt on the original session timeline (real pauses
    restored, drift ≤1 ms), dedicated 1080p slide feed upstream, 100% slide
    coverage. En references pending (machine draft + human-verified core).
- **Measured facts**: mTEDx visual signal is sparse (~12% text frames; 58/100
  talks near-zero) → honest negative stratum; Chinese-LiPS slides are
  never-occluded 1080p (chi_sim OCR 47–151 tokens/slide) → clean upper bound.
- **Scripts** in `code/scripts/`: `build_mtedx_v_manifest.py`,
  `extract_frames_by_manifest.py`, `build_chinese_lips_longform.py`
  (`--timeline-dir` for original timeline), `score_visual_signal.py`
  (`--backend ocr|vlm`), `translate_zh_en_draft.py`.

## Next decision (see docs/RESEARCH_GOAL_20260731.md)

1. Measure the real slide dwell-time and slide-to-related-speech lead/lag
   distributions instead of assuming a 30–60 s window.
2. Rebuild the 206-item probe around once-per-slide cached evidence, with strong
   OCR, structured VLM, oracle text-equivalent, stale, and cross-talk controls.
3. Measure cold visual encoding, dwell-normalized amortized cost, evidence-ready
   misses, and on-path retrieval separately. Scale only if content specificity
   and beyond-OCR gates pass.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
