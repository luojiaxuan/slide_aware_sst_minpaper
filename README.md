# Slide-Aware Simultaneous Speech Translation — investigation record

> **Status (2026-08-01).** The 21-talk MCIF translation subset is now
> materialized without extracting or reading references. All 21 videos passed
> frozen hash checks and visual inspection. A 1 s reference-free pass produced
> 283 visually reviewed transition candidates and 304 conservative causal
> states including initial states; each state unlocks only after two stable
> frames. This establishes **visual/timing readiness, not translation benefit**.
> See
> [docs/MCIF_VISUAL_READINESS_20260801.md](docs/MCIF_VISUAL_READINESS_20260801.md).
> The controlled-acoustic input path is also ready: official SLR17/SLR28
> archives and disjoint source pools are frozen, and 75 full-talk ACL dev
> variants pass duration, SNR, clipping, hash, and no-reference checks. See
> [docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md](docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md).
> ACL dev 的 source-side screen 也已可执行：468 个 source segments 与 468 个
> frame observations 完成时间对齐和 Tesseract OCR；严格 exact-match diagnostic
> 找到 149 个独立 `segment × current-frame` anticipation events，其中 183/344
> 个去嵌套候选具有至少 10 秒的保守提前量。该数字只说明 OCR-sufficient
> headroom，不是人工 event density 或 ST gain。100 个双标注 packet 已生成并通过
> no-target/reference/model-output 检查，当前状态仍是
> `PENDING_DOUBLE_ANNOTATION`。见
> [docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md](docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md)。
>
> The earlier Chinese-LiPS diagnostic remains a warning: with audio input and
> the slide supplied through a
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
> **Current exploration strategy:** study whether causally available,
> persistent slide semantics can improve SimulST without putting vision on the
> streaming critical path. ACL dev is an explicit story-discovery stage across
> content attribution, noisy-speech robustness, pixels beyond OCR, and
> evidence-selection/integration. Correct versus matched stale/wrong slides is
> a shared validity control, not the only allowed paper outcome. After the dev
> evidence identifies a useful mechanism, freeze one main claim and analysis
> before touching held-out evaluation. The authoritative strategy, audit, data
> update, and experiment stages are in
> [docs/PAPER_STORY_DECISION_20260731.md](docs/PAPER_STORY_DECISION_20260731.md).
> The previous dual-route document remains the detailed `C0-C7` contract, not
> the current narrative. Lip video and slide+lip hybrid experiments remain out
> of scope.

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

## Current research direction (revised 2026-07-31)

- **Research space:** pre-available slide semantics for SimulST, compiled once
  per persistent slide and reused by the streaming speech path.
- **Development exits:** current-content attribution and earlier commits;
  robustness under controlled acoustic corruption; useful visual relations
  beyond OCR; or a selective integration method with a better quality-latency-
  hallucination trade-off than naive prompting.
- **Shared validity controls:** audio/document-only, naive OCR prompt, correct
  slide, time/type/budget-matched stale/wrong slide, and empty slide slot.
- **Selection discipline:** use ACL dev and other declared development material
  to discover which mechanism is real, report the full exploratory matrix, then
  freeze the main claim, metric, slices, and decision rule before held-out data.
- **Project-held-out long-form source:** the 21-talk MCIF translation subset
  used by the official IWSLT 2026 SimulST development corpus, En→Zh primary and
  En→De replication. Its input-side visual readiness now passes: 21 videos,
  7,110 s, 283 reviewed transitions and 304 causal states. Eligible-event
  density, power and outcome-side freeze gates remain open.
- **Replication benchmark:** ACL60/60 dev/eval En→Zh, using its external term
  annotations and direct lineage to *Do Slides Help?*. The verified Figshare v2
  supplement adds 884 real talk-video frames covering all 10 talks.
- **Private diagnostic:** Chinese-LiPS-Long for slide dwell/change timing and
  Chinese ASR/terminology probes. It is not a paper-grade ST ranking set without
  independent human references and confirmed use scope.
- **Historical assets:** mTEDx-V and Chinese-LiPS derived artifacts remain on
  Hugging Face, but they no longer define the main paper. mTEDx/TED media are
  blocked from new use without explicit permission.
- **Scripts** in `code/scripts/`: `build_mtedx_v_manifest.py`,
  `extract_frames_by_manifest.py`, `build_chinese_lips_longform.py`
  (`--timeline-dir` for original timeline), `score_visual_signal.py`
  (`--backend ocr|vlm`), `translate_zh_en_draft.py`.

The July 17 direction plan is retained only as history in
[docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md); do not execute it as the
current benchmark contract.

## Next execution milestone

1. Complete independent double source-side annotation of the frozen 100-row ACL
   dev workspace; report event density, negatives, boundary agreement, and
   subtype distribution. The media packets and isolated A/B sheets are ready;
   no annotation label is complete yet.
2. Freeze candidate inventory, source-only packets and target scoring; compare
   audio/document-only, token-budget-matched OCR, current image/semantic
   evidence, and matched stale/wrong evidence under native and controlled-noise
   audio.
3. Require a content-specific advance in first stable correct target decisions,
   preserved final quality and a coherent noise interaction. A small aggregate
   BLEU change without these controls is not sufficient.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
