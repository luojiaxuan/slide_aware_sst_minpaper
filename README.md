# Slide-Aware Simultaneous Speech Translation — investigation record

> **Status (2026-08-01).** The 21-talk MCIF translation subset is now
> materialized without extracting or reading references. All 21 videos passed
> frozen hash checks and visual inspection. A 1 s reference-free pass produced
> 283 visually reviewed transition candidates and 304 conservative causal
> states including initial states; each state unlocks only after two stable
> frames. This establishes **visual/timing readiness, not translation benefit**.
> The hash-bound 304-state private Qwen3-VL-32B source-only prescreen is now
> complete and frozen at private HF revision
> [`5da477ff`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/5da477ff7d199dbded0ffe44d6b41b9cd8c8e75d).
> Final QA has 304/304 unique rows, 0 parse failures and 0 empty contexts;
> 303 rows contain a spatial-relation candidate. This confirms descriptive
> material exists, but not that descriptions are correct or pixels beat OCR.
> The output cannot become a human label, eligibility decision, sample filter,
> suggested annotation answer, or paper result.
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
> headroom，不是人工 event density 或 ST gain。v1 的 100 个 media packets 已生成并通过
> no-target/reference/model-output 检查，但 v1 annotation order 已被方法审计否决，不能直接
> 使用其 A/B sheets。v2 已生成 100-frame blinded author view，当前状态是
> `AUTHORING_VIEW_READY_NO_LABELS`。见
> [docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md](docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md)。
> Media workspace 已冻结在 private HF revision
> [`3199207c`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-events/tree/3199207c66b159ab39f662a32e0f6d633c9c2b79)。
> Identifier-hardened author-only r4 view 已冻结在 private HF revision
> [`bbbbdbf5`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-event-author-v2/tree/bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a)；
> r3 在任何人工标签产生前已 superseded。
>
> The frozen Chinese-LiPS five-condition diagnostic is now complete. With
> Qwen3-Omni-30B over 206 segments, current slide, same-talk wrong slide,
> unrelated-talk scientific slide, and blank image all produce similar gains
> over audio-only. `slide - wrong` is only `+0.113 chrF` with descriptive 95% CI
> `[-0.693, +0.894]`, while `blank - none` is `+1.545 [0.644, 2.482]`.
> Therefore **current-page semantic use is not established**; the simplest
> explanation is a generic vision-slot/decoding perturbation. This remains a
> single-talk, machine-reference mechanism diagnostic, not paper evidence. See
> [docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md](docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md)
> and **read [docs/FINDINGS.md](docs/FINDINGS.md) first** for claim confidence
> and falsification criteria.
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
> A narrower 2026-08-01 collision audit rates the broad claim **Level 2 high
> overlap** because OmniFusion already combines scientific-talk slides, SimulST
> and earlier/stable commitments. The remaining paper space is the frozen causal
> event study: correct-target commit lead before audio sufficiency, content
> controls and acoustic-noise interaction. See
> [docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md](docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md).
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

What survived testing is narrower: supplying any image slot changes quality and
latency, but current-page content does not outperform wrong or blank controls in
the Chinese-LiPS probe. What remains untested is the paper-specific question:
whether causally earlier, content-matched slide evidence produces stable target
decisions before the audio becomes sufficient, especially under controlled
noise and across multiple talks.

Lineage and neighbours (survey retained in
[latex/sections/02_related_work.tex](latex/sections/02_related_work.tex)):
RASST (retrieval-augmented terminology for SST), Do-Slides-Help (EMNLP'25,
offline ASR), OmniFusion (En→X SimulST with synchronous vision), Caglayan'20
line (simultaneous *text* MT with caption images), MCIF. These works occupy the
broad slide/visual-anticipation claim; this project must contribute causal
content attribution and stronger event-level evidence rather than another
aggregate image gain.

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

1. Complete v2 frame-only canonical item authoring, hash-lock the questions,
   using the implemented localhost-only blinded authoring UI, then use the
   sequential backend for two independent full-causal-prefix audio trajectories
   and the implemented localhost-only validator UI for a disjoint two-person
   frame-only cohort. v1 media remains
   reusable, but its A/B annotation sheets are superseded. No human label is
   complete yet.
2. Freeze candidate inventory, source-only packets and target scoring; compare
   audio/document-only, token-budget-matched unordered OCR, layout/structure-
   preserving text, current raw image, and matched stale/wrong evidence under
   native and controlled-noise audio. This ladder separates “context helps”
   from “raw pixels are necessary.”
3. Require a content-specific advance in first stable correct target decisions,
   preserved final quality and a coherent noise interaction. A small aggregate
   BLEU change without these controls is not sufficient. The fail-closed event
   scorer and exact 16-condition acoustic grouping are implemented in
   [docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md](docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md);
   it now replays immutable tokenizer IDs, binds source media/extractor identity,
   separates pre-run contract from post-run result attestation, commits raw
   annotation/adjudication artifacts before output, captures start/end process-tree
   isolation, and includes an executable external audio broker. Its synchronized
   talk-level frontier prevents every condition/event/noise stream from seeing a
   later audio time until all current-time hypotheses commit. The production
   Qwen3-Omni worker and audited shard merger are now implemented and tested;
   paper-grade fresh generation remains blocked on the read-only `network=none`
   narrow-mount container rebuild and unfinished human outcome artifacts. No ACL
   system result exists yet.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
