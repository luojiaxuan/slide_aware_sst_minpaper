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
> **Current research goal:** compare two semantic-context routes in one
> falsifiable pilot. Route B1 precompiles typed, non-terminological context from
> paper/deck/slide materials before the talk and keeps the VLM off the streaming
> path. Route A tests whether image-specific relations add anything beyond
> matched slide OCR/layout propositions. The authoritative route decision,
> shared baselines, metrics, and GO gates are in
> [docs/DUAL_ROUTE_DECISION_20260731.md](docs/DUAL_ROUTE_DECISION_20260731.md).
> [docs/ACL_PAPER_BLUEPRINT_20260731.md](docs/ACL_PAPER_BLUEPRINT_20260731.md)
> remains the detailed Route A contract, not the sole active plan. Lip video and
> slide+lip hybrid experiments remain out of scope. The independent ACL-style
> cross-review and remaining rejection risks are in
> [docs/ACL_BLUEPRINT_REVIEW_20260731.md](docs/ACL_BLUEPRINT_REVIEW_20260731.md).

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

## Current paper direction (decided 2026-07-31)

- **Route B1, conditional GO:** determine whether propositions, discourse
  referents, relations, and scope compiled before a talk improve SimulST beyond
  term memory, entities/abstract prompting, and PDF BM25/RAG.
- **Route A, HOLD inside the same pilot:** promote a vision-aware claim only if
  image-specific visual relations beat schema- and token-matched slide
  OCR/layout propositions on predeclared `visual_relation` events.
- **Route B0, NO-GO:** whole-PDF prompts, generic RAG, term/entity extraction,
  or a new gate over those payloads are strong baselines, not contributions.
- **Primary benchmark:** MCIF long-form scientific talks, En→Zh primary and
  En→De replication. It has 21 talks, video/audio and professional references,
  and is the official IWSLT 2026 SimulST development corpus.
- **Replication benchmark:** ACL60/60 dev/eval En→Zh, using its external term
  annotations and direct lineage to *Do Slides Help?*.
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

Build the shared Phase-A dual-route futility screen:

1. Freeze MCIF and ACL60/60 revisions/licenses and reconstruct causal slide
   timelines, including measured dwell and slide-to-speech lead/lag.
2. Connect a long-form IWSLT 2026 runner and reproduce `C0-C3`: audio-only,
   terminology memory, entities/abstract prompt, and PDF phrase boost +
   pretranslated BM25/RAG.
3. Compile `C4-C6`: document proposition/discourse memory, slide OCR/layout
   propositions, and image-specific visual relations. Keep extraction offline
   before the talk and match online context budgets.
4. Blind-label 200-300 context-critical events, then run `C0-C7` on five
   ACL60/60 dev talks at native and +5 dB. Keep all 21 MCIF talks held out until
   one route passes its gate and the final system is frozen.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
