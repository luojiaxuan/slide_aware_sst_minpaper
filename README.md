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
> **Current paper story:** after giving a SimulST system the same strong
> document/PDF context, test whether the causally available correct current
> slide produces earlier stable correct decisions than a time-, type-, and
> budget-matched same-talk stale/wrong slide. This is the only primary
> estimand; non-term document representations and pixels-beyond-OCR are gated
> secondary questions. The authoritative paper identity, audit, data update,
> and kill test are in
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

## Current paper direction (decided 2026-07-31)

- **Single primary question:** does correct current-slide content beat a
  matched same-talk stale/wrong slide after both receive the same frozen strong
  document context?
- **Primary outcome:** talk-weighted difference in stable correct decisions
  emitted before source audio resolves the forced-choice ambiguity; SESOI is
  +5 percentage points with a -1 pp final-correctness non-inferiority margin.
- **Secondary only:** C1-C4 baseline variants and C6 pixels beyond OCR/layout.
  Pixel nulls are inconclusive without a gold visual-relation positive control,
  at least 15 eligible talk clusters, and an equivalence test.
- **Project-held-out long-form source:** the 21-talk MCIF translation subset
  used by the official IWSLT 2026 SimulST development corpus, En→Zh primary and
  En→De replication. It becomes a visual-tier confirmatory set only after all
  21 corresponding videos and causal slide timelines pass QA.
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

Run the data-density and oracle headroom screen before building the full
automatic ladder:

1. Import the verified *Do Slides Help?* supplement into a frame-only ACL60/60
   inference view; strip the bundled source transcript and use conservative
   frame availability.
2. Blind-label 80-120 candidate evidence-opportunity events on the five ACL dev
   talks and estimate non-term/current-slide/image-specific density.
3. Freeze candidate inventory, source-only packets, and target scoring, then
   compare document-only, correct source-only oracle, and matched wrong oracle.
   Continue only if correct evidence reaches the +5 pp primary SESOI across at
   least 3/5 talks.
4. Only after the oracle passes, reproduce `C0-C3`, compile `C4-C6`, and run the
   frozen automatic ladder. Keep ACL eval and all 21 MCIF talks held out.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
