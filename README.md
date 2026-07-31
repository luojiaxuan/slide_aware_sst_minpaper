# Slide-Aware Simultaneous Speech Translation — investigation record

> **Status (2026-07-31): the naive form of the core hypothesis was tested and
> did not hold.** With audio input and the slide supplied through a vision
> encoder (Qwen3-Omni-30B, 206 segments), a *correct* slide and a *wrong* slide
> produce statistically indistinguishable gains in both quality
> (+2.63 vs +2.29 chrF) and latency (AL −0.202 vs −0.199): the benefit comes
> from image presence, not slide content. **Read
> [docs/FINDINGS.md](docs/FINDINGS.md) first** — it is the single source of
> truth and marks which older documents are superseded.

## Original hypothesis (for the record)

Slides in talks temporally precede the speech that discusses them, so the visual
channel could be a zero-latency lookahead: an asynchronously harvested slide
context might improve terminology, resolve word senses, and let the model commit
earlier. Three mechanisms were posited — M1 anticipation, M2 recognition support,
M3 target-form supply — with slide language as a stratification variable.

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

## Current status (2026-07-17)

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

## Open decision (see docs/FINDINGS.md §5)

1. Re-test on **high-ambiguity speech** (noisy / accented / jargon-dense) where
   the baseline leaves headroom — current baseline chrF is 78.3 on clean,
   scripted lecture audio.
2. Add **relevance selection** (RASST chunkwise retriever over the visual
   channel) so injected evidence is not ~96% irrelevant to the current segment.
3. Write up the **negative result + benchmark + methodology** as the
   contribution.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- No media redistribution (TEDx CC BY-NC-ND); manifests + scripts only.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
