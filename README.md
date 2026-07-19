# Vision-as-Anticipation: Slide-Aware Simultaneous Speech Translation

## Background

Simultaneous speech translation (SST) must commit output before hearing the
future. Slides in talks/lectures **temporally precede** the speech that discusses
them — so the visual channel is a legal, zero-latency lookahead: an
asynchronously harvested slide glossary can supply upcoming terminology (often in
the *target* language, for X→En with English slides) without touching the audio
critical path. This project builds the method, the benchmark, and the
faithfulness evaluation for that claim.

Thesis: **vision buys latency** — visual evidence shifts the quality–latency
Pareto (earlier commits at equal quality; higher terminology accuracy at equal
lagging), and a policy must gate it (wrong-slide rejection, no
visible-but-unspoken hallucination).

Lineage: extends RASST (retrieval-augmented SST with prepared glossaries) by
making the glossary self-provisioning from the slide stream. Key neighbors, all
differentiated in [latex/sections/02_related_work.tex](latex/sections/02_related_work.tex):
Do-Slides-Help (EMNLP'25, offline ASR), OmniFusion (arXiv 2512.00234, En→X
SimulST, synchronous vision on the decoding path), Caglayan'20 line (simultaneous
*text* MT with caption images), MCIF (offline En→X benchmark from ACL talks).
Scoop-check verdict (2026-07-17): Level 2–3, delta defensible, window closing —
see `~/research_idea/step1–7.md`.

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

## TODOs (priority order)

1. ~~Oracle anticipation kill test~~ — **DONE 2026-07-19, PREMISE SURVIVES**
   (Qwen3-32B, 224 runs: oracle vs none pooled ΔchrF +7.0 p=0.03; hard stratum
   +18.8 p<1e-4, termR 0.12→0.41; slide-OCR condition ≈ baseline → extraction is
   the bottleneck; wrong-slide neutral). Full report:
   [docs/killtest/KILLTEST_RESULTS.md](docs/killtest/KILLTEST_RESULTS.md).
2. ~~Chinese-LiPS En references~~ — **S2 COMPLETE**: all 21 test videos rebuilt
   on the original session timeline (11.1 h, drift <10 ms) + 3,908 machine-draft
   refs, all on HF. Verified-core post-edit deferred; machine-draft tier labeled.
3. ~~VLM visual-signal pass over mTEDx-V~~ — DONE (visual_signal_vlm.json,
   31/100 talks reclassified, sparsity confirmed real).
4. Streaming policy implementation (async slide worker + evidence gating) and
   faithfulness metrics (wrong-slide adoption, visible-but-unspoken rate).
   **Probe-scale evidence in**: VLM terms + oracle-gate = +7.3 chrF pooled
   (p<0.001) ≈ oracle upper bound; naive LLM gate fails (no selectivity) →
   need-prediction is the method's core problem.
5. ACL 60/60 integration — recon DONE (Anthology-hosted mp4s, zero obstacles,
   [docs/ACL6060_INTEGRATION.md](docs/ACL6060_INTEGRATION.md)); execution ~1 day.
   NEW: trie-constrained contextual biasing logits processor (injection-form
   study verdict; see docs/NIGHT_REPORT_20260719.md).
6. Paper: intro/method drafts; related work is written.

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- No media redistribution (TEDx CC BY-NC-ND); manifests + scripts only.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
