# Benchmark Plan: Direction Decision and Test-Set Construction

Date: 2026-07-17. Status: **decided — X→En primary, En→Zh secondary, X→X dropped.**
Principle: test sets must be reviewer-proof (real slides, human or human-verified
references, no synthetic visual evidence in eval); training data is cheap and out
of scope here.

## 1. Direction decision

### X→En — PRIMARY

Reasons, in order of weight:

1. **The unique scientific phenomenon lives here.** Non-English talks routinely
   project English slides (observed directly in our mTEDx probe: Greek speech,
   English slide text). For X→En the slide supplies the *target-side surface form*
   of upcoming terms — vision doesn't just disambiguate, it hands the system its
   output vocabulary ahead of time. En→X has no equivalent (slides = source
   language), and ASR papers cannot exploit it by construction. This is the
   sharpest delta against both Do-Slides-Help (ASR) and OmniFusion (En→X).
2. **Avoids OmniFusion's home field.** They own En→{De,It} SimulST on MCIF;
   competing there head-on makes us a baseline-chaser.
3. **Assets already exist.** mTEDx-V is built, alive-checked, human-referenced,
   stratified. Chinese-LiPS gives the clean-visual stratum with the user's own
   verification language (zh).
4. **RASST lineage continuity** (X→En terminology injection → visual self-provisioning).

### En→Zh — SECONDARY (nearly free, run late)

MCIF (CC-BY 4.0, ACL talks, real slides, En→Zh) is ready-made and
user-verifiable on both sides. One experiment table: does the same policy
transfer to En→X where slides are source-language? Expected: terminology gains
shrink (no target-side supply), anticipation gains persist. This *contrast* is
itself evidence for the target-side-supply mechanism, and it meets OmniFusion on
a direction they didn't evaluate (they did De/It SimulST). Verify MCIF reference
provenance during integration.

### X→X — DROPPED

No assets, no verification ability, no reviewer constituency. Zero-shot X→X can
be one analysis paragraph if the model happens to support it, not a benchmark.

## 2. Test suite design (two strata + one transfer set)

| Stratum | Dataset | Directions | Visual regime | References | Status |
|---|---|---|---|---|---|
| S1 realistic-noisy | **mTEDx-V** (100 talks, ~18 h) | es/fr/it/ru/el→en | Sparse: ~12% frames w/ text, 58 near-zero talks (measured) | Human (TED translators, via mTEDx) | **DONE** (HF: gavinlaw/mtedx-v-eval) |
| S2 clean-strong | **Chinese-LiPS-Long** | zh→en | Dedicated 1080p slide feed, 100% coverage, never occluded | Machine-draft + human post-edit (two-tier, FLORAS-style test/test_verified) | Audio+manifests done; **En refs = the one real cost** (§3) |
| T transfer | **MCIF** subset | en→zh | Real ACL slides (source-lang) | MCIF-provided | Integration only, no construction |

Design rationale: S1 and S2 bracket the deployment space. S2 (slides always
available and perfect) answers *"does visual anticipation work at all?"* — if no
gain here, the thesis dies honestly. S1 answers *"does the policy survive
reality?"* — sparse/absent slides must not hurt (faithfulness metrics: wrong-slide
adoption, visible-but-unspoken hallucination). Stratified reporting on S1 uses the
input-side visual_signal labels already computed. **No synthetic slides in any
test set** — synthetic slide generation (à la Do-Slides-Help) is a *training*
technique; putting it in eval hands reviewers a free rejection reason.

## 3. Cost accounting (the only real bill: Chinese-LiPS En references)

Test split totals: 21 videos, 3,908 clips, ≈9.1 h speech (avg 8.4 s/clip,
measured on 3 rebuilt videos).

| Item | Scale | Cost | Who |
|---|---|---|---|
| Orig-timeline reconstruction | 21 videos | Script exists (`build_chinese_lips_longform.py --timeline-dir`); raw JSONs pulled per video via HTTP-range from test.zip; ~1 day compute/IO | done for 3, mechanical for rest |
| Machine-draft En refs (slide-context-aware) | 3,908 clips ≈ 200k zh chars | API $5–15, half a day (`translate_zh_en_draft.py`, needs API key) | automated |
| **Verified core** post-edit (FLORAS-style `test_verified`) | 6 videos ≈ 1,150 clips ≈ 2.8 h speech | **6–8 h user time** at 150–200 clips/h given good drafts | user (zh/en bilingual) |
| Full-set post-edit (optional, later) | 3,908 clips | 20–26 h user time — defer; machine-draft tier is labeled as such | optional |
| mTEDx-V | — | $0 (done) | — |
| MCIF integration | — | ~0 (download + protocol adapter) | automated |

Total to a defensible benchmark: **one API run + one weekend of post-editing.**
Reviewer story: "verified core translated by machine draft + bilingual human
post-edit; full set machine-draft (clearly labeled); realistic stratum uses
existing human references (mTEDx)." This matches accepted practice (FLORAS
test/test_verified; MuST-C/mTEDx volunteer refs).

## 4. Long-form status of Chinese-LiPS

Resolved: the raw release's per-segment JSONs carry original session timestamps
(startTime/endTime). We reconstruct the full session timeline with real
inter-segment silence (drift ≤1 ms; e.g. 130_42_M_TY: 43.0 min session = 37.6 min
speech + 5.4 min gaps). This *is* the original long streaming speech minus
discarded invalid spans; the PPT stream reconstructs on the same timeline
(freeze-frame across gaps). No need to re-source original videos.

## 5. Execution order

1. Kill test first (oracle anticipation probe on S2's 3 rebuilt videos:
   wait-k ± current-slide terms ± oracle future terms → 3 Pareto curves).
2. If premise survives: extend orig-timeline rebuild to all 21 videos; run
   machine drafts; user post-edits 6-video verified core.
3. VLM pass over mTEDx-V visual labels (when API key available) to upgrade
   OCR-lower-bound stratification.
4. MCIF En→Zh integration last (transfer table).
