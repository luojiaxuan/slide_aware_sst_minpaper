# Benchmark Plan: Direction Decision and Test-Set Construction

Date: 2026-07-17. Status: **decided — X→En primary; En→X control on ACL 60/60; X→X dropped.**
Principle: test sets must be reviewer-proof (real slides, human or human-verified
references, no synthetic visual evidence in eval); training data is cheap and out
of scope here.

## 1. Direction decision

### X→En — PRIMARY

Reasons, in order of weight:

1. **Mechanism decomposition and where each is measurable.** Slide benefit
   decomposes into three mechanisms: **M1 anticipation** (slides precede the
   speech discussing them by 30-60 s; direction-agnostic), **M2 recognition
   support** (slide terms in the *source* language help the model hear/segment
   rare terms - the mechanism Do-Slides-Help measures for ASR; also helps
   translation, any direction), and **M3 target-form supply** (slide terms in
   the *target* language hand the system the output string to copy; requires
   slide-lang = target-lang, common in X->En because English is the lingua
   franca of slides - observed directly in our probe: Greek speech, English
   slides). The direction asymmetry that matters most is on the *measurement*
   side: for X->En, whether an output token is a "term" is externally
   verifiable (English Wikipedia/Wikidata linking); for En->X no equally
   credible target-side term criterion exists. Slide language is a per-talk
   stratification variable (Chinese-LiPS zh slides = M1+M2; mTEDx Greek talk =
   M1+M3, Russian talk = M1+M2; ACL 60/60 = M1+M2). Paired hypotheses: H1
   (anticipation) - Pareto gains appear in all regimes; H2 (target supply) -
   terminology gains and slide-string copy rates are largest in M3 talks.
2. **Avoids OmniFusion's home field.** They own En→{De,It} SimulST on MCIF;
   competing there head-on makes us a baseline-chaser.
3. **Assets already exist.** mTEDx-V is built, alive-checked, human-referenced,
   stratified. Chinese-LiPS gives the clean-visual stratum with the user's own
   verification language (zh).
4. **RASST lineage continuity** (X→En terminology injection → visual self-provisioning).

### En→X on ACL 60/60 — SECONDARY (terminology-credible control)

ACL 60/60 (IWSLT 2023): academic talks with real slides (slide frames already
proven recoverable by Do-Slides-Help, which used it for ASR), professional gold
translations En->10 languages, and **third-party tagged terminology**. Roles:
(a) the terminology-credibility stratum - term metrics against an external tag
set instead of self-constructed lists (imperfect tags like trivial "model"
entries handled by a pre-registered frequency-based filter, reporting raw and
filtered numbers); (b) the M1+M2 control condition for H2 (English slides =
source language); (c) direct lineage comparability - RASST was evaluated on
ACL6060-zh (HF: gavinlaw/rasst-demo-acl6060-zh-segments), and Do-Slides-Help
reports ASR gains on the same talks, so ASR-vs-translation slide gains can be
contrasted on identical material. Primary target language: zh (user-verifiable);
optionally +de/ja. Earlier objection "ACL6060 too easy / BLEU saturated"
applies to corpus-level BLEU only, not to term-level accuracy, Pareto, or
faithfulness metrics - it does not block this role.

### MCIF — OPTIONAL (demoted)

Same ACL-talk domain as ACL 60/60 but without tagged terms or the user's prior
results; retains one exclusive property (long-form instruction protocol).
Integrate only if long-form En->Zh becomes a reviewer ask.

### X→X — DROPPED

No assets, no verification ability, no reviewer constituency. Zero-shot X→X can
be one analysis paragraph if the model happens to support it, not a benchmark.

## 2. Test suite design (three strata + optional)

| Stratum | Dataset | Directions | Visual regime | References | Status |
|---|---|---|---|---|---|
| S1 realistic-noisy | **mTEDx-V** (100 talks, ~18 h) | es/fr/it/ru/el→en | Sparse: ~12% frames w/ text, 58 near-zero talks (measured) | Human (TED translators, via mTEDx) | **DONE** (HF: gavinlaw/mtedx-v-eval) |
| S2 clean-strong | **Chinese-LiPS-Long** | zh→en | Dedicated 1080p slide feed, 100% coverage, never occluded | Machine-draft + human post-edit (two-tier, FLORAS-style test/test_verified) | Audio+manifests done; **En refs = the one real cost** (§3) |
| S3 term-credible | **ACL 60/60** | en→zh (opt. de/ja) | Real ACL slides (source-lang) | Professional gold (IWSLT) + tagged terms | Integration + frame recovery |
| T optional | MCIF subset | en→zh | Real ACL slides | MCIF-provided | Only if long-form En→X asked |

Term measurement: S3 uses the external tagged-term set (transparent frequency
filter, both raw and filtered reported); S1/S2 X->En term-hood is determined by
English Wikipedia/Wikidata entity linking on the reference side - external and
reproducible, avoiding self-constructed term lists (the credibility weakness of
un-annotated Chinese-LiPS).

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
| ACL 60/60 integration | dev+eval, en→zh | ~0 refs (gold exists); video/frame recovery + term filter ~1-2 days | automated |
| MCIF integration (optional) | — | ~0 (download + protocol adapter) | automated |

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
4. ACL 60/60 integration: recover talk videos/frames, adapt term filter,
   run En→Zh control condition.
5. MCIF only if reviewers ask for long-form En→X.
