# Benchmark Plan: Direction Decision and Test-Set Construction

Date: 2026-07-17. Status: **decided — direction-general framing; X→En primary evidence; En→X control on ACL 60/60; optional X→X generality stratum (mTEDx es→fr/it); unified base model across all strata.**
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
   side, and it is quantitative, not qualitative: term-hood and correct
   renderings are verified via Wikidata cross-lingual entity records (QID ->
   target-language label + aliases, matched with normalization), which works in
   every direction but with unequal label coverage - near-universal for
   English, substantially lower for zh/fr technical terms. Each stratum
   therefore reports its term-coverage rate and scores terms only on the
   covered subset; X->En stays primary as the highest-coverage, fullest-
   measurement direction. Slide language is a per-talk
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

### X→X — no dedicated benchmark, but an optional generality stratum (S4)

What was dropped (and stays dropped): constructing non-En↔non-En test sets as a
main line — no verification ability, no credible term criterion. What is IN: the
**unified-system framing**. One multilingual base model + one direction-agnostic
policy is evaluated across all strata; unified multilingual models are exactly
where term confusion / mishearing / language confusion bite hardest, which is
the deployment pain the visual channel treats (goes in the intro motivation).
Cheap honest X→X evidence exists: mTEDx ships human-translated X→X pairs
(es-fr, es-it, es-pt, fr-es, fr-pt, it-es) over the same talk domain and the
same video-recovery pipeline as mTEDx-V. Optional **S4**: extract es→fr/it with
the existing script. Reporting: XCOMET-vs-AL Pareto and slide-string copy-rate
primary; term accuracy secondary on the Wikidata-covered subset (fr label
coverage reported). Do NOT lead with corpus BLEU/XCOMET deltas - context
injection historically moves aggregate metrics little (the Do-Slides-Help /
ACL6060-saturation lesson); the signal concentrates in term-level and
latency-side metrics. Paper label:
**direction-general**, never "X→X" (avoids promising untestable pairs, keeps
the M3/X→En delta sharp, stays out of OmniFusion's unified-architecture game).

## 2. Test suite design (three strata + optional)

| Stratum | Dataset | Directions | Visual regime | References | Status |
|---|---|---|---|---|---|
| S1 realistic-noisy | **mTEDx-V** (100 talks, ~18 h) | es/fr/it/ru/el→en | Sparse: ~12% frames w/ text, 58 near-zero talks (measured) | Human (TED translators, via mTEDx) | **DONE** (HF: gavinlaw/mtedx-v-eval) |
| S2 clean-strong | **Chinese-LiPS-Long** | zh→en | Dedicated 1080p slide feed, 100% coverage, never occluded | Machine-draft + human post-edit (two-tier, FLORAS-style test/test_verified) | Audio+manifests done; **En refs = the one real cost** (§3) |
| S3 term-credible | **ACL 60/60** | en→zh (opt. de/ja) | Real ACL slides (source-lang) | Professional gold (IWSLT) + tagged terms | Integration + frame recovery |
| S4 optional | mTEDx X→X (es→fr/it) | es→fr, es→it | Same talks/pipeline as S1 | Human (mTEDx) | Pareto + copy-rate primary; Wikidata-covered term accuracy secondary (coverage reported) |
| T optional | MCIF subset | en→zh | Real ACL slides | MCIF-provided | Only if long-form En→X asked |

Term measurement (unified protocol, all strata): entity-link reference-side
terms to Wikidata QIDs, take target-language labels+aliases, match with
normalization; report per-stratum term-coverage rate and score only the covered
subset. S3 additionally uses the official tagged-term set (transparent
frequency filter, raw and filtered both reported). This keeps every term claim
externally verifiable and avoids self-constructed lists (the credibility
weakness of un-annotated Chinese-LiPS). Expected coverage: En >> zh/fr - the
quantitative form of the direction asymmetry.

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
| **Verified core** post-edit (FLORAS-style `test_verified`) | 6 videos ≈ 1,150 clips | **DEFERRED** (2026-07-19: user unavailable) — zh stratum ships machine-draft-labeled; headline quality claims lean on S1 human refs (mTEDx) and S3 gold (ACL 60/60) | deferred |
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
