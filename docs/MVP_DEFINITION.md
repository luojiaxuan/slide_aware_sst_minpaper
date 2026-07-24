# MVP Definition — the minimal workable, publishable paper

Date: 2026-07-24. Purpose: pin down the smallest coherent paper the current
evidence supports, its single load-bearing claim, and the one experiment that
decides viability.

## The scoped thesis (what we can actually defend)

**Vision-as-anticipation for simultaneous X→En speech translation.** In talks,
slides precede the speech that discusses them, and for X→En the slide text is
in the *target* language (English is the lingua franca of slides). An
asynchronously harvested slide glossary is therefore a zero-latency lookahead
that supplies the target-side form of upcoming terminology, improving
terminology translation on the hard segments where an audio-only streaming
baseline fails.

**Scope boundary (a strength, not a hedge).** The gain is direction-specific.
En→Zh (source-language slides, M2 recognition-support only) shows *no separable
visual gain* over a wrong-slide control under deterministic decoding
(slide−wrong = −0.1, n.s.; copy-rate 0.00). This negative control is exactly
what the M1/M2/M3 decomposition predicts and is what makes the X→En positive
result credible rather than a generic "context helps" artifact.

## The single decision experiment (in flight, Follow-up 12)

X→En (el probe, 40 seg) under deterministic decoding, 4 conditions.
**MVP is viable iff slide > wrong AND oracle > wrong on X→En** (the visual
signal beats the prompt-stabilization control on the primary direction).
- If yes → MVP stands; S3 becomes the clean mechanism contrast; write the paper.
- If no → the observed gains are prompt-stabilization on all directions; pivot
  required (see fallbacks).

## Evidence already in hand (all ours, all reproducible)

1. **Premise** (el→en, vLLM): oracle future terms +7.0 pooled / +18.8 hard
   (p<1e-4), term recall 0.12→0.41. Anticipation has real value.
2. **Extraction** decisive: VLM slide reading recovers +14.1 hard chrF (75% of
   oracle) where OCR is neutral-to-harmful.
3. **Injection form** resolved: deterministic decoding removes the serving-
   nondeterminism confound; prompt injection is powerful (+6.5 hard, p<0.001)
   and benign. Gate demotes to a token-budget optimization; trie biasing is a
   recorded negative result.
4. **Mechanism** (H2): En→Zh copy-rate 0.00 vs X→En active copy channel —
   direct cross-regime evidence that M3 is the load-bearing mechanism.
5. **Benchmark** built and on HF: S1 mTEDx-V (100 talks, live-recoverable video,
   human refs, VLM visual-signal stratification), S2 Chinese-LiPS-Long (21
   sessions, 11.1 h, 1080p slide feed), S3 ACL 60/60 (En→Zh control, tagged
   terms). No synthetic slides in any test set.
6. **Workable system**: session-level streaming pipeline runs end-to-end on a
   full talk with continuous cross-segment state (el: chrF 61.7→62.2 pooled,
   hard 43.9→46.1 under deterministic-style decoding).

## The MVP paper shape (EMNLP-short / workshop-viable, ACL-long with more scale)

- **C1 Setting + mechanism**: vision-as-anticipation, M1/M2/M3, slide-language
  as the stratification variable.
- **C2 Benchmark**: S1/S2/S3, real slides only, Wikidata cross-lingual term
  protocol, per-direction faithfulness.
- **C3 Method + findings**: async off-critical-path slide worker; the
  extraction→injection→determinism chain; X→En gains with En→Zh negative
  control; need-prediction as an open sub-problem (AUC 0.62 pilot).

The honest headline is not "vision always helps SST" but **"vision helps X→En
terminology under speech uncertainty, via target-form supply, when injected
under deterministic decoding — and we show precisely where it does not."** That
is a defensible, mechanism-grounded contribution that OmniFusion/Do-Slides-Help
do not make.

## Fallbacks if the decision experiment fails

1. **Re-anchor on the benchmark + diagnostic contribution** (C2 + the
   determinism/protocol findings + the mechanism controls) — a "how to evaluate
   slide-aware SST honestly, and what breaks" paper. Still publishable; the
   serving-nondeterminism finding alone is a useful negative result.
2. **Deepen X→En with stronger extraction** (image-input VLM translator rather
   than term injection) if term injection is the ceiling.

## What NOT to do

- Do not scale to more languages/data before C3's headline number (slide>wrong
  on X→En) is confirmed.
- Do not report corpus-BLEU deltas as the headline (they are near-zero by
  construction; the signal is term-level and hard-stratum).
- Do not ship synthetic slides in any test set.
