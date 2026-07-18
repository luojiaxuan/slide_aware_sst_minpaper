# Slide/Context-Aware SST MVP

## Decision

Do not replace Chinese-LiPS now, but do not frame the paper as pure
vision-aware SST yet. The safer current positioning is:

> Slide/context-aware simultaneous speech translation under latency constraints.

In this framing, OCR terms, VLM captions, slide/deck topic, and future
speaker/topic context are all evidence sources. Raw vision is an ablation and a
possible later strengthening, not the only reason the paper exists.

## Why This Is Safer

Chinese-LiPS is a presentation dataset, not a natural image caption dataset and
not a speech translation dataset. Its strongest visual signal may be slide text,
terms, formulas, diagrams, and topic cues. If the paper promises generic visual
reasoning and OCR wins, the story weakens. If the paper promises latency-aware
context selection from slide evidence, OCR being strong is an expected and useful
result.

The core research question is:

> Can pre-available or streaming slide-derived evidence improve Chinese-to-English
> simultaneous speech translation under a fixed latency budget, while avoiding
> distractor-context hallucination?

## Data Strategy

### Main Dataset

Keep Chinese-LiPS as the main dataset because it matches the target application:
Chinese speech, presentation slides, speaker explanation around slides, and
technical/domain terminology.

Current caveat: the staged raw train archive `train.z01` through `train.z08`
recovers 29,322 train items with PPT frames, not the full 30,341 train rows. The
archive appears to be missing tail bytes or a final part. Treat the 29,322-row
frame-backed subset as the active train subset, not the complete train split.

### Human Diagnostic Subset

Do not rely on Qwen pseudo references alone for paper-grade SST claims. Build a
500-1,000 item human English translation diagnostic subset from Chinese-LiPS
test or train/test mix, stratified by evidence type:

- `ocr_support`: key terms are likely present in slide OCR.
- `visual_non_ocr`: frames exist but OCR is sparse; diagrams/images/layout may
  matter.
- `term_homophone`: transcript contains technical or homophone-prone terms.
- `distractor_risk`: visible slide text is abundant but weakly anchored in the
  spoken transcript.
- `latency_critical`: early translation may need slide/topic context before the
  source sentence is complete.
- `no_context`: little useful slide-derived evidence is available.

Training can use pseudo references. The main evaluation table needs at least one
human-reference subset.

### ST-Native Sanity Check

Add an ST-native no-visual control, such as BSTC, to validate the streaming
translation/latency pipeline independently of Chinese-LiPS pseudo references.
This is not a replacement for Chinese-LiPS because it lacks slides.

## MVP Context Sources

Use cheap, automatic, reproducible slide-derived context first:

| Source | MVP role | Main use |
| --- | --- | --- |
| Current/nearby slide OCR | Primary baseline | Terms, named entities, formulas, titles |
| OCR-derived glossary | Primary baseline | Terminology control |
| VLM slide caption/summary | Beyond-OCR ablation | Diagrams, charts, visual semantics |
| Deck/topic metadata | Low-cost prior | Domain and section context |
| Speaker bio/external docs | Later extension | Do not include in MVP main line |
| Manual context | Human evaluation only | Do not build large manual context for MVP |

## Experiment Shape

Main conditions:

- `NoContext`: streaming transcript/audio only.
- `OCRTerms`: compact slide OCR and OCR-derived terms.
- `VLMSummary`: VLM slide summary without raw OCR bulk.
- `OCRPlusVLM`: combined slide-derived text context.
- `NaiveAllContext`: raw retrieved context, expected to overuse.
- `Policy`: latency-aware evidence selection and rejection.
- `WrongContext`: random/previous/next slide evidence as a negative control.
- `Oracle`: verified supporting evidence upper bound.

Raw image or image-token fusion should remain an optional ablation until the
textual context pipeline proves there is signal and the slices show beyond-OCR
headroom.

## Success Criteria

The MVP succeeds if:

1. Relevant slide-derived context improves term or named-entity accuracy over
   `NoContext`.
2. The gain is visible under low-latency streaming settings.
3. Wrong or random context does not produce the same gain.
4. The policy reduces distractor adoption compared with `NaiveAllContext`.
5. VLM context improves a clearly defined `visual_non_ocr` or low-OCR slice, or
   the paper can honestly report that OCR/terminology is the dominant useful
   evidence in slide-based SST.

## Immediate Next Steps

1. Generate a diagnostic subset candidate sheet with slice labels.
2. Run fresh pseudo-reference generation on the 29,322 frame-backed train subset.
3. Add OCR/VLM extraction only as reproducible automatic context generation.
4. Build distractor-context runs early; do not wait until the final ablation.
5. Prepare a human English translation workflow for 500-1,000 selected items.
