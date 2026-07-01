# Vision-Aware Pivot Plan

Pivot the scaffold from slide-only context toward visual-evidence-aware streaming speech translation. Keep the `slidesst` package name for now and treat slides as one evidence source alongside video OCR, scene/object/action evidence, glossary, background, and history.

The core rule is faithfulness: visual evidence may ground spoken content, but it must not introduce visible objects, actions, or facts that were not spoken.

## First Implementation Pass

- Extend schema with optional video, visual context, reference metadata, hard labels, and visual evidence fields.
- Keep all existing slide-aware toy data and commands runnable.
- Add VASR-like manifest ingestion without downloading or copying media.
- Add mock keyframe/OCR/VLM providers so tests and toy experiments run without GPU or API keys.
- Add visual evidence building, visual conditions, visual metrics, and table generation.

## First Paper Pilot

- Build 50 candidate examples from local manifests.
- Generate or import English candidate references.
- Export a human review sheet and import verified/edited references.
- Run V0/V2/V3/V5/V6/V7 on matched context and V5/V6 under mismatch.
- Report hard-label metrics, visual hallucination, wrong visual adoption, evidence selection, and latency separately from BLEU.
