# Visual Evaluation Metrics

Keep BLEU, HDA, Term F1, context overuse, and latency. Add visual hard-label metrics:

- `VGA`: visual grounded accuracy for labels with `requires_visual=true`.
- `DRA`: deixis resolution accuracy for `label_type=visual_deixis`.
- `OAA`: object/action accuracy for `label_type in {object, action}`.
- `OGA`: OCR grounded accuracy for labels with `requires_ocr=true`.
- `VHR`: visual hallucination rate, based on unspoken visual distractors.
- `WVAR`: wrong visual adoption rate from `wrong_video`, `wrong_clip`, and `negative_visual` evidence.
- `evidence_precision` and `evidence_recall`: selected supporting evidence quality when labels are available.

Main tables should separate matched visual gains from mismatch robustness. Naive visual injection is expected to improve some grounded labels while increasing VHR/WVAR; the policy should reduce VHR/WVAR while preserving most grounded-label gains.
