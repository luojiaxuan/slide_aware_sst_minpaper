# Experiment matrix

## Context conditions

| ID | Condition | Inputs | Purpose |
|---|---|---|---|
| C0 | no_context | transcript/audio only | Base SST difficulty |
| C1 | glossary_only | correct + distractor glossary terms | Check term benefit |
| C2 | ocr_terms_current | compact current/nearby slide OCR terms | Primary slide-context baseline |
| C3 | vlm_summary_current | VLM slide summary/caption without raw OCR bulk | Beyond-OCR visual semantics |
| C4 | ocr_plus_vlm | compact OCR terms + VLM summary | Combined automatic slide context |
| C5 | naive_all_context | raw retrieved context | Upper-ish but unsafe context injection |
| C6 | policy | small evidence packet chosen by policy | Main method |
| C7 | oracle | only verified supporting evidence | Upper bound |
| C8 | raw_image_optional | raw image/image-token model input | Optional ablation after text-context signal |

## Context mismatch settings

| ID | Description |
|---|---|
| M0 | matched current slide |
| M1 | previous slide as current |
| M2 | next slide as current |
| M3 | random same-lecture slide |
| M4 | random same-domain slide |
| M5 | noisy OCR current slide |
| M6 | distractor glossary size 10/100/1000 |

## Diagnostic evidence slices

| ID | Description | Expected use |
|---|---|---|
| S0 | ocr_support | OCR/slide terms should help |
| S1 | visual_non_ocr | Frames exist but OCR is sparse; VLM/visual semantics may help |
| S2 | term_homophone | Technical or homophone-prone transcript terms |
| S3 | distractor_risk | Abundant visible text weakly anchored in speech |
| S4 | latency_critical | Deixis or early translation likely needs context |
| S5 | no_context | Little useful slide evidence; methods should not hallucinate |

## Metrics

- BLEU: sacreBLEU on full references.
- COMET: optional if references and compute available.
- HDA: homophone disambiguation accuracy.
- Term F1: normalized target-term precision/recall.
- COR: context overuse rate, i.e., generated terms from non-supporting evidence.
- WSAR: wrong-slide adoption rate on mismatch settings.
- Latency: token delay or chunk delay; StreamLAAL if alignment is available.

## Minimum tables for paper

1. Main results: C0-C7 on M0 matched setting.
2. Robustness: C2/C5/C6 on M1-M4 mismatch settings.
3. Ablation: policy minus pinyin, minus temporal prior, minus conflict penalty.
4. Dataset stats: number of examples by diagnostic evidence slice.
5. Qualitative case table: 3-5 examples.
