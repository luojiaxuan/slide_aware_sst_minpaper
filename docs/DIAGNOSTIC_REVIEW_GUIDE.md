# Diagnostic Review Guide

这份文档说明如何审核
`qwen3_32b_diagnostic500_review_sheet_20260707`。目标是把当前
Qwen3-vs-Qwen3 self-BLEU 诊断集升级为可用于 paper claim 的人工诊断集。

## Artifact

- HF repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
- HF tag: `qwen3_32b_diagnostic500_review_sheet_20260707`
- HF path: `annotation/qwen3_32b_diagnostic500_review_sheet_20260707/`
- Local staging path:
  `outputs/chinese_lips_train/annotation/diagnostic_review_sheet_500_qwen3_context_experiments_20260707.csv`

The sheet contains Chinese-LiPS-derived text, so keep it private unless upstream
redistribution permission is obtained.

## Review Goals

1. Produce an independent English reference for each row.
2. Mark whether the row truly needs visual context, OCR context, both, or
   neither.
3. Mark supporting evidence ids that legitimately help translate spoken
   content.
4. Mark system hypotheses that hallucinate visible but unspoken information.
5. Record reviewer notes for ambiguous or low-quality rows.

## Columns To Fill

| Column | Allowed values | Meaning |
| --- | --- | --- |
| `human_reference_en` | free-form English | Independent translation of the spoken Chinese transcript. Translate speech, not the slide. |
| `reference_quality` | `accept`, `edit`, `reject`, `unclear` | Quality of `candidate_reference_en`. Use `edit` when it is mostly correct but needs changes. |
| `requires_visual` | `true`, `false`, `unclear` | Whether non-OCR visual evidence is needed to translate the spoken content correctly. |
| `requires_ocr` | `true`, `false`, `unclear` | Whether on-screen text/OCR is needed to translate the spoken content correctly. |
| `supporting_evidence_ids` | pipe-separated ids | Evidence ids from `v4_evidence_packet` or `v6_evidence_packet` that support the translation. |
| `hallucination_conditions` | pipe-separated condition names | Conditions whose hypothesis adds visible but unspoken content. |
| `reviewer_notes` | free-form text | Explain uncertainty, missing context, bad source transcript, or why an evidence id is supporting. |

## Faithfulness Rule

Visual evidence can disambiguate spoken words, deixis, objects, actions, and
on-screen text. It must not add a visible object, action, label, statistic, or
fact unless that content is actually spoken or clearly referred to by the
speaker.

Examples:

- If the speaker says "this process" and the slide shows an insurance claim
  workflow, using "insurance claim workflow" can be supporting if the transcript
  makes the reference clear.
- If the speaker only says "this is very important" and the slide shows "KFC",
  adding "KFC" is a hallucination unless KFC is spoken or clearly referred to.
- Proper nouns or technical terms copied from OCR are acceptable only when the
  spoken transcript anchors them.

## Suggested Review Order

1. Rows where `reference_audit_severity=review`.
2. Rows where `reference_audit_flags` contains `copied_visual_text` or
   `length_ratio_review`.
3. Rows where `hyp_V4_ocr_plus_visual`, `hyp_V6_policy_visual`, and
   `hyp_V8_wrong_visual` differ materially.
4. Rows with residual Chinese characters in any hypothesis.
5. Remaining rows for coverage across domains and lecture ids.

## Output Back Into The Pipeline

After review, save the completed CSV as a new private HF artifact and import it
into a new verified diagnostic JSONL. The next evaluation should use
`human_reference_en` as the reference and should compute HDA, evidence
precision/recall, visual hallucination, and wrong-visual adoption only on rows
with completed labels.
