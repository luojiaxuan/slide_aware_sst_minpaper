# Qwen3-VL Context QA and Repair, 2026-07-06

This note records the post-run quality audit for the Qwen3-VL enriched
Chinese-LiPS train artifacts. Local paths are Hyper00 staging paths, not durable
sources of truth until uploaded to Hugging Face.

## Artifacts

- Host/container: Hyper00,
  `sglang-omni-jaxan-vision-sst-0701`
- Repo path in container: `/data/projects/slide_aware_sst_minpaper/repo`
- Base enrichment run:
  `outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711`
- Repair run:
  `outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750`
- Final challenge:
  `outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl`
- Final evidence:
  `outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl`
- Final diagnostic sample:
  `outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.jsonl`
  and `.csv`
- Final QA report:
  `outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json`

## Initial Finding

The initial combined Qwen3-VL artifact had the expected 29,322 unique challenge
rows, but QA found a provenance problem:

| Metric | Initial value |
| --- | ---: |
| Challenge rows | 29,322 |
| Evidence rows | 437,122 |
| `raw_parse_failures` | 3,387 |
| `raw_parse_failure_no_ocr` | 3,387 |
| `raw_parse_failure_with_fallback_summary` | 3,387 |
| `no_ocr_with_summary` | 3,388 |

The failing rows were not safe to treat as Qwen3-enriched context because their
metadata contained truncated raw model JSON while the parsed context had fallen
back to generic summaries such as "OCR not provided in source metadata."

## Review Decision

Claude diagnostic review classified the issue as `NEEDS_FIX`. Accepted
must-fix conclusions:

- Do not salvage partially truncated raw JSON for the full failing subset.
- Prove the failure subset and rerun it with tighter generation settings.
- Add a repeatable QA script and tests so the final artifact can be audited
  before Hugging Face upload or pseudo-reference generation.

## Repair Sequence

| Step | Input rows | Prompt/settings | Remaining failures |
| --- | ---: | --- | ---: |
| Sample validation | 128 | `max_new_tokens=512`, batch 32 | 0 |
| Full repair | 3,387 | `max_new_tokens=512`, batch 32, 4 workers on 2 H200s | 128 |
| Second repair | 128 | `max_new_tokens=768`, batch 16 | 32 |
| Compact repair | 32 | `configs/qwen3_compact_repair_prompt.txt`, `max_new_tokens=384` | 4 |
| Strict repair | 4 | `configs/qwen3_strict_repair_prompt.txt`, `max_new_tokens=256` | 0 |

The final merge kept the original 29,322-row order and replaced exactly 3,387
ids. Evidence and the diagnostic sample were rebuilt from the repaired challenge
file.

The compact and strict prompts are consumed by the tracked
`repo/scripts/enrich_visual_context.py --prompt-file` CLI argument. The merge is
now reproducible with `repo/scripts/merge_visual_context_repairs.py`; later
repair files override earlier rows for the same id.

## Final QA

| Metric | Final value |
| --- | ---: |
| Challenge rows | 29,322 |
| Unique ids | 29,322 |
| Duplicate ids | 0 |
| Missing visual context | 0 |
| Missing enrichment metadata | 0 |
| Empty context | 0 |
| Missing raw model output | 0 |
| `raw_parse_failures` | 0 |
| `raw_parse_failure_no_ocr` | 0 |
| `raw_parse_failure_with_fallback_summary` | 0 |
| `no_ocr_with_summary` | 1 |
| Evidence rows | 526,597 |

The remaining `no_ocr_with_summary` item is `021_26_F_RWLS_171`, an image-like
misty mountain/water scene with objects but no OCR text. It is not a fallback
row and has valid Qwen3-VL metadata.

The final report also checked raw-output provenance directly: all 29,322 rows
have non-empty `context_enrichment.raw_output`; among the 3,387 repaired ids,
the shortest raw output is 137 characters.

`raw_parse_failures=0` is a structural provenance check: it means the stored raw
model output is present and parseable as a JSON object. It is not an independent
claim that the visual content is correct or that every stored field has been
semantically validated against the slide.

Evidence internal consistency after rebuild:

| Source type | Challenge expected | Evidence actual | Delta |
| --- | ---: | ---: | ---: |
| `video_action` | 30,722 | 30,722 | 0 |
| `video_object` | 148,235 | 148,235 | 0 |
| `video_ocr` | 238,320 | 238,320 | 0 |
| `video_scene` | 29,322 | 29,322 | 0 |
| `video_spatial` | 79,998 | 79,998 | 0 |

This table checks that the evidence index was rebuilt from the repaired
challenge without source-count drift. It is an internal consistency check, not
an independent validation of visual content correctness.

Diagnostic sample stats after repair:

| Slice | All rows | Selected rows |
| --- | ---: | ---: |
| `ocr_support` | 20,668 | 337 |
| `visual_non_ocr` | 3,911 | 142 |
| `term_homophone` | 2,879 | 128 |
| `latency_critical` | 1,561 | 115 |
| `distractor_risk` | 358 | 101 |
| `no_context` | 6,531 | 2 |

The `no_context` all-row count rose from the reconstructed pre-repair count of
5,336 to 6,531. This is expected under the current slice rules. Before repair,
all 3,387 failed rows had empty OCR and were classified as `visual_non_ocr`.
After repair, 1,195 of those rows had OCR terms but did not meet the
OCR-transcript overlap, term/homophone, latency, or distractor thresholds, so
they moved from `visual_non_ocr` to `no_context`. No rows outside the 3,387
repaired-id set are needed to explain the slice-count shift.

Repaired-id slice delta:

| Slice | Before repair | After repair | Delta |
| --- | ---: | ---: | ---: |
| `ocr_support` | 0 | 1,864 | +1,864 |
| `visual_non_ocr` | 3,387 | 2 | -3,385 |
| `term_homophone` | 485 | 485 | 0 |
| `latency_critical` | 118 | 118 | 0 |
| `distractor_risk` | 0 | 133 | +133 |
| `no_context` | 0 | 1,195 | +1,195 |

## Verification Commands

Failure subset extraction command, run before repair while the combined artifact
still contained fallback rows:

```bash
python3 scripts/audit_visual_context_quality.py \
  --input outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --output-json outputs/chinese_lips_train/qa/qwen3_vl_context_initial_qa.json \
  --failures-output outputs/chinese_lips_train/qa/qwen3_parse_failures.jsonl \
  --max-examples 20
```

The failure subset predicate is intentionally more conservative than the
`raw_parse_failures` metric: it exports rows with missing raw output,
non-parseable raw output, or the known fallback scene-summary template. Exported
rows preserve their original JSONL lines rather than being reserialized through
the schema.

Repair command shapes:

```bash
python3 scripts/enrich_visual_context.py \
  --input outputs/chinese_lips_train/qa/qwen3_parse_failures.jsonl \
  --output outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repaired_failed_rows.jsonl \
  --provider qwen_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:<0-or-1> \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --batch-size 32 \
  --prefetch-batches 1 \
  --resume
```

```bash
python3 scripts/enrich_visual_context.py \
  --input outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair768_remaining_parse_failures.jsonl \
  --output outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair_compact32.jsonl \
  --provider qwen_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 384 \
  --batch-size 16 \
  --prefetch-batches 1 \
  --max-ocr-terms 24 \
  --prompt-file configs/qwen3_compact_repair_prompt.txt
```

```bash
python3 scripts/enrich_visual_context.py \
  --input outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair_compact32_remaining_parse_failures.jsonl \
  --output outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair_strict4.jsonl \
  --provider qwen_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 256 \
  --batch-size 4 \
  --prefetch-batches 1 \
  --max-ocr-terms 12 \
  --prompt-file configs/qwen3_strict_repair_prompt.txt
```

Merge command shape:

At execution time, the `--base` path below still pointed to the initial combined
artifact. The script writes through a temporary file and then replaces the same
output path after validating row and replacement counts.

```bash
python3 scripts/merge_visual_context_repairs.py \
  --base outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --repair outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repaired_failed_rows.jsonl \
  --repair outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair768_remaining.jsonl \
  --repair outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair_compact32.jsonl \
  --repair outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/repair_strict4.jsonl \
  --output outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --expected-rows 29322 \
  --expected-replacements 3387 \
  --log-json outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/merge_log.json
```

Post-hoc idempotence check with the committed merge driver:

| Check | Value |
| --- | --- |
| Rows | 29,322 |
| Replaced rows | 3,387 |
| Unique repair ids | 3,387 |
| Overridden repair ids | 128 |
| Final SHA-256 | `ac447b395db16448536d80407313fa2205c5a11271dc2bdd3b924876b32fe09e` |
| Merge-driver check SHA-256 | `ac447b395db16448536d80407313fa2205c5a11271dc2bdd3b924876b32fe09e` |

```bash
python3 scripts/audit_visual_context_quality.py \
  --input outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --baseline outputs/chinese_lips_train/data/challenge_verified_qwen_vl_context.jsonl \
  --evidence outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl \
  --sample-stats outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.stats.json \
  --output-json outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json \
  --max-examples 20
```

Evidence and diagnostic rebuild commands:

```bash
python3 scripts/build_visual_context_index.py \
  --config configs/chinese_lips_train.yaml \
  --input outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --output outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl
```

```bash
python3 scripts/sample_diagnostic_subset.py \
  --input outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl \
  --output-jsonl outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.jsonl \
  --output-csv outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv \
  --stats-json outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.stats.json \
  --target-size 500 \
  --per-slice 100 \
  --seed 13
```

```bash
python3 -m pytest \
  tests/test_audit_visual_context_quality.py \
  tests/test_merge_visual_context_repairs.py \
  tests/test_enrich_visual_context.py
python3 -m compileall \
  scripts/audit_visual_context_quality.py \
  scripts/merge_visual_context_repairs.py \
  scripts/enrich_visual_context.py \
  src/slidesst
```

## Status

The Qwen3-VL enriched train artifacts are now suitable for Hugging Face upload
and for the next pseudo-reference generation pass. Upload is still pending, so
the canonical reusable copy does not yet exist outside Hyper00 local staging.
