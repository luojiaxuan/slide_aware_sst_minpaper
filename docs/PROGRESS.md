# Progress

This file is the Git source-of-truth progress log. Keep milestone state here
instead of chat transcripts, Notion pages, or shared-machine scratch paths.

## 2026-07-01 Chinese-LiPS Test Pilot

- Created the public GitHub repo:
  <https://github.com/luojiaxuan/slide_aware_sst_minpaper>.
- Staged Chinese-LiPS test/pilot assets on Hyper00.
- Built the full test split:
  - 3,908 Chinese-LiPS test challenge items.
  - 75,634 visual evidence records.
- Output root on Hyper00:
  `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_test`.
- Ran mock stress comparison over all 3,908 test items for visual/context
  conditions including no context, OCR, VLM summary, naive visual context,
  policy visual context, and oracle.
- Added local HF reference generation support and generated 200
  `Qwen/Qwen2.5-7B-Instruct` pseudo references.
- Audit result for Qwen200: 187 pass, 9 review, 4 reject.
- Main teacher failure modes: Chinese leakage, copied visual/OCR text,
  placeholder text, and overlong expansion.

## 2026-07-03 Train Split and SFT Seed

- Added train metadata and SFT export code:
  - `repo/scripts/build_chinese_lips_manifest.py`
  - `repo/scripts/build_training_data.py`
  - `repo/configs/chinese_lips_train.yaml`
- Downloaded and extracted `processed_train.zip`.
- Built initial train artifacts from `meta_train.csv` with missing frames
  allowed:
  - Manifest: `/data/datasets/chinese_lips/train/chinese_lips_train_manifest.jsonl`
  - Challenge items:
    `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified.jsonl`
  - Evidence index:
    `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence.jsonl`
- Generated 500 train pseudo references with Qwen2.5-7B.
- Audit result for train Qwen500: 471 pass, 26 review, 3 reject.
- SFT pass rows: `outputs/chinese_lips_train/training/qwen500_sft_pass.jsonl`,
  471 rows.

## 2026-07-03 Raw Train PPT Frame Recovery

- Downloaded raw train split files:
  `/data/datasets/chinese_lips/train.z01` through `train.z08`.
- Standard `zipinfo`/`unzip` could not list the archive because the central
  directory was absent from the staged split files.
- Added `repo/scripts/recover_zip_ppt_frames.py` to recover local ZIP members
  by streaming local headers without relying on the central directory.
- Recovery result:
  - 118,519 local ZIP members scanned.
  - 29,323 PPT members seen.
  - 29,317 new PPT frames written.
  - 5 existing frames skipped.
  - 1 EOF failure:
    `train/031_19_M_TY/PPT/031_19_M_TY_160_PPT.mp4`.
- Rebuilt train artifacts without allowing missing frames:
  - 29,322 train manifest rows.
  - 29,322 challenge rows.
  - 29,322 evidence rows.
- Caveat: Chinese-LiPS train has 30,341 rows, so 1,019 rows currently lack
  recoverable PPT frames from the staged raw archive.

## 2026-07-06 Slide/Context-Aware Reframe

- Added `code_plan/SLIDE_CONTEXT_AWARE_MVP.md`.
- Updated start-here, MVP decisions, and experiment matrix docs.
- Durable decision: keep Chinese-LiPS, but frame the first paper as
  slide/context-aware SST under latency constraints, not pure vision-aware SST.
- Added `repo/scripts/sample_diagnostic_subset.py`.
- Generated diagnostic samples:
  - Train:
    `outputs/chinese_lips_train/annotation/diagnostic_sample_500.jsonl`
    and `.csv`.
  - Test:
    `outputs/chinese_lips_test/annotation/diagnostic_sample_500.jsonl`
    and `.csv`.
- Train sample before VLM enrichment:
  - selected 500
  - `visual_non_ocr`: 500
  - `term_homophone`: 201
  - `latency_critical`: 175
- Test sample:
  - selected 500
  - `ocr_support`: 389
  - `visual_non_ocr`: 165
  - `term_homophone`: 138
  - `latency_critical`: 495
  - `distractor_risk`: 29
- Interpretation: test split confirms Chinese-LiPS is strongly slide/OCR-context
  heavy. Train had recovered frames but lacked OCR/VLM metadata, so automatic
  slide context enrichment became the next concrete step.

## 2026-07-06 Qwen2.5-VL Slide Context Enrichment Pilot

- Added `repo/scripts/enrich_visual_context.py`.
- Added `repo/tests/test_enrich_visual_context.py`.
- Validated `Qwen/Qwen2.5-VL-3B-Instruct` on Hyper00 as a pipeline and
  throughput pilot.
- Smoke test showed usable slide OCR-like terms and scene summaries.
- Long run:
  - Run id: `qwen_vl_train_20260706_095137`
  - Run directory:
    `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/enrichment/qwen_vl_train_20260706_095137`
  - Model: `Qwen/Qwen2.5-VL-3B-Instruct`
  - GPUs: 8 H200 processes, one shard per GPU.
  - Input: 29,322 frame-backed train challenge items.
- Completion:
  - 8 shards completed: 29,322 enriched challenge items total.
  - Combined output:
    `outputs/chinese_lips_train/data/challenge_verified_qwen_vl_context.jsonl`
  - Enriched evidence:
    `outputs/chinese_lips_train/index/evidence_qwen_vl_context.jsonl`
  - Evidence rows: 310,601.
  - Enriched diagnostic sample:
    `outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen_vl_context.jsonl`
    and `.csv`.
- Enriched train slice counts:
  - all 29,322:
    - `ocr_support`: 20,920
    - `visual_non_ocr`: 4,882
    - `term_homophone`: 2,879
    - `latency_critical`: 1,561
    - `distractor_risk`: 256
    - `no_context`: 5,855
  - selected 500:
    - `ocr_support`: 317
    - `visual_non_ocr`: 149
    - `term_homophone`: 151
    - `latency_critical`: 45
    - `distractor_risk`: 120

### Status Update

This Qwen2.5-VL run is not the final data construction teacher. It validated the
pipeline, sharding, prompt, parsing, and evidence rebuild. For dataset quality,
the next enrichment run should use the newer cached `Qwen/Qwen3-VL-8B-Instruct`
or a stronger Qwen3-VL variant if available.

## 2026-07-06 Qwen3-VL Migration

- Rationale: training-set construction quality matters more than the small-model
  throughput pilot. Qwen3-VL is the newer and stronger Qwen VL generation, and
  Hyper00 already has `Qwen/Qwen3-VL-8B-Instruct` in the shared HF cache.
- Code change: `repo/scripts/enrich_visual_context.py` now uses
  `AutoModelForImageTextToText` instead of the Qwen2.5-specific model class, so
  the same enrichment entry point can run Qwen3-VL and future VL models.
- Smoke validation: `Qwen/Qwen3-VL-8B-Instruct` successfully enriched two test
  examples on Hyper00 and produced compact JSON without markdown fences.
- Active Qwen3 run:
  - Run id: `qwen3_vl_train_20260706_164650`
  - Run directory:
    `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/enrichment/qwen3_vl_train_20260706_164650`
  - Initial stability check: 8 shard processes ran successfully, one per H200,
    about 18.4GB GPU memory per process, no Traceback/OOM detected.
  - Resource update: the original low-batch run was stopped. Shards 0-7 are
    partial and resumable because the enrichment command uses `--resume`, but
    no shard should be resumed with the old one-sample-per-GPU configuration.
    Current partial line counts are shard_0=1361, shard_1=1372, shard_2=1354,
    shard_3=1523, shard_4=1490, shard_5=1259, shard_6=1384, shard_7=1332.
- GPU utilization profiling:
  - Code change: `repo/scripts/enrich_visual_context.py` now supports
    `--batch-size` for batched Qwen-VL image/text generation. It also flushes
    pending batched items before writing missing-frame or already-contextualized
    skip items, so JSONL output order is preserved when skip writes are
    interleaved with batched generation.
  - Batched generation safety: the Qwen-VL processor tokenizer is forced to
    `padding_side="left"` before generation. Hyper00 real-model smoke checks
    confirmed left padding and produced semantically consistent OCR and scene
    outputs for batch=1 vs batch=2 and duplicate-image batch inputs. Exact
    object/action wording is not guaranteed to be byte-identical across batch
    shapes, so downstream quality checks should compare semantic fields rather
    than raw strings. New rows now record `batch_size` in
    `visual_context.metadata.context_enrichment` so single-sample and batched
    generations can be audited separately.
  - Tests: Hyper00 container passed `python3 -m pytest
    tests/test_enrich_visual_context.py` with batch ordering and missing-frame
    skip coverage; local syntax check passed
    `python3 -m compileall scripts/enrich_visual_context.py src/slidesst`.
  - Single-process steady-state profiles showed that larger batch alone was not
    enough on representative random train frames. With `max_new_tokens=256`,
    batch 64/80/96 reached only 72.9/73.2/74.7% average utilization, despite
    p50 utilization near 98-100%. Adding `--prefetch-batches 1` improved those
    points to 76.4/78.2/80.3%, confirming that the bottleneck was batch-level
    CPU/image/processor gaps rather than raw model compute.
  - Code change: `repo/scripts/enrich_visual_context.py` now splits Qwen batch
    work into `prepare_batch` and `extract_prepared`, and supports
    `--prefetch-batches 1` to overlap the next batch's CPU image/processor work
    with the current batch's GPU generation. Hyper00 container tests passed
    `python3 -m pytest tests/test_enrich_visual_context.py` with 6 tests,
    including prefetched missing-frame ordering coverage and a Qwen-shaped fake
    extractor concurrency test for processor locking and output order.
  - Validated high-utilization configuration: 2 workers on one H200, each with
    `--batch-size 48`, `--max-new-tokens 256`, and `--prefetch-batches 1`,
    sustained 91.9% GPU utilization after warmup with about 110GB peak memory.
    A 2-GPU, 4-worker short run with `--batch-size 56` per worker sustained
    GPU0 93.3/93.9/97.2% and GPU1 91.5/91.2/94.2% average utilization after
    20/40/60 seconds, with about 122GB peak memory per GPU. The profiling
    evidence and command shapes are recorded in
    [`docs/QWEN3_GPU_PROFILING_20260706.md`](QWEN3_GPU_PROFILING_20260706.md).
  - Active Qwen3 production enrichment:
    - Run id: `qwen3_vl_train_bs56x2_2gpu_20260706_214711`
    - Run directory:
      `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711`
    - Configuration: GPU 0 and 1 only, 2 workers per GPU, 4 shards total,
      `--batch-size 56`, `--max-new-tokens 256`, `--prefetch-batches 1`,
      `--resume`.
    - Startup validation: all four workers reached 616 rows per shard while
      GPU0/GPU1 were both at 100% utilization and about 122GB memory per GPU.
    - Safety status: this full-train run is the longer image-distribution
      validation for the batch56/two-workers-per-GPU configuration. Treat
      downstream Qwen3-VL artifacts as pending until all shards complete and
      combine/schema/sample checks pass.
    - Monitoring: Codex heartbeat automation
      `monitor-vision-aware-sst-qwen3-run` checks this thread every 30 minutes
      for utilization, shard counts, worker status, and completion handling.
  - Completion:
    - Shards completed: shard_0=7,331, shard_1=7,331, shard_2=7,330,
      shard_3=7,330.
    - Combined challenge:
      `outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl`,
      29,322 rows, 29,322 unique ids.
    - Initial coarse checks passed for row counts and ids, but the stricter QA
      pass found 3,387 rows with truncated raw model JSON and generic fallback
      context. Those rows were not accepted as final Qwen3-VL context.
  - QA and repair:
    - Added `repo/scripts/audit_visual_context_quality.py` and
      `repo/tests/test_audit_visual_context_quality.py`.
    - Claude diagnostic review agreed that the truncated rows should be rerun,
      not salvaged from partial raw JSON.
    - Targeted repair run:
      `outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750`.
    - Repair sequence: 512-token rerun fixed 3,259/3,387 rows, 768-token rerun
      fixed 96/128 remaining rows, compact prompt fixed 28/32 remaining rows,
      and strict prompt fixed the final 4 rows.
    - The final merge replaced exactly 3,387 ids and preserved the 29,322-row
      challenge order.
    - Final QA report:
      `outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json`.
    - Final QA metrics: 29,322 rows, 29,322 unique ids, 0 duplicate ids,
      0 missing visual contexts, 0 missing enrichment metadata, 0 empty
      contexts, 0 missing raw model outputs, 0 raw parse failures, and 1 valid
      no-OCR image scene.
    - Rebuilt evidence:
      `outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl`,
      526,597 rows. Internal source-count deltas are 0 for `video_action`,
      `video_object`, `video_ocr`, `video_scene`, and `video_spatial`, meaning
      the index rebuild is consistent with the repaired challenge fields.
    - Rebuilt diagnostic sample:
      `outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.jsonl`
      and `.csv`, 500 rows.
    - Final selected sample stats:
      `ocr_support`: 337, `visual_non_ocr`: 142, `term_homophone`: 128,
      `latency_critical`: 115, `distractor_risk`: 101, `no_context`: 2.
    - The all-row `no_context` count increased because 1,195 repaired rows
      moved from the previous empty-OCR `visual_non_ocr` bucket to valid
      non-overlapping OCR/context rows that do not meet other diagnostic slice
      thresholds.
    - Detailed QA record:
      [`docs/QWEN3_CONTEXT_QA_20260706.md`](QWEN3_CONTEXT_QA_20260706.md).

## 2026-07-06 Hugging Face Bundle Preparation

- Added `repo/scripts/package_hf_dataset_bundle.py` and
  `repo/tests/test_package_hf_dataset_bundle.py`.
- Verified upstream `BAAI/Chinese-LiPS` metadata on Hugging Face:
  - Revision: `db96948538811029011eee44602438a26710ecd9`
  - License: `cc-by-nc-sa-4.0`
  - Access: gated; terms restrict redistribution of derived works outside the
    research group unless the upstream maintainers grant permission.
- Prepared a private/gated HF upload bundle on Hyper00:
  `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_vl_context_v1`
- HF repo id: `gavinlaw/slide-context-sst-chinese-lips`, variant
  `qwen3_vl_context_v1`.
- Bundle source Git commit:
  `c983c91cbfaaa5f400be556b0fcbb9cd24b6258e`.
- Bundle contents:
  - `data/challenge_verified_qwen3_vl_context.jsonl.gz`, 29,322 rows,
    11,814,902 bytes.
  - `index/evidence_qwen3_vl_context.jsonl.gz`, 526,597 rows, 19,092,763 bytes.
  - `annotation/diagnostic_sample_500_qwen3_vl_context.jsonl.gz`, 500 rows.
  - `annotation/diagnostic_sample_500_qwen3_vl_context.csv`, 500 rows.
  - `annotation/diagnostic_sample_500_qwen3_vl_context.stats.json`.
  - `qa/qwen3_vl_context_qa.json`.
  - `README.md` and `manifest.json` with checksums and access notes.
- Uploaded the bundle to the private HF dataset repo:
  <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
- HF commit:
  `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`
- HF tag: `qwen3_vl_context_v1`
- Verified repo privacy via the HF API: `private=True`.
- Pre-generation gate on the HF revision passed:
  - HF revision checked:
    `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`
  - All uploaded file SHA-256 values matched `manifest.json`.
  - Row counts matched: 29,322 challenge rows, 526,597 evidence rows, and 500
    diagnostic rows.
  - QA fields matched the uploaded report: `missing_raw_output=0` and
    `raw_parse_failures=0`.
  - Re-parsing `context_enrichment.raw_output` with the tracked enrichment parser
    matched stored `visual_context` fields for all rows.
  - Required enrichment metadata keys were present for all rows.

## 2026-07-07 Qwen3-32B Reference Pilot

- Fresh pseudo-reference generation gate used the private HF source revision:
  `gavinlaw/slide-context-sst-chinese-lips@a83770446ded4599bf9d95d2b77cdcc7fe359ef7`.
- Added Qwen3-32B reference-generation support:
  - `repo/configs/chinese_lips_qwen3_hf_reference_pilot.yaml`
  - batched `hf_transformers` generation in `repo/scripts/generate_references.py`
  - Qwen3 `enable_thinking: false` chat-template passthrough
  - optional `system_prompt`
  - `repo/scripts/repair_references.py` for targeted CJK/overlong repairs
  - `repo/scripts/package_reference_generation_bundle.py` for HF artifact packaging
- The first Qwen3-32B smoke exposed default `<think>` output. After setting
  `enable_thinking: false`, the 2-item smoke passed audit.
- 100-row pilot:
  - Base artifact:
    `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs.jsonl`
  - Targeted repair artifact:
    `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl`
  - Repair fixed 3 rows with residual Chinese terms.
  - Final audit: 100 rows, 84 pass, 16 review, 0 reject.
- GPU utilization tuning:
  - batch=16 triggered low-util alerts: 67-83% average in 10-second windows.
  - batch=32 reached 88.3% average in a 10-second manual sample.
  - batch=48 reached 100% in all 10 manual samples, used about 122GiB H200
    memory, completed 48 rows in 45.9 seconds, and audited as 43 pass,
    5 review, 0 reject.
  - Short-pilot recommended single-GPU Qwen3-32B teacher setting was batch=48
    plus targeted repair, pending validation on the longer diagnostic-500
    length distribution.
- Uploaded the repaired 100-row pilot to the private HF dataset repo:
  - HF repo:
    <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
  - HF commit:
    `ee785604ba51a5c65335de12bfcfd99d3c4febff`
  - HF tag:
    `qwen3_32b_reference_pilot_20260706`
  - Path:
    `reference_pilots/qwen3_32b_reference_pilot_20260706/`
- Detailed record:
  [`docs/QWEN3_REFERENCE_PILOT_20260706.md`](QWEN3_REFERENCE_PILOT_20260706.md).
- Diagnostic 500 follow-up:
  - batch=48 OOMed on the first full-diagnostic batch because longer examples
    pushed H200 memory past the available headroom.
  - batch=40 was selected for diagnostic-scale generation. It reached 90%
    average utilization on the 80-row tune run and passed two monitor windows
    during the 500-row run: 97% and 94% average utilization.
  - Base generation:
    `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_v3_batch40.jsonl`
  - Targeted repair:
    `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_repaired.jsonl`
  - Targeted repair fixed 5 rows with residual Chinese characters.
  - Final diagnostic 500 audit: 435 pass, 65 review, 0 reject, 0
    `target_cjk_chars`.
  - Uploaded the repaired diagnostic 500 artifact to the private HF dataset
    repo:
    - HF commit:
      `5ca0c090fc6d76ac50938924b28a57b1026c3043`
    - HF tag:
      `qwen3_32b_reference_diagnostic500_20260707`
    - Path:
      `reference_pilots/qwen3_32b_reference_diagnostic500_20260707/`

## 2026-07-07 Qwen3-32B Diagnostic 500 Context Ablation

- Added batched final-state experiment support:
  - `repo/scripts/run_batched_reference_experiments.py`
  - `repo/configs/chinese_lips_qwen3_diagnostic500_eval.yaml`
- Ran 7 repaired-diagnostic-500 conditions with `Qwen/Qwen3-32B`:
  `V0_no_context`, `V2_ocr_only`, `V3_visual_caption_only`,
  `V4_ocr_plus_visual`, `V5_naive_all_visual`, `V6_policy_visual`, and
  `V8_wrong_visual`.
- GPU utilization:
  - `V5` batch=192 tune reached about 95% average utilization.
  - Full run monitor windows reached 96% and 100% while running `V0`-`V5`.
  - `V6` OOMed at batch=192, then completed with batch=128 and a 99% monitor
    window.
- Output completeness:
  - Each condition has exactly 500 outputs.
  - `V0`-`V5` were generated with batch=192.
  - `V6` and `V8` were generated with batch=128.
- Diagnostic self-BLEU results against repaired Qwen3-32B diagnostic
  references:
  - `V0_no_context`: 76.50
  - `V2_ocr_only`: 83.41
  - `V3_visual_caption_only`: 83.88
  - `V4_ocr_plus_visual`: 85.17
  - `V5_naive_all_visual`: 84.75
  - `V6_policy_visual`: 83.24
  - `V8_wrong_visual`: 81.66
- Interpretation boundary:
  - These BLEU values are pipeline sanity signals only. References and
    hypotheses both come from `Qwen/Qwen3-32B`, so the table can reward
    stylistic self-agreement and evidence overlap rather than translation
    correctness.
  - `V0`-`V5` used batch=192 while `V6`/`V8` used batch=128 after a V6 OOM;
    cross-condition ranking is therefore confounded until uniform-batch or
    batch-sensitivity checks are run.
  - Do not use the current table as a paper-grade claim that one condition is
    better than another.
- Metric caveat:
  - Diagnostic 500 does not yet have manual `hard_label`, `supporting_ids`, or
    verified hallucination labels.
  - HDA, evidence precision/recall, and paper-grade visual hallucination
    metrics are not meaningful yet.
- Uploaded the full experiment bundle to the private HF dataset repo:
  - HF commit:
    `3cc7249d45eca71a4f0b5c06a6b0773efead128a`
  - HF tag:
    `qwen3_32b_diagnostic500_experiments_20260707`
  - Path:
    `experiments/qwen3_32b_diagnostic500_experiments_20260707/`
- Detailed record:
  [`docs/QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md`](QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md).
- Claude hostile review flagged the initial interpretation as too strong
  because the table is Qwen3-32B self-BLEU and V0-V5 used batch=192 while
  V6/V8 used batch=128. Accepted fixes:
  - Reframed BLEU as diagnostic self-BLEU only.
  - Added explicit per-condition batch-size provenance to
    `repo/configs/chinese_lips_qwen3_diagnostic500_eval.yaml`.
  - Verified the final streaming transcript equals `source_transcript` for all
    500 rows.
  - Added tests covering final-state selection and V8 wrong-visual packet
    isolation.
- Batch160 sensitivity:
  - Full V4/V5/V6/V8 rerun used one selected Hyper00 H200, `cuda:0`.
  - Stable GPU monitor windows: V4 99%, V5 100%, V6 91%.
  - V0 batch160 tune was only 83%, so a full all-condition uniform-batch run
    was not continued under the 90% utilization rule.
  - Self-BLEU deltas relative to the parent experiment:
    - `V4_ocr_plus_visual`: 85.1712 to 85.2877, delta +0.1165.
    - `V5_naive_all_visual`: 84.7461 to 84.9740, delta +0.2279.
    - `V6_policy_visual`: 83.2369 to 83.5166, delta +0.2797.
    - `V8_wrong_visual`: 81.6593 to 81.6652, delta +0.0060.
  - Uploaded the batch160 sensitivity bundle to the private HF dataset repo:
    - HF commit:
      `03f59f1babc0c37e778e8f415bc85ab5fb36f573`
    - HF tag:
      `qwen3_32b_diagnostic500_batch160_visual_policy_20260707`
    - Path:
      `experiments/qwen3_32b_diagnostic500_batch160_visual_policy_20260707/`
- Human review sheet:
  - Added `repo/scripts/export_diagnostic_review_sheet.py` and
    `repo/tests/test_export_diagnostic_review_sheet.py`.
  - Generated a 500-row CSV with source transcript, candidate Qwen3 reference,
    reference audit flags, visual/OCR context, V4/V6 evidence packets, all 7
    parent-run hypotheses, and blank human review columns for reference quality,
    visual/OCR requirement, supporting evidence ids, hallucination conditions,
    and notes.
  - Local path:
    `outputs/chinese_lips_train/annotation/diagnostic_review_sheet_500_qwen3_context_experiments_20260707.csv`
  - Uploaded the review sheet bundle to the private HF dataset repo:
    - HF commit:
      `3d681ebe85babdacffe5e984bf59af6cade9c2f1`
    - HF tag:
      `qwen3_32b_diagnostic500_review_sheet_20260707`
    - Path:
      `annotation/qwen3_32b_diagnostic500_review_sheet_20260707/`

## 2026-07-31 Dual-Route Research Decision

- Corrected the collision boundary: EGTA and RASST are terminology-only. They
  rule out a term-only retrieval/gating contribution, not broader proposition,
  discourse, relation, or vision-aware context.
- Audited the IWSLT 2026 extra-context systems. Named entities, abstract prompt,
  phrase boosting, paper pretranslation, and BM25 retrieval are mandatory
  baselines and cannot be the new method.
- Froze the authoritative decision in
  [`docs/DUAL_ROUTE_DECISION_20260731.md`](DUAL_ROUTE_DECISION_20260731.md):
  - Route B1 is conditional GO for typed, non-terminological context compiled
    before the talk;
  - Route A is HOLD inside the same pilot and requires image-specific visual
    relations to beat matched slide OCR/layout propositions;
  - generic Route B0 is NO-GO.
- Defined shared conditions `C0-C7`, term-masked context-critical evaluation,
  matched wrong/shuffled/stale controls, and separate B1/A futility gates.
- Reserved the 21-talk MCIF translation subset as project-held-out. Five
  ACL60/60 dev talks are for route screening only.
- Stopped Chinese-LiPS pseudo-reference expansion and new multimodal training
  pending the shared pilot.

## Superseded Open Items (before paper-story audit)

1. Build automatic reference-free `C1-C2` packets from the frozen ACL60/60
   dev PDFs, implement C3 causal retrieval, and pass the pinned one-talk
   no-reference dry run.
2. Reproduce `C0-C3`: audio-only,
   terminology memory, entities/abstract, and phrase boost + PDF BM25/RAG.
3. Freeze the typed-memory schema and build offline `C4-C6` extraction QA.
   Slide-derived entries may be precomputed but only unlock at their real
   stable-slide timestamps.
4. Blind-label 200-300 term/entity-masked context-critical events on the five
   ACL60/60 dev talks.
5. Run the native/+5 dB `C0-C7` futility screen with matched packet budgets and
   apply the B1/A gates once. Do not access MCIF outputs before a route, system,
   selector and evaluator are frozen.

## 2026-07-31 Phase-A Data and Runner Freeze

- Downloaded and verified the official ACL60/60 attachment:
  SHA256 `5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc`.
  Frozen 10 full talks, 468 dev + 416 eval gold segments, multilingual text/XML
  and tagged-terminology file hashes.
- Built separate five-talk ACL dev inference/scoring views. Inference rows expose
  audio, paper and paper abstract only; scoring paths never enter the inference
  process. Official tagged terms are labels, not C1 runtime hints.
- Corrected MCIF scope: HF revision
  `e24065b919758263cfe5d157057278affe76ea7b` contains 100 long-media talks;
  the IWSLT translation subset contains 21 talks.
- Downloaded the official IWSLT `mcif-long-trans.zip`, verified SHA256
  `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7`,
  and froze the 21 talk IDs from matching audio/PDF filenames without reading
  reference contents.
- Pinned the IWSLT baseline, SimulStream, OmniSTEval and four Qwen model
  revisions in `code/configs/phase_a_c0_c3.yaml`.
- Added a thin context adapter that validates talk/WAV order, exact C0-C3 packet
  schema, scoring-key isolation and 256-token-per-channel ASR/MT budgets. The upstream
  baseline is not vendored because its frozen commit has no LICENSE file.
- Added the only allowed contract-driven launcher. It rejects C3 until causal
  retrieval exists, verifies upstream/toolkit commits and exact offline HF
  snapshots, derives a hash-bound non-reusable run directory, and never accepts
  scoring paths.
- Reconstructed all 468 ACL dev gold segment offsets by exact PCM matching
  against the full talk WAVs and generated a separate OmniSTEval-compatible
  scoring bundle. Adjacent official gold segments can overlap, which is now
  covered by the builder logic.
- Detailed record:
  [`docs/PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](PHASE_A_DATA_RUNNER_FREEZE_20260731.md).

## 2026-07-31 Paper Story and Real-Frame Data Audit

- Locked `docs/DUAL_ROUTE_DECISION_20260731.md` at SHA256
  `ccb8376cbca48328ec1640dae6bd4aa516b07e8161a51c406e6936be7cd48767`
  before literature judgment.
- Ran independent primary-paper search and Scoop-Check. Both returned
  Level 3 / partial overlap.
- Added 2026 collision boundaries:
  - OmniFusion and BOOM occupy slide-aware live/simultaneous translation;
  - VAPO occupies look-then-listen, image/OCR, mismatched-slide, and visual
    interference for slide ASR;
  - visual-context SiMT occupies image-conditioned anticipation/READ-WRITE;
  - EGTA/RASST and Context Helps/DoCIA occupy terminology selection and
    discourse-context components.
- Froze a single paper identity in
  [`docs/PAPER_STORY_DECISION_20260731.md`](PAPER_STORY_DECISION_20260731.md):
  current-slide content attribution after strong document context, measured as
  correct-vs-matched-stale/wrong stable decisions before source-audio
  disambiguation. C1-C4 and pixels-beyond-OCR are gated secondary questions.
- Downloaded the official *Do Slides Help?* Figshare v2 supplement under CC BY
  4.0 and verified outer SHA256
  `f771d3f6f03026ad1510cf6840b47df3406b06b804926ab3ae18af99f663d4cc`.
- Verified its real ACL60/60 frame coverage: dev 468 + eval 416 = 884 frames
  over all 10 talks. These are video frames, not the paper's synthetic
  transcript-to-LaTeX training slides.
- Recorded source, hashes, coverage, local staging, and transcript-leakage
  firewall in
  [`data/manifests/do_slides_help_figshare_v2_20260731.json`](../data/manifests/do_slides_help_figshare_v2_20260731.json).
- Changed the immediate execution order: import a frame-only inference view,
  blind-label 80-120 source-side forced-choice candidates, then run
  document-only versus correct/wrong source-only oracle packets. Automatic
  C3-C6 and GPU inference are conditional on oracle headroom.
- A separate hostile ACL review rejected the first broad “evidence sufficiency
  ladder” revision for circular oracle construction, non-nested conditions,
  five-talk power, subjective timing, underpowered pixel nulls, and missing
  MCIF visual readiness.
- Revised the final story to one primary contrast: after the same frozen C3
  document context, compare correct current-slide C5 against a time/type/budget-
  matched same-talk stale/wrong C5 control. Primary SESOI is +5 pp in stable
  correct decisions before source-audio disambiguation; final-correctness
  non-inferiority margin is -1 pp.
- Added independent candidate/source-packet/target-scoring freezes, fixed-prefix
  forced-choice timing intervals, source-only oracle packets, conservative
  slide-state transitions, a gold visual-relation positive control, and an
  MCIF gate requiring at least 15 eligible talks before confirmatory claims.

### Exploration-strategy correction

- Rejected the premature assumption that the project must select one paper
  story and one primary estimand before running development experiments.
- Reclassified ACL dev as an explicit multi-route discovery stage covering
  current-content anticipation, noisy-speech robustness, pixels beyond OCR, and
  evidence selection/integration. Correct versus matched stale/wrong evidence
  remains a common content-use validity control, not the only allowed outcome.
- OmniFusion is now treated as a weak application precedent rather than a paper
  that closes usable simultaneous ST with slides; its latency and attribution
  gaps remain part of the opportunity.
- The rigor boundary moves to the dev/held-out split: record the complete
  declared development matrix, choose the most defensible route from dev
  evidence, then commit/push a frozen primary claim and analysis before reading
  ACL eval or MCIF results.

## 2026-08-01 MCIF Visual Readiness

- Added an inference-safe MCIF subset materializer that selectively extracts
  21 audio/PDF files and timing metadata, downloads only the matching 21 MP4s
  at frozen HF revision `e24065b9...`, and verifies bytes/SHA256.
- Materialized 21 talks / 919 segments / 7,105.53 s audio / 7,110.39 s video.
  Reference files were not extracted or read, and the inference tree contains
  no reference/transcript/translation directory.
- Added a 10 s visual coverage audit: 711 frames and 21 contact sheets. Visual
  inspection confirmed slide-dominant real conference recordings in all talks,
  with persistent text, charts, formulas, layouts and progressive builds.
- Added a 1 s patch-grid visual-state detector that suppresses small speaker
  overlay motion. Calibrated v2 recovered same-template text changes without
  materially increasing high-motion talk counts.
- Full v2 result: 7,111 frames, 283 transition candidates, 283 stable
  confirmations, 304 causal states including initial states, and 2.388
  candidates/minute.
- Inspected all 21 transition sheets. No speaker-only false trigger was
  observed. The timeline remains a candidate artifact because sub-threshold
  local updates may be missed; it is not independent human ground truth.
- Each causal state now has a conservative half-open availability interval,
  stable evidence-frame path/SHA256 and transition window. No future state is
  exposed before stable confirmation.
- Corrected the noise source: SLR119 is AliMeeting. The selected RIR source is
  OpenSLR 28 (Apache 2.0); MUSAN remains OpenSLR 17 (CC BY 4.0).
- Detailed record:
  [`docs/MCIF_VISUAL_READINESS_20260801.md`](MCIF_VISUAL_READINESS_20260801.md).

## 2026-08-01 Controlled Acoustic Pipeline

- Downloaded official MUSAN/SLR17 and RIR/Noise SLR28 archives; verified the
  published MD5 values, local SHA256, licenses, resource-page snapshots, and
  archive integrity.
- Deterministically selected and extracted 130 sources: 32 babble speech, 16
  generic noise, 16 music, and one real RIR for each of development and
  confirmatory. Source overlap between the two pools is zero.
- Added a full-talk corruption materializer with stable per-talk/condition
  seeds, exact source hashes/offsets/wrap counts, source-only activity masks,
  SNR scaling, peak guards, and onset-aligned RIR convolution.
- Calibrated `energy_vad_v1` only from the five clean ACL dev waveforms. The
  initial `p95-35 dB` rule marked one talk almost fully active and was rejected;
  the frozen rule is `max(-50 dBFS, p95-15 dB)`.
- Materialized 75 ACL60/60 dev variants: five talks × 15 conditions. All output
  hashes/durations pass; maximum target/achieved SNR error is below `4.9e-7 dB`;
  post-PCM16 recomputation is within `0.00027 dB`; PCM saturation count is zero;
  active fractions are 49.6%–88.8%.
- Uploaded the source-only bundle to the private HF dataset
  `gavinlaw/slide-aware-sst-controlled-acoustic-dev` at immutable commit
  `d28c499c8845c4991b5ccea27bc9a2ad520f51fa`, tag
  `acl6060-controlled-acoustic-v1-20260801`. Verified the 75-WAV remote inventory,
  privacy, all metadata bytes, and one downloaded WAV against local staging.
- The artifacts are controlled acoustic interventions, not real noisy-talk
  recordings. No ST inference or target/reference access occurred.
- Detailed record:
  [`docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md`](CONTROLLED_ACOUSTIC_PIPELINE_20260801.md).

## 2026-08-01 ACL60/60 Visual Timeline v1

- Imported all 468 real ACL dev talk-video frames without opening the upstream
  metadata JSON containing `sentence`; counts match the five frozen talks.
- Froze an every-observed-frame causal policy: each midpoint frame unlocks only
  at its filename timestamp, ends at the next observation, and no visual state
  exists before the first frame.
- Audited 97 pixel-threshold transition candidates and 60 stratified negatives.
  Positives were high precision, but negatives exposed clear full-slide and
  incremental-reveal misses on white text-heavy slides.
- Rejected transition-based state compression for v1. The 468 observation
  states are inference-ready; the transition inventory remains diagnostic only.
- Detailed record:
  [`docs/ACL6060_VISUAL_TIMELINE_20260801.md`](ACL6060_VISUAL_TIMELINE_20260801.md).

## 2026-08-01 ACL60/60 Source Event Seed v1

- Froze 100 transcript/target/model-output-free annotation packets: 20 per talk,
  balanced between high-precision transition states and deterministic random
  observations.
- Each packet records causal frame timing/hash, full-talk audio identity/hash,
  and a bounded listening window; all 100 observation IDs are unique.
- Froze the 0.96 s prefix boundary protocol, forced-choice schema, evidence
  subtypes, negative retention, target firewall and two-annotator requirement.
- No event labels exist yet. The seed status is
  `PENDING_DOUBLE_SOURCE_SIDE_ANNOTATION`; it must not be reported as event
  density or oracle headroom.
- Annotation guide:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V1.md).

## 2026-08-01 ACL60/60 Source Event Workspace and OCR Headroom

- Built a source-only 468-segment timing manifest from English source XML and
  gold segment WAV alignment. The builder has no target/reference input and the
  local manifest is explicitly excluded from inference inputs.
- Ran deterministic Tesseract 5.5.2 OCR over all 468 dev frame observations:
  26,921 tokens, 10,090 lines, zero empty frames. Token boxes, confidence,
  engine settings, hashes and observation intervals are preserved.
- Added a conservative exact-match diagnostic: a candidate counts only if the
  first source segment containing it begins while a causal current frame still
  shows it. Source segment start underestimates within-segment lead.
- Found 901 overlapping n-gram matches, 344 candidates after nested-subphrase
  removal and 149 independent segment-frame events. Median candidate lead is
  10.325 s; 183 candidates are at least 10 s early. This is automatic
  OCR-sufficient headroom, not a human event count or translation result.
- The frozen seed contains 38/100 packets with an automatic future candidate
  and 31/100 with a future phrase. A 50-row high-lead/hash-random review set and
  visual sheets were generated locally; semantic usefulness remains pending
  human audit.
- Materialized 100 source-only annotation packets: 100 frames, 100 mono
  PCM16/16 kHz clips totaling 6,498.51 s, and isolated annotator A/B JSONL
  files. All packet/media hashes, WAV lengths, unique ids and forbidden-field
  scans pass. No transcript, target/reference or model output is present.
- Local workspace is 205 MB and currently `PENDING_HF_UPLOAD` to private
  `gavinlaw/slide-aware-sst-acl6060-source-events`. Labels remain
  `PENDING_DOUBLE_ANNOTATION`.
- Detailed record:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V1.md).
