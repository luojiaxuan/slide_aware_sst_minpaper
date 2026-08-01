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
- Uploaded the 205 MB workspace to private HF dataset
  `gavinlaw/slide-aware-sst-acl6060-source-events` at revision
  `3199207c66b159ab39f662a32e0f6d633c9c2b79`, tag
  `acl6060-source-event-workspace-v1-20260801`. Verified privacy, the complete
  307-file inventory and byte identity for metadata, one annotation sheet and
  one downloaded WAV. Labels remain `PENDING_DOUBLE_ANNOTATION`.
- Detailed record:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V1.md).

## 2026-08-01 ACL60/60 Annotation Protocol v2

- Independent methodology review rejected direct use of v1 A/B sheets. Two
  independently authored forced-choice questions have no common answer space,
  and seeing the frame answer before audio contaminates audio sufficiency.
- Found a second P0: v1 windows include only five seconds before evidence and
  cannot rule out an earlier answer in the full causal source prefix. v1 media
  remains valid, but its annotation protocol is superseded.
- Froze v2 as frame-only canonical authoring, question hash lock, author
  full-audio relevance/gold check, two blinded talk-start causal-audio response
  trajectories, then validation by a separate two-person frame-only cohort and
  append-only adjudication. Author/audio/frame role IDs must all be distinct.
- The audio grid starts at exact `t_evidence` and advances by 0.96 s. Validators
  fill every `insufficient|uncertain|option_id` response through the restricted
  `t_evidence+60s` endpoint; scorer derives first stable correct after unsealing
  gold. Right-censoring is explicit.
- Removed sampling-prior leakage: author/validator views omit stratum, replace
  original A-numbers with 100 salted opaque ids, globally hash-randomize author
  order, and independently randomize option order per validator. Real mapping
  remains scorer-side only.
- Implemented stage builders, author post-lock audio-review bundles, immutable
  content/annotation hashes, actual-media hash revalidation, field allowlist
  checks, full trajectory/grid validation, physically separate audio/frame
  bundles, opaque per-validator option IDs/item order, two disjoint validator
  cohorts, agreement metrics and adjudication flags. Nine packager/scorer tests cover
  locks, grids, minimum stable evidence, packet completeness, negative
  denominators, agreement and modality separation.
- Implemented a sequential HTTP backend that withholds audio before question-only
  lock, returns only `0..g_k`, rejects out-of-order responses, and emits a
  hash-chained append-only interaction log. Two backend tests cover causal WAV
  clipping, export and tamper rejection. Formal deployment must deny annotators
  direct filesystem access to the audio root.
- Implemented append-only adjudication sheets keyed to raw report-row hashes.
  Conflicts remain `primary_eligible=null` until a locked decision exists, so
  disagreement cannot be silently converted to a negative. Question-only
  answerability is a non-overridable hard failure; positive adjudicated boundaries
  must remain on the causal prefix grid.
- Implemented the frozen-design prevalence estimator with per-stratum sample
  count checks, finite-population correction, standard error and 95% CI. It
  refuses to estimate while any adjudication or source exclusion remains unresolved.
- Replaced the public fixed-salt opaque IDs with scorer-secret HMAC IDs; public
  enumeration of `A001--A020` can no longer recover selection stratum.
- Materialized author view r3: 100 frames, 0 WAV, 6,333,781 bytes including card. All frame
  hashes and forbidden-field checks pass. No human labels exist.
- Uploaded the r3 author-only view to private HF revision
  `2fb266d168e0abbf4ace17d3f5de9503a8c46cd6`, tag
  `acl6060-source-event-author-v2-r3-20260801`. Remote inventory is 100 JPG,
  0 WAV plus card/sheet metadata; mapping and HMAC secret are absent. Three
  forced downloads were byte-verified.
- Froze the ten `talk × stratum` pool sizes and inclusion probabilities. Raw
  balanced-seed yield is not overall prevalence; the latter requires
  inverse-probability weighting over 468 observations.
- Detailed record:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V2.md).

## 2026-08-01 Chinese-LiPS Five-Condition Visual Control Freeze

- The historical 206-segment Qwen3-Omni run showed nearly identical gains from
  current and same-talk wrong slides, but omitted cross-talk and blank-image
  controls and did not record an immutable model revision.
- Froze a complete same-revision rerun with `none`, `slide`, `wrong`,
  `cross_talk`, and `blank`: 1,030 expected records over 206 items.
- Added deterministic cross-talk frame assignment, a hashed blank-image
  control, resumable multi-worker GPU sharding, strict completion checks and a
  paired chrF/AL analyzer.
- This remains a private, single-talk, machine-reference mechanism diagnostic;
  it cannot be used as paper-grade model ranking or talk-level inference.
- Frozen contract:
  [`docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md`](CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md).

## 2026-08-01 Annotation v2 Independent-Audit Fixes

- A final independent audit found three P1 issues after the earlier v2 freeze:
  the log verifier trusted a valid hash chain without fully enforcing event
  order/release boundaries, adjudication could place a positive boundary at an
  isolated final grid point, and plaintext timing/media identifiers could
  bypass opaque packet IDs.
- The verifier now enforces the complete ordered event-state machine, exact
  release/submission boundaries, monotonic server timestamps and completion
  ordering. Rehashed adversarial logs for all three failure modes are tests.
- Positive adjudication now must leave the configured two-observation stable
  tail; the final isolated point is rejected.
- Generated local author view r4 with the same 100 frames but no `talk_id`,
  absolute timing or raw frame SHA in author rows. Author/frame views use a
  scorer-secret media binding; audio timing sheets are server-private and the
  HTTP UI exposes prefix indices only. r3 is superseded before any labels were
  collected.
- Uploaded r4 to the same private HF repo at immutable revision
  `bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a`, tag
  `acl6060-source-event-author-v2-r4-20260801`. Verified private visibility,
  103-file inventory, 100 JPG / 0 WAV, absence of mapping/secret filenames and
  byte identity for README, authoring sheet and one frame.
- The follow-up audit found one remaining Stage-2 leak: the post-lock author
  audio-review JSON reintroduced talk/timing/raw-frame identifiers. Split it
  into an editable public sheet and a scorer-private media manifest; freeze now
  rejects schema/task-field drift before merging author outcomes. The author-
  facing Stage-2 bundle now omits the same linkable identifiers as Stage 1.
- Final independent regression review reproduced the hardened Stage-2 artifact
  and all prior adversarial cases. It found no remaining P0/P1: the public
  author-audio sheet contains none of the five linkable fields, strict merge
  rejects task/schema drift, and wrong release boundaries, early completion,
  non-monotonic timestamps and an isolated final positive are all rejected.

## 2026-08-01 Visual Control Launch Preparation

- Materialized the frozen 206-item five-condition input matrix on Hyper00 at
  run root
  `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_2gpu_20260801_132051`.
- The exact 70.5 GB Qwen3-Omni revision was absent from Hyper00 and `/data` had
  only 35 GB free. Reused the byte-identical historical Hyper01 snapshot,
  transferred its full HF cache directly host-to-host into Hyper00 personal
  `/data01/jaxan`, and verified a zero-difference 25-file SHA256 manifest.
- Copied that verified persistent cache to the canonical container's 1 TB
  `/dev/shm` runtime cache and again obtained a zero-difference SHA256 manifest.
  No existing artifact was deleted and no second compute container was created.
- After the GPU fleet became free, preflight reported all eight cards below
  1 GB. Launched only GPU `0/1` in the existing canonical container from
  `main@f76f922`, one model worker and one spawned process-prefetch child per GPU,
  at run root
  `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801_153400`.
- The first formal 10-second `nvidia-smi dmon` window was GPU0 `89.2%`, GPU1
  `94.0%`, two-GPU mean `91.6%`; this passes the frozen `90%+` continuation gate.
  A second window was `88.1%/96.6%`, two-GPU mean `92.35%`. At the documentation
  checkpoint the two shards had `249/241 = 490/1,030` rows. The run later
  completed at exactly `515/515 = 1,030/1,030`; completion binds both shard
  hashes, exact model revision and input hash, all workers exited, and logs had
  no traceback/OOM/IPC/resource-sharer error.

## 2026-08-01 Visual Control Analysis and Private Freeze

- Completed the frozen 10,000-sample paired bootstrap over all 206 items. The
  analyzer now parallelizes independent contrasts and caches SacreBLEU segment
  statistics; the optimized full-size serial benchmark takes about 2.8 seconds
  and matches naive resampling exactly. The complete project suite passes
  `148 tests`.
- `slide - none` is `+1.772 chrF [0.752, 2.782]`, but `wrong - none` is
  `+1.659 [0.546, 2.726]` and `blank - none` is `+1.545 [0.644, 2.482]`.
  Page specificity is absent: `slide - wrong = +0.113 [-0.693, 0.894]`.
  Structured-slide specificity is also absent: `cross_talk - blank = -0.119
  [-1.205, 0.824]`.
- The latency pattern agrees with the same diagnosis. Correct slide is not
  faster than same-talk wrong slide (`slide - wrong` AL `+0.051 s
  [-0.045, 0.152]`), while a blank image alone is faster than no image by
  `0.118 s [0.021, 0.216]`.
- Conclusion: this single-talk machine-reference probe measures a generic
  vision-slot/decoding perturbation, not demonstrated use of current-page
  semantics. It is a negative mechanism diagnostic and not paper evidence.
- Packaged 1,030 outputs, completion/analysis, runtime logs and GPU provenance
  without raw media. Uploaded to private HF
  `gavinlaw/slide-context-sst-chinese-lips` at immutable revision
  `4923b253e87bd94487dace77576ad66e4ea9d8b9`, tag
  `chinese_lips_visual_controls_v1_qwen3_omni_20260801_canonical`, path
  `experiments/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801/`.
  Privacy, tag target, inventory and all 10 downloaded files were byte-verified.
- The first metadata-only HF revision `d0c4453...` and its tag without the
  `canonical` suffix are superseded. The canonical revision fixes the Hub task
  category and includes the README hash in the manifest.

## 2026-08-01 Event-Level Timing Scorer

- Implemented the paper-specific development estimand rather than relying on
  aggregate BLEU/AL: talk-equal risk difference in first stable correct target
  decisions by the conservative audio-insufficient boundary.
- Kept source timing/expected media identities, source artifact tree, actual
  source-only evidence packets, control pairs, scientific config, no-target
  pre-run inference contract, post-run result attestation, start/end live
  environment audits, raw outcome commitment/tree, causal-audio schedule, broker
  audit, release log, tokenizer artifact tree, target realizations, system
  trajectories, frozen model artifact tree and scoring config as nineteen hash-bound inputs. The scorer rejects missing/duplicate
  matrix rows, unknown conditions, time reversal, early trajectory termination,
  endpoint overflow and model/acoustic-condition-specific audio-time grids.
- Stable correctness requires a final all-correct tail of at least two
  observations. An isolated final positive is right-censored; retractions,
  overcommit and forbidden realization adoption remain explicit outputs.
- Matched-wrong controls are type-specific for OCR, semantic and relation
  evidence. Noise effects are difference-in-differences over the same content
  contrast, not correct-vs-audio-only BLEU gaps.
- Aligned the scoring config with the frozen acoustic manifest: native, 12
  babble variants (`+10/+5/0/-5 dB × 3 seeds`), generic noise, music and RIR.
  Seed replicates are averaged within event/talk and do not inflate talk count.
- Independent adversarial review found four P1 gaps in the first scorer draft:
  acoustic-specific time grids, technical-token false matches, unaudited
  matched-control/provenance claims and permissive config parsing. All four
  are fixed: one grid is required across every condition/acoustic variant;
  technical symbols/acronyms have regression cases; source-only control pairs
  bind recomputed payload/token hashes and availability; a no-target inference
  manifest plus mount/open-file audit bind run/config/Git/isolation evidence;
  all schemas reject extra, mistyped or non-finite fields.
- The second adversarial pass found decimal/sign matching, self-declared packet
  metadata, mutable v1 parameters and unpaired noise uncertainty. The final
  implementation preserves leading decimals, Unicode signs, technical dotted
  names, percent and version tokens; freezes the exact 16×9 matrix and all
  analysis constants; jointly bootstraps DiD/curves by talk; and explicitly
  reports undefined severity-correlation draws.
- Replaced arbitrary packet dictionaries with a strict source-context schema
  and deterministic renderer. Each context item binds by text/index/hash to a
  source-derived artifact that in turn binds upstream slide/document bytes and
  extractor revision. Gold text injected only into the packet now fails.
- The scorer pins a tokenizer-only artifact tree by file bytes as well as model
  identity/revision and replays exact token IDs. Mutable symlinks and local
  tokenizer edits fail even though libraries ignore `revision=` for local dirs.
- The live environment capture CLI token-matches the exact run marker, enumerates
  marker workers plus descendants, records PID/PPID/process-start ticks/cwd/
  entrypoint/executable/environment, mounts and `/proc/<pid>/fd`, and binds every Python child
  entrypoint to the clean audited worktree. Formal runs require read-only rootfs
  and matching start/end process-identity trees. Container destination roots,
  host mount-source roots and scoring-host protected roots are separate path
  namespaces; actual target/outcome roots must map to the forbidden host set.
- Added a pre-run contract builder that validates all source/target/outcome/audio/
  tokenizer/config inputs, derives the actual image and complete worker identity
  tree from `workers_start`, then exclusively writes the contract and atomic
  ready marker. This replaces impossible single-environment hashes for multi-GPU
  workers and makes generation wait on a byte-addressed contract. The builder and
  scorer both require the live worker command to bind the same contract/ready
  paths; the worker-side wait helper rehashes and parses the contract before use.
- A third independent review found that self-declared source media, substring
  process matching and `future_audio_access=false` could still pass ordinary
  reindex/launch bugs. Source events now freeze media/extractor identity per
  condition. Every trajectory observation binds a monotonic external-broker
  prefix schedule and release record. Full-audio roots must be absent from the
  inference mounts/open files.
- A fourth independent read-only audit found seven P1 provenance risks and no
  P0. The four generation-critical issues are closed in code: canonical source
  PCM/full/prefix/provenance hashes and exact time/sample boundaries; executable
  Unix-socket broker with server-enforced interaction ordering and hash-chained records;
  pre-run contract plus post-run trajectory attestation; and start/end read-only
  runtime audits. Target/outcome bytes are precommitted and linked to excluded
  host roots. The current broad writable `/data` diagnostic container still
  cannot satisfy the formal isolation contract, so fresh paper-grade ACL
  generation remains blocked on rebuilding the same canonical container after
  active workloads finish and completing human outcome artifacts.
- A fifth independent two-agent audit found the release-only protocol still
  allowed a worker to prefetch full audio and backfill nominally early outputs.
  Replaced it with a release/observation-commit state machine: every exact
  hypothesis hash is committed before the next prefix, the final observation is
  also committed, and scorer verifies the complete ordinal/hash chain. The same
  pass made broker/contract readiness atomic, filtered schedules to eligible
  events, requires read-only `network=none`, snapshots file bytes once for both
  parsing and attestation, and binds a strict in-process model config plus full
  model artifact tree. It also confirmed one explicit remaining blocker: no
  production inference worker currently calls the barrier/broker APIs.
- A final independent audit found four further P1 proof gaps: future audio could
  cross condition/event/acoustic streams, host model bytes were not mapped to the
  worker load path, scientific config was not runtime-bound, and ready hash was
  recomputed from the published path. The fix uses one synchronized audio-time
  frontier across every stream of a talk; exact read-only model/config mounts;
  worker-side config rehash/parse; start/end container and mount-topology
  identity; and a ready hash derived from the builder-validated bytes. Focused
  tests now include a cross-condition future-audio rejection.
- Follow-up review found same-time clean/noisy prefixes could still coexist in
  one worker. Sessions are now single-stream, each talk permits only one
  in-flight release even at the same frontier time, scorer replays that serial
  order, and a multi-acoustic test covers the rejection. Internal model/audio
  state isolation remains an explicit production-worker requirement and formal
  generation blocker rather than a broker-only claim.
- Added executable `+5 pp/-1 pp/3-of-5` exploratory point-estimate gate
  components and talk-cluster intervals for early/final/forbidden/overcommit.
  The gates are not labeled as statistical non-inferiority tests.
- Added 34 focused scorer/broker tests including CLI artifact/hash round trip,
  real Unix-socket prefix delivery and adversarial future-slide/extractor/
  prefix/source/root cases. The project environment passes the complete suite:
  `145 passed` in both the project environment and a CPU-only no-`torch` test
  environment; two Qwen-VL unit tests now mock their dtype-only torch dependency.
- This is scoring readiness only. Human source-event labels, target
  realizations and system trajectories do not yet exist, so there is no ACL
  dev effect estimate.
- Contract:
  [`docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md`](ACL6060_EVENT_TRAJECTORY_SCORING_V1.md).

## 2026-08-01 Production Causal Inference Worker

- Implemented `run_causal_event_inference_worker.py` and
  `merge_causal_event_worker_shards.py` for in-process Qwen3-Omni generation.
  The worker batches only across talks; within a talk it permits one released
  prefix at a time, commits the exact hypothesis hash before advancing, and
  assigns an independent session to every event/condition/acoustic stream. It
  carries no persistent model KV or audio cache across prefixes.
- Closed all three P1 findings from the independent read-only review. Contract
  and ready bytes are now read once into one validated snapshot; the merger also
  verifies the ready marker. The contract freezes exact read-only model and
  tokenizer mounts; workers hash both trees before load and after generation,
  then replay every rendered evidence packet through the model processor's
  actual tokenizer and require exact token-ID equality.
- Each done marker now binds contract/schedule/evidence hashes, worker
  index/count, deterministic talk partition, PID/process-start ticks, complete
  start-audit process-tree hash and canonical output path. Merge maps every
  shard to one exact audited command and rejects missing, duplicate, stale,
  overlapping or externally substituted workers before validating the complete
  event/condition/acoustic/prefix matrix.
- Added regressions for processor-tokenizer drift, deterministic worker
  provenance, two acoustic conditions and two events sharing the same talk and
  frontier time. The focused worker/scorer suite passes 37 tests; the complete
  project suite passes `155 tests` with only two upstream `pypinyin` deprecation
  warnings. All three affected CLIs also pass direct `--help` import/startup
  checks. A formal model smoke and ACL run remain intentionally blocked until
  human source/target outcomes are frozen and the same canonical container is
  rebuilt with read-only rootfs, `network=none` and narrow mounts.

## 2026-08-01 Frame-Only Authoring Unblock

- Confirmed the canonical r4 author sheet is still exactly `0/100`: every row
  is `pending`, so no human outcome may be inferred or replaced by model labels.
- Added a localhost-only authoring service over the blinded r4 bundle. It shows
  only opaque packet ID and current frame, supports normalized drag-to-box
  evidence localization, 2--4 source options, canonical answer, evidence type,
  negative/exclusion labels and atomic resume writes. Immutable task identity,
  workspace path confinement, symlink rejection and the existing v2 row
  validator remain enforced server-side.
- The canonical input remains unchanged. The separate local working sheet is
  `author_view_v2_blinded_r4/authoring_working_question-author-01.jsonl`, now
  initialized at `0/100`; it is not frozen and cannot enter later stages until
  `freeze-author` succeeds.
- Unit tests cover candidate/negative saves, resume, invalid candidate fields,
  immutable task drift, path escape and symlink media. Real-browser checks on
  the 100-frame bundle verified desktop/mobile rendering, responsive navigation,
  slide loading and drag-box coordinates with no console warnings/errors. The
  complete project suite now passes `158 tests` with the same two upstream
  `pypinyin` deprecation warnings.

## 2026-08-01 Narrow Collision Audit and Frame Validation UI

- Locked the exact claim that pre-audio slides can make correct SimulST target
  text appear earlier and that this value may grow under acoustic degradation.
  The immutable source hash is
  `4c176a7bd3c41dd5846356a74fb020db945e5eb19db00fb449c4220a13f7f66f`.
- Re-audited the narrow claim against full primary texts and current official
  records. Verdict is `Level 2 - High Overlap`: OmniFusion already matches
  slide-aware scientific-talk SimulST and earlier/stable commitments; Caglayan
  et al. 2020 already establishes visual anticipation in text SiMT; multimodal
  RL SiMT already conditions READ/WRITE on images.
- Froze the remaining executable paper boundary: first-stable-correct target
  timing before audio sufficiency, correct-versus-wrong/stale/empty content
  attribution, OCR/raw-image separation, matched final quality and a controlled
  acoustic-noise difference-in-differences. Generic BLEU/COMET/AL gains are not
  a sufficient contribution. See
  `docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md`.
- Verified from *Do Slides Help?* that ACL60/60 evaluation uses real midpoint
  video frames, while its large MuST-C training augmentation groups transcript
  in eight-sentence chunks, asks Llama 3 for LaTeX slides and renders PDF/images.
  It is an ASR/term precedent, not evidence for online target timing or real
  slide semantic necessity.
- Added a localhost-only Stage-4 frame-validator service. It exposes only the
  current frame, locked question and validator-specific opaque options; saves
  support/answer/subtype/confidence/note to an independent `0600` atomic working
  sheet; and enforces task lock, immutable rows, one-validator ownership,
  workspace confinement and symlink rejection. It never exposes canonical
  answer, audio, talk/timing, raw media SHA or another validator's labels.
- Unit regressions cover supported/unsupported saves, invalid answers, resume,
  task drift, path escape and symlink media. A real-frame browser smoke verified
  save/resume, desktop and 375 px mobile layouts, zero horizontal overflow and
  no console warnings/errors. An independent Claude review found no P0 or
  correctness/blinding P1; its HTTP-handler coverage P1 was closed with real
  localhost GET/POST, routing, security-header, malformed-body, oversized-body
  and bounds tests. The complete suite passes `163 tests` with two
  upstream `pypinyin` deprecation warnings.
- No human label was created by this work. Stage 4 remains causally blocked until
  question authoring, author audio review and two disjoint audio trajectories
  are frozen.

## 2026-08-01 MCIF Reference-Free VLM Screen Input Freeze

- Confirmed the statistical role split before doing more computation: five ACL
  dev talks are discovery/futility only; five ACL eval talks cannot produce a
  two-sided sign-test p-value below `0.0625`; the planned confirmatory unit is
  the 21-talk MCIF subset, with at least 15 talks required to contain eligible
  events. No segment-level pseudo-replication or noise-seed inflation is allowed.
- Added `build_mcif_visual_context_screen.py` and a Qwen3-VL source-screen prompt.
  The builder verifies the sealed MCIF source manifest, complete 21-talk state
  set, contiguous intervals, state IDs, root confinement, symlink rejection and
  every evidence-frame SHA256 before producing model input.
- Materialized all 304 causal states, not a model-selected subset. Frozen input
  SHA256 is `62fa1fb540ae279ddafcb2b8449a8cde355ea6fe828dd7ec6534d04c6d605073`;
  frame-binding-set SHA256 is
  `a2399ad294035557803f3a75d1ebe65613da99b89a5f36597124fd0c83bf4ad9`.
  Source transcript, target/reference and model output were not consumed.
- This artifact is a private automatic prescreen, not annotation or evidence of
  translation benefit. Its output cannot drop states, supply suggested answers
  to human authors, define `image_needed`, or enter paper statistics. The Git
  contract is `data/manifests/mcif_visual_context_screen_input_v1_20260801.json`;
  generated data remains local pending the Qwen3-VL run and private HF upload.
- Four new regressions cover complete source-only construction, reference-seal
  rejection, frame-hash drift, path escape, timeline gaps and symlink media. The
  complete project suite passes `170 tests` with the same two upstream
  `pypinyin` deprecation warnings.

## 2026-08-01 MCIF Qwen3-VL-32B Source Screen Completion

- Ran all 304 frozen MCIF causal states on Hyper00 with
  `Qwen/Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb`,
  two H200 GPUs and no transcript/reference/translation input. The pre-output
  contract was already pushed at `ab37bfaa4c924a0df68764ce5f3a87f55eecc6a5`.
- Added a fail-closed finalizer and compact repetition-repair prompt. The first
  384-token pass had 98 truncated raw JSON rows; a same-prompt 1024-token repair
  left six repetition loops; the compact repair resolved all six. The final
  artifact has 304 unique rows, 21 talks, zero parse failures, zero empty
  contexts and no local absolute paths. Full tests pass `177 tests` with the two
  existing upstream `pypinyin` warnings.
- Descriptive coverage is high: 303 OCR rows, 302 object rows, 169 action rows
  and 303 spatial-relation rows. This is not an `image_needed` label or a
  pixels-beyond-OCR result; many relations are trivial layout statements and all
  descriptions remain unverified model output.
- Uploaded the portable bundle to private HF dataset
  [`gavinlaw/slide-aware-sst-mcif-source-prescreen@5da477ff`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/5da477ff7d199dbded0ffe44d6b41b9cd8c8e75d),
  tag `mcif-source-only-qwen3-vl-32b-v1`. Repo privacy and remote
  `SHA256SUMS` bytes were verified. Output SHA256 is
  `55a6dafe5ebd1fc5f37226de7ce48e601d61e1fb914287a81bdb8f92b0479682`.
- Batch 16/32/64 did not meet the 90% utilization policy; the best observed
  10-second average was 78%. The bottleneck is Transformers autoregressive
  decode/preparation/variable-length tail behavior. Future larger VLM rollout
  must use a continuous-batching serving path.
- Added a hash-bound lexical structural triage over the portable output. After
  removing a `video feed` connectivity false positive, 192/304 rows across all
  21 talks contain at least one model-described table/chart/connectivity/formula/
  emphasis/mapping pattern; 111 relation rows are simple-layout-only. An
  eight-frame deterministic agent spot check supported the broad categories but
  found one hallucinated explicit arrow over a real left-to-right pipeline.
  These diagnostics are not labels. They motivate a three-step development
  baseline: unordered OCR, structure-preserving text, then raw image. The full
  project suite now passes `184 tests` with the same two upstream warnings.
- Audited the actual source timing and resolution behind the 304 detector
  thumbnails before implementing the strong OCR baseline. The thumbnails are
  only 320 px and correspond to the centers of the detector's one-second
  `fps=1` buckets. At nominal bucket-start timestamps, 4 initial states had
  source-frame MAE above 100/255 (maximum 231.66). At `t+0.5s`, all 304 align
  below 12/255 (mean 1.83, maximum 9.74). Added a source-only native-resolution
  materializer that moves visual availability to the actual capture time and
  leaves `[0.0, 0.5)` context-free. The old Qwen3-VL screen is now explicitly a
  morphology prescreen, not a causal raw-image baseline.
- Materialized the corrected native evidence as 304 PNGs across 21 talks. All
  frame dimensions, source-video hashes, detector-frame hashes and alignment
  checks pass; manifest SHA256 is `4e1008ab...9cccc`. Uploaded 308 paths to the
  private MCIF source-prescreen repo at revision `4e80dd0a...ae49`, tag
  `mcif-native-causal-evidence-v1`; the remote checksum manifest is byte-identical.

## 2026-08-01 MCIF Native OCR / Structure Evidence Completion

- 在全部 304 个 native causal frames 上完成 matched flat PP-OCRv6 与
  PP-StructureV3 抽取；21 talks 全覆盖，6 shards 为 `51,51,51,51,50,50`，0 failed
  rows。source transcript、audio、target 和 reference 均未读取。
- 冻结 PaddleOCR `3.7.0`、PaddleX `3.7.2`、PaddlePaddle GPU `3.3.0`、
  `paddle_dynamic` 和 13-model byte-tree manifest。PP-Chart2Table 的 tied embedding
  已做 runtime identity 检查；PaddleOCR-VL 被明确排除，因为它不是 OCR baseline。
- 修复 PP-Structure table reconciliation 对 `KMeans(n_clusters=0)` 的批级失败污染：runner
  递归隔离单帧，并只对精确 trigger 应用冻结 fallback。17 rows 最终通过
  `disable_table_recognition` 完成，保留 flat OCR 和非 table structure；0 rows 丢失。
- 最终产物含 7,123 flat OCR items 和 1,780 structure blocks。严格可机器读取的 non-flat
  union 为 65 rows / 18 talks：chart 53、table 7、formula 5；另有 17 rows 只是 table
  detection placeholders，不能当作 R1 table serialization。
- 44-row hash-deterministic visual QA 支持 positive strata 的 broad precision，但在
  layout/plain strata 中看到漏检 numerical tables、small chart、relation diagram 和
  semantic illustration。因此自动 tiers 不能过滤 raw-image condition、定义
  `image_needed` 或估计 event prevalence。
- Hyper01 正式配置只占 GPU 0/1，每卡 3 workers；冻结 10 秒平均 utilization 为
  98.6% / 98.9%。全套本地回归为 `227 passed`，仅有两个既有 `pypinyin`
  deprecation warnings。
- Portable 6.2 MB bundle 已上传 private HF：
  [`gavinlaw/slide-aware-sst-mcif-source-prescreen@09004d42/ppstructurev3_source_screen_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/09004d4262278b26a1f2f014fdd908427f55797a/ppstructurev3_source_screen_v1)，
  tag `mcif-ppstructurev3-source-screen-v1`。22 个远端文件已全量重下载，21-entry
  `SHA256SUMS` 全部通过。
- R0/R1 input extraction 至此完成。下一步是冻结 source-event packets / target scoring，
  再做 document、unordered OCR、structured text、raw image、matched wrong 的 native/noisy
  oracle headroom；当前仍没有 MCIF translation output 或 paper effect estimate。

## 2026-08-01 Raw-image Event Contract V2

- 审计发现 v1 causal worker 硬编码 `images=None`，只能比较由 slide 抽出的文本，不能回答
  `vision > OCR`。保留 v1 不变，新增 v2 的 `correct_image / matched_wrong_image` 条件和
  `image content specificity / image over relation / relation over OCR` 三个增量。
- image packet 现在绑定 artifact JSON、native media bytes、source tree、只读 worker mount、
  固定 textual token IDs 与 processor 实际展开的 visual token count。任一路径、字节、
  processor 或 worker command 漂移都会 fail closed。
- Qwen3-Omni runner 按 image/non-image 拆分 homogeneous batches，在 message 中按
  `image -> prompt text -> causal audio` 排列，再恢复原条件顺序。float audio/image features
  都转换到 frozen model dtype。
- 当前 direct-image 路径会为每个 audio prefix 重复 image encoding，必须报告为 on-path
  quality diagnostic；在另行实现和审计 visual cache/compiler 前，不宣称 zero-latency 或
  off-path vision。
- 新增 8 个 v2 回归，覆盖 v1 拒绝、artifact/media tamper、路径逃逸、modality 顺序、visual
  token 漂移、batch 顺序恢复与 broker 集成。全套测试为 `235 passed`，两个既有 warning。
- 当前没有新 data artifact。下一步物化 304-state `R0/R1/R2` portable ladder，构造
  visual-token-matched wrong-image controls，再上传 private HF revision。

## 2026-08-01 MCIF R0/R1/R2 Evidence Ladder Completion

- 新增 fail-closed builder，在 clean Git `e991969` 上重新验证 304 native PNG 的 bytes、
  dimensions、timing、ID 与 PPStructure frame binding，并拒绝 transcript/reference/target、
  symlink、路径逃逸、provenance drift 和不完整 state matrix。全套回归为 `248 passed`。
- 物化 304 rows / 21 talks：R0 为不含 bbox 的 flat OCR text；R1 为 label、normalized bbox、
  reading order 和 chart/table/formula serialization，344 个 image tags 被降为 explicit
  visual placeholders；R2 引用同一 native PNG，不复制 images。
- ladder SHA256 为 `8f77312b...94f7f`，row-binding set 为 `84874eaf...1a8b`；第二次构建
  6/6 files byte-identical，本地 5-entry `SHA256SUMS` 全通过。
- 上传 private HF revision
  [`b13bd204/source_evidence_ladder_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/b13bd2045644f90a6de6be19f77a4af3acaa924f/source_evidence_ladder_v1)，
  tag `mcif-source-evidence-ladder-v1`。远端 6 files 已全量重下载并逐字节验证，5 个 payload
  checksums 全通过。
- 下一步冻结 Qwen3-Omni processor/image token budget，构造 deterministic
  `matched_wrong_image`；当前仍没有 event labels、translation output 或 paper effect。

## 2026-08-01 MCIF Qwen3-Omni Visual-token Controls Completion

- 在 clean Git `7e9b3e3` 上用
  `Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695`
  与 `transformers==5.6.0` 处理全部 304 个 R2 images；冻结 7-file processor manifest、
  prompt/message order、9 种 `image_grid_thw` 和 450--2040 visual-token counts。9/9
  representative images 加入 1 秒 dummy audio 后 token count 不变。
- 初始实现要求跨 talk candidate 天然拥有相同 grid/token，但正式数据的
  `mcif:EqmWoxNDIr:S000` 是自然单例，contract 因而 fail closed。最终没有降为近似
  compute matching：优先天然同尺寸/同 grid；缺失时冻结 aspect-preserving
  `contain + center + black pad + bicubic` transform，并将最终 image 再送入同一 processor。
- 304/304 cross-talk controls 与 283/304 causally-prior same-talk stale controls 全覆盖可定义
  states；203 个 cross-talk controls 天然同尺寸，101 个使用 transform spec。587 个最终
  processor inputs 的 `image_grid_thw` 和 visual-token count 全部与 source 精确相同。
- CPU-only 正式构建与独立第二次构建的 6/6 files byte-identical；全套测试为
  `261 passed`，仅有两个既有 `pypinyin` warnings。上传 private HF revision
  [`b2c9a409/visual_token_controls_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/b2c9a4093cb14cf15e26ff72efe941406bbaf59f/visual_token_controls_v1)，
  tag `mcif-qwen3-omni-visual-token-controls-v1`；远端 6 files 已全量重下载并逐字节验证。
- Artifact 只冻结 candidate original-image references 与 transform specs，不物化 101 张
  transformed control images，也不选择最终 paper control。下一步在 source-event packet
  compiler 中生成并 hash-bind 最终 bytes，冻结 target scoring，再执行 native/noisy oracle
  headroom；当前仍没有 MCIF event label、translation output 或 paper effect。

## 2026-08-01 MCIF Visual Control Media Completion

- 新增 state-level source-only media builder；它拒绝 source/control row drift、candidate
  identity drift、forbidden target/reference flags、path traversal、processor drift 和非原子
  结束态。全套回归为 `269 passed`，仅有两个既有 `pypinyin` warnings。
- 在 clean Git `62328c7` 上物化 101 个 `fit_pad_to_source_canvas` cross-talk controls，输出
  25,580,679 bytes PNG；其余 203 个 cross-talk 和 283 个 same-talk stale identity inputs
  继续引用 canonical native media，不复制图像。
- 304 rows / 21 talks 的 587 个最终 control inputs 已全部从保存后的 bytes 重过 frozen
  Qwen3-Omni processor，`image_grid_thw` 与 visual-token count 全匹配。正式构建与独立
  rebuild 的 105/105 files byte-identical；104-entry `SHA256SUMS` 全通过。
- 上传 private HF revision
  [`0001171c/visual_control_media_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/0001171cf661d605c6fa344df7cd3f90d291d194/visual_control_media_v1)，
  tag `mcif-qwen3-omni-visual-control-media-v1`；远端 105 files 已全量重下载并逐字节验证。
- 该产物刻意不生成 `SourceEventTiming` 或 `EvidencePacketSpec`：304 visual states 不是 304
  target events。下一步先冻结 event inventory、audio-insufficient boundaries 和 target
  scoring，再做 state-to-event packet join；当前仍没有 MCIF event label、translation output
  或 paper effect。

## 2026-08-01 MCIF Outcome Candidate Inventory Completion

- 在 source evidence ladder、visual controls 和 media bytes 全部冻结之后，新增 outcome-only
  extractor；它按 `audio-segments.yaml` 的 919 个 physical rows 对齐 official En/Zh/De/It
  references，并显式处理 En 的 14 个 talk-level quote wrappers 与 It 的 21 个 wrappers。
  不使用会跨行误解析的整文件 CSV reader。
- 候选定义为 English reference 中第一次出现、且在该 segment start 时仍由当前 causal R0
  OCR state 精确可见的 token/phrase。严格嵌套项被折叠；记录 earliest contiguous evidence
  state 与保守 lead lower bound。输出 954 candidates / 21 talks，其中 phrase 732、token 222；
  689 个 lead 至少 5 秒、458 个至少 10 秒，最大 148.111 秒。
- 每个候选明确标记 `AUTOMATIC_REFERENCE_AWARE_CANDIDATE_NOT_GOLD_EVENT`，human
  eligibility、Zh/De/It acceptable/forbidden realizations、audio-insufficient 和 first-sufficient
  boundaries 均为空。它只证明存在可审计的 anticipation candidate pool，不证明 translation
  gain、pixels > OCR 或 event prevalence。
- Builder 绑定 official archive、source ladder、inference timing 和 clean Git `2778222`，拒绝
  talk/timing/hash/source-only contract drift、非因果状态窗口和非原子覆盖。9 个定向测试及全套
  `278 passed` 通过，仅有两个既有 `pypinyin` warnings；第二次完整构建 10/10 files
  byte-identical，9-entry `SHA256SUMS` 全通过。
- 产物上传到独立 private HF dataset
  [`gavinlaw/slide-aware-sst-mcif-outcomes@64dee522/outcome_candidate_inventory_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/64dee5225e609fc0e900c7d6cd239ae6c702dc5c/outcome_candidate_inventory_v1)，
  tag `mcif-outcome-candidate-inventory-v1`。repo privacy 已确认，完整 snapshot 回下载后 data
  bundle、dataset card 与 9 个 payload checksums 全部逐字节验证。
- 下一步从 954 candidates 生成按 talk、lead、phrase/token 与 visual state 分层的 blinded
  annotation sheet，先冻结 event eligibility 与 target realizations，再独立标注 audio
  sufficiency。只有通过这两个 gate 的事件才能进入 state-to-event packets 和 native/noisy
  oracle headroom。

## 2026-08-01 MCIF En-to-Zh Target-event Author Workspace Completion

- 没有按 lead 或模型判断挑选“好看”候选，而是把 954 candidates 按 candidate-bearing
  segment 穷举合并为 355 items；保留每个 segment 的全部 1--18 个局部 options，但最终最多
  允许冻结一个 event，避免同一语音片段形成重复统计单位。21 talks 全覆盖，每 talk 2--37
  items；该不均衡是候选密度，不是 prevalence。
- Author view 使用 deterministic shuffled opaque item/option IDs，只包含 current slide、R0/R1、
  English source segment、Chinese reference 和空 annotation fields；talk/segment/state/candidate
  真 ID 与 causal binding 位于物理分离的 scorer mapping。De/It 不进入当前 author view。
- Builder 重放 first-source-occurrence、current R0 exact visibility、causal state interval、earliest
  evidence lead、reference content、row hashes 与 173 张 native image bytes/dimensions；任何 premature
  human label 或 model output flag 均 fail closed。12 个定向测试和全套 `290 passed` 通过，仅有
  两个既有 `pypinyin` warnings。
- 正式 workspace 含 180 files / 60,140,669 bytes。独立 rebuild 逐字节一致；author/scorer/root
  checksum entries 分别为 175/1/177 且全部通过。
- 上传 private HF revision
  [`0785a37f/target_event_author_workspace_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/0785a37f6537363b5cd0a8db0ead730298b12a1b/target_event_author_workspace_v1)，
  tag `mcif-target-event-author-workspace-v1`；180-file workspace 全量回下载并逐字节验证。
- 当前状态仍是 `AUTHOR_VIEW_READY_NO_HUMAN_LABELS`。Target-event author 可访问
  `author_view/`；未来 audio-only validators 必须使用独立 view/repo，不能看到 slides、OCR、
  references、candidate options、scorer mapping 或 author labels。下一步实现 authoring UI 与
  freeze validator，不自动填充 human labels。

## 2026-08-01 MCIF Target-event Annotation Protocol / UI Completion

- 冻结 machine-readable config
  `code/configs/mcif_target_event_annotation_v1.json`，SHA256 `95b8dc69...60b9c`。它固定 R0
  lexical scope、one-event-per-segment、五种 completed statuses、eligible/noneligible gates 与
  audio-validator forbidden fields；启动时 code/config 不一致会 fail closed。
- 新增 working-sheet initializer、row validator 和 create-once freeze compiler。Eligible row
  必须选定一个 option、canonical English event、至少一个 Chinese acceptable realization、
  explicit/paraphrased target alignment 与 supported slide evidence。Noneligible row 不能残留
  scoring answers；pending row 不能保存 partial labels。
- Freeze 同时验证 author input、mutable working sheet、scorer mapping 的 exact hashes 与
  annotator identity，只输出 `TARGET_EVENT_AUTHORED_PENDING_AUDIO_SUFFICIENCY`；audio boundary、
  `primary_eligible` 和 `SourceEventTiming` 保持为空。
- Localhost UI 已在真实 355-item workspace 启动并初始化 `0600` working sheet。Desktop/mobile
  浏览器检查均加载真实 1920×1080 slide，mobile horizontal overflow 为 0，console 0
  error/warning；未保存测试标签，当前 progress 仍为 0/355。
- Protocol/freezer targeted tests 21、server tests 11；全套 `322 passed`，两个既有 warnings。
  实现 commit `b6cd276`。Author workflow 与 freeze 命令见
  `docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md`。
- 下一 gate 是真实 human authoring；在 355/355 freeze 前不生成 audio task，不编译 event packets，
  不启动 MCIF ST inference。R1/R2 visual-only event discovery 保持独立，不由这批 R0 events
  替代。

## 2026-08-01 MCIF Beyond-OCR Candidate Inventory Completion

- 新增独立 outcome-side builder，绑定 919-row official references、304-state corrected
  R0/R1/R2 ladder、304-row source-only Qwen3-VL output 及 clean Git `eb601f6`。VLM 自带的旧
  nominal timing 被明确忽略，所有 causal current/earliest-contiguous 判断只使用 ladder 的
  `t+0.5s` intervals。
- R1 strict 只解析 `chart_markdown`、`table_html`、`formula_latex` 的实际可见内容；不读取
  serialized `model_input_text`，HTML/LaTeX markup 不算 evidence，并排除当前 R0 candidates。
  先前 naive 23 个 R1-only 候选收敛为 2 个真实候选 / 2 talks：`graduate school` 与 `metric`。
- R2 只读 `scene_summary`、`objects`、`actions`、`spatial_relations`，完全排除 `ocr_text`，并
  排除当前 R0 与所有实际 R1 block lexical candidates。最终得到 150 个 proposals / 21 talks /
  118 segments；122 个 lead≥5 秒，86 个≥10 秒，最大 173.677 秒。
- R2 抽样同时包含 `sql lambda`、流程/关系等可能有效项和 `content`、`presentation` 等明显
  泛化噪声。每行因此保持 `visual_evidence_correct`、`ocr_insufficient`、event eligibility、
  target realizations 与 audio sufficiency 为空；这批数据不能被解释为 `pixels > OCR`。
- 构建器对 references/ladder row hash、VLM source-only boundary、exact model/prompt provenance、
  raw-to-canonical 去重/空列表补全、first source occurrence、causal interval、连续 evidence、
  create-once 与 checksum fail closed。目标测试 11、全套 `333 passed`，仅有两个既有
  `pypinyin` warnings。
- 正式 6-file bundle / 473,097 bytes 与独立 rebuild byte-identical，5-entry `SHA256SUMS`
  全通过。上传 private HF revision
  [`01defe41/beyond_ocr_candidate_inventory_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/01defe410b4fde07c647d8ed241dfbe501b5d691/beyond_ocr_candidate_inventory_v1)，
  tag `mcif-beyond-ocr-candidate-inventory-v1`；远端 6 files 全量回下载并逐字节验证。
- 下一步构造与 R0 authoring 物理隔离的 R1/R2 validation workspace。人工先判断 VLM 描述是否
  符合 pixels、候选是否真的无法由 R0 OCR 得到、是否构成可计分 En→Zh event；通过后才进入
  独立 audio-sufficiency 与 event-packet gate。
