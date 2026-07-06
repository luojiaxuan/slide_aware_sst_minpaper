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
- Planned final train artifacts:
  - `outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl`
  - `outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl`
  - `outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.*`

## Open Items

1. Upload reusable derived artifacts to Hugging Face and record repo revisions in
   `docs/SOURCE_OF_TRUTH.md`.
2. Generate fresh pseudo references using the enriched Qwen3-VL context artifacts.
3. Build OCR-only, VLM-summary, OCR+VLM, policy, and wrong-context experiment
   runs on the enriched split.
4. Select and send a 500-1,000 item diagnostic set for human English
   translation.
5. Add an ST-native no-visual sanity check dataset such as BSTC for pipeline
   validation.
