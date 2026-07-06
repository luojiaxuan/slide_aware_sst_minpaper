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

## 2026-07-06 Qwen-VL Slide Context Enrichment

- Added `repo/scripts/enrich_visual_context.py`.
- Added `repo/tests/test_enrich_visual_context.py`.
- Validated `Qwen/Qwen2.5-VL-3B-Instruct` on Hyper00.
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

## Open Items

1. Upload reusable derived artifacts to Hugging Face and record repo revisions in
   `docs/SOURCE_OF_TRUTH.md`.
2. Generate fresh pseudo references using the enriched Qwen-VL context artifacts.
3. Build OCR-only, VLM-summary, OCR+VLM, policy, and wrong-context experiment
   runs on the enriched split.
4. Select and send a 500-1,000 item diagnostic set for human English
   translation.
5. Add an ST-native no-visual sanity check dataset such as BSTC for pipeline
   validation.
