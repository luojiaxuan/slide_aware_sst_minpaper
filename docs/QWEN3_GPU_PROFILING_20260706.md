# Qwen3-VL GPU Profiling, 2026-07-06

This note records the profiling evidence used to choose the active Qwen3-VL
Chinese-LiPS train enrichment configuration. Local scratch outputs are not a
source of truth, but the paths below let a follow-up agent inspect raw logs while
they remain on Hyper00.

## Context

- Host: Hyper00, hostname `node-radixark-16-0000`
- Container: `sglang-omni-jaxan-vision-sst-0701`
- Repo path: `/data/projects/slide_aware_sst_minpaper/repo`
- Input: `outputs/chinese_lips_train/data/challenge_verified.jsonl`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Dtype: `bfloat16`
- Final active run id: `qwen3_vl_train_bs56x2_2gpu_20260706_214711`

Before the final run, `gpu-idle-docker-cleanup` preflight selected two free H200
GPUs. At launch time, GPUs 0 and 1 were free and GPUs 2 and 3 were occupied by
another user's `sglang-omni-jingwengu` container.

## Single-Process Batch Sweep

Representative random train frames were profiled with the real
`QwenVLSlideContextExtractor` path. The first sweep used a single process on one
H200 with `max_new_tokens=256`.

| Batch size | Rows | Rows/s | Avg util | P50 util | P90 util | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 192 | 3.074 | 72.9% | 97% | 100% | 69,957 MiB |
| 80 | 240 | 3.262 | 73.2% | 98% | 100% | 91,027 MiB |
| 96 | 288 | 3.381 | 74.7% | 98% | 100% | 109,413 MiB |

Interpretation: generation itself often reached full utilization, but end-to-end
utilization was pulled down by batch-level CPU/image/processor gaps.

## Prefetch Sweep

After adding `--prefetch-batches 1`, the same process shape improved but did not
reach the 90% target.

| Batch size | Rows | Rows/s | Avg util | P50 util | P90 util | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 192 | 3.225 | 76.4% | 97% | 100% | 70,483 MiB |
| 80 | 240 | 3.447 | 78.2% | 98% | 100% | 91,027 MiB |
| 96 | 288 | 3.591 | 80.3% | 98% | 100% | 109,413 MiB |

Interpretation: prefetch helped, but a single process still left too much idle
time between GPU-heavy sections.

## Multi-Worker Validation

The final efficient shape uses two independent workers per GPU. This overlaps
one worker's CPU and processor gap with the other worker's generation phase.

### Single GPU, Two Workers

Scratch run directory:
`/data/tmp/vision_sst_profile/dual_worker_bs48_256_prefetch`

Configuration:

```bash
python3 scripts/enrich_visual_context.py \
  --provider qwen_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:2 \
  --dtype bfloat16 \
  --max-new-tokens 256 \
  --batch-size 48 \
  --prefetch-batches 1 \
  --resume
```

Two workers on one H200 processed 192 rows each. After the first 20 seconds,
average utilization was 91.9%, P50/P90 were both 100%, and peak memory was
109,883 MiB.

### Two GPUs, Four Workers

Scratch run directory:
`/data/tmp/vision_sst_profile/four_worker_2gpu_bs56_256_prefetch`

Configuration: 2 GPUs, 2 workers per GPU, 224 rows per worker, `batch_size=56`,
`max_new_tokens=256`, `prefetch_batches=1`.

| GPU | Avg util after 20s | Avg util after 40s | Avg util after 60s | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| 0 | 93.3% | 93.9% | 97.2% | 122,331 MiB |
| 1 | 91.5% | 91.2% | 94.2% | 122,331 MiB |

This setting satisfies the 90% post-warmup target while keeping about 20GB H200
memory headroom per active GPU in the short validation. The full-train run later
completed successfully and served as the longer image-distribution validation
for this configuration.

## Active Production Run

Run directory:
`/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711`

Shard layout:

| Shard | Device | Offset | Limit |
| --- | --- | ---: | ---: |
| 0 | `cuda:0` | 0 | 7,331 |
| 1 | `cuda:0` | 7,331 | 7,331 |
| 2 | `cuda:1` | 14,662 | 7,330 |
| 3 | `cuda:1` | 21,992 | 7,330 |

Launch command shape per shard:

```bash
python3 scripts/enrich_visual_context.py \
  --input outputs/chinese_lips_train/data/challenge_verified.jsonl \
  --output outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711/shard_<N>.jsonl \
  --provider qwen_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:<0-or-1> \
  --dtype bfloat16 \
  --max-new-tokens 256 \
  --batch-size 56 \
  --prefetch-batches 1 \
  --offset <offset> \
  --limit <limit> \
  --resume
```

Startup validation reached 616 rows per shard with GPU0/GPU1 both at 100%
instantaneous utilization and about 122GB memory per GPU.

The run finished successfully on 2026-07-06. The initial combined artifact had
the expected challenge row count and ids, but stricter post-run QA found 3,387
fallback rows caused by truncated raw model JSON. A targeted repair pass rebuilt
the final challenge, evidence, and diagnostic sample. The final repaired
artifact has 29,322 challenge rows, 526,597 evidence rows, 500 diagnostic rows,
0 missing raw model outputs, and 0 raw parse failures. The repaired Qwen3-VL
context bundle was uploaded to the private HF dataset repo at revision
`a83770446ded4599bf9d95d2b77cdcc7fe359ef7` with tag `qwen3_vl_context_v1`. See
[`docs/QWEN3_CONTEXT_QA_20260706.md`](QWEN3_CONTEXT_QA_20260706.md) for the
quality audit and repair record.

## Failure Handling Notes

The enrichment script currently fails the worker if `Image.open(...).convert`
raises on a corrupt or truncated frame. With `--prefetch-batches 1`, an image
decode failure from the next batch surfaces when the main loop calls
`future.result()`. Because the production command uses `--resume`, already
written rows remain usable, but the same bad row will crash again on restart
until it is inspected or excluded.

If this happens, inspect the shard log, identify the first unwritten source row
from the shard offset plus current output line count, verify the frame path, and
either regenerate/replace that frame or create a documented skip/passthrough
decision before resuming the shard.
