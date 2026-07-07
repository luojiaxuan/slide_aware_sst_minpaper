# Source of Truth

This project uses Git and Hugging Face as the only durable sources of truth.
Notion is not used for project knowledge, progress records, reusable artifact
links, or handoff state.

## Canonical Locations

| Material | Canonical home | Current status |
| --- | --- | --- |
| Code, configs, scripts, tests | GitHub: `luojiaxuan/slide_aware_sst_minpaper` | Active on `main` |
| Paper notes and implementation plans | This Git repo under `paper/`, `code_plan/`, and `docs/` | Active |
| Lightweight progress and artifact index | This Git repo under `docs/` | Active |
| Reusable datasets and generated data artifacts | Hugging Face Hub | Qwen3-VL bundle uploaded to a private dataset repo |
| Reusable checkpoints/adapters | Hugging Face Hub | None yet |
| Local staging and active runs | Hyper00 `/data` paths | Temporary, not canonical |

## Current Git Pointers

- GitHub repo: <https://github.com/luojiaxuan/slide_aware_sst_minpaper>
- Current project framing: [`code_plan/SLIDE_CONTEXT_AWARE_MVP.md`](../code_plan/SLIDE_CONTEXT_AWARE_MVP.md)
- Agent handoff plan: [`code_plan/AGENT_START_HERE.md`](../code_plan/AGENT_START_HERE.md)
- Experiment matrix: [`code_plan/EXPERIMENT_MATRIX.md`](../code_plan/EXPERIMENT_MATRIX.md)
- Progress log: [`docs/PROGRESS.md`](PROGRESS.md)
- Qwen3-VL GPU profiling evidence:
  [`docs/QWEN3_GPU_PROFILING_20260706.md`](QWEN3_GPU_PROFILING_20260706.md)
- Qwen3-VL context QA and repair record:
  [`docs/QWEN3_CONTEXT_QA_20260706.md`](QWEN3_CONTEXT_QA_20260706.md)
- Qwen3-32B reference pilot:
  [`docs/QWEN3_REFERENCE_PILOT_20260706.md`](QWEN3_REFERENCE_PILOT_20260706.md)
- Qwen3-32B diagnostic-500 context ablation:
  [`docs/QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md`](QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md)

## Dataset and Artifact Truth

### Upstream Dataset

- Source dataset: `BAAI/Chinese-LiPS`
- Hugging Face source: <https://huggingface.co/datasets/BAAI/Chinese-LiPS>
- Local staging path on Hyper00: `/data/datasets/chinese_lips`
- Local staging is cache/scratch only. It is not a source of truth.

### Reusable Derived Artifacts

The Qwen3-VL derived train artifacts are uploaded to a private Hugging Face
dataset repo. Because
`BAAI/Chinese-LiPS` is gated and its terms restrict redistribution of derived
works outside the research group without maintainer permission, any HF repo for
these artifacts should be private or otherwise access-controlled unless that
permission is obtained.

| Artifact | Local path | Canonical or intended HF destination | Upload status |
| --- | --- | --- | --- |
| Chinese-LiPS frame-backed train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`, tag `qwen3_vl_context_v1` | Not uploaded as a separate raw artifact |
| Qwen2.5-VL pilot enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen2.5-VL pilot enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen3-VL HF upload bundle | `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_vl_context_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`, tag `qwen3_vl_context_v1` | Uploaded; repo privacy verified as `private=True` |
| Qwen3-VL enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl` | `data/challenge_verified_qwen3_vl_context.jsonl.gz` in the private HF repo above | Uploaded; 29,322 rows; QA passed |
| Qwen3-VL enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl` | `index/evidence_qwen3_vl_context.jsonl.gz` in the private HF repo above | Uploaded; 526,597 rows; internal consistency check passed |
| Qwen3-VL train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv` | `annotation/diagnostic_sample_500_qwen3_vl_context.*` in the private HF repo above | Uploaded; 500 rows |
| Qwen3-VL context QA report | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json` | `qa/qwen3_vl_context_qa.json` in the private HF repo above | Uploaded |
| Qwen3-32B repaired reference pilot | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `ee785604ba51a5c65335de12bfcfd99d3c4febff`, tag `qwen3_32b_reference_pilot_20260706`, path `reference_pilots/qwen3_32b_reference_pilot_20260706/` | Uploaded; 100 rows; final audit 84 pass, 16 review, 0 reject |
| Qwen3-32B repaired diagnostic 500 references | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_repaired.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `5ca0c090fc6d76ac50938924b28a57b1026c3043`, tag `qwen3_32b_reference_diagnostic500_20260707`, path `reference_pilots/qwen3_32b_reference_diagnostic500_20260707/` | Uploaded; 500 rows; final audit 435 pass, 65 review, 0 reject |
| Qwen3-32B diagnostic 500 context-ablation experiment bundle | `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_32b_diagnostic500_experiments_20260707` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `3cc7249d45eca71a4f0b5c06a6b0773efead128a`, tag `qwen3_32b_diagnostic500_experiments_20260707`, path `experiments/qwen3_32b_diagnostic500_experiments_20260707/` | Uploaded; 7 conditions x 500 outputs plus tables and manifest |
| Train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen_vl_context.csv` | Same private HF repo if retained as a pilot artifact | Not uploaded |
| Test diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_test/annotation/diagnostic_sample_500.csv` | TBD | Not uploaded |

Do not make this derived Chinese-LiPS repo public unless upstream permission is
confirmed.

## Active Compute Context

- Host: `hyper00`
- Hostname: `node-radixark-16-0000`
- Container: `sglang-omni-jaxan-vision-sst-0701`
- Repo in container: `/data/projects/slide_aware_sst_minpaper`
- Dataset staging: `/data/datasets/chinese_lips`
- HF cache: `/root/.cache/huggingface`
- Resource policy: use at most 2 GPUs by default on Hyper00, and first make the
  active GPU utilization sustain at least 90%. For the current Qwen3-VL
  enrichment, the selected monitored setting is 2 H200 GPUs with 2 workers per
  GPU, `--batch-size 56` per worker, `--max-new-tokens 256`, and
  `--prefetch-batches 1`. A 2026-07-06 Hyper00 short run sustained at least
  91% utilization after warmup on both active GPUs with about 122GB peak memory
  per GPU. The full-train run completed on 2026-07-06. Post-run QA found 3,387
  fallback rows caused by truncated raw model JSON, then a targeted Qwen3-VL
  repair pass replaced exactly those rows. Final QA has 0 missing raw model
  outputs and 0 raw parse failures; the rebuilt evidence index also passes an
  internal source-count consistency check against the repaired challenge.

## Active Runs

| Run | Host/container | Model | Input | Output | Status |
| --- | --- | --- | --- | --- | --- |
| `qwen3_vl_train_bs56x2_2gpu_20260706_214711` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | `outputs/chinese_lips_train/data/challenge_verified.jsonl` | final repaired artifacts under `outputs/chinese_lips_train/{data,index,annotation,qa}/` plus shard outputs under `outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711/` | Completed and repaired locally; final QA has 29,322 challenge rows, 526,597 evidence rows, 500 diagnostic rows, 0 missing raw model outputs, 0 raw parse failures; uploaded to private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` |
| `qwen3_parse_failure_repair512_20260706_231750` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | 3,387 failed Qwen3-VL rows from the initial combined artifact | repair outputs under `outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/` | Completed locally; 512/768/compact/strict passes repaired all initial parse failures |
| `qwen3_32b_reference_pilot_20260706` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | 100 rows from private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` diagnostic sample | `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl` and HF path `reference_pilots/qwen3_32b_reference_pilot_20260706/` | Completed; uploaded to private HF commit `ee785604ba51a5c65335de12bfcfd99d3c4febff`; tag `qwen3_32b_reference_pilot_20260706` |
| `qwen3_32b_reference_diagnostic500_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | 500 rows from private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` diagnostic sample | `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_repaired.jsonl` and HF path `reference_pilots/qwen3_32b_reference_diagnostic500_20260707/` | Completed; batch=40 after batch=48 OOM; final audit 435 pass, 65 review, 0 reject; uploaded to private HF commit `5ca0c090fc6d76ac50938924b28a57b1026c3043`; tag `qwen3_32b_reference_diagnostic500_20260707` |
| `qwen3_32b_diagnostic500_experiments_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | repaired diagnostic 500 references plus Qwen3-VL evidence index | 7 condition outputs under `outputs/chinese_lips_train/experiments/qwen3_32b_diagnostic500/` and HF path `experiments/qwen3_32b_diagnostic500_experiments_20260707/` | Completed; uploaded to private HF commit `3cc7249d45eca71a4f0b5c06a6b0773efead128a`; tag `qwen3_32b_diagnostic500_experiments_20260707`; BLEU best is `V4_ocr_plus_visual` at 85.17, but hard-label/evidence metrics are not paper-grade until manual labels are added |
| `qwen3_vl_train_20260706_164650` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | `outputs/chinese_lips_train/data/challenge_verified.jsonl` | partial shards under `outputs/chinese_lips_train/enrichment/qwen3_vl_train_20260706_164650/` | Paused/superseded; do not resume with old one-sample-per-process settings |

## Current Durable Decisions

1. Keep Chinese-LiPS as the main dataset.
2. Do not frame the first paper as pure vision-aware SST.
3. Use the safer umbrella: slide/context-aware simultaneous speech translation
   under latency constraints.
4. Treat OCR terms as a strong baseline, not a failure.
5. Treat VLM captions/summaries as beyond-OCR evidence to test on low-OCR and
   visual-non-OCR slices.
6. Use Qwen pseudo references for scale, but build a human English diagnostic
   subset for paper-grade evaluation.
7. Use Git docs for progress and Hugging Face for reusable data/model artifacts.

## Current Next Actions

1. Add manual hard-label/supporting-evidence annotations for the diagnostic
   500 before claiming HDA, evidence precision/recall, or paper-grade visual
   hallucination metrics.
2. Inspect why `V6_policy_visual` underperforms `V4_ocr_plus_visual` on BLEU
   before scaling policy experiments.
3. Decide whether to scale Qwen3-32B pseudo references beyond diagnostic 500
   after manual diagnostic labels and metric semantics are fixed.
