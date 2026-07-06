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
| Reusable datasets and generated data artifacts | Hugging Face Hub | Not uploaded yet |
| Reusable checkpoints/adapters | Hugging Face Hub | None yet |
| Local staging and active runs | Hyper00 `/data` paths | Temporary, not canonical |

## Current Git Pointers

- GitHub repo: <https://github.com/luojiaxuan/slide_aware_sst_minpaper>
- Current project framing: [`code_plan/SLIDE_CONTEXT_AWARE_MVP.md`](../code_plan/SLIDE_CONTEXT_AWARE_MVP.md)
- Agent handoff plan: [`code_plan/AGENT_START_HERE.md`](../code_plan/AGENT_START_HERE.md)
- Experiment matrix: [`code_plan/EXPERIMENT_MATRIX.md`](../code_plan/EXPERIMENT_MATRIX.md)
- Progress log: [`docs/PROGRESS.md`](PROGRESS.md)

## Dataset and Artifact Truth

### Upstream Dataset

- Source dataset: `BAAI/Chinese-LiPS`
- Hugging Face source: <https://huggingface.co/datasets/BAAI/Chinese-LiPS>
- Local staging path on Hyper00: `/data/datasets/chinese_lips`
- Local staging is cache/scratch only. It is not a source of truth.

### Reusable Derived Artifacts

Reusable derived artifacts are not yet uploaded to Hugging Face. Until upload,
they are staged locally and recorded here with intended destinations.

| Artifact | Local path | Intended HF destination | Upload status |
| --- | --- | --- | --- |
| Chinese-LiPS frame-backed train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified.jsonl` | TBD, likely `<hf-owner>/slide-context-sst-chinese-lips` | Not uploaded |
| Qwen2.5-VL pilot enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen2.5-VL pilot enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen3-VL enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl` | TBD, likely `<hf-owner>/slide-context-sst-chinese-lips` | Running locally |
| Qwen3-VL enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl` | TBD, same dataset repo as above | Pending Qwen3-VL completion |
| Qwen3-VL train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv` | TBD, same dataset repo as above or Git if kept as lightweight metadata | Pending Qwen3-VL completion |
| Train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen_vl_context.csv` | TBD, same dataset repo as above or Git if kept as lightweight metadata | Not uploaded |
| Test diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_test/annotation/diagnostic_sample_500.csv` | TBD | Not uploaded |

Before any paper-grade or cross-repo reuse, upload the derived JSONL/CSV files to
Hugging Face and record the exact repo revision here.

## Active Compute Context

- Host: `hyper00`
- Hostname: `node-radixark-16-0000`
- Container: `sglang-omni-jaxan-vision-sst-0701`
- Repo in container: `/data/projects/slide_aware_sst_minpaper`
- Dataset staging: `/data/datasets/chinese_lips`
- HF cache: `/root/.cache/huggingface`
- Resource policy: use at most 2 GPUs by default on Hyper00, and first make the
  active GPU utilization sustain at least 90%. For Qwen3-VL enrichment, the
  next longer trial should use 1 GPU with `--batch-size 64` for memory
  headroom. Treat `--batch-size 96` as opt-in only after a wider image-size
  distribution profile, and do not expand to a second GPU until the single-GPU
  run remains efficient on a longer shard.

## Active Runs

| Run | Host/container | Model | Input | Output | Status |
| --- | --- | --- | --- | --- | --- |
| `qwen3_vl_train_20260706_164650` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | `outputs/chinese_lips_train/data/challenge_verified.jsonl` | `outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl` | Paused after partial shards; next step is a monitored 1-GPU `--batch-size 64` longer trial |

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

1. Upload reusable derived Chinese-LiPS artifacts to a Hugging Face dataset repo
   and record the revision.
2. Run Qwen3-VL enrichment and fresh pseudo-reference generation on
   `challenge_verified_qwen3_vl_context.jsonl`.
3. Build OCR/VLM vs OCR-only vs wrong-context experiments on the enriched train
   and test splits.
4. Prepare human English translation workflow for the diagnostic 500-1,000 item
   subset.
