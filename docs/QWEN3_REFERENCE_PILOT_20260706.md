# Qwen3-32B Reference Pilot, 2026-07-07

这份记录沉淀 Qwen3-32B pseudo-reference pilot 的可复现状态。Git 记录代码、
配置和结论；可复用数据 artifact 已上传到 private Hugging Face dataset repo。

## 输入

- Source HF repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
- Source HF revision:
  `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`
- Source HF tag: `qwen3_vl_context_v1`
- Input file:
  `annotation/diagnostic_sample_500_qwen3_vl_context.jsonl`
- Local staging path on Hyper00:
  `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/hf_revision/a83770446ded4599bf9d95d2b77cdcc7fe359ef7`

## Teacher 配置

- Teacher: `Qwen/Qwen3-32B`
- Runtime: `hf_transformers`
- Device: one Hyper00 H200 selected by `gpu-idle-docker-cleanup` preflight.
- Dtype: `bfloat16`
- Decoding: `temperature=0.0`, `max_new_tokens=192`
- Qwen3 thinking: disabled with `enable_thinking: false`
- Final throughput setting: `batch_size=48`
- Config: `repo/configs/chinese_lips_qwen3_hf_reference_pilot.yaml`

## 代码变更

- `repo/src/slidesst/translation/adapters.py`
  - 支持 Qwen3 `enable_thinking: false` 透传到 chat template。
  - 支持 optional `system_prompt`。
  - 支持 batched HF generation 和通用 `complete_prompts`。
- `repo/scripts/generate_references.py`
  - 支持 `--batch-size`，当 translator 支持 batch 时批量生成。
- `repo/scripts/repair_references.py`
  - 对 `target_cjk_chars` 和 `length_ratio_high` 等问题项做 targeted repair。
- `repo/scripts/package_reference_generation_bundle.py`
  - 打包 reference pilot、audit CSV/summary 和 manifest，用于 HF 上传。
- `repo/src/slidesst/data/reference_audit.py`
  - 英文 visual text 匹配改为词边界匹配，避免 `man` 命中 `many`。
  - `copied_visual_text` 从 automatic reject 降为 review，因为 proper nouns
    和术语可能同时出现在语音和视觉上下文里。

## 质量结果

初始 2 条 smoke 暴露了 Qwen3 默认输出 `<think>` 的问题；关闭 thinking 后
2/2 audit pass。随后 batch=8 smoke 为 8/8 pass。

100 条 pilot 的初始生成中有 3 条残留中文字符：

- `观赏性`
- `漂流`
- `布朗`

Targeted repair 修复了这 3 条。最终 repaired 100 条结果：

| Artifact | Rows | Audit result |
| --- | ---: | --- |
| `pilot_100_refs_repaired.jsonl` | 100 | 84 pass, 16 review, 0 reject |
| `pilot_100_reference_audit_repaired.csv` | 100 | `copied_visual_text`: 8, `length_ratio_review`: 10 |

`review` 不代表自动失败，主要用于人工抽查：

- `copied_visual_text` 多数是语音中实际出现的 proper nouns 或术语也出现在
  slide/visual context。
- `length_ratio_review` 是较长但通常合理的英语展开。

## GPU 利用率调参

所有 GPU run 前都执行了 `gpu-idle-docker-cleanup` preflight，只使用选出的空闲
GPU。调参结论：

| Setting | Sample | Utilization | Peak memory | Decision |
| --- | ---: | ---: | ---: | --- |
| batch=16 | 100 rows | 67-83% average in 10s monitor windows | about 84GiB | 不达标 |
| batch=32 | 64 rows | 88.3% average in one 10s manual window | about 106GiB | 接近但不稳 |
| batch=48 | 48 rows | 100% for all 10 manual samples | about 122GiB | 当前推荐 |

Batch=48 生成 48 条用了 45.9 秒，没有 OOM；audit 为 43 pass、5 review、0
reject。它是当前单卡 Qwen3-32B reference generation 的推荐 pilot setting。

## Hugging Face Artifact

- HF repo:
  <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
- HF commit:
  `ee785604ba51a5c65335de12bfcfd99d3c4febff`
- HF tag:
  `qwen3_32b_reference_pilot_20260706`
- Path in repo:
  `reference_pilots/qwen3_32b_reference_pilot_20260706/`
- Local bundle:
  `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_32b_reference_pilot_20260706`

Uploaded files include:

- `reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl.gz`
- `reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_reference_audit_repaired.csv`
- `reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_reference_audit_summary_repaired.json`
- `manifest_qwen3_32b_reference_pilot_20260706.json`
- `README_qwen3_32b_reference_pilot_20260706.md`

The HF repo must remain private because upstream Chinese-LiPS is gated.

## 下一步

1. 用 batch=48 + repair pipeline 跑完整 diagnostic 500。
2. 若 diagnostic 500 的 reject 仍为 0，再决定是否扩展到更大训练子集。
3. 基于 repaired references 构建 OCR-only、VLM-summary、OCR+VLM、policy、
   wrong-context 实验输入。
