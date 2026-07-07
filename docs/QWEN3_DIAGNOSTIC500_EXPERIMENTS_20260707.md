# Qwen3-32B Diagnostic 500 Experiments, 2026-07-07

这份记录沉淀 repaired diagnostic 500 上的第一轮 context-ablation 实验。Git 记录
代码、配置、结论和限制；可复查的模型输出、表格和 manifest 已上传到 private
Hugging Face dataset repo。

## 输入和产物

- Source HF repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>
- Source context revision:
  `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`
- Source reference tag:
  `qwen3_32b_reference_diagnostic500_20260707`
- Reference HF commit:
  `5ca0c090fc6d76ac50938924b28a57b1026c3043`
- Experiment HF commit:
  `3cc7249d45eca71a4f0b5c06a6b0773efead128a`
- Experiment HF tag:
  `qwen3_32b_diagnostic500_experiments_20260707`
- Path in HF repo:
  `experiments/qwen3_32b_diagnostic500_experiments_20260707/`
- Local run root on Hyper00:
  `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/experiments/qwen3_32b_diagnostic500`

HF bundle includes 7 compressed `outputs.jsonl.gz` files, paper-table CSV/TeX
files, a README, and a manifest with SHA-256 checksums and row counts. Each
condition has 500 outputs.

## 代码和配置

- Runner:
  `repo/scripts/run_batched_reference_experiments.py`
- Config:
  `repo/configs/chinese_lips_qwen3_diagnostic500_eval.yaml`
- Evaluation:
  `repo/scripts/evaluate.py`
- Table generation:
  `repo/scripts/make_paper_tables.py`

The runner uses the existing evidence selection and policy logic from
`run_stream_translate.py`, but evaluates only the final transcript state. This
keeps Qwen3-32B batched generation efficient. It is therefore a context
ablation over final-state transcript translation, not yet a full simultaneous
latency experiment.

## 条件

| Condition | Meaning |
| --- | --- |
| `V0_no_context` | transcript only |
| `V2_ocr_only` | OCR evidence only |
| `V3_visual_caption_only` | VLM visual summary/caption evidence only |
| `V4_ocr_plus_visual` | OCR plus visual evidence |
| `V5_naive_all_visual` | all available visual/context evidence |
| `V6_policy_visual` | policy-selected evidence |
| `V8_wrong_visual` | deliberately mismatched visual evidence |

## GPU 利用率和批大小

所有 GPU run 前都执行了 `gpu-idle-docker-cleanup` preflight，只使用选出的空闲
GPU。正式实验只用了单张 H200；第二张 GPU 保持可用但没有参与 Qwen3-32B
进程。

| Stage | Batch size | Result |
| --- | ---: | --- |
| `V5` 64-row tune | 32 | 10 秒窗口平均约 55%，不达标 |
| `V5` 128-row tune | 128 | 10 秒窗口平均约 80.5%，仍不达标 |
| `V5` 192-row tune | 192 | 10 秒窗口平均约 95.1%，可用 |
| Full run `V0`-`V5` | 192 | monitor 窗口分别达到 96% 和 100% |
| Full run `V6` at 192 | 192 | 第二批 OOM，约 137.5GiB 已用时额外申请失败 |
| Rerun `V6`/`V8` | 128 | `V6` monitor 窗口平均 99%，完成 |

最终 artifact 记录的 batch size:

| Condition | Batch size |
| --- | ---: |
| `V0_no_context` | 192 |
| `V2_ocr_only` | 192 |
| `V3_visual_caption_only` | 192 |
| `V4_ocr_plus_visual` | 192 |
| `V5_naive_all_visual` | 192 |
| `V6_policy_visual` | 128 |
| `V8_wrong_visual` | 128 |

## 主结果

BLEU is computed against the repaired Qwen3-32B diagnostic references. It is a
useful first-pass ranking signal, not a substitute for human reference
validation.

| Condition | BLEU |
| --- | ---: |
| `V0_no_context` | 76.50 |
| `V2_ocr_only` | 83.41 |
| `V3_visual_caption_only` | 83.88 |
| `V4_ocr_plus_visual` | 85.17 |
| `V5_naive_all_visual` | 84.75 |
| `V6_policy_visual` | 83.24 |
| `V8_wrong_visual` | 81.66 |

Initial interpretation:

- OCR/visual context clearly improves over no-context on this diagnostic slice.
- `V4_ocr_plus_visual` is best in BLEU, slightly ahead of naive all-visual.
- `V8_wrong_visual` is worse than the matched visual conditions, which is the
  expected direction for context mismatch sensitivity.
- `V6_policy_visual` underperforms `V4_ocr_plus_visual`; this needs diagnosis
  before scaling policy experiments.

## 当前不能过度解读的指标

The diagnostic 500 artifact does not yet have manual `hard_label`,
`supporting_ids`, or verified hallucination labels. Therefore:

- HDA has `n=0` and is not meaningful.
- Evidence precision/recall are not meaningful.
- Visual hallucination / object/action grounding metrics are placeholders or
  weak automatic signals.
- The current result is suitable for pipeline validation and a first ablation,
  not final paper claims.

## 输出质量扫描

No condition produced `<think>` leakage. Residual CJK rows are low but non-zero:

| Condition | Residual CJK rows | Evidence phrase rows |
| --- | ---: | ---: |
| `V0_no_context` | 2 | 1 |
| `V2_ocr_only` | 1 | 1 |
| `V3_visual_caption_only` | 3 | 1 |
| `V4_ocr_plus_visual` | 4 | 1 |
| `V5_naive_all_visual` | 3 | 1 |
| `V6_policy_visual` | 3 | 1 |
| `V8_wrong_visual` | 2 | 1 |

These are model hypothesis outputs, not pseudo references. They were not
repaired because repair would change method behavior. They should be kept for
system comparison and inspected during case-study selection.

## 下一步

1. 给 diagnostic 500 增加人工 hard-label、supporting evidence ids 和 hallucination
   review labels。
2. 诊断 `V6_policy_visual` 为什么不如 `V4_ocr_plus_visual`，重点看 evidence
   selection 是否过窄、是否丢掉 OCR terms、以及 policy prompt 是否保守。
3. 在人工标签和指标语义修好前，不扩大宣传性结论；最多扩大 pseudo-reference
   generation 用于模型训练或更多探索。
