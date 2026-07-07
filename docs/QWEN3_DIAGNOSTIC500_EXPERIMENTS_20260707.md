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

BLEU is computed against repaired Qwen3-32B diagnostic references, while the
systems under test also use `Qwen/Qwen3-32B`. Treat these numbers as
self-BLEU/pipeline sanity signals only. They are not valid method rankings until
the same outputs are scored against independent or human references.

| Condition | BLEU |
| --- | ---: |
| `V0_no_context` | 76.50 |
| `V2_ocr_only` | 83.41 |
| `V3_visual_caption_only` | 83.88 |
| `V4_ocr_plus_visual` | 85.17 |
| `V5_naive_all_visual` | 84.75 |
| `V6_policy_visual` | 83.24 |
| `V8_wrong_visual` | 81.66 |

Current interpretation boundary:

- The table verifies that the pipeline can run all intended conditions and
  preserve 500 outputs per condition.
- Because references and hypotheses come from the same teacher family and the
  reference generator used full evidence, conditions with similar evidence can
  receive self-agreement credit.
- Because `V0`-`V5` ran with batch=192 and `V6`/`V8` ran with batch=128,
  cross-condition ranking also has a batch-shape confound.
- Do not claim that `V4` is better than `V6`, or that visual context improves
  translation quality, from this table alone.

## 当前不能过度解读的指标

The diagnostic 500 artifact does not yet have manual `hard_label`,
`supporting_ids`, or verified hallucination labels. Therefore:

- HDA has `n=0` and is not meaningful.
- Evidence precision/recall are not meaningful.
- Visual hallucination / object/action grounding metrics are placeholders or
  weak automatic signals.
- The current result is suitable for pipeline validation and a first ablation,
  not final paper claims.

## Claude Review and Batch160 Sensitivity

Claude hostile review marked the original interpretation as `NEEDS_FIX`.
Accepted issues:

- BLEU is Qwen3-32B self-BLEU because references and hypotheses were generated
  by the same model family.
- The parent run had a batch-shape confound: `V0`-`V5` used batch=192 while
  `V6`/`V8` used batch=128.
- The actual per-condition batch sizes needed to be recorded in Git config, not
  only in prose.

Rejected with evidence:

- Reference source text mismatch: on the repaired diagnostic 500, every final
  streaming partial equals `source_transcript`.
- V8 correct-evidence leakage: the runner appends wrong evidence to the item,
  but the V8 packet selector keeps only `wrong_video`, `wrong_clip`, or
  `negative_visual`; a unit test now locks this behavior.

Follow-up sensitivity run:

- HF commit:
  `03f59f1babc0c37e778e8f415bc85ab5fb36f573`
- HF tag:
  `qwen3_32b_diagnostic500_batch160_visual_policy_20260707`
- Path in HF repo:
  `experiments/qwen3_32b_diagnostic500_batch160_visual_policy_20260707/`
- Local run:
  `outputs/chinese_lips_train/experiments/qwen3_32b_diagnostic500/runs_batch160_visual_policy/`

| Condition | Original batch | Original BLEU | Batch160 BLEU | Delta |
| --- | ---: | ---: | ---: | ---: |
| `V4_ocr_plus_visual` | 192 | 85.1712 | 85.2877 | +0.1165 |
| `V5_naive_all_visual` | 192 | 84.7461 | 84.9740 | +0.2279 |
| `V6_policy_visual` | 128 | 83.2369 | 83.5166 | +0.2797 |
| `V8_wrong_visual` | 128 | 81.6593 | 81.6652 | +0.0060 |

GPU monitor windows for the full batch160 sensitivity run were 99% on V4,
100% on V5, and 91% on V6. A V0 batch160 tune reached only 83%, so the
all-condition uniform-batch rerun was stopped under the project GPU utilization
rule.

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
2. 准备 independent/human references；在此之前不要把 Qwen3-vs-Qwen3 BLEU
   当作方法排名。
3. 统一 batch=128 重跑，或至少重跑一个 batch=192 条件到 batch=128 来量化
   batch-shape sensitivity。
4. 在人工标签和指标语义修好前，不扩大宣传性结论；最多扩大 pseudo-reference
   generation 用于模型训练或更多探索。

## Diagnostic Review Sheet

为了推进人工审核，已生成并上传 diagnostic 500 review sheet：

- Script: `repo/scripts/export_diagnostic_review_sheet.py`
- Local CSV:
  `outputs/chinese_lips_train/annotation/diagnostic_review_sheet_500_qwen3_context_experiments_20260707.csv`
- HF commit:
  `3d681ebe85babdacffe5e984bf59af6cade9c2f1`
- HF tag:
  `qwen3_32b_diagnostic500_review_sheet_20260707`
- HF path:
  `annotation/qwen3_32b_diagnostic500_review_sheet_20260707/`

The sheet has 500 rows and includes the candidate Qwen3 reference, reference
audit flags, visual/OCR context, V4/V6 evidence packets, all 7 parent-run
hypotheses, and blank columns for human reference, reference quality,
requires-visual/requires-OCR labels, supporting evidence ids, hallucination
conditions, and reviewer notes.
