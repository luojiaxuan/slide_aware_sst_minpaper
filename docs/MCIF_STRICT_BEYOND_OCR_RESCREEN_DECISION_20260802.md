# MCIF Strict Beyond-OCR Re-screen Decision

日期：2026-08-02

状态：`DECISION_FROZEN_IMPLEMENTATION_PENDING`

## 当前判断

MCIF exploratory screen 没有否定整个 paper 空间，但否定了宽泛的
`raw image > flat OCR on arbitrary slide-aligned segments` 叙事。当前仍有两个有效信号：

- correct raw image 相对 matched wrong image 在 clean/noisy 下都有正的 talk-cluster bootstrap
  interval，说明模型不是只利用 image presence；
- slide 在 speech 前可用，允许视觉处理离开 online audio critical path。这一因果时序优势仍是
  simultaneous ST 特有的研究对象。

同时，当前 screen 不能回答严格的 beyond-OCR 问题。34 条中 10 条 candidate 的全部 tokens 已
分别出现在 flat OCR 中，21 条至少 50% tokens 已出现。整句 prefix-AUC chrF 也会稀释少量术语
或提前 commit 的收益。

## 模型容量结论

本次使用 `Qwen/Qwen3-Omni-30B-A3B-Instruct`。它的总参数量不小，但 vision tower 为
27 layers、hidden size 1152。`Qwen/Qwen3-VL-235B-A22B-Instruct` 的官方 config 也是
27 layers、hidden size 1152；直接换成 235B 并不是换一个更大的 image encoder，主要变化是
multimodal reasoning/text backbone、训练与融合能力。

Qwen3-VL/Qwen3.5 不能作为当前 audio+image runner 的 drop-in replacement。利用 slide
pre-availability，更合理的 test 是先用强 image-text model 离线生成 bounded structured visual
evidence packet，再让同一个 Qwen3-Omni online ST backbone 消费 packet。首选 strong compiler
候选为 `Qwen/Qwen3.5-122B-A10B`；`Qwen/Qwen3-VL-235B-A22B-Instruct` 作为容量/家族 sensitivity，
均需在 launch 时冻结 immutable revision。

官方模型信息：

- <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json>
- <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/blob/cbaeae4914228a41d84c560a6d36ac971afa66bc/config.json>
- <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>

## 修订 Kill Test

### Candidate contract

重筛 30--50 条，并在 GPU inference 前冻结以下人工判断：

1. flat OCR 单独不足以恢复目标 evidence；
2. image pixels 足以恢复；
3. evidence 在 speech 中出现前至少可用 5 秒；
4. 按 `chart/color/legend relation`、`diagram/arrow/entity link`、`spatial/order/count`、
   `OCR-error-only` 分层；最后一层只作 control，不能冒充 semantic beyond-OCR；
5. candidate token overlap 只作自动审计，不能替代人工 OCR-sufficiency 判断。

### Evidence conditions

所有 online ST 条件固定同一个 Qwen3-Omni revision、audio prefixes、decode policy 与 token budget：

| condition | pre-speech evidence |
| --- | --- |
| `audio_only` | none |
| `flat_ocr` | 当前 flat OCR |
| `strong_vlm_packet` | strong compiler 从 correct image 生成的 bounded structured packet |
| `wrong_vlm_packet` | 同一 compiler 从 matched wrong image 生成的 packet |
| `human_oracle_packet` | 人工从 pixels 写出的同 schema、同预算 packet |

先跑 clean 与 +5 dB babble。direct raw-image Omni 只保留为 fusion diagnostic，不再作为唯一主方法。

### Primary metrics

- candidate/event correctness；
- first correct realization time 与 stable commit time；
- strong packet 相对 OCR/wrong packet 的 paired effect；
- sentence chrF/COMET 只作整体 no-harm 与 secondary quality 指标。

### 三路判定

1. `human_oracle_packet <= flat_ocr`：数据/任务没有足够 beyond-OCR 空间，停止扩大模型；
2. `human_oracle_packet > flat_ocr`，但 `strong_vlm_packet <= flat_ocr`：空间存在，瓶颈在 visual
   compiler/fusion，可研究抽取、训练或 gating；
3. `strong_vlm_packet > flat_ocr` 且 `strong_vlm_packet > wrong_vlm_packet`，同时无整体质量退化：
   paper 主线成立，进入大样本、noise curve 与 selective gating。

这个实验能区分数据问题、visual extraction 问题和 online fusion 问题。单纯在旧 34 条上更换
更大总参数模型不能作出这一区分，因此不执行原矩阵的盲目 scale-up。
