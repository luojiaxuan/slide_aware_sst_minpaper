# MCIF Beyond-OCR Exploratory Screen V1

日期：2026-08-02

状态：`INPUT_BUNDLE_READY_INFERENCE_NOT_STARTED`

## 目标

先用一个小而完整的因果矩阵回答是否值得投入正式人工标注：

```text
audio-only / OCR / raw image / wrong image x clean / +5 dB babble
```

这个阶段是 `exploratory_screen_not_paper_gold`。它允许从 automatic、reference-aware
candidate inventory 中挑选高置信候选并用 reference 评分，但 reference、source transcript、
candidate text 和 candidate identity 均不能进入模型 worker。只有当 `raw_image` 在同一样本上
同时优于 `ocr` 和 `wrong_image`，才对对应 positive samples 启动正式人工验证。

## 冻结范围

- 34 个候选，来自 16 个 talk、34 个不同 segment；
- 每个 talk 最多 3 条，每个 segment 恰好最多 1 条；
- slide evidence 至少领先 speech 5 秒；segment 长度为 3--24 秒；
- 候选必须出现在 English source reference 中、且不出现在当前 flat OCR；
- 配置：`code/configs/mcif_beyond_ocr_exploratory_screen_v1.json`；
- config SHA256：`863e6af816e9f6a51f149a03b3a6677ce8bfa8e6530cf9c15cff1f63641390af`。

候选筛选是 automatic inventory 上的一次人工高置信 prescreen，不是 visual correctness、OCR
insufficiency、target realization 或 audio sufficiency 的正式 annotation。不能把 34 条写成
gold events，也不能把本次结果当作 held-out confirmatory evidence。

## 输入与控制

每个 candidate 生成 clean 和 deterministic `babble_p5_s0` 两个音频输入。后者使用 development
noise pool 中 5 个 `babble_speech` source、global seed `20260801`，speech-active SNR 为 +5 dB。
当前构建的 34 条 achieved SNR 范围为 `[4.9999996, 5.0000004]` dB。

四个 evidence conditions 为：

| condition | 模型输入 |
| --- | --- |
| `audio_only` | 当前 audio prefix |
| `ocr` | pre-available flat OCR + 当前 audio prefix |
| `raw_image` | 当前 native causal slide + 当前 audio prefix |
| `wrong_image` | cross-talk matched wrong image + 当前 audio prefix |

`wrong_image` 与 correct image 的 Qwen3-Omni `image_grid_thw` 和 visual-token count 精确匹配。
模型固定为 `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
`26291f793822fb6be9555850f06dfe95f2d7e695`，1 秒 audio chunk、96 max new tokens、
deterministic decoding。总输入 68 条，总结果应为 `68 x 4 = 272` 条。

## Scoring Gate

runner 保存每个 1 秒 prefix 的 raw hypothesis。primary exploratory metric 是该样本相对完整人工
Chinese reference 的 `prefix_auc_sentence_chrf`；clean 与 noisy 分别报告。单个
`candidate x acoustic condition` 的 positive rule 是严格：

```text
raw_image > ocr AND raw_image > wrong_image
```

同时报告 final sentence chrF、四个 condition 的 aggregate mean，以及按 talk cluster 的
10,000 次 bootstrap。bootstrap 只用于描述小样本不确定性，不把该 prescreen 升格为正式统计检验。

## Reference Firewall

builder 物理生成两个 sibling roots：

- `inference_bundle/`：只含 68 个 source-side input rows、audio 和 correct/wrong images；
- `scorer_private/`：含 candidate mapping、English/Chinese references 和 selection provenance。

GPU host 只能收到 `inference_bundle/`。`scorer_private/` 保留在本地 scorer 侧，推理结束后才按
opaque `screen_id` join。构建检查拒绝 reference/transcript/translation/candidate 类字段，并确认
Chinese target reference 未出现在 serialized inference rows 中。

## Source of Truth

### Git

- repo：<https://github.com/luojiaxuan/slide_aware_sst_minpaper>
- config：`code/configs/mcif_beyond_ocr_exploratory_screen_v1.json`
- bundle builder：`code/scripts/build_mcif_beyond_ocr_exploratory_screen.py`
- inference runner：`code/scripts/omni_speech_vision_probe.py`
- analyzer：`code/scripts/analyze_mcif_beyond_ocr_exploratory_screen.py`

### Local Staging

| artifact | local path | status |
| --- | --- | --- |
| complete build | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/outcomes/mcif_beyond_ocr_exploratory_screen_v1_863e6af8_b2` | `PENDING_HF_UPLOAD` |
| inference-only bundle | `.../inference_bundle` | 68 rows；39 MB；可传 GPU worker |
| scorer-only mapping | `.../scorer_private` | 34 rows；不得传 GPU worker |

`SHA256SUMS` 已对 140 个文件全量验证。HF upload 等 inference/analysis 完成后作为一个完整、
可复用 private revision 执行；上传前 Git 只记录本地 staging 状态，不宣称已有远端 revision。

## 下一步

1. 从包含本 contract 的精确 Git commit 在 Hyper00 canonical container 启动 2-GPU、2-shard run；
2. 验证 272/272 unique rows、四条件、两种 acoustic conditions 和 immutable model revision；
3. 在本地 scorer 侧运行 analyzer，列出 clean/noisy positives；
4. 仅当存在 positive samples 时，才为这些样本构建正式人工 visual/OCR/target validation；
5. 把结果、失败模式与 reusable artifacts 同步到 Git 和 private Hugging Face immutable revision。
