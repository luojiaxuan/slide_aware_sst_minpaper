# MCIF Beyond-OCR Exploratory Screen V1

日期：2026-08-02

状态：`SCREEN_COMPLETE_FORMAL_VALIDATION_PENDING`

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
| complete r5 run | `.../results/mcif_beyond_ocr_exploratory_v1_c098d9f_2gpu_r5_bs16_20260802_033300` | 272/272 rows；analysis complete |

完整 artifact 已上传到 private
[`gavinlaw/slide-aware-sst-mcif-outcomes@d6b01409`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/d6b01409200379bc22b1aecd485d5ebc4fe2b4d1/beyond_ocr_exploratory_screen_v1)，
tag `mcif-beyond-ocr-exploratory-screen-v1`。独立回下载 161 个实际 files，其中
`ARTIFACT_SHA256SUMS` 覆盖的 160 files 全部逐字节验证通过；repo visibility 确认为 private。

## 运行结果

正式结果只来自 Hyper00 r5：Git `c098d9f64b9d162991b3bcc1182cf78cc67bb2dd`、GPU
2/5、每卡一个 worker、两个 shards、`batch_items=16`。两个 shard 各 136 rows，合计
272/272 unique results；model revision 唯一且正确，两个 worker 均以 0 退出。完整 aggregate
summary 位于 `data/mcif_beyond_ocr_exploratory_screen_v1_summary.json`。

| acoustic | audio-only AUC | OCR AUC | raw-image AUC | wrong-image AUC | positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 22.589 | 27.140 | 26.701 | 24.258 | 9/34 |
| +5 dB babble | 7.749 | 11.180 | 11.563 | 6.081 | 17/34 |

按 16 talks 做 cluster bootstrap：

- clean `raw - OCR = -0.902`，95% CI `[-2.475, 0.324]`；
- noisy `raw - OCR = -0.173`，95% CI `[-4.619, 3.454]`；
- clean `raw - wrong = +2.141`，95% CI `[1.026, 3.281]`；
- noisy `raw - wrong = +4.747`，95% CI `[2.275, 7.433]`。

因此当前正确结论是：**raw image 相对 matched wrong image 有 content-sensitive signal，但 raw
image 整体没有打败 OCR。** 不能写成 `vision > OCR` aggregate result。样本级 gate 共得到
26 个 `candidate x acoustic` positives，覆盖 20/34 unique candidates；其中 6 条在 clean/noisy
都通过，优先进入正式人工验证：

| candidate | clean raw-OCR / raw-wrong | noisy raw-OCR / raw-wrong |
| --- | ---: | ---: |
| `Mila and Microsoft Research` | +8.708 / +11.352 | +30.872 / +36.761 |
| `bounding box` | +7.709 / +8.772 | +19.358 / +25.260 |
| `three speech bubbles` | +6.978 / +12.486 | +1.124 / +1.087 |
| `performance drop is` | +1.956 / +3.280 | +1.021 / +1.167 |
| `blue and orange lines` | +0.879 / +0.346 | +1.934 / +5.651 |
| `belief and action` | +5.240 / +5.318 | +0.075 / +13.658 |

最后四条 margin 较小，尤其 `belief and action` 的 noisy `raw-OCR=+0.075`，人工验证必须检查
是否真是 candidate-relevant advance，而不是一般生成波动。

## GPU 运行诊断

- 初次启动因 container 缺少 `accelerate` 在 0 rows 失败；补装 `accelerate==1.14.0` 后恢复；
- `batch_items=32` 使利用率降到约 40%，已停止并保留独立 FAILED run；
- 每卡两个 model workers 在 OCR 阶段把 H200 推到约 140.4 GiB，进入 image 条件前停止；
- 安全 r5 的两卡 10 秒 mean utilization 约 80--83%，低于 90% 目标；瓶颈是每个 prefix
  重复 multimodal preprocessing。失败 run 遗留的 processor children 已定向清理；r5 没有
  traceback/OOM/IPC error。

这个规模上继续重启会增加总 GPU 消耗，所以 r5 完成后停止扩卡。任何更大规模 run 前必须先
实现 image/audio prefix feature cache 或更深的 preprocessing queue，并重新量测吞吐和利用率。

## 下一步

1. 为 20 条 unique positives 构建正式人工 visual/OCR/target validation，优先完成 6 条
   clean/noisy robust positives；
2. 人工确认 pixels 包含正确证据、flat OCR 确实不足、raw-image advance 与 candidate 相关，
   并排除 hallucination 或一般 decoding variation；
3. 只有通过人工验证的样本才进入后续 audio-sufficiency/commit-time analysis；
4. human validation artifact 完成后，以新的 private immutable revision 上传，不改写本次
   screen revision。

其中前两步的 zero-label packet 已完成并冻结在 private HF revision
[`34e8f9b1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/34e8f9b16ef06dc9503066d3446304400c750c22/beyond_ocr_positive_validation_v1)，
tag `mcif-beyond-ocr-positive-validation-v1`。它含 2 个 visual roles 与 2 个 outcome roles，
每个 6 items；38 files 回下载、37 manifest-bound files 校验通过。当前仍是 0 human labels，
等待四个互不重合的真实 annotators。详见
[`MCIF_BEYOND_OCR_POSITIVE_VALIDATION_V1.md`](MCIF_BEYOND_OCR_POSITIVE_VALIDATION_V1.md)。
