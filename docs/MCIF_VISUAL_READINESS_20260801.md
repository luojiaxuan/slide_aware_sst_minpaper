# MCIF Visual Readiness 与 Paper Go 条件

日期：2026-08-01

状态：**21-talk input-side visual readiness 通过；translation benefit 未测试。**

## 结论

这个 paper 空间站得住，但 novelty 不能只写成“给 SimulST 加 vision”。真正属于
SimulST 的机制是：slide evidence 在 source speech 足以消歧之前已经出现并持续存在，
因此系统可以让正确 target decision 更早稳定提交，而且 slide 编译不必进入每个 audio
chunk 的 critical path。离线 ST/ASR 没有“把正确 target text 提前到哪个时刻”的同一问题。

MCIF 现在已经证明数据形态适合检验该机制：21 个 talk 全部是以真实 slide 为主体的会议
录屏，speaker 只是小窗；slide 含术语、图表、公式、layout relation 和渐进动画，并持续
多个 speech chunks。它不是 *Do Slides Help?* 那类由 text terms 合成 PDF image 的设定。

但目前没有任何 MCIF translation output。视觉数据可用不等于 paper 已成立。

## 冻结数据与产物

| 项目 | 结果 |
| --- | --- |
| HF source | `FBK-MT/MCIF@e24065b919758263cfe5d157057278affe76ea7b` |
| IWSLT archive | SHA256 `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7` |
| talks / segments | 21 / 919 |
| audio / video | 7,105.53 s / 7,110.39 s |
| 10 s coverage audit | 711 frames；21/21 contact sheets inspected |
| 1 s state pass | 7,111 frames；283 transition candidates |
| causal states | 304，含 21 个 initial states |
| stable confirmation | 283/283；每个新 state 至少等待两个稳定 pair |
| transition QA | 21/21 sheets inspected；未观察到 speaker-only false trigger |
| references | 未解压、未读取；inference tree 无 `ref`/`transcript`/`translation` 目录 |

Git manifests：

- [`../data/manifests/mcif_translation_subset_materialized_20260801.json`](../data/manifests/mcif_translation_subset_materialized_20260801.json)
- [`../data/manifests/mcif_visual_coverage_10s_20260801.json`](../data/manifests/mcif_visual_coverage_10s_20260801.json)
- [`../data/manifests/mcif_visual_state_candidates_v2_20260801.json`](../data/manifests/mcif_visual_state_candidates_v2_20260801.json)
- [`../data/manifests/mcif_visual_qa_20260801.json`](../data/manifests/mcif_visual_qa_20260801.json)

本地持久化 staging：
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9`。
本地 media/frames 是可重建 cache；Git 中的 scripts、参数、hashes 与 QA record 是当前
轻量 source of truth。

## Reproduction

以下命令从 frozen manifests 重建当前产物。`MCIF_ROOT` 只指向持久化本地 cache，不能指向
Git worktree；命令不会解压或读取 IWSLT archive 中的 references。

```bash
cd code
MCIF_ROOT=/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9

PYTHONPATH=. .venv/bin/python scripts/materialize_mcif_subset.py \
  --sources-manifest ../data/manifests/phase_a_sources_20260731.json \
  --files-manifest ../data/manifests/mcif_files_e24065b9.jsonl \
  --iwslt-archive /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/iwslt2026/mcif-long-trans.zip \
  --output-root "$MCIF_ROOT" \
  --portable-manifest-out ../data/manifests/mcif_translation_subset_materialized_20260801.json \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9 \
  --workers 4

PYTHONPATH=. .venv/bin/python scripts/audit_mcif_visual_coverage.py \
  --inference-manifest "$MCIF_ROOT/manifests/inference.jsonl" \
  --output-root "$MCIF_ROOT/qa/coverage_10s" \
  --portable-summary-out ../data/manifests/mcif_visual_coverage_10s_20260801.json \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/qa/coverage_10s \
  --interval-sec 10 \
  --workers 4

PYTHONPATH=. .venv/bin/python scripts/detect_mcif_visual_state_candidates.py \
  --inference-manifest "$MCIF_ROOT/manifests/inference.jsonl" \
  --output-root "$MCIF_ROOT/qa/state_candidates_v2" \
  --portable-summary-out ../data/manifests/mcif_visual_state_candidates_v2_20260801.json \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/qa/state_candidates_v2 \
  --interval-sec 1 \
  --frame-width 320 \
  --p75-threshold 0.03 \
  --changed-patch-fraction-threshold 0.12 \
  --debounce-samples 2 \
  --stable-pairs 2 \
  --stability-p75-threshold 0.02 \
  --max-confirmation-samples 6 \
  --workers 4
```

## Causal state contract

`detect_mcif_visual_state_candidates.py` 在 1 s、320 px 灰度帧上使用 6x8 patch-grid
difference。候选条件是 `patch p75 >= 0.03` 或至少 12% patches 的 MAE `>= 0.05`；相邻
两秒内的变化合并。新 state 只有在后续两个 non-candidate pairs 的 `patch p75 < 0.02`
后解锁。

每个 causal state 记录：

- 半开可用区间 `[availability_start_sec, availability_end_sec)`；
- stable evidence frame 的 local path 与 SHA256；
- transition window；
- 严格单调的 state id。

该算法能忽略只占少数 patches 的 speaker 小窗运动，也能保留同模板正文替换和重要渐进
动画。它仍可能漏掉很小的局部更新，因此产物叫 candidate timeline，不叫 ground-truth
slide boundary。需要精确 timing 的 event 若落在 transition window 内，必须排除 primary。

## Reference-free Qwen3-VL source screen

已冻结覆盖全部 304 个 causal states 的 private VLM screen input，而不是根据像素或 OCR
挑选一部分 states。输入 SHA256 是
`62fa1fb540ae279ddafcb2b8449a8cde355ea6fe828dd7ec6534d04c6d605073`；frame binding
set SHA256 是
`a2399ad294035557803f3a75d1ebe65613da99b89a5f36597124fd0c83bf4ad9`。Git manifest：
[`../data/manifests/mcif_visual_context_screen_input_v1_20260801.json`](../data/manifests/mcif_visual_context_screen_input_v1_20260801.json)。

构建命令：

```bash
cd code
MCIF_ROOT=/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9
PYTHONPATH=. .venv/bin/python scripts/build_mcif_visual_context_screen.py \
  --causal-states "$MCIF_ROOT/qa/state_candidates_v2/causal_states.jsonl" \
  --source-manifest ../data/manifests/mcif_translation_subset_materialized_20260801.json \
  --state-root "$MCIF_ROOT/qa/state_candidates_v2" \
  --output "$MCIF_ROOT/prescreen/qwen3_vl_source_screen_v1/input.jsonl" \
  --summary-out ../data/manifests/mcif_visual_context_screen_input_v1_20260801.json \
  --portable-output-label ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/prescreen/qwen3_vl_source_screen_v1/input.jsonl
```

Prompt 是
[`../code/configs/mcif_qwen3_vl_source_screen_v1.txt`](../code/configs/mcif_qwen3_vl_source_screen_v1.txt)，
明确分开 visible text 与 flat OCR 会丢失的 chart/table/formula/layout/emphasis relation。
该 pass 只做 source-side feasibility triage：不读取 transcript/reference/translation，不得删
state，不得给 human author 看 suggested answer，也不得作为 `image_needed` 标签或 paper
结果。它不是 blueprint 中受 oracle gate 约束的 automatic integration/compiler。

## 什么结果才足以支撑 paper

仅有小幅 aggregate BLEU/COMET 提升不够。至少需要以下证据链：

1. **Temporal value：** 在 source audio 仍不足以完成 locked forced choice 时，correct
   current state 提高 first-stable correct target decision rate，并带来正的 target commit
   advance；更早但错误或随后撤回的 token 不计。
2. **Content specificity：** correct state 优于 time/type/token-budget-matched stale/wrong
   state，而不是 correct 与 wrong 都同幅提升。当前 Chinese-LiPS diagnostic 没通过这一点。
3. **Final safety：** final event correctness 和整体 translation quality 不退化；记录
   revision、hallucination 与 wrong-evidence adoption。
4. **Acoustic interaction：** native 单列；在 full-talk controlled noise 下，correct-state
   benefit 随 SNR 下降形成可解释 dose-response，wrong-state 不同步扩大。
5. **Strong baselines：** audio-only、whole PDF/RAG、term/entity memory、token-budget-matched
   OCR、direct image、empty 和 stale/wrong 必须同表。若 pixels 不胜 OCR，paper 可以转向
   contextual integration，但不能声称 raw vision 必要。
6. **Breadth：** 效果需跨 talks 稳定，并至少在第二个 system family 或语言方向保持方向
   一致。单模型、单 slice 的偶然 gain 只能写限定结论。

开发期继续使用现有 practical signal：early correct decision 至少约 +5 pp、final
correctness 不低于 -1 pp；它们是资源投入门槛，不是尚未注册的显著性结论。

## Noise source correction

- MUSAN 是 OpenSLR `SLR17`，CC BY 4.0，用于 babble/music/noise。
- Room Impulse Response and Noise Database 是 OpenSLR `SLR28`，Apache 2.0，用于 real/
  simulated RIR 与额外 noise。
- 旧文档中的 `SLR119` 是 AliMeeting（CC BY-SA 4.0），不是 AcousticRooms/RIR 数据。

所有 corruption 必须对完整连续 talk 施加，并记录 source id、seed、VAD/RMS 定义、目标
与 achieved SNR；不能把 noise seeds 当作独立 talks。

## 下一步

1. SLR17/SLR28 freeze 与 full-talk noise/RIR materialization 已完成，见
   [`CONTROLLED_ACOUSTIC_PIPELINE_20260801.md`](CONTROLLED_ACOUSTIC_PIPELINE_20260801.md)。
2. ACL dev 468-state frame timeline 已完成，见
   [`ACL6060_VISUAL_TIMELINE_20260801.md`](ACL6060_VISUAL_TIMELINE_20260801.md)；100-row
   balanced seed 也已冻结，下一步完成独立双标注。
3. MCIF 304-state reference-free VLM screen input 已冻结；完成私有 source-only prescreen、
   QA 和 private HF upload，但不得用其输出修改 human inventory。
4. 冻结 candidate/source-packet/target-scoring 三件套，先跑 oracle headroom：document、
   OCR、correct semantic/relation、matched wrong，覆盖 native 与 noise。
5. 只有看到 content-specific early-commit 或稳定 robustness signal 后，才投入 automatic
   VLM compiler、selection/gating 与 GPU inference。
