# Agent Start Here

更新日期：2026-08-01

本项目当前研究 **persistent, pre-available slide semantics for SimulST**。最终 paper
identity 尚未冻结；当前权威的探索策略与 held-out freeze 边界是
[`../PAPER_STORY_DECISION_20260731.md`](../PAPER_STORY_DECISION_20260731.md)；
[`../DUAL_ROUTE_DECISION_20260731.md`](../DUAL_ROUTE_DECISION_20260731.md) 只保留
`C0-C7` control contract。先完整阅读这些文件、[`../SOURCE_OF_TRUTH.md`](../SOURCE_OF_TRUTH.md) 和
[`../FINDINGS.md`](../FINDINGS.md)，不要从历史 Chinese-LiPS MVP 继续执行。

## 当前决策

- ACL dev 同时探索四条 paper route：current-slide anticipation、noisy-speech
  robustness、pixels beyond OCR、evidence selection/integration。
- Correct current slide 与 time/type/budget-matched same-talk stale/wrong slide 是所有
  route 共用的 content-use control，不是提前冻结的唯一 primary estimand。
- 完整记录开发矩阵；根据 effect size、跨 talk 稳定性、failure analysis 与方法价值选择
  story。选择后必须先 commit/push frozen confirmatory contract，再看 ACL eval/MCIF。
- 不能用低 power pixel null 宣称 OCR sufficient。
- term/entity extraction、abstract prompt、phrase boosting、whole-PDF prompt 和
  BM25/RAG 都是 strong baselines，不是 contribution。
- Lip video、slide+lip hybrid、新的 Chinese-LiPS pseudo-reference 和新 multimodal
  training 均不在当前执行范围。

## 数据角色

- ACL60/60 的 5 个 dev talks：Phase-A multi-route development screen。
- ACL60/60 eval：通过 gate 后的 replication。
- *Do Slides Help?* Figshare v2 supplement 已验证覆盖 ACL60/60 全部 10 talks、884
  个真实 video frames；原 metadata 含 transcript，必须先构建 frame-only inference view。
- MCIF 当前 HF revision 有 100 个 long-media talks；其中官方 IWSLT translation
  subset 的 21 talks 是 project-held-out long-form source。21 videos 与 causal-state
  input QA 已通过：283 transitions、304 states；eligible-event/MDE/output gates 尚未
  通过。系统冻结前禁止读取 references 或运行 outputs。
- Chinese-LiPS：private timing/ASR diagnostic only，不是 paper ranking 主数据。

## 当前实验矩阵

使用 dual-route contract 中的 `C0-C7`：

| ID | Condition |
| --- | --- |
| `C0` | audio-only |
| `C1` | term memory |
| `C2` | entities/abstract static prompt |
| `C3` | phrase boost + pretranslated PDF BM25/RAG |
| `C4` | non-term document propositions/discourse |
| `C5` | frozen C3 + current-slide OCR/layout propositions |
| `C6` | C5 + image-specific visual relations |
| `C7` | matched same-domain wrong/shuffled/stale controls for C4-C6 |

Talk 前可以预计算整套 deck，但 future slide 不得提前可见：slide-derived entries 只在
真实 stable-slide timestamp 后解锁。C7 必须继承 correct condition 的可用时间、token
budget 和 selection path。

## 下一步

1. MCIF input-side visual readiness 已完成；先读
   [`../MCIF_VISUAL_READINESS_20260801.md`](../MCIF_VISUAL_READINESS_20260801.md)；
2. 数据与 runner revisions 已冻结；再读
   [`../PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](../PHASE_A_DATA_RUNNER_FREEZE_20260731.md)；
3. Controlled acoustic v1 已完成；读
   [`../CONTROLLED_ACOUSTIC_PIPELINE_20260801.md`](../CONTROLLED_ACOUSTIC_PIPELINE_20260801.md)；
4. ACL dev 468-state frame-only timeline 已完成；读
   [`../ACL6060_VISUAL_TIMELINE_20260801.md`](../ACL6060_VISUAL_TIMELINE_20260801.md)；
5. 100-row source event seed 已冻结；按
   [`../ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](../ACL6060_SOURCE_EVENT_ANNOTATION_V1.md)
   完成双标注，估计 term、semantic、relation、noise interaction 的 density；
6. 独立冻结 candidates、source-only packets、target scoring，运行 document-only、OCR、
   correct semantic/relation oracle、matched wrong oracle，以及 native/noisy audio；
7. 为有 practical headroom 的 route 构建小规模 automatic conditions，比较 naive prompt、
   selection/gating 和 direct image；所有 gold route 都无 headroom 才停止；
8. 从开发证据选择最终 story，冻结 primary claim/metric/config 后再运行 ACL eval 与
   21-talk MCIF。

## 禁止捷径

- 不用 image-vs-none 或 aggregate BLEU 单独证明 vision；
- 不把 terminology gain 计入 Route B1 的 non-term primary metric；
- 不隐藏开发期失败条件，不在同一 held-out 数据上选 story 又声称 confirmatory；
- 不在 MCIF 上调 prompt、threshold、selector 或 context schema；
- 不把预计算成本写成 zero latency。
