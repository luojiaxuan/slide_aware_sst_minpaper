# Agent Start Here

更新日期：2026-07-31

本项目当前研究 **pre-talk semantic context for long-form SimulST**。唯一权威路线与
Phase-A contract 是
[`../DUAL_ROUTE_DECISION_20260731.md`](../DUAL_ROUTE_DECISION_20260731.md)。先完整
阅读该文件、[`../SOURCE_OF_TRUTH.md`](../SOURCE_OF_TRUTH.md) 和
[`../FINDINGS.md`](../FINDINGS.md)，不要从历史 Chinese-LiPS MVP 继续执行。

## 当前决策

- Route B1 conditional GO：talk 前从 paper/deck/slides 编译 typed、non-term context
  memory；streaming path 不运行 VLM，只做冻结的 causal lookup。
- Route A HOLD：在同一 pilot 中检验 image-specific visual relations 是否超过 matched
  slide OCR/layout propositions。只有 A-GO 通过才执行
  [`../ACL_PAPER_BLUEPRINT_20260731.md`](../ACL_PAPER_BLUEPRINT_20260731.md)。
- Route B0 NO-GO：term/entity extraction、abstract prompt、phrase boosting、whole-PDF
  prompt 和 BM25/RAG 都是 strong baselines，不是 contribution。
- Lip video、slide+lip hybrid、新的 Chinese-LiPS pseudo-reference 和新 multimodal
  training 均不在当前执行范围。

## 数据角色

- ACL60/60 的 5 个 dev talks：Phase-A futility screen。
- ACL60/60 eval：通过 gate 后的 replication。
- MCIF 21 talks：project-held-out confirmatory benchmark；系统冻结前禁止访问 outputs。
- Chinese-LiPS：private timing/ASR diagnostic only，不是 paper ranking 主数据。

## 当前实验矩阵

使用 dual-route contract 中的 `C0-C7`：

| ID | Condition |
| --- | --- |
| `C0` | audio-only |
| `C1` | term memory |
| `C2` | entities/abstract static prompt |
| `C3` | phrase boost + pretranslated PDF BM25/RAG |
| `C4` | PDF-derived proposition/discourse memory |
| `C5` | C4 + slide OCR/layout propositions |
| `C6` | C5 + image-specific visual relations |
| `C7` | matched same-domain wrong/shuffled/stale controls for C4-C6 |

Talk 前可以预计算整套 deck，但 future slide 不得提前可见：slide-derived entries 只在
真实 stable-slide timestamp 后解锁。C7 必须继承 correct condition 的可用时间、token
budget 和 selection path。

## 下一步

1. 冻结 ACL60/60 与 MCIF revisions、licenses、talk ids 和 hashes；
2. 接通 long-form SimulST runner，先复现 `C0-C3`；
3. 冻结 typed-memory schema，构建 `C4-C6` extraction QA；
4. blind 标注 200-300 个 term/entity-masked context-critical events；
5. 在 ACL60/60 dev 跑 native/+5 dB `C0-C7`，按 B1/A gates 作一次路线决策；
6. 只有通过的路线进入 MDE/power、frozen selector 和 one-shot MCIF run。

## 禁止捷径

- 不用 image-vs-none 或 aggregate BLEU 单独证明 vision；
- 不把 terminology gain 计入 Route B1 的 non-term primary metric；
- 不在看到 outputs 后定义 event slice 或修改 gate；
- 不在 MCIF 上调 prompt、threshold、selector 或 context schema；
- 不把预计算成本写成 zero latency。
