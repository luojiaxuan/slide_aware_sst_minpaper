# Agent Start Here

更新日期：2026-07-31

本项目当前研究 **current-slide content attribution after strong document context**。
唯一权威的 paper identity 与执行优先级是
[`../PAPER_STORY_DECISION_20260731.md`](../PAPER_STORY_DECISION_20260731.md)；
[`../DUAL_ROUTE_DECISION_20260731.md`](../DUAL_ROUTE_DECISION_20260731.md) 只保留
`C0-C7` control contract。先完整阅读这些文件、[`../SOURCE_OF_TRUTH.md`](../SOURCE_OF_TRUTH.md) 和
[`../FINDINGS.md`](../FINDINGS.md)，不要从历史 Chinese-LiPS MVP 继续执行。

## 当前决策

- Primary treatment/control 都接收同一 frozen strong document packet，只比较 correct
  current slide 与 time/type/budget-matched same-talk stale/wrong slide。
- 唯一 primary estimand 是 audio 消歧前 stable correct decision 的 talk-weighted risk
  difference；SESOI +5 pp，final correctness non-inferiority margin -1 pp。
- C1-C4 baseline variants 与 C6 pixels-beyond-OCR 都是 gated secondary；不能替代
  primary failure，也不能用低 power pixel null 宣称 OCR sufficient。
- term/entity extraction、abstract prompt、phrase boosting、whole-PDF prompt 和
  BM25/RAG 都是 strong baselines，不是 contribution。
- Lip video、slide+lip hybrid、新的 Chinese-LiPS pseudo-reference 和新 multimodal
  training 均不在当前执行范围。

## 数据角色

- ACL60/60 的 5 个 dev talks：Phase-A futility screen。
- ACL60/60 eval：通过 gate 后的 replication。
- *Do Slides Help?* Figshare v2 supplement 已验证覆盖 ACL60/60 全部 10 talks、884
  个真实 video frames；原 metadata 含 transcript，必须先构建 frame-only inference view。
- MCIF 当前 HF revision 有 100 个 long-media talks；其中官方 IWSLT translation
  subset 的 21 talks 是 project-held-out long-form source。其 visual tier 只有在 21
  个 videos 与 causal slide timelines 通过 QA 后才是 confirmatory benchmark；系统
  冻结前禁止读取 references 或运行 outputs。
- Chinese-LiPS：private timing/ASR diagnostic only，不是 paper ranking 主数据。

## 当前实验矩阵

使用 dual-route contract 中的 `C0-C7`：

| ID | Condition |
| --- | --- |
| `C0` | audio-only |
| `C1` | term memory |
| `C2` | entities/abstract static prompt |
| `C3` | phrase boost + pretranslated PDF BM25/RAG |
| `C4` | non-term document propositions/discourse（secondary） |
| `C5` | frozen C3 + current-slide OCR/layout propositions（primary） |
| `C6` | C5 + image-specific visual relations（secondary） |
| `C7` | matched same-domain wrong/shuffled/stale controls for C4-C6 |

Talk 前可以预计算整套 deck，但 future slide 不得提前可见：slide-derived entries 只在
真实 stable-slide timestamp 后解锁。C7 必须继承 correct condition 的可用时间、token
budget 和 selection path。

## 下一步

1. 数据与 runner revisions 已冻结；先读
   [`../PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](../PHASE_A_DATA_RUNNER_FREEZE_20260731.md)；
2. 导入 Figshare frames，生成不含 `sentence` 的 frame-only inference manifest，并用
   保守 timestamp 构建 current/stale/wrong state；
3. blind 标注 80-120 个候选 opportunity events，先估计各层 evidence density；
4. 独立冻结 candidates、source-only packets、target scoring，只跑 document-only、
   correct oracle、matched wrong oracle。Oracle 不通过就停止自动 C3-C6 与 GPU 大跑；
5. Oracle 通过后才复现 `C0-C3`、构建 primary C5 与 gated C4/C6 conditions；
6. ACL eval 与 21-talk MCIF 只在自动 ladder、MDE 和 evaluator 冻结后运行。

## 禁止捷径

- 不用 image-vs-none 或 aggregate BLEU 单独证明 vision；
- 不把 terminology gain 计入 Route B1 的 non-term primary metric；
- 不在看到 outputs 后定义 event slice 或修改 gate；
- 不在 MCIF 上调 prompt、threshold、selector 或 context schema；
- 不把预计算成本写成 zero latency。
