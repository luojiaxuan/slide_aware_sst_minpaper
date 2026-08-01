# MCIF Beyond-OCR Reliability Audit

日期：2026-08-01

审计对象：Git `main@50185d2` 的
[`MCIF_BEYOND_OCR_VALIDATION_V1.md`](MCIF_BEYOND_OCR_VALIDATION_V1.md)、config、workspace、
validator/freezer/join、UI 与 tests。

结论：**v1 在 0 labels 时 superseded。它可以保留为 role-firewall、hash、UI 和内部
calibration artifact，但不能产生 paper gold event，不能支持 `pixels > OCR`，也不能进入 audio
sufficiency 或 inference。43872/43873 服务已停止；两个 v1 working sheets 保持 0/152。**

## 为什么是 No-Go

### 1. Dual-role 不是重复标注

一名 visual validator 与一名 target author 回答不同问题。两者 joint pass 只能表示两个不同
gate 各被一个人接受，不能给出任何 role 内 agreement、个体偏差或 label reliability。

### 2. OCR 判断被 pixels/VLM 污染

v1 visual 页面同时展示 slide、R0、R1、candidate 和 proposed evidence。Annotator 在看过
pixels 与 VLM proposal 后再判断 R0/R1 是否足够，会把已知视觉答案带回 text-only
counterfactual，并产生 confirmation bias。`pixels > OCR` 的关键对比因此不能由该 instrument
识别。

### 3. Disagreement 被错误折叠为 rejection

v1 没有 A/B labels、agreement report 或 append-only adjudication。`uncertain` 与确定的 `no`
都会直接成为 rejected candidate；single-role errors 不可观察，也没有 unresolved/missing 状态。

### 4. Target author 自定义又自批准 gold

同一人同时决定 eligibility、canonical English event、acceptable/forbidden Chinese
realizations 和 alignment。Schema 只能检查非空与枚举合法，不能证明双语语义正确性或
realization completeness。

### 5. 缺少 instrument-level reliability gate

152 proposals 来自 21 talks、120 segments、91 states；同一 talk、segment 和 state 内相关，
不能当 152 个独立样本。R1 只有 2 items，只能作为案例。v1 report 没有 primitive confusion
matrix、chance-corrected agreement、talk-cluster interval 或 adjudication rate。

## V2 最小合同

1. 两名 disjoint visual validators 完整覆盖 152 items；不使用只覆盖 20% 的 overlap sample。
2. 每名 visual validator 必须按以下顺序 append-only lock：
   `candidate + R0` → `+R1` → `pixels` → `VLM/descriptor fidelity`。后续 stage 不能修改前一
   stage，也不能提前释放。
3. R0/R1 sufficiency 在看 pixels/VLM 前冻结；pixel support 在看 descriptor 前冻结。
4. Target 由一名 author 定义 event/realizations，再由 disjoint bilingual validator 先独立判断
   eligibility/alignment，锁定后才可 accept/edit/reject author scoring text。
5. 任一 primitive disagreement、任一 `uncertain`、target edit/reject 都进入 role-specific
   append-only adjudication。Raw labels 不覆盖；未解决项保持 missing，不能计为 negative。
6. Author、visual A/B、target validator 与 adjudicator ids 必须代码级 disjoint。
7. Pre-label instrument gate 冻结为：每个 load-bearing categorical field exact agreement
   `>=0.80`、Gwet AC1 `>=0.67`、adjudication rate `<=0.25`。这是本项目的 go/no-go，不包装成
   通用领域阈值；未通过时修 guideline 并全部重标，不能靠 adjudication 清洗。

## 必须报告

- 每个 primitive field 的 3×3 confusion matrix、annotator marginals 与 exact agreement；
- category-specific positive/negative agreement；
- Gwet AC1 primary、Cohen's kappa secondary，均给 talk-cluster bootstrap 95% CI；
- pre-adjudication composite-gate agreement；
- target scoring text unchanged/edited/rejected 比例；
- adjudication rate、原因与 positive/negative/unresolvable 数量；
- raw 与 adjudicated yield，以及按 tier/talk/segment/state 的分布；
- paper estimand 按 talk 等权，R1 `n=2` 不形成一般性 claim。

## 与论文 claim 的边界

V2 通过后也只证明存在可靠的 beyond-OCR candidate events。论文仍必须在 matched system
outputs 上证明 raw image 优于 layout/structure-preserving text、correct evidence 优于
stale/wrong evidence，并在 talk-level first-stable-correct timing 与 controlled noise interaction
上达到预先冻结的 effect gate。Aggregate BLEU 或 adjudicated event count 不能替代这些结果。
