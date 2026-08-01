# Dual-Route Decision：Semantic Vision 与 Pre-Talk Context for SimulST

更新日期：2026-07-31

状态：**当前权威的路线选择与 Phase-A 实验 contract。**

本文档取代“只能做 live-slide semantic evidence”的单一路线判断，但不废弃
[`ACL_PAPER_BLUEPRINT_20260731.md`](ACL_PAPER_BLUEPRINT_20260731.md) 中已经冻结的
Route A 细节。后者现在是 Route A 的专项 blueprint；本文件决定两条路线如何在同一
pilot 中比较、何时升级、何时停止。

## 1. 纠正与结论

用户的纠正成立：EGTA 和 RASST 都是 **terminology-only** 路线。

- RASST 从流式 speech/text state 检索 terminology hints，并学习是否、何时使用；
- EGTA 从 document terminology memory 选择 acronym、model、dataset、method、entity
  等短术语，并在 ASR/decoder 侧适配；
- 它们没有系统研究 proposition、discourse referent、relation、scope、slide layout
  或 image-specific semantic evidence。

因此 EGTA/RASST 只封住“再做一个 term retriever/gate”这条退路，并没有封住广义
context-aware SimulST，更没有封住 vision-aware context。

但 generic context 也不是空白。IWSLT 2026 已有系统在 talk 开始前处理 ACL paper：

- CUHKSZ 把 named entities 或 abstract 直接注入 prompt；
- MLLP-VRAIN 做 phrase boosting、paper pretranslation 和 BM25 translation memory；
- NeMo 从 title/author/abstract 或全文提取 entities 做 context biasing。

所以最终决策不是在“vision”和“任意 context”之间二选一，而是：

1. **Route B1：conditional GO，优先执行。** 研究超越术语、实体、摘要和 PDF-RAG 的
   pre-talk typed context memory；
2. **Route A：HOLD，嵌入共享 pilot。** 检验 image-specific relation 是否在同预算下
   超过 slide OCR/layout propositions；
3. **Route B0：NO-GO。** 单纯 whole-PDF prompt、abstract/entities、phrase boosting
   或 BM25 top-k 不构成新 paper；
4. 不把 A/B 写成两个并列贡献。先用共享 pilot 决定论文身份。

## 2. 两条路线的精确定义

### Route A：Semantic Live Vision SST

研究当前 live slide 是否提供 **slide text 无法等价表达的 image-specific semantic
evidence**，以及 once-per-slide evidence state 能否在不阻塞 audio path 的情况下被
因果读取。

核心对象：

- chart trend、series-to-value correspondence、increase/decrease relation；
- formula structure、symbol-role mapping、superscript/subscript；
- diagram topology、arrow direction、grouping、spatial correspondence；
- visual emphasis、highlight、color/shape legend；
- speech 中的 deictic reference 与当前 slide region 的绑定。

Route A 不是：

- image presence 与 audio-only 的简单比较；
- 把 OCR text 包装成 VLM 输出；
- 每个 audio chunk 重跑视觉 encoder；
- “vision encoder 很快，所以 vision 有用”的系统论证；
- lip video 或同步 phonetic vision。

Route A 的致命问题只有一个：在同模型、同 prompt、同 schema、同 token budget 下，
`image + OCR` 是否超过 `OCR/layout-only`。如果否，论文不得声称 pixels necessary。

### Route B0：Generic Pre-Talk Context

以下方案已被相邻工作实质覆盖，直接 NO-GO：

- 全文 PDF、abstract 或 named entities 直接放入 prompt；
- 仅做 term/entity extraction 与 biasing；
- 仅用 BM25/embedding 从全文检索相似句；
- 预翻译全文后做 translation-memory retrieval；
- 用一个新 gate 决定是否注入上述文本，但没有新的 context type 或科学问题。

它们应作为 strong baseline，而不是 contribution。

### Route B1：Beyond-Terminology Pre-Talk Multimodal Context

在 talk 开始前，把允许访问的 paper PDF、deck 和 slide/video 材料编译为一个冻结的
typed memory。SimulST 运行时不调用视觉 encoder、不更新 memory、不做 test-time
learning；只允许当前 causal audio hypothesis 触发廉价、冻结、可复现的 lookup。

**预计算不等于提前可见。** PDF/document propositions 在 talk 开始前可用；由第
`k` 张 slide 产生的 `SLIDE_TEXT_RELATION`/`VISUAL_RELATION` 即使已离线编码，也只能
在该 slide 稳定出现的时间 `a_k` 后解锁，并在 slide change 后转为 stale。整套 deck
从一开始全部可检索只能作为显式标注的 `deck-known-in-advance` ablation，不能混入
live-slide primary setting。

这里“不做 on-policy”具体表示：

- 不在 audio critical path 上运行 VLM；
- 不根据 test talk 的翻译 reward 更新 selector 或 memory；
- 不读取未来 audio、future commits 或 references；
- 可以使用当前 audio prefix 做 deterministic/frozen retrieval，否则无法知道哪段
  pre-talk context 与当前 speech 有关。

候选 memory types：

| Type | 内容 | 是否已有强先例 |
| --- | --- | --- |
| `TERM` | acronym、model、dataset、method、proper name、target rendering | RASST/EGTA 已覆盖，baseline |
| `ENTITY` | author、institution、location、paper entity | CUHKSZ/NeMo 已覆盖，baseline |
| `ABSTRACT` | 全局 topic/summary prompt | CUHKSZ 已覆盖，baseline |
| `DOC_PROPOSITION` | definition、claim、method relation、comparison、condition | 尚未被 terminology work 覆盖 |
| `DISCOURSE_REFERENT` | this/it/these 等候选 antecedent、section-local referent | prior-sentence context 有先例，pre-talk material setting 未充分覆盖 |
| `RELATION_SCOPE` | negation、increase/decrease、greater/less、cause/result、argument role | 不是 term list 可表达的单位 |
| `TALK_OUTLINE` | section order、topic transitions、expected concept progression | generic document context 的窄化、需实证 |
| `SLIDE_TEXT_RELATION` | OCR + layout 后恢复的标题、分组、表格/公式文本关系 | slide-derived structured context |
| `VISUAL_RELATION` | 只能从 pixels/regions 稳定恢复的 chart/diagram/emphasis relation | Route A 的关键增量 |

运行时 action space 固定为：

```text
{NONE, TERM, DOC_PROPOSITION, SLIDE_TEXT_RELATION, VISUAL_RELATION}
```

每次注入有统一 token budget、来源 id、时间可用性和 confidence。`NONE` 是正式动作，
避免 context 一定要用的错误前提。

## 3. 真正的研究问题

### Route B1 主问题

> Can non-terminological context compiled before a talk improve simultaneous
> speech translation beyond term memory, entity/abstract prompting, and PDF
> retrieval, without placing a multimodal model on the streaming path?

重点不是“更多 context 是否涨 BLEU”，而是：

1. 哪类 context 在哪类歧义上有用；
2. 是否让系统在相同 final quality 下更早作出正确 commit；
3. correct memory 是否显著优于 same-domain wrong/shuffled memory；
4. 增益是否在去除 term/entity spans 后仍存在；
5. 视觉材料究竟新增了哪类 PDF/text 无法提供的 context。

### Route A 主问题

> When is image-specific slide evidence necessary beyond schema- and
> token-matched slide text for causal SimulST?

Route A 只在 `VISUAL_RELATION` slice 上成立。若 pooled BLEU 上涨但 image-needed slice
无增量，不能靠 aggregate metric 宣称 vision necessity。

## 4. Collision Matrix

| 工作 | 已覆盖 | 对本项目的约束 |
| --- | --- | --- |
| RASST | streaming terminology retrieval 与使用决策 | `TERM` 必须是 baseline；term gain 不能算 B1 novelty |
| EGTA | evidence-grounded terminology memory、激活与 shuffled-memory controls | 只与 term-only fallback 构成 material collision，不封住 proposition/vision context |
| CUHKSZ IWSLT 2026 | named entities 与 full abstract static prompt | B0 baseline；B1 必须超过它且控制 token budget |
| MLLP-VRAIN IWSLT 2026 | PDF phrase boosting、pretranslation、BM25 top-k memory | PDF RAG 是 mandatory baseline；retrieval 本身不是 novelty |
| NeMo IWSLT 2026 | document-derived entity inventory 与 ASR biasing | entity recognition gain 必须单列，不可归入 non-term context |
| Context Helps (ACL 2021) | previous source/target sentences 改善 pronoun、homophone、segmentation | 证明 discourse context 有价值；B1 差异在 external pre-talk material、typed controls 和 long-form speech |
| Document-level SimulMT (MT Summit 2021) | neighboring-sentence document context + wait-k | generic document context 并非新概念；B1 必须落到可标注现象与外部材料 |
| Visual-context SiMT (EMNLP/WMT 2020) | caption image 帮助 anticipation、低延迟翻译 | “视觉帮助 anticipation”不是新 claim；需 scientific-talk speech、persistent context 和 beyond-text controls |
| Do Slides Help? | slide-conditioned offline ASR；真实 ACL slides | B1/A 必须是 long-form SimulST，不可复述 slide helps recognition |
| LECTRANS | academic lecture slide image/OCR + transcript-conditioned translation | generic multimodal lecture translation 被占据；必须保留 raw speech、causal commits 和 matched context controls |
| OmniFusion | synchronous image+speech SimulST | 不能声称 first visual SimulST；也不采用其高成本 synchronous fusion 作为主系统 |

**Novelty 结论：**

- Route B0：Level 2 / material collision，NO-GO；
- Route B1：Level 3 / partial overlap，conditional GO；
- Route A：Level 3 / partial overlap，只有 `VISUAL_RELATION > SLIDE_TEXT_RELATION`
  后才可升级；
- A+B 若只是功能堆叠，不会自动提高 novelty。

## 5. 共享 Phase-A Pilot

两条路线共用同一数据、runner、audio outputs、token budget 和 scoring pipeline，避免
先分别建设两个完整系统。

### 5.1 Conditions

| ID | Condition | 角色 |
| --- | --- | --- |
| `C0` | audio-only | 基础线 |
| `C1` | term memory | RASST/EGTA-style strong baseline |
| `C2` | entities/abstract static prompt | CUHKSZ-style baseline |
| `C3` | phrase boost + pretranslated PDF BM25/RAG | MLLP-style baseline |
| `C4` | PDF-derived proposition/discourse memory | Route B1 核心 |
| `C5` | C4 + slide OCR/layout propositions | slide text 增量 |
| `C6` | C5 + image-specific visual relations | Route A 增量 |
| `C7` | C4-C6 的 same-domain wrong、shuffled、stale controls | 因果内容控制 |

所有条件：

- talk 前完成 context compilation；
- online 时不运行 VLM；
- PDF memory 在 talk start 解锁；slide-derived memory 按真实 slide stable timestamp
  解锁，禁止未来页 leakage；
- context packet 使用相同最大 token budget；
- 检索器看相同 causal audio prefix；
- 模型、decoding、read/write policy、audio preprocessing 和 random seed 相同；
- 每条 evidence 保留 source、type、confidence、available time 和 selected time。
- C7 wrong/shuffled/stale controls 继承 correct evidence 的解锁时间和 packet budget，
  只改变内容或 temporal identity。

Phase A 可以预计算 C6 来优先检验“信息本身是否有增量”，但这不等于 Route A 已解决
live system cost。若 A-GO 通过，confirmatory run 必须回到 Route A blueprint，按真实
slide-change event 测量 once-per-slide extraction、evidence-ready lead/lag 和 audio-path
lookup；预计算结果只能作为 content upper bound。

### 5.2 数据与 noise

Phase A 仅使用 ACL60/60 的 5 个 dev talks，作为 futility screen：

- native audio 是 Route B1 的 primary pilot setting；
- `+5 dB` babble 是 Route A 的 primary pilot stress setting；
- `0 dB` 可作 secondary dose response；
- 不能用 5 个 talks 宣称 paper-level significance；
- 不能用极端 noise 中才出现的 gain 代替 native usefulness。

MCIF 官方 IWSLT translation subset 的 21 talks（当前 HF revision 的 100-talk media
pool 中的一部分）保持 project-held-out。在 route、representation、selector、prompt、model、
metric 和 gate 冻结前，不运行 MCIF outputs。ACL60/60 eval 用于最终 replication；若
IWSLT ACL-TALKS blind test 可访问，再作为独立验证。

## 6. Context-Critical Annotation

先在 5 个 dev talks 标注 200-300 个候选 event，每个 event 对应一个可判定的 target
span 或 commit decision。标注时不得查看系统输出。

互斥 primary labels：

- `term_or_entity`；
- `discourse_referent`；
- `lexical_sense`；
- `relation_or_scope`；
- `anticipation_or_reordering`；
- `visual_relation`；
- `no_context_needed`。

约束：

- 与 official term/entity/acronym list 重叠的 span 一律不计入 non-term primary score；
- `visual_relation` 必须由 annotator 判断 linear OCR 不足以恢复；
- 每个 event 记录最早可用的 external evidence、首次相关 speech time 和 gold target；
- 报告双人 annotation agreement、disagreement adjudication 和每 talk/type 数量；
- 不能在看到 C4/C6 的结果后重定义 slice。

## 7. Metrics

### Primary targeted metrics

- non-term context event 的 span/contrastive accuracy；
- first-correct-emission time 和 final-correct stability；
- matched final accuracy 下的 commit lead；
- wrong-context adoption；
- visible-but-unspoken hallucination；
- retraction/flicker 与 over-commit rate。

### Secondary aggregate metrics

- XCOMET、chrF、BLEU；
- term/entity recall，必须与 non-term score 分开；
- AL、LAAL、LongYAAL/computation-aware latency；
- on-path lookup latency、packet tokens、GPU seconds、RTF；
- cold pre-talk compilation cost 单独报告，不伪装为零成本。

## 8. GO / NO-GO Gates

这些是 dev futility gates，不是最终论文显著性判定。

### B1-GO

必须同时满足：

1. `C4` 或 `C5` 的 correct memory 明确优于其 C7 wrong/shuffled control；
2. `max(C4,C5) > max(C1,C2,C3)`，且比较只在 term/entity-masked 的 non-term events
   上进行，context token budget matched；
3. targeted accuracy 至少提高 5 个百分点，或在 final accuracy 不降时让首次正确
   commit 明显提前；
4. 方向在至少 3/5 talks 一致，不能由单 talk 或单类事件驱动；
5. wrong-context adoption、hallucination 和 final quality 没有不可接受的回退。

任何以下情况触发 B1 NO-GO：

- gain 只来自 terms/entities；
- C2/C3 已解释全部收益；
- correct 与 shuffled/wrong memory 等价；
- 只在 0 dB 或更差的合成噪声中出现；
- 只能增加 final metric，不能定位到预注册 context-critical events。

### A-GO

必须同时满足：

1. `C6 > C5` 于 pooled `visual_relation` events；
2. C5/C6 使用相同 extractor family、prompt、schema、decoding 和 output budget，唯一
   information difference 是 pixels；
3. correct visual relation 优于 stale/wrong/shuffled relation；
4. 增益分布在多个 talks，不由单张 chart 或单个 noise seed 驱动；
5. image-needed events 数量足够支撑后续 talk-cluster power analysis。

若 `C5 > C4` 但 `C6 = C5`，保留 slide-derived structured context，不声称 raw
vision。若 `C6` 只在极端 noise 下有效，Route A 不升级为主 paper。

## 9. Pilot 后的论文身份

| 结果 | 决策 |
| --- | --- |
| B1 通过，A 未通过 | 做 context-aware SimulST；标题与 abstract 不使用 vision-aware 主 claim |
| A 通过，B1 未通过 | 做 semantic live-vision paper；遵循 Route A blueprint 的完整 causal protocol |
| B1 与 A 都通过 | 做 multimodal pre-talk context paper；A 是 typed memory 的 image-specific source，不是第二个并列系统 |
| 只有 C1 通过 | RASST/EGTA replication，停止 |
| 只有 C2/C3 通过 | IWSLT context-system replication，停止 |
| 全部不通过 | 报告内部 negative result，停止主 paper 投资 |

当前先验排序：

| 维度 | Route A | Route B1 |
| --- | --- | --- |
| novelty ceiling | 高 | 中高 |
| 通过概率 | 低到中 | 中 |
| 数据依赖 | 真实 timeline + 足量 image-needed events | PDF/deck + context-critical labels |
| online latency 风险 | 低，若预计算成功 | 最低 |
| 最危险 baseline | matched slide text/OCR semantics | term/entity/abstract/PDF-RAG |
| 已有负面证据 | 206-segment correct-vs-same-talk-wrong 不显著 | EGTA 暗示多数 gains 可能由 terminology 解释 |
| 当前动作 | HOLD inside shared pilot | GO first |

## 10. 执行顺序

1. 冻结 ACL60/60 dev/eval 与 MCIF revisions、licenses、talk ids；
2. 接通 long-form SimulST runner，先复现 C0-C3；
3. 离线构建 typed memory schema，并对 C4-C6 做 extraction QA；
4. blind 标注 200-300 个 context-critical events；
5. 在 5 个 ACL dev talks 上运行 native 与 +5 dB 的 C0-C7；
6. 按 B1/A gates 作唯一一次路线决策；
7. 只有 route 通过后，做 MDE/power audit、冻结 selector 与 one-shot MCIF protocol；
8. MCIF 运行后按 talk-cluster inference 报告结果，再做 ACL eval replication。

现阶段不应：

- 训练新的 multimodal model；
- 扩充 Chinese-LiPS pseudo-reference；
- 把 lips 加回实验；
- 先在 MCIF 调 prompt/threshold；
- 只跑 image-vs-none；
- 以 aggregate BLEU 取代 typed event analysis。

## 11. Primary Sources

- Yang and Nakamura, 2026, *When to Use Extra Context: Evidence-Grounded
  Terminology Adaptation for Simultaneous Speech Translation*:
  <https://arxiv.org/abs/2607.17766>
- Luo et al., 2026, *RASST: Retrieval-Augmented Simultaneous Speech
  Translation*: <https://arxiv.org/abs/2601.22777>
- IWSLT 2026, *Speech-to-Text with Extra Context*:
  <https://iwslt.org/2026/simultaneous>
- CUHKSZ IWSLT 2026 system paper:
  <https://aclanthology.org/2026.iwslt-1.13/>
- MLLP-VRAIN IWSLT 2026 system paper:
  <https://aclanthology.org/2026.iwslt-1.24/>
- NeMo IWSLT 2026 system paper:
  <https://aclanthology.org/2026.iwslt-1.23/>
- Zhang et al., 2021, *Beyond Sentence-Level End-to-End Speech Translation:
  Context Helps*: <https://aclanthology.org/2021.acl-long.200/>
- Iranzo-Sánchez et al., 2021, *Studying The Impact Of Document-level Context
  On Simultaneous Neural Machine Translation*:
  <https://aclanthology.org/2021.mtsummit-research.17/>
- Caglayan et al., 2020, *Simultaneous Machine Translation with Visual
  Context*: <https://aclanthology.org/2020.emnlp-main.184/>
- Sinhamahapatra and Niehues, 2025, *Do Slides Help?*:
  <https://aclanthology.org/2025.emnlp-main.814/>
- Imankulova et al., 2020, *Towards Multimodal Simultaneous Neural Machine
  Translation*: <https://aclanthology.org/2020.wmt-1.70/>
- Koneru et al., 2025, *OmniFusion: Simultaneous Multilingual Multimodal
  Translations via Modular Fusion*: <https://arxiv.org/abs/2512.00234>
- *LECTRANS: A Multimodal Translation Benchmark for Academic Lectures*:
  <https://openreview.net/forum?id=b129e506359e5d129d72d135c11e28938b7e34d8>
- Bentivogli et al., 2025, *MCIF: Multimodal Crosslingual Instruction-Following
  Benchmark from Scientific Talks*: <https://arxiv.org/abs/2507.19634>
