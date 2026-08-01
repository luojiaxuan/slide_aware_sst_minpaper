# Pre-Audio Slide Evidence Collision Audit

日期：2026-08-01

状态：**基础命题是 Level 2 high overlap；项目可以继续，但 paper 必须由 event-level
causal measurement 和 solid result 支撑，不能只做普通 slide-aware SimulST。**

## 审计结论

当前窄问题是：slide 在支持该内容的 speech 之前已经可见，能否让 SimulST 更早产生正确且
稳定的 target；当 acoustic evidence 退化时，这种作用是否增强。

这个问题有明确的 streaming 意义，但宽泛 novelty 已被占据：

- [Caglayan et al., 2020](https://aclanthology.org/2020.emnlp-main.184/)
  已经用预先可见的 image 补偿尚未读取的 source context，并在 text SiMT 中展示 target
  anticipation；
- [Ive et al., 2021](https://aclanthology.org/2021.eacl-main.281/)
  已经让 image 影响 READ/WRITE policy；
- [OmniFusion](https://arxiv.org/abs/2512.00234) 已经在 MCIF scientific-talk
  speech + aligned slide 上运行 chunked SimulST，并把 image 条件下的低 latency 归因于
  earlier and more stable commitments；
- [Maergner et al., 2011](https://aclanthology.org/2011.iwslt-papers.4/)
  更早已经使用 lecture 前可获得的 slides/materials 做 simultaneous lecture translation
  的 vocabulary/language-model adaptation。

因此，不能声称 first visual SimulST、first slide-aware SimulST、first visual anticipation，
也不能把“slide 在 speech 前出现”单独作为 technical novelty。OmniFusion 与当前原始命题在
`problem + key insight + application` 三个 native axes 上重合；原始命题没有指定新机制，
不能把 ambiguity 当作 mechanism difference。审计结论是 **Level 2 - High Overlap**。

## 仍然存在的 paper 空间

OmniFusion 只给 aggregate quality-latency curves。它没有回答以下因果问题：

1. current correct slide 是否在 audio 尚不足以判断时，提前了 `first stable correct target`；
2. 该提前是否显著强于 time/type/token-budget matched wrong/stale slide 和 empty slot；
3. OCR-only 是否已经解释全部收益，raw image 是否在 chart/layout/formula/visual emphasis
   事件上提供额外信息；
4. correct-over-wrong 的 content-specific effect 是否随 SNR 下降而增加；
5. 提前是否在 final quality、hallucination 和 computation-aware latency 上付出不可接受代价。

项目的可发表 delta 不是“vision improves SimulST”，而是：

> 在冻结的 source-side events 上，测量 causally available correct slide 是否在 supporting
> speech 充分之前推进稳定且正确的 target decision，并用 wrong/stale/empty/OCR controls
> 和 controlled acoustic corruption 识别这一推进来自什么内容、在何时出现。

这仍是 planned delta，不是现有结果。若只得到 generic image-slot、普通 BLEU/COMET 或 AL
变化，paper 会是 incremental；若得到稳定的 content-attributable commit lead、跨 talk 一致性
和清晰的 noise interaction，则可以形成 solid measurement/causal paper，后续再由 dev 结果
决定是否需要 selector/gating 或其他 integration method。

## Closest-work boundary

| Work | 已占据部分 | 未覆盖的关键部分 |
| --- | --- | --- |
| OmniFusion | scientific-talk slide + SimulST；with/without image；earlier/stable commitment 描述 | event correctness、audio-sufficiency boundary、wrong/stale/empty、inference-time OCR control、acoustic-noise interaction |
| Caglayan et al. 2020 | pre-observed image 补偿 missing source；low-latency anticipation | speech、slides、continuous audio、noise、causal event metric |
| Ive et al. 2021 | image-conditioned READ/WRITE | speech、scientific slides、event-level correctness |
| Context Helps 2021 | ST context、random context、BLEU/DAL/flicker | external pre-audio slide；其 robustness 是 segmentation error，不是 acoustic SNR |
| Do Slides Help? 2025 | ACL60/60 real midpoint frames；image 与 OCR/VLM-extracted terms；slide ASR | translation、online timing、controlled noise；大规模 MuST-C training slides 是 transcript→Llama 3 LaTeX→PDF/image synthetic augmentation |
| BOOM 2026 | live lecture + current slide screenshot + OmniFusion | live event-level slide benefit；paper 自己说明 full-talk downstream evaluation 不模拟 live use |

## 实验判据

现有 [`ACL6060_EVENT_TRAJECTORY_SCORING_V1.md`](ACL6060_EVENT_TRAJECTORY_SCORING_V1.md)
是正确的下一步，不应退回 aggregate metric：

- primary outcome：talk-equal、audio-insufficient boundary 前的 first-stable-correct risk
  difference；
- content attribution：`correct - matched wrong/stale`，不是 `image - audio-only`；
- noise estimand：同一 content contrast 在 native/noisy 条件间的 difference-in-differences，
  三个固定 seed 在 event/talk 内平均，不能扩充样本数；
- safety：final correctness、forbidden realization adoption、overcommit、hallucination、
  computation-aware AL/LAAL；
- representation：audio/document-only、naive OCR、raw image/semantic relation、matched wrong、
  empty slot 同时保留。OCR 获胜仍是有效 contextual SimulST 出口，不能事先强迫 raw vision
  获胜；
- selection：ACL dev 探索完整 matrix，选定 story 后先在 Git 冻结 claim/metric/SESOI/config，
  再读取 ACL eval 或 MCIF outcome。

现有探索 gate `+5 pp early / -1 pp final / 3-of-5 talks` 只作为 development
point-estimate screen，不得写成正式 non-inferiority test。

## 执行决策

继续当前 source-event annotation 与 causal inference 路线。当前最高优先级不是再跑 GPU，
而是完成 100 个 blinded source questions 和互不重叠的 audio/frame validation cohorts，得到
真实 eligible-event density 与 audio-sufficiency boundaries。没有 human freeze 前，不生成或
解读 paper-grade ACL system outputs。

## 审计 provenance 与限制

- locked idea 原文：

  > 可以就用这个, 我觉得paper空间是很明确的, 因为vision as slide可以提前语音出现,
  > 从而使得target text更早出现, 这个是流式独有的, 离线ST/ASR都不存在这个特点,
  > 然后我们可以加噪音看什么时候vision会逐渐起到更多的作用, 你觉得这个paper站得住吗?
  > 唯一的问题在于我们需要真的做到很solid的指标提升, 不然会被说incremental.
  > 继续下一步吧

- immutable idea lock SHA256：
  `4c176a7bd3c41dd5846356a74fb020db945e5eb19db00fb449c4220a13f7f66f`；
- 审计使用已保存的 full primary PDFs/text，并增量检索 2018--2026 official paper records；
- OpenAlex 返回 HTTP 504，Semantic Scholar/DBLP rate-limited，OpenReview Python connector
  不可用；这些是 coverage limitations，不作为“没有相关工作”的证据；
- verdict-driving papers 均由 ACL Anthology/arXiv/OpenReview official primary source 核实。
