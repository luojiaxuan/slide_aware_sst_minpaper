# Research Goal：异步、持久的 Slide Semantic Evidence for SimulST

更新日期：2026-07-31

## 一句话目标

研究 **once-per-slide semantic evidence** 能否在不阻塞音频流的前提下改善
simultaneous speech translation（SimulST）：slide 出现或变化时只编码一次，把
可审计的语义证据缓存为持续状态，在该 slide 的驻留窗口内由流式语音按需、因果
地读取；重点检验 noisy、accented、jargon/entity-dense speech，以及 raw/structured
visual evidence 何时提供 strong OCR 无法提供的信息。

## Scope lock

1. **主线只有 semantic slide/screen evidence。** 目标信息包括文字、版式、公式、
   图表关系、视觉强调和跨区域对应，不是说话人的嘴形。
2. **lip video 退出主实验。** Lip 是 phonetic evidence，需要帧级同步、持续编码和
   完全不同的数据/控制组；offline robust AVST 已较成熟，而本项目的优势恰好是
   slide 可以提前出现并持续较长时间。
3. **不做 slide+lip hybrid。** Hybrid 会扩大数据、计算和归因成本，却不帮助回答
   当前核心问题。AVMuST-TED 只留作 related-work 参考，不再是数据候选或 blocker。
4. **image 是提前到达的 context/evidence state，不是每个 audio chunk 都重编码的
   同步输入。** 当前 slide 通常可支撑约 30--60 秒 speech，但这是待从真实 timeline
   验证的经验假设，不作为未经测量的实验事实。
5. **不预设 raw pixels 必须胜过所有文本代理。** Strong OCR、VLM-derived structured
   evidence 和 oracle text-equivalent 都是必要对照。若 raw/direct visual embedding
   没有额外收益，就采用更便宜、可审计的结构化证据，不声称 raw-vision necessity。

## 系统与因果协议

对第 (k\) 张 slide，记稳定出现时间为 (a_k\)，下一次 change 为 (a_{k+1}\)，
驻留时长为 (D_k=a_{k+1}-a_k\)。视觉 worker 在 change event 后执行一次：

```text
I_k --(detect + encode once)--> E_k --(atomic cache update)--> persistent evidence state
audio prefix x_<=t + decoder state y_<m --(retrieve/gate)--> small packet P_t
```

- (E_k\) 可包含 OCR text、术语、实体、公式、图表关系、layout/emphasis 和来源坐标；
  glossary 只是 (E_k\) 的一种投影，不是完整表示。
- 在 (a_k \le t < a_{k+1}\) 内，所有 audio chunks 复用同一个 (E_k\)，除非证据
  被显式判 stale。系统不得为等待 vision 阻塞 READ/WRITE；证据未 ready 时退化为
  audio-only。
- Live-slide setting 只能使用时间 (t\) 前已观察到并稳定的 slide。只有在单独声明
  `deck-known-in-advance` 条件时，才允许使用未来页，避免把整套 deck 偷换成 lookahead。
- 视觉 evidence 按与当前 causal audio hypothesis 的匹配度、模型不确定性和 staleness
  检索。一个 slide 长时间存在不等于它的全部内容在每个 chunk 都相关。

## 可辩护的论文空间

| 邻近工作 | 已覆盖 | 本项目必须证明的差异 |
| --- | --- | --- |
| OmniFusion | scientific-talk speech+image SimulST；image 在推理路径上带来额外时延 | change-triggered once-per-slide encoding、跨 chunk cache、非阻塞读取、cold/amortized/on-path 分项成本，以及视觉内容必要性控制 |
| EGTA / contextual SST | 从预先存在的 document terminology memory 按 streaming speech 选择文本证据 | evidence 来自随时间变化的 live visual state；保留非纯文本结构；测 slide-speech temporal alignment。若最终只剩静态术语 routing，则会与 EGTA 实质碰撞 |
| Do Slides Help? | slide-conditioned offline ASR；大量训练图像由 transcript terms 合成 PDF/image | real slides、speech translation、causal commits、错误/过期证据和计算时延；不能用合成文字图片支撑 raw-vision claim |
| Lip-based AVST | phonetic vision 对 noisy/offline ST 的鲁棒性 | 不属于本项目主张；related work 中用于区分 continuous phonetic vision 与 sparse persistent semantic vision |

因此主问题不是泛化的“image 是否有用”，而是：

> Can semantic evidence computed once from a live slide, cached across its dwell
> window, and causally selected during speech improve the quality--latency
> frontier of noisy and terminology-heavy SimulST; and on which slices is
> information beyond strong OCR actually necessary?

## 预注册 Kill Tests

### A. 内容与表示

固定同一模型、causal audio prefix、decoding policy、evidence token budget 和
计算预算，至少比较：

- `A0`: audio only；
- `A1`: audio + topic/metadata；
- `A2`: audio + strong OCR（含标准清洗、去重和语言识别）；
- `A3`: audio + VLM-derived structured semantic evidence；
- `A4`: audio + cached raw/direct visual representation（模型支持时）；
- `A5`: audio + oracle text-equivalent（人工覆盖可见文字、标签、公式和必要关系）。

每种 visual condition 都要配 matched current slide、same-talk stale/wrong slide 和
cross-talk/cross-domain wrong slide。报告 chrF/BLEU、XCOMET、term/entity recall、
visible-but-unspoken hallucination、wrong-evidence adoption、copy rate、AL/LAAL。

**通过门槛分三层：**

1. `correct > stale/wrong` 的 paired 95% CI 下界大于 0，证明内容和时间对齐真的
   被使用，而不是 image presence 或 domain priming；
2. `A3 > A2` 在预注册的 beyond-OCR slice 上成立，才能声称 layout、formula、
   chart relation 或 emphasis 带来 OCR 之外的 semantic vision value；
3. 只有 `A4 > max(A3, A5)` 时才声称 direct raw-pixel representation 不可替代。
   否则 method 应选择最便宜、最可审计且效果相当的结构化 evidence。

若只通过第 1 层而不通过第 2 层，结果是 slide-derived contextual SST，而不是
raw-vision paper；此时必须证明 temporal state、异步成本或 evidence-selection protocol
相对 EGTA 构成实质增量，否则停止正向方法 claim。

### B. 时间与持久性

在保持 evidence 内容相同的前提下比较：

- `B0`: live causal current-slide cache；
- `B1`: 同一 slide 每个 audio chunk 同步重编码；
- `B2`: evidence 在真实 encode 完成时间后才可见；
- `B3`: stale previous slide；
- `B4`: 打乱 slide change timestamps；
- `B5`: oracle alignment，仅作 upper bound。

系统必须证明增益不是未来泄漏，也不是把整套 deck 当作预知文本。核心 temporal
结果是：一次生成的 evidence 能在多个后续 chunks 中稳定复用，并且 current、stale、
wrong 三类状态产生可解释的差异。

### C. 计算与时延

不能把 “vision 先算” 直接写成 zero latency。每个系统至少报告：

- slide-change detection latency $L_{detect}$；
- once-per-slide encoding latency $L_{encode}$；
- evidence-ready time 与首次相关 speech 的 lead/lag；
- cache lookup、retrieval 和 injection 的 on-path latency $L_{onpath}$；
- dwell-normalized amortized cost
  $(L_{detect}+L_{encode})/D_k$；
- end-to-end wall clock、GPU seconds、RTF，以及 computation-aware AL/LAAL。

**通过门槛：** async cached condition 在质量不降的前提下显著降低 synchronous
per-chunk vision 的计算成本，并且相对 audio-only 不产生可测的 audio-path latency
回退；若 evidence 未及时 ready，系统应跳过而不是阻塞。

## 数据角色

### Chinese-LiPS

当前最重要的 clean/strong-slide 内部候选：连续 session timeline、独立 1080p slide
feed、speech 和 transcript 同时存在。它用于测 slide dwell distribution、构建真实
change events，并验证 semantic evidence 的上界。Derived artifacts 继续只放 private
HF；paper-grade 排名仍需独立/人工翻译 reference 和许可范围确认。

现有 206-segment Qwen3-Omni probe 不能代表目标架构：它按 segment 重新送入 image，
没有跨 chunk/segment cache；wrong image 又来自同一 lecture。它只说明在该 clean
setting 下 `correct - same-talk wrong = +0.34 chrF (p=0.22)`，没有证明局部 slide
内容被利用。

### 其他 strata

- ACL 60/60：term-credible control，前提是媒体和翻译使用范围确认；
- mTEDx-V：历史上用于 sparse/noisy negative stratum，但当前 TED 条款风险不允许
  把重新下载或自动抽取视为默认可行；
- AVMuST-TED：退出执行计划，仅保留历史许可审计和 related-work 记录。

## Novelty audit

当前主问题为 **Level 3 — Medium Overlap / partial overlap**：

- OmniFusion 是任务先例，但没有 once-per-slide persistent cache、异步成本分解、
  OCR/text-equivalent/wrong-image controls，也未证明 vision content 的必要性；详见
  [`OMNIFUSION_REASSESSMENT_20260731.md`](OMNIFUSION_REASSESSMENT_20260731.md)。
- EGTA 对“文本 memory + streaming selection”构成 **Level 2 / material collision**。
  因而本项目不能把 VLM 输出简单压成术语表后宣称 multimodal novelty；必须保留
  live visual state、非文本关系、temporal alignment 和 async cost protocol。
- “image-vs-none 涨分”“视觉在音频前提供”“首次 vision SimulST”都不是可守住的
  novelty claim。

当前可守住、但仍待实验验证的 delta 是：

> 把 slide 建模为低频更新、长时间驻留的 semantic evidence state，而不是与每个
> speech chunk 同步融合的第二条流；在真实 causal timing 下分离 evidence content、
> temporal alignment、representation sufficiency 和 amortized computation。

## Paper go/no-go

1. **完整 semantic paper：** A1/A2、B、C 通过；至少一个可审稿复核的数据来源
   具有人工 reference 和足够 beyond-OCR 样本。
2. **Raw-vision claim：** 还需要 A3 通过；否则只能声称 slide-derived structured
   evidence，不得声称 pixels 本身不可替代。
3. **Diagnostic/negative-result paper：** 两个模型、两个数据来源、人工 reference、
   完整 matched controls 和 latency accounting 后，即使 OCR 足够，也可报告可信边界。
4. **停止条件：** 只有 image-presence gain、只有 machine-reference gain、只有
   极端合成噪声 gain、correct 不优于 wrong/stale，或最终方法等价于已有静态 textual
   context routing 时，停止正向方法论文。

## 接下来三步

1. 从已有 long-form timeline 重建真实 slide change event，报告 dwell time 的
   median、P25/P75/P90 和 slide-to-first-related-speech lead/lag；验证 30--60 秒假设。
2. 把现有 206-item speech probe 改成 persistent evidence protocol：补 strong OCR、
   structured VLM、oracle text-equivalent、same-talk stale 和 cross-talk wrong controls，
   并人工核查小型 reference/beyond-OCR subset。
3. 实现 async cache simulator/runner，分开测 cold encode、amortized cost、ready-time
   miss 和 on-path retrieval；只在这三步通过后扩大数据或训练模型。

## 主要相关工作

- Yang and Nakamura, 2026, *When to Use Extra Context: Evidence-Grounded
  Terminology Adaptation for Simultaneous Speech Translation*:
  <https://arxiv.org/abs/2607.17766>
- Koneru et al., 2025, *OmniFusion: Simultaneous Multilingual Multimodal
  Translations via Modular Fusion*: <https://arxiv.org/abs/2512.00234>
- Caglayan et al., EMNLP 2020, *Simultaneous Machine Translation with Visual
  Context*: <https://aclanthology.org/2020.emnlp-main.184/>
- Sinhamahapatra and Niehues, EMNLP 2025, *Do Slides Help?*:
  <https://aclanthology.org/2025.emnlp-main.814/>
- Cheng et al., ICCV 2023, *MixSpeech* / AVMuST-TED（excluded lip route）:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_MixSpeech_Cross-Modality_Self-Learning_with_Audio-Visual_Stream_Mixup_for_Visual_Speech_ICCV_2023_paper.html>
