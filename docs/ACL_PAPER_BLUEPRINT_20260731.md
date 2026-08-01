# ACL Paper Blueprint：Causal Live-Slide Evidence for SimulST

更新日期：2026-07-31

状态：**历史 Route A 的详细 confirmatory contract。当前 paper identity、ordered
evidence ladder 与 oracle-first priority 以
[`PAPER_STORY_DECISION_20260731.md`](PAPER_STORY_DECISION_20260731.md) 为准；本文只在
`C6 > C5` 后提供 image-specific confirmatory 细节。实验结果尚未产生，所有正向结论
均为待检验假设。**

## 1. 结论先行

这篇 paper 不应写成“把 image 接到 simultaneous speech translation（SimulST）
模型上”，也不应写成“视觉总是有用”。最可守住的问题是：

> **When does a live slide provide useful semantic evidence beyond strong text
> context for SimulST under controlled acoustic corruption, and how much of
> that evidence should a causal system consume?**

系统把 slide 当作**提前到达、低频更新、长时间驻留的 semantic state**：每次 slide
变化只解析一次，缓存可追溯的 evidence；后续 audio chunks 按需读取一个有预算的
packet。音频流永不等待 vision。论文同时回答四件事：

1. 正确且时间对齐的 slide 内容是否真的改善 noisy SimulST；
2. 哪些增益 strong OCR / PDF RAG 已经能解释，哪些需要 layout、formula、chart
   relation 或 visual emphasis；
3. 何时应该使用 `NONE`、短 OCR 或更丰富的 semantic evidence；
4. once-per-slide cache 是否在 computation-aware latency 下优于 per-chunk vision。

**主实验不包含 lip video。** Lip 是连续、同步的 phonetic evidence；slide 是提前、
持久的 semantic evidence。二者的数据、因果机制和计算路径不同，放在同一模型表里
既不公平，也会削弱论文问题。Lip-based AVST/AVSR 只作为 related work 和独立
diagnostic 参照，不作为 MCIF/ACL60/60 主表 baseline。

## 2. Paper claim 与 novelty delta

### 2.1 暂定标题

**When Does a Slide Help? Causal Evaluation of Live Slide Evidence for
Simultaneous Speech Translation**

更保守的备选标题：

- **Beyond OCR? Live Slide Evidence for SimulST under Acoustic Corruption**
- **Persistent Slide Evidence for Simultaneous Speech Translation**

### 2.2 一句话 delta

> Unlike IWSLT-2026 PDF-RAG and LECTRANS, which use static document or slide
> context for transcript/context-conditioned translation, we treat a live
> slide as a causally changing, once-encoded semantic state in raw-audio
> SimulST, and measure when structured evidence beyond OCR helps under matched
> current/stale/wrong controls and explicit evidence/latency budgets.

### 2.3 Route A 的 primary contribution

本文只把 **causal benchmark/protocol + empirical finding** 作为 primary
contribution：在完整 long-form speech 上联合控制 acoustic corruption、live
current/stale/wrong slide、representation sufficiency 和 causal timing，并给出
OCR-sufficient、image-needed 与 harmful-context 的可复核边界。

change-triggered cache 是实现该 protocol 的必要系统设施，不单列为算法 novelty。
Evidence-budget policy 是 conditional secondary contribution：只有在不看 test 的前提
下通过 G2 才进入 abstract，否则只作为分析工具。不得在结果出来后从 benchmark、
system、policy、raw vision 四种论文中任选一种叙事。

禁止声称：

- “first visual SimulST”或“first multimodal SimulST”；
- “first adaptive multimodal gate”；
- “raw pixels are necessary”，除非 direct image 超过 structured evidence 和
  oracle text-equivalent；
- “real-world noise robustness”，除非加入独立的真实录制验证；
- “zero-latency vision”。异步不等于零计算。

## 3. Collision audit：论文空间到底在哪里

当前 novelty 判定为 **Level 3：medium overlap / partial overlap**。没有单篇工作覆盖
“live slide temporal state + raw-audio SimulST + noisy causal intervention +
beyond-OCR controls + evidence budget”，但多个相邻方向分别覆盖了其中一部分。

| 工作 | 已经做了什么 | 本文必须保留的差异 |
| --- | --- | --- |
| *Do Slides Help?* (EMNLP 2025) | 在 ACL60/60 的真实 slide 上做 offline ASR；训练扩增才使用 transcript terms 合成 PDF/image | raw speech translation、long-form causal commits、noise dose-response、strong OCR/structured/wrong/stale controls |
| IWSLT 2026 extra-context + MLLP-VRAIN | ACL paper PDF；KeyBERT phrase boosting、预翻译 memory、BM25 top-k RAG；context 带来质量增益 | live current slide 而非整篇静态 PDF；非文本结构；temporal alignment；matched wrong/stale；选择视觉 evidence 的成本 |
| LECTRANS (ARR 2026 submission) | 383 h academic lectures；slide image/OCR + transcript translation；直接讨论 slide 何时 helpful/noisy | 其 baseline 是 aligned ASR transcript 的 segment-level MT；本文必须做 raw-audio unsegmented SimulST、真实 timing、noise 和 async cost |
| OmniFusion | image+speech SimulST 先例；其论文报告相对自身 cascade 的时延改善 | 跨论文 latency 不直接排名；在同一 runner 中实现 competent cached raw-vision/KV 对照，比较 persistence、content necessity 和 computation-aware cost |
| EGTA / RASST | streaming speech 从 document terminology memory 选择、使用 term hints | 它们只覆盖 terminology，不封住广义 context/vision；但如果本文最终只剩术语表/top-k，就会直接碰撞。Route A 必须保留 image-specific relation 与 matched text control |
| XLAVS-R、AV-TranSpeech | noisy speech 下 lip video 改善 offline AVST/S2ST | continuous phonetic vision，不是 persistent semantic context；不作为相同样本的直接 baseline |
| reliability-gated AVSR | 按 acoustic/visual quality 决定何时信任 lips | “when to use vision”本身不是 novelty；本文 policy 的独特对象是预计算 slide evidence 的最小充分预算 |

### 3.1 最危险的碰撞

1. **LECTRANS** 已经占据“academic lecture multimodal translation + slide text/image +
   when slides help or add noise”的泛化故事。本文必须强调 raw-audio SimulST、实时
   slide state、noise intervention 和 matched causal controls。
2. **IWSLT 2026 context track** 已经把 PDF context、ASR phrase boosting、RAG top-k
   放进正式 SimulST protocol。“多给一点 context”不再构成新方法。
3. **adaptive AVSR** 已经覆盖按模态质量 gating。本文不能仅用一个 acoustic
   uncertainty gate 作为算法贡献。

## 4. Research questions 与可证伪假设

### RQ1：内容特异性

在 native、+10、+5、0、-5 dB 下，`correct-current` 是否稳定优于
`same-talk-stale`、`cross-talk-wrong` 和 `timestamp-shuffled`？

- **H1：** 视觉收益随 acoustic ambiguity 增加，但 correct-vs-wrong 差值只在
  slide 与 speech 语义相关时出现。
- **反证：** correct 与 wrong/stale 相同，说明只是 domain priming、image presence
  或 prompt regularization。

### RQ2：OCR 之外还有什么

在人工预注册的 `image_needed` pooled slice 上，image+OCR VLM semantics 是否超过
OCR-only LLM semantics？`layout_required`、`chart_relation`、
`formula_structure`、`visual_emphasis` 只是该 pooled slice 的描述性 subtype，不能
分别挑显著结果。

- **H2：** beyond-OCR 增益集中在非线性结构，而不是普通可见术语。
- **反证：** structured 与 OCR 等价，则论文应降级为 slide-derived contextual SST；
  不得声称 raw/semantic vision necessity。

### RQ3：何时用、用多少

选择 `{NONE, OCR-32, SEM-32, SEM-96}` 的 policy，能否以更少 evidence tokens 和
更低 GPU cost 达到 always-on semantic evidence 的质量？

- **H3：** 大量 chunks 无需视觉；小部分高不确定、强相关 chunks 需要短 packet；
  只有少量 beyond-OCR chunks 需要长 packet。
- **反证：** always-on 在 matched compute 下始终更好，或 policy 只学到固定 noise
  threshold，则不把 selection 作为独立贡献。

### RQ4：持久性与系统成本

once-per-slide async cache 与 per-chunk direct vision 相比，是否降低 GPU seconds、
RTF 和 computation-aware LongYAAL，同时保持质量？

- **H4：** 真实 dwell window 足够长，使 cold encode cost 可被后续 chunks 摊销。
- **反证：** slide 变化过快、evidence 经常晚于相关 speech、或 retrieval on-path
  cost 抵消节省。

## 5. 数据集决策

### 5.1 推荐层级

| 角色 | 数据 | 规模/性质 | 用途 | 当前动作 |
| --- | --- | --- | --- | --- |
| **主 held-out test** | **MCIF translation subset** | 当前 HF revision 有 100 个 long-media talks；其中 21 个 ACL 2023 talks（约 1 h 58 m）构成 translation/IWSLT subset，含原始 MP4/WAV、人工 English transcript、专业 De/It/Zh translations；CC BY 4.0 | long-form En→Zh 主方向；En→De 复制；真实 talk video/slide；接 IWSLT 2026 protocol | 冻结 HF media-pool revision 与官方 21-talk IWSLT archive，重建 slide timeline，不改 test labels |
| **development + tagged-term replication** | **ACL60/60** | dev 5 talks + eval 5 talks；专业 10 语翻译；第三方术语标签；音频本身刻意选得较清晰 | dev 调 policy/noise；eval 做 En→Zh 复制与 term metrics；直接延续 *Do Slides Help?* | 当前 repo 只有 dev 5 talks，补齐 eval 与官方 term annotations |
| **private diagnostic** | Chinese-LiPS-Long | 21 sessions/约 11.1 h private reconstruction；1080p slide；连续 timeline；无独立 paper-grade ST refs | 测 dwell/lead-lag、change detector、zh ASR/术语机制；不做 headline ST 排名 | 只在 private HF revision 上运行；先人工小样本，不扩大 pseudo refs |
| **未来外部验证** | LECTRANS | 383 h，162 min human-verified eval；当前匿名 ARR submission 声称 accepted 后发布 | 发布且许可明确后，验证 academic-lecture generalization | 不作为当前 blocker，不依赖尚未发布的数据 |

### 5.2 为什么 MCIF 取代 ACL60/60 成为主集

- 21 个独立 talks 比 ACL60/60 eval 的 5 talks 更适合 talk-level inference；
- 同时有 video、speech 和专业 En→De/It/Zh references；
- 已进入 IWSLT 2026 long-form SimulST dev protocol，可直接复用 SimulStream、
  OmniSTEval、XCOMET-XL 和 LongYAAL；
- ICLR 2026 MCIF 结果显示，当前 MLLM 加 video 经常无增益甚至退化，说明存在真实
  headroom，也使 negative result 有价值。

MCIF 不是天然 noisy benchmark：其构建排除了 inaudible、excessive-noise 和 distant-
microphone 样本。因此主表必须分开报告 native audio 与受控 corruption，不能把合成
噪声冒充自然分布。

原始 MCIF translation benchmark 的 `long/short × fixed/mixed` 共享同一批 21 talks，
不是独立的
train/dev/test split。为避免最严重的 model-selection leakage，本项目把 **全部 21 个
MCIF talks 设为 project-held-out test**：不能用其 references、模型输出或逐 talk
结果改 prompt、extractor、policy、noise level 或 metric。开发只使用 ACL60/60 dev
五个 talks 和不参与 ST 排名的 Chinese-LiPS diagnostics；系统冻结后一次性跑 MCIF。
MCIF references 是公开可见而非 blind test，因此论文必须如实称 `project-held-out`；
若 IWSLT organizer evaluation server 仍可用，再补真正 blind 的官方 test replication。

首次 MCIF inference 前必须提交并推送一个 frozen-run commit，包含 model revision、
code/config、all prompts、action budgets、noise manifests、primary hypotheses 和 metric
versions 的 SHA256。Inference container 不挂载 references；全部 21 talks 输出完成并
写入 append-only run ledger 后，scorer 才能读取 references。One-shot evaluator 检测到
completion marker 后拒绝覆盖；仅允许预先定义的 infrastructure failure 重跑，并保留
旧输出、failure log 与原因。这个机制不能把 public test 变成 blind test，但能减少
项目内部的反复试分。

### 5.3 为什么不把 lips 当主 baseline

同一个 causal comparison 要求每个 talk 都有稳定的 face track、可见嘴部和当前
slide。当前 ACL60/60 五个 dev video 的本地审计中，三个是 slide-only，两个只有很小
的 speaker thumbnail；MCIF 也不是以 lip ROI 质量设计的数据集。用另一套 MuAViC/
LRS3-T 数据跑 lip model，不是同一样本上的 evidence comparison。

因此 paper 把任务明确限定为 `slide-conditioned SimulST`，related work 里只比较
机制与成本。Appendix 不放跨 MuAViC/LRS3 数据集的 lip 数字，避免制造不可比较的
排名；只有未来构造出同一样本 face-visible subset 时才允许 paired lip comparison。

## 6. Noise protocol：真实动机，受控因果实验

### 6.1 主协议

先保留 native audio；随后对**完整连续 talk**混噪，再交给 streaming runner，不能
按 gold segments 独立混噪。这样保持噪声连续性、speaker pauses 和 causal timing。

| 条件 | 设置 | 目的 |
| --- | --- | --- |
| Native | 原始音频 | 不依赖合成噪声的 anchor |
| Babble dose-response | MUSAN babble，+10/+5/0/-5 dB，3 个固定 seeds | 模拟 overlapping speech；主因果曲线 |
| Noise-type generalization | MUSAN music、generic noise，0 dB；训练/调参不见该类型 | 检查是否只适配 babble |
| Reverberation | OpenSLR 28 Room Impulse Response and Noise Database 的一个预注册 moderate RIR | 检查 room degradation |

MUSAN/SLR17 采用 CC BY 4.0；RIR/Noise SLR28 采用 Apache 2.0。旧版误写的
SLR119 实际是 AliMeeting，不是 RIR source。Babble/noise source ids 在 dev/test
间隔离；固定 VAD 版本后，只在 speech-active frames 上计算 speech/noise RMS 与 SNR。
所有 source ids、seed、混合脚本、VAD revision、RMS/SNR 定义和 manifest 必须进
Git/HF。Noise seeds 是同一 talk 的 repeated measures，不能当作额外独立样本。

### 6.2 Reviewer 对“合成噪声不真实”的防守

论文把 noise 写成**controlled acoustic intervention**，用于建立视觉价值随可听性
变化的 dose-response，而不是声称代表会议现场噪声分布。Native 结果始终单列。
小规模、预注册环境的 record-and-replay validation 是外部效度增强项，不阻塞第一轮
futility screen。若最终没有它，标题、abstract 和 claims 一律写 `under controlled
acoustic corruption`，不能写 `noise-robust` 或 `real-world noisy speech`。

## 7. Evidence conditions 与强基线

所有条件使用同一 base model、audio prefix、policy、decode budget 和可比的 context
token budget。先比较可审计 evidence，再决定是否值得 direct image fusion。

| ID | 条件 | 说明 |
| --- | --- | --- |
| B0 | Audio-only SimulST | 现代 long-form streaming backbone；不是旧 Local Agreement probe |
| B1 | Noise-robust audio-only | denoiser / noise augmentation / robust ASR-SST；防止把普通鲁棒性收益错算给 vision |
| B2 | IWSLT-PDF context | PDF 一次预处理；KeyBERT/phrase boosting + pretranslated BM25 RAG，复现 MLLP-VRAIN 类 strong static-text baseline |
| R0 | Linear OCR | 只给 normalized visible strings；无 bbox、reading order 或 LLM semantic rewrite |
| R1 | OCR + 2D layout | 与 R0 相同文字，加 bbox/reading-order/region hierarchy；隔离 layout serialization 的价值 |
| R2 | OCR-only text-mode VLM semantics | 与 R3 使用同一 VLM revision、prompt/schema/decoding，但只读 R1、不传 image；输出 entities/terms/relations |
| R3 | Image + OCR VLM semantics | 与 R2 完全相同，仅增加 current image pixels；隔离 image-specific incremental information |
| R4 | Human relation oracle | 人工写出可见文字与必要的 layout/formula/chart relation；测试 extractor error 与信息上限 |
| R5 | Cached raw visual/KV | current slide 只编码一次，跨 chunks 复用模型支持的 visual state；competent direct-vision baseline |
| R6 | Per-chunk raw image | 每 chunk 重编码同一 image，只作为 cost ablation，不冒充 strong baseline |

每个 R0–R6 至少配四种 alignment：

- `current-correct`；
- `same-talk-stale`，来自前一页或语义不相关页；
- `cross-talk-wrong`，topic 尽量匹配与完全不匹配各一组；
- `timestamp-shuffled`，保留 evidence 分布但破坏 temporal alignment。

R0→R1→R2→R3→R4 是嵌套因果链；R2/R3 的 weights、prompt、schema、decoding 和
token budget 必须完全相同，唯一处理差异是 pixels。R3-vs-R2 才是 beyond-OCR
primary contrast，不能用 R3-vs-linear-R0 支撑
image necessity。必须额外报告 `visible-but-unspoken hallucination`、
`wrong-evidence adoption` 和 `copy rate`。仅有 BLEU/COMET 提升不足以证明
evidence 使用正确。

## 8. 方法：Persistent Slide Evidence State

### 8.1 Causal pipeline

```text
video frames -> causal slide-change detector -> stable current slide I_k
I_k -> one-time OCR/VLM extraction -> evidence state E_k + provenance boxes
audio prefix x_<=t + decoder state y_<m -> relevance/uncertainty features z_t
(z_t, E_k) -> budget policy a_t -> packet P_t -> SimulST decoder
```

关键约束：

- `E_k` 只在 slide 稳定出现后生成；未知未来 slide 不可见；
- evidence 未 ready 时执行 `NONE`，不能阻塞 audio READ/WRITE；
- 一个 slide 的视觉编码只发生一次，后续只做 cheap retrieval/packetization；
- packet 中每条 claim 保留 slide id、region box、extractor version 和 confidence；
- previous evidence 在 change 后立即标 stale，除非 policy 明确使用 historical memory。

### 8.2 Leakage firewall 与输入白名单

`current-slide` extractor 只能读取时间 $t$ 前稳定出现的 current frame pixels、frame
timestamp、slide id 和固定模型权重；禁止读取 audio、ASR hypothesis、human/gold
transcript、paper PDF、target reference、test annotation、gold alignment 或未来 frame。
PDF context 是单独的 B2 condition，不能悄悄进入 R0–R5。

Policy 只能读取 causal audio-only state 和已经 ready 的 evidence。Test annotations
只用于 scoring/slice analysis。Wrong/stale sampling 只能使用预冻结的 slide-only
embedding、时间和 source talk id，不能用 source/target transcript、reference 或模型
输出做 topic matching。每条 evidence 与 control 都记录 input field provenance、model
revision、prompt hash 和 available-at timestamp；违反白名单的 run 直接作废。

### 8.3 Evidence schema

```json
{
  "slide_id": "talk/slide_0042",
  "available_at_ms": 183420,
  "ocr_blocks": [{"text": "...", "bbox": [0, 0, 1, 1]}],
  "entities": [{"surface": "...", "type": "...", "bbox": [0, 0, 1, 1]}],
  "relations": [{"subject": "...", "predicate": "increases", "object": "..."}],
  "formulae": [{"linearized": "...", "bbox": [0, 0, 1, 1]}],
  "emphasis": [{"item": "...", "kind": "highlight|title|callout"}],
  "extractor": {"model": "...", "revision": "...", "prompt_sha256": "..."}
}
```

### 8.4 Evidence-budget policy

动作先固定为：

```text
A = {NONE, OCR-32, SEM-32, SEM-96}
```

测试时只允许 causal features：

- audio-only hypothesis entropy/margin、revision rate 和 stability；
- 非 oracle 的 acoustic quality estimate；
- current hypothesis 与 evidence 的 retrieval relevance；
- slide age、evidence ready/stale 状态；
- packet novelty，避免重复注入同一内容。

不能使用 reference、gold term、oracle SNR、未来 transcript 或未来 slide。主分析先
使用预注册 heuristic；learned cost-sensitive classifier/ranker 只作为 secondary，
不先训练大型 fusion model。开发集上为每个 decision point 做 counterfactual rollout，
label 定义为：在接近该点最佳质量的动作中，选择 evidence tokens、GPU cost 和
hallucination risk 最小者。具体 non-inferiority margin 必须在看 held-out test 前，
依据 dev bootstrap variance 冻结。

对照包括：always-none、always-full、固定 acoustic threshold、relevance-only、随机
matched activation、learned policy 和 oracle hindsight policy。Oracle 只给 upper bound。

## 9. Streaming backbone 与 protocol

主协议采用 IWSLT 2026 long-form SimulST，而不是当前 probe 的 segment-level 1 s
Local Agreement：

- 输入为 unsegmented complete-talk audio；
- 输出 SimulStream-compatible logs；
- 质量与 latency 用 OmniSTEval / official metric implementation；
- 报 computation-aware 和 computation-unaware 两套结果；
- primary backbone 冻结为 `Qwen3-Omni-30B-A3B-Instruct` 的 IWSLT 2026 direct
  adaptation protocol；variable VAD chunks、wait policy、multi-turn/KV-cache 设置
  在 MCIF scoring 前进 Git；
- cascaded baseline 复现强 ASR + Qwen MT + PDF context；
- cascaded system、第二语言和其他 noise levels 都是 replication/secondary，不能
  在 primary 失败后被提升为 headline；
- OmniFusion 若无法按同协议复现，只能作为 related-work precedent，不做跨论文
  latency 排名；同协议的 R5 cached raw visual/KV 才是效率主对照。

模型版本必须在 contract 冻结时选“当时最新、可复现且许可允许”的 open model，
而不是继续沿用 Qwen2.5。现有 Qwen3-Omni probe 只作为历史证据，不锁定最终 backbone。

## 10. Annotation 与 benchmark slices

从 ACL60/60 dev 和不参与 ST 排名的 Chinese-LiPS diagnostic 先盲抽 300–500 个
speech-slide events，用于定义 slices 与开发 extractor/policy；两个 annotators 独立
标注，冲突 adjudicate。Annotators 不看模型输出。系统冻结后可用同一 guideline
独立标注 MCIF test events 作 slice analysis，但不得据此改方法或阈值。

必备 labels：

- `ocr_sufficient`：可见线性文字已足够；
- `layout_required`：区域对应、层级或 reading order 决定含义；
- `chart_relation`：趋势、比较、坐标/legend 对应；
- `formula_structure`：公式结构不能由普通 OCR 可靠表达；
- `visual_emphasis`：高亮、圈选、标题层级提供取舍信号；
- `no_visual_support`：slide 与当前 speech 无直接证据关系；
- `stale_mismatch`：旧页内容具有误导风险。

同时标注：speech 中的 term/entity、audio-only 可辨识度、slide evidence 是否先于
首次相关 speech、evidence 是否能唯一消歧。报告 inter-annotator agreement、slice
数量和每个 talk 的分布，禁止只挑方法成功例。

## 11. Metrics 与统计协议

### 11.1 Quality / terminology

- XCOMET-XL / COMET 为主，chrF、BLEU 为辅助；
- term exact/normalized recall、entity/acronym accuracy、formula/visual-relation QA；
- ASR WER 与 term WER，用于区分 recognition support 和 translation improvement；
- hallucination、wrong-evidence adoption、visible-string copy rate。

### 11.2 Latency / compute

- LongYAAL / StreamLAAL，computation-aware 与 unaware 都报；
- output flicker、erasure、first correct emission；
- end-to-end RTF、GPU seconds、peak memory；
- slide-change detect、cold encode、evidence-ready lead/lag；
- dwell-normalized amortized encode cost、on-path retrieval latency；
- policy activation rate、evidence tokens、ready-time misses。

### 11.3 Policy quality

- beneficial-event precision/recall；
- action calibration；
- 相对 hindsight oracle 的 regret；
- 同等质量下的 token/GPU reduction，或同等 cost 下的质量提升。

### 11.4 Inference unit

独立统计单位是 **talk**，不是 segment 或 noise seed。主置信区间和 randomization/
bootstrap 按 talk cluster；noise level/seed 作为 repeated measures。MCIF 的 21 talks
做主推断，ACL60/60 eval 的 5 talks 只作 replication。所有 primary contrasts、slice
和 non-inferiority margin 在跑 held-out test 前冻结。

### 11.5 Confirmatory statistical contract

唯一 primary setting 在首次 MCIF run 前冻结为：

- dataset：全部 21 个 MCIF project-held-out talks；
- direction：En→Zh；
- backbone：`Qwen3-Omni-30B-A3B-Instruct` direct protocol；
- corruption：+5 dB MUSAN babble，三个 test-only source/seed 的 talk-level 平均；
- metric：XCOMET-XL；
- independent unit：talk。

采用有序 gatekeeping，前一项失败后后续只作 descriptive：

1. **H1 content specificity：** R3 `current-correct` 相对预冻结的
   `0.5 × same-talk-stale + 0.5 × cross-talk-wrong` counterfactual，talk-cluster
   95% CI 下界大于 0，且平均增益至少 **1.0 XCOMET-XL point**；
2. **H2 image-specific information：** 仅在 H1 通过后检验；在盲标的 pooled
   `image_needed` slice 上，R3 image+OCR semantics 超过同模型/同 prompt/schema/
   decoding 的 R2 OCR-only text-mode semantics，talk-cluster 95% CI 下界大于 0，
   且平均增益至少 **1.0 XCOMET-XL point**。该 slice 必须覆盖至少 12 个 MCIF
   talks；不足时 H2 自动降为 descriptive，不能声称 beyond-OCR superiority；
3. **H3 selectivity：** 仅作 conditional secondary；相对 R3 always-on 减少至少
   50% evidence tokens/activations，并落在 dev-frozen non-inferiority margin 内。

Native、其他 SNR/noise types、四个 image-needed subtypes、En→De、cascaded backbone
和 ACL60/60 eval 都是 secondary/replication。四个 subtype 用 Holm correction；其余
报告 effect size 与 CI，不允许在 primary 失败后挑其中一个改写 headline。

在冻结 MCIF 前，用 ACL60/60 dev 的 talk-level variance 做 simulation-based MDE/power
分析。若 21 talks 对 1.0-point minimum effect 明显不足，不得靠 segment-level p-value
补救：应在看 MCIF 输出前增加另一个 licensed independent talk corpus，或把论文降为
effect-estimation/benchmark paper并明确不做 confirmatory superiority claim。

## 12. Kill gates 与 pivot 规则

| Gate | 最低证据 | 不通过时怎么做 |
| --- | --- | --- |
| G0：content specificity | 上述唯一 H1 在 MCIF En→Zh、+5 dB、primary backbone/XCOMET 上通过；ACL dev 只能提前判 futility，不能宣告通过 | 停止正向 vision claim；报告 domain priming/null result |
| G1：OCR headroom | H1 通过后，R3 在 pooled `image_needed` slice 上超过 schema/token-matched R2，且错误采纳不增加 | 降级为 contextual SST；不声称 beyond-OCR |
| G2：selectivity | learned policy 相对 always-on 至少减少 50% evidence tokens/activation，并落在 dev-frozen quality non-inferiority margin 内 | policy 不作为贡献；保留 benchmark/system finding |
| G3：async efficiency | 同信息/模型的 R5 cached visual/KV 相对 R6 per-chunk image 降低 GPU seconds/RTF，且 computation-aware LongYAAL 不回退 | 不声称 latency advantage；只报告 evidence efficacy |
| G4：generalization | MCIF effect 在 ACL60/60 eval 或 En→De replication 中方向一致，并报告完整 CI | 限定为单 corpus finding，降低投稿 claim |

以下情况直接停止“大模型训练”并先写 diagnostic：

- 增益只在 -5 dB 这种极端条件出现；
- correct 与 wrong/stale 不可区分；
- structured 与 strong OCR 不可区分，且 temporal/cache 也无新发现；
- 结果只对 machine-generated references 成立；
- 方法最终等价于 PDF keyword list 或 BM25 RAG。

## 13. 最小完整实验矩阵

### Phase A：不训练的 causal futility screen

- 数据：ACL60/60 的 5 个 dev talks；Chinese-LiPS 只作 ASR/timing diagnostic；
- noise：native、+5、0 dB babble；
- evidence：none、current OCR、current structured、stale、cross-talk wrong；
- backbones：一个 direct omni + 一个 cascaded strong baseline；
- 目标：只做 **futility screen**、pipeline debugging 和 slice-density/MDE 估计，不训练
  selector，也不能凭 5 talks 宣告 G0/G1 通过。若 correct-current 在所有 dev talks/
  两个非极端条件都没有一致正向迹象则停止；有正向迹象只表示值得承担 final-run 成本。

### Phase B：冻结 benchmark 与 policy

- 完成 full slide timeline、300–500 event annotation、noise manifests；
- 冻结 MCIF held-out talk ids、ACL60/60 dev/eval、primary target 和 metrics；
- 先冻结 heuristic；learned selector 仅作为 secondary，可用 dev counterfactual rollouts；
- 完成 MDE/power audit、frozen-run commit、one-shot evaluator 和 no-reference inference
  environment；冻结 action budget、non-inferiority margin 和所有 contrasts。

### Phase C：完整 evaluation

- 冻结系统后只运行一次 MCIF 21-talk project-held-out protocol，En→Zh 主表，
  En→De replication；不得在语言间重新调 policy；
- ACL60/60 eval En→Zh term replication；
- native + babble dose-response + held-out noise type + RIR；
- always-on/selective、OCR/structured/direct、current/stale/wrong；
- talk-cluster CIs、quality-latency-compute Pareto。

### Phase D：只在 G0–G3 通过后训练

先尝试 parameter-efficient evidence adapter 或 policy distillation；不从头训练 omni
model。训练目标必须对应已证明的 headroom，否则只会把 benchmark artifact 放大。

## 14. 预计 paper tables / figures

1. **Table 1：** datasets、talk 数、时长、targets、slide coverage、face/lip 可用性、
   native acoustic condition、license；
2. **Figure 1：** once-per-slide causal timeline 与 persistent evidence state；
3. **Table 2：** primary +5 dB H1/H2 与 nested R0→R5 contrasts；
4. **Table 3：** correct/stale/wrong/shuffled causal controls；
5. **Table 4：** pooled image-needed confirmatory result；OCR-sufficient 与四个
   subtypes 作为 Holm-corrected/descriptive analysis；
6. **Figure 2：** quality–LongYAAL–GPU cost Pareto，always-on vs selective；
7. **Table 5：** activation、tokens、ready misses、cold/amortized/on-path cost；
8. **Figure 3：** noise dose-response 与 correct-minus-wrong gap；
9. **Table 6：** MCIF→ACL60/60 / En→Zh→En→De generalization；
10. **Appendix：** annotation guideline、prompts、noise manifests、lip mechanism
    discussion、negative examples。

## 15. Reviewer attacks 预演

| 质疑 | 必须提前准备的回答 |
| --- | --- |
| Why not OCR? | R0→R4 是嵌套因果链；只有 schema/token-matched R3-vs-R2 在 pooled image-needed slice 上的差值才支撑 image-specific claim |
| Why not the full paper PDF? | 复现 IWSLT PDF phrase boosting + BM25 RAG；live slide 的差异必须来自当前状态、temporal alignment 或非文本结构 |
| Synthetic noise is unrealistic | native 单列；full-talk causal mixing；VAD-aware SNR；dev/test noise-source 隔离；无 record-and-replay 时只声称 controlled acoustic corruption |
| Why not denoise the audio? | noise-robust/denoised audio-only 是必要 baseline，与 visual 条件组合形成 factorial comparison |
| The model just copies slide text | 报 copy rate、unspoken hallucination、wrong-evidence adoption、oracle text-equivalent |
| The gate is not novel | 同意；不声称 generic gating。贡献是 live persistent evidence benchmark、cost budget 和实证边界 |
| Slides are known in advance | 主 setting 只看 current observed slide；整套 deck 只作为明确标注的 upper bound |
| Async does not remove latency | 报 cold encode、ready time、on-path、amortized、GPU seconds 和 computation-aware latency |
| Lip video is a stronger visual signal | 它解决 phonetic robustness 且需要连续 face track；当前问题是提前到达的 semantic state；related work 与独立 diagnostic 分开比较 |
| MCIF is public dev data | 如实写 project-held-out；MCIF 前推送 frozen hash；inference 无 refs；append-only one-shot ledger；优先补 IWSLT blind evaluation |
| Five dev talks are underpowered | Phase A 只判 futility；primary inference 在 21 MCIF talks；事前 MDE 不足则先增加独立 corpus，不用 segment p-values |

## 16. 八周执行路线

### Week 1：资产与时间线

- 拉取/校验 MCIF immutable revision；补齐 ACL60/60 eval 与 term tags；
- causal slide-change extraction；统计 dwell P25/median/P75/P90、change error、
  slide-to-first-related-speech lead/lag；
- 输出 `data/manifests/` schema 和 HF revision pointer。

### Week 2：强 baseline

- IWSLT SimulStream/OmniSTEval runner；
- audio-only、noise-robust、PDF-RAG、current-slide OCR；
- full-talk noise mixer 与 fixed manifests。

### Week 3：semantic evidence 与 controls

- structured extractor + provenance；
- current/stale/cross-talk/shuffled packet builder；
- nested R0→R5 conditions 与 leakage whitelist；
- ACL60/60 五个 dev-talk Phase-A futility screen。

### Week 4：人工 slices 与 go/no-go

- 双人标注 300–500 events；
- 估计 image-needed density、talk-level variance 和 1.0 XCOMET MDE；
- dev 只能触发停止，不能宣告 G0/G1 通过；统计 power 不足则先扩 independent talks。

### Week 5：selection policy

- counterfactual rollout；
- 冻结 heuristic；learned/oracle actions 只作 secondary；
- 推送 frozen-run commit，验证 no-reference inference 与 one-shot ledger。

### Week 6：完整 MCIF

- En→Zh all talks；先完成唯一 +5 dB primary，再统一解锁 secondary scoring；
- cluster inference、failure analysis、compute accounting。

### Week 7：replication

- ACL60/60 eval En→Zh；MCIF En→De；
- 只在 G0–G3 通过后尝试轻量 adapter。

### Week 8：paper

- 冻结 tables/figures；
- 完成 introduction、method、benchmark、results、limitations；
- artifact card、reproduction commands、licenses、negative-result appendix。

## 17. Route A 升级后的执行动作

不要立即训练，也不要继续扩 Chinese-LiPS pseudo-reference。先按
[`DUAL_ROUTE_DECISION_20260731.md`](DUAL_ROUTE_DECISION_20260731.md) 完成共享
`C0-C7` pilot。只有 A-GO 通过后，才继续本文的 **Route-A confirmatory package**：

1. 拉取 MCIF + ACL60/60 eval，冻结 revisions 和 licenses；
2. 重建 causal slide timeline，实测 30–60 s dwell 假设；
3. 接通 IWSLT 2026 long-form runner；
4. 将共享 pilot 中通过的 text/context baselines 与 `R0/R1/R2/R3/stale/wrong` 映射到
   冻结的 Route A config；
5. 完成 image-needed annotation、MDE/power audit 和 frozen heuristic；G0/G1 只能由
   one-shot MCIF confirmatory run 判定。

## 18. Primary sources

- *Do Slides Help? Improving Speech Recognition with Slides*:
  <https://aclanthology.org/2025.emnlp-main.814/>
- IWSLT 2026 Simultaneous ST / Speech-to-Text with Extra Context:
  <https://iwslt.org/2026/simultaneous>
- *MLLP-VRAIN UPV System for the IWSLT 2026 Simultaneous Speech Translation
  Task*: <https://aclanthology.org/2026.iwslt-1.24/>
- *Test-Time Adaptation of an Offline Multimodal Foundation Model for
  Simultaneous Speech Translation*:
  <https://aclanthology.org/2026.iwslt-1.27/>
- *LECTRANS: A Multimodal Translation Benchmark for Academic Lectures*:
  <https://openreview.net/pdf/b129e506359e5d129d72d135c11e28938b7e34d8.pdf>
- *MCIF: Multimodal Crosslingual Instruction-Following Benchmark from
  Scientific Talks*: <https://arxiv.org/abs/2507.19634>
- MCIF project and dataset pointers: <https://mt.fbk.eu/mcif/>
- *XLAVS-R: Cross-Lingual Audio-Visual Speech Representation Learning for
  Noise-Robust Speech Translation*: <https://aclanthology.org/2024.acl-long.697/>
- MUSAN: <https://www.openslr.org/17/>
- Room Impulse Response and Noise Database (SLR28): <https://www.openslr.org/28/>
