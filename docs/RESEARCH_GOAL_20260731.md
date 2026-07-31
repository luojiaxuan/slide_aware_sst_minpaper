# Research Goal：何时 SimulST 真正需要原始视觉证据？

更新日期：2026-07-31

## 一句话目标

在严格因果、计算时延可计量的 simultaneous speech translation（SimulST）
设置中，判断 **raw visual evidence 何时提供了强文本代理无法提供的信息**：
一类是 slide/screen 的 semantic evidence，另一类是 lips 的 phonetic
evidence；只有单路证据分别通过预注册 kill test，才研究二者的 hybrid。

这是一项先诊断、后建模的研究。当前不把论文预设为“某个 multimodal
fusion 方法必然有效”，也不把任意 image-vs-none 增益解释为视觉内容增益。

## 当前判断

1. **不直接做 hybrid。** semantic vision 和 lip vision 的信息类型、数据要求、
   控制组和时延成本不同。现在混合会让正向结果无法归因，负向结果也无法诊断。
2. **不退化成一般 contextual SST。** EGTA 已经实现了从 document memory 中按
   当前 streaming speech evidence 选择术语，并在 MCIF-dev 和 ACL60/60-dev 上
   做了 shuffled-memory 与 activation audit。只做 OCR/topic/glossary routing 的
   外部新颖性风险很高。
3. **semantic 路线仍未被否定，但当前证据只支持 talk-level priming。**
   Qwen3-Omni speech+image probe 中，correct slide 与 same-talk wrong slide 的
   差异为 `+0.34 chrF`（`p=0.22`），无法证明 segment-specific slide content
   被利用。当前 wrong control 不是 unrelated-domain control。
4. **lip 路线的空白更清楚，但数据合法性和在线成本是硬约束。** MixSpeech、
   XLAVS-R 等已经覆盖 offline/noise-robust AVST，SimulLR 已覆盖 simultaneous
   lip reading；相对空白是 causal online audio-visual ST，而不是一般 lip ST。
5. **论文空间应放在 necessity，而不是 modality presence。** 核心问题不是
   “加 image 是否涨分”，而是“在强 OCR/text、强 audio-only robustness、匹配
   噪声控制和 computation-aware latency 下，raw pixels 是否仍有不可替代价值”。

## 三条路线的论文空间

| 路线 | 已有工作边界 | 本项目可守住的 delta | 当前风险 | 决策 |
| --- | --- | --- | --- | --- |
| Semantic slide | Caglayan et al. 做 image-conditioned simultaneous **text** MT；OmniFusion 是高延迟且缺少视觉必要性控制的 speech+image SimulST 任务先例；Do Slides Help? 做 offline ASR，并主要用 transcript 生成 synthetic PDF/images 扩充训练 | 对 raw slide、strong OCR、oracle text-equivalent、same-talk/cross-talk wrong slide 做因果对照，回答 raw pixels 何时超越文本代理 | 当前真实 speech probe 未显示 correct > same-talk wrong；可用样本中真正 beyond-OCR 的视觉语义稀疏 | 保留，先跑最低成本的 decisive controls |
| Lip | MixSpeech/AVMuST-TED、MuAViC、XLAVS-R 主要是 offline robust AVST；SimulLR 是 online lip reading，不是 translation | causal lip prefixes 在真实噪声下是否改善 SimulST quality-latency frontier，并超过强 audio denoising/robustness baseline | 视频编码成本、嘴形歧义、可用翻译数据少；TED 当前条款造成高风险 | 与 semantic kill test 并行做小规模可行性验证 |
| Hybrid | 两种证据理论上互补：lip 给 phonetic evidence，slide 给 semantic evidence | 只有在两路各自成立后，检验二者是否有正 interaction 或按需 routing 能否形成更优 Pareto frontier | 数据必须同时含 face、slide、连续音频和可靠翻译；实验矩阵和归因成本翻倍 | 暂不作为 MVP 或首篇 paper claim |

## 预注册 Kill Tests

### A. Semantic necessity test

固定同一模型、同一 causal audio prefix、同一 decoding policy 和同一计算预算，
至少比较：

- `A0`: audio only；
- `A1`: audio + topic/metadata；
- `A2`: audio + strong OCR/VLM-transcribed slide text；
- `A3`: audio + raw correct slide；
- `A4`: audio + same-talk wrong slide；
- `A5`: audio + cross-talk/cross-domain wrong slide；
- `A6`: audio + oracle text-equivalent（人工转录 slide 中所有可见文字、图表标签、
  公式和必要的结构化描述）。

在 clean、真实噪声、accented、jargon/entity-dense 分层上报告 chrF/BLEU、
XCOMET、term/entity recall、hallucination、copy rate、AL/LAAL 和 wall-clock
computation-aware latency。

**通过门槛：** `A3 > max(A2, A6)` 的 paired 95% CI 下界大于 0，并且该增益
集中在预先定义的 beyond-OCR 样本；同时 `A3 > A4` 和 `A3 > A5`，分别证明
segment-specific evidence 和排除任意 image/domain priming。如果 raw slide 只与
OCR/oracle text 打平，就把 semantic channel 定义为 text/context channel，不再
声称 raw vision necessity。

### B. Lip necessity test

使用严格 causal video prefix，至少比较：

- `B0`: audio only；
- `B1`: 强 noise-robust audio-only / speech enhancement baseline；
- `B2`: lip only；
- `B3`: audio + aligned lips；
- `B4`: audio + temporally shuffled lips；
- `B5`: audio + speaker-matched wrong lips。

按 clean 和多个真实/合成 SNR 分层，并报告与 A 相同的翻译质量和
computation-aware latency，同时加入 ASR/WER 诊断以定位收益来自 recognition
还是 translation。

**通过门槛：** `B3 > max(B1, B4, B5)` 的 paired 95% CI 下界大于 0，在至少
一个预注册的现实噪声区间形成更优 quality-latency Pareto point，且 clean 条件
没有不可接受退化。只有极端 `-20 dB` 才有效不能支撑一般 online AVST claim。

### C. Hybrid interaction test

仅当 A、B 均通过后，做 `audio / +slide / +lip / +slide+lip` 的 2x2 factorial
实验。Hybrid 需要满足下列至少一项：

- interaction term 的 paired 95% CI 下界大于 0；
- 在相同计算预算下形成单路方法都达不到的 Pareto point；
- 一个预注册 router 在不同 failure slice 上可靠选择 semantic 或 phonetic
  evidence，并优于 always-on 两路编码。

否则首篇论文只保留通过门槛的单路证据。

## 数据与许可结论

### AVMuST-TED

- 仓库声称 706 小时、English→Spanish/French/Italian/Portuguese，face-centered
  `224x224`、25 fps，并提供 clean 与 `{-20,-10,0,10,20} dB` 噪声结果。
- 这是 sentence/clip-level offline AVST 资源，不是现成的 SimulST benchmark；
  online 评估需要重新构造 causal stream、commit policy 和 computation-aware
  latency。
- 仓库 README 写 `CC-BY-NC 4.0`，根目录 `LICENSE` 却是 MIT，二者已经不一致；
  更关键的是媒体和字幕来自 TED/TEDx。
- TED 于 2024-05-07 更新的当前 Terms of Use 明确将 research dataset、ML/AI
  training、evaluation 和 automated extraction 排除在普通 educational use 外，
  除非另有书面许可。因此不能仅凭 AVMuST-TED 仓库声明就把它视为低风险、
  可重新下载和可复现的数据源。

**执行规则：** 在拿到 TED/数据作者书面许可或机构法律确认前，AVMuST-TED
只能用于文献和数据设计参考，不能成为本项目不可替代的训练/主测试集。

### Chinese-LiPS

它目前是最接近 hybrid 的内部候选：连续 session timeline、face、独立 slide
feed 和 speech 同时存在。但 derived artifacts 必须继续保存在 private HF repo；
paper-grade 结论还缺独立/人工翻译 reference、许可范围确认和严格 long-form
SimulST protocol。当前 206-segment probe 不能替代这些要求。

## Novelty audit

当前主问题的判断修正为 **Level 3 — Medium Overlap / partial overlap**。任务邻近
不等于强 prior；详细复核见
[`OMNIFUSION_REASSESSMENT_20260731.md`](OMNIFUSION_REASSESSMENT_20260731.md)：

- OmniFusion 是 scientific-talk speech+image SimulST 的任务先例，但其
  computation-aware AL 约为 `5.5–10s`，所谓“快约 1 秒”来自 E2E 相对自建
  cascade，不是 image 相对 audio-only；offline 加 image 令 OmniFusion inference
  time 从 `1.98s` 增至 `3.15s`。它没有 OCR/text-equivalent、wrong-image 或
  noisy-speech controls，因此没有覆盖 raw-vision necessity 这一核心问题。
- 如果退化成一般 contextual SimulST，EGTA 仍构成 **Level 2 / material
  collision**：document terminology memory、stream-conditioned selection、
  shuffled-memory control 和 terminology 指标均已存在。
- Lip-only online AVST 的重合度较低，但 MixSpeech/XLAVS-R 已覆盖 noisy AVST，
  SimulLR 已覆盖 online lip processing；delta 必须同时包含 translation、causal
  streaming 与计算时延，缺一项都会退回已有工作。

因此目前不能写“首次把 vision 用于 SimulST”，但 OmniFusion 也不封死低时延、
可归因的 raw-vision 研究空间。可辩护的 delta 是：

> 不再把 image-vs-none 当作视觉有效性的证据，而是在统一 causal SimulST
> protocol 中，用 text-equivalent、matched wrong-vision、noise-robust audio 和
> computation-aware controls，判定 semantic 与 phonetic raw vision 各自在何种
> failure slice 上不可替代。

这句话仍是目标，不是已证实贡献。

## Paper go/no-go

1. **Semantic paper：** A 通过，且至少一个可公开或可审稿复核的数据集具备
   人工 reference 和 beyond-OCR 样本量。
2. **Lip paper：** B 通过，且主数据的许可允许训练、评估和必要的可复现发布。
3. **Hybrid paper：** A、B、C 全部通过。
4. **Diagnostic/negative-result paper：** A/B 至少完成两个模型、两个数据来源、
   人工 reference 和完备 controls；即使 raw vision 不通过，也能形成可信的
   benchmark/methodology 结论。
5. **停止条件：** 只有 image-presence gain、只有 machine-reference gain、只有
   极端噪声 gain，或无法给出合法可复现的数据协议时，不写正向方法论文。

## 接下来三步

1. 先补 semantic test 的 `cross-talk wrong image + strong OCR + oracle
   text-equivalent`，复用现有 206 条 speech run，验证 domain priming 与 raw
   vision necessity；同时人工核查一个小型 reference subset。
2. 对合法可用的 20–50 条 face+speech 样本做 causal lip pipeline smoke，测
   encoder wall-clock cost，并检查 aligned-vs-shuffled lips 在 `0/-10 dB` 是否
   有方向正确的差异；此阶段不训练大模型。
3. 根据 A/B 的 paired effect 和许可结果选择 semantic 或 lip 主线。两路都不
   通过时，转为严谨的 visual-necessity benchmark/negative-result paper；两路
   都通过时才设计 hybrid routing。

## 主要相关工作

- Yang and Nakamura, 2026, *When to Use Extra Context: Evidence-Grounded
  Terminology Adaptation for Simultaneous Speech Translation*:
  <https://arxiv.org/abs/2607.17766>
- Koneru et al., 2025, *OmniFusion: Simultaneous Multilingual Multimodal
  Translations via Modular Fusion*: <https://arxiv.org/abs/2512.00234>
- Cheng et al., ICCV 2023, *MixSpeech* / AVMuST-TED:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_MixSpeech_Cross-Modality_Self-Learning_with_Audio-Visual_Stream_Mixup_for_Visual_Speech_ICCV_2023_paper.html>
- Lin et al., 2021, *SimulLR*: <https://arxiv.org/abs/2108.13630>
- Caglayan et al., EMNLP 2020, *Simultaneous Machine Translation with Visual
  Context*: <https://aclanthology.org/2020.emnlp-main.184/>
- Sinhamahapatra and Niehues, EMNLP 2025, *Do Slides Help?*:
  <https://aclanthology.org/2025.emnlp-main.814/>
- TED Terms of Use（2024-05-07 更新）:
  <https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use>
