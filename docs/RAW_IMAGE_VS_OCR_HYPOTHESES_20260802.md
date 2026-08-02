# Raw Image vs. OCR Hypotheses

日期：2026-08-02

状态：`HYPOTHESES_ONLY_PENDING_HUMAN_RESCREEN_AND_CONTROLLED_TESTS`

## Candidate Provenance Correction

当前 researcher prescreen 页面顶部的 `candidate` **不是术语识别结果**，也不表示系统已经确认
English source 与 Chinese target 存在术语级对应。它来自以下高召回词面流程：

1. source-only `Qwen/Qwen3-VL-32B-Instruct` 从 slide 生成 `scene_summary`、`objects`、
   `actions`、`spatial_relations`；`ocr_text` 不参与 R2 proposal；
2. descriptor 和 official English source 分别被拆成候选 n-grams：长度至少 5 的非 stopword
   unigram，以及 2--4 gram phrase；
3. 只保留同时出现在视觉 descriptor 与该 talk official English source、且在该 source segment
   首次出现的 normalized lexical overlap；
4. `candidate_kind=token|phrase` 只表示 n-gram 形态，不表示 terminology；
5. official Chinese reference 在 outcome-side join 后展示给 reviewer，但 pipeline 没有运行双语
   word alignment、术语抽取、target realization authoring 或人工确认。

因此，`method`、`based`、`content` 等泛词进入候选池是预期的 high-recall false positive。系统只
知道 English candidate 的词面出现在 source reference；它不知道 Chinese reference 是否有稳定
对应表达。`event_scoreable` 的人工判断正是用来补上这个缺口。此前把这些 candidates 统称为
“术语”是不准确的，后续统一称为 `lexical/event proposals`。

页面中的 `OCR token N%` 也只是 candidate tokens 在 flat OCR token set 中的 exact coverage，
不是 OCR 语义充分性、phrase adjacency、layout preservation 或模型可用性的结论。

## Raw Image 可能胜过 OCR 的情况

以下均为待验证机制，不是当前结果或既定术语。

### 1. Flat OCR 丢失非文本语义关系

包括颜色/图例映射、箭头与流程拓扑、跨区域 entity link、空间分组、顺序和计数。此类关系若与
source/target event 明确对应，是最强的 strict beyond-OCR 候选。观察例为 natural-language span
与 logical-form span 的 `color-coded correspondence`；计分对象应是“颜色标注的对应关系”，而
不是泛化的“系统对应关系”。

### 2. Layout 与 typography 可能帮助选择相关证据

Dense table、粗体、highlight、caption、title、row/column grouping 可能让 raw-image model 更容易
找到与当前 speech 有关的内容，而 full flat OCR 会把相关文本和大量数字串联在一起。`MASSalign`
所在表格是一个待测 diagnostic：图片可能突出粗体 method/row，但当前自动 candidate `method`
本身不可计分。

这一机制不等于 OCR 缺少信息。若 caption-only、structure-aware OCR 或 OCR retrieval 能恢复同样
效果，则只能主张 context selection/representation 的工程收益，不能主张 pixels 语义不可替代。
Raw vision 是否真的更会利用粗体或忽略数字也必须实测；vision encoder 下采样可能反而损害小字
与 dense table。

### 3. OCR recognition 或 serialization failure

包括小字、低对比度、stylized text、公式、竖排、错词、漏词和错误 reading order。若图片中只是
可读文字而 OCR 失败，归入 `OCR-error-only` control，不冒充 semantic beyond-OCR。换行或分词本身
不构成失败：例如 `target` 与 `embedding` 都被识别且语义可重组时，仍应判 OCR sufficient。

### 4. 非文本 object/icon/shape grounding

图片可能直接展示未被文字命名的 object、icon 或 shape。只有当该视觉实体确实支持一个具体、
可计分的 source/target event，且 flat OCR 的上下文不能可靠推断它时，才是有效候选。仅仅“图片
画了 speech bubble”不够；若 OCR 已包含 cartoon/dialog/quoted utterances，仍可能 OCR sufficient。

### 5. Pre-speech context 与 noisy audio 的交互

Slide 可在对应 speech 前几十秒出现，所以 visual processing 可移出 online audio critical path。
当 candidate-bearing acoustic span 被噪声破坏时，提前得到的 visual prior 可能提高 event
correctness 或 first/stable realization time。该效应必须通过 clean/noise curve、matched wrong
image 和相同 online decoding contract 验证；“source 本身容易翻译”不影响 scoreability，但会让
clean-condition headroom 很小。

## Required Baselines

后续任何 raw-image advantage 至少需要区分：

| Condition | 回答的问题 |
| --- | --- |
| `audio_only` | speech 本身是否已经充分 |
| `full_flat_ocr` | naive 完整 OCR context 的效果 |
| `caption_title_header_ocr` | 去掉 dense 数字后，普通文本是否已足够 |
| `structure_aware_ocr` | layout/table structure 是否可由 text representation 恢复 |
| `retrieved_ocr` | OCR relevance selection 是否能替代 raw image |
| `raw_image` | 原始 pixels 在相同 causal availability 下的效果 |
| `wrong_image` | 收益是否来自 matched visual content，而非 image presence |
| `human_oracle_packet` | 数据是否存在可利用 headroom，以及模型 visual extraction 是否是瓶颈 |

必须匹配 model revision、audio prefixes、decode policy、evidence availability、输出预算和可比的
context budget。结果解释冻结为：

- raw image 只赢 `full_flat_ocr`：可能只是 OCR clutter/selection 效应；
- raw image 赢 `full_flat_ocr`，但不赢 structured/retrieved OCR：layout/context selection 有用，
  但 pixels 非必要；
- raw image 同时赢 structured/retrieved OCR 和 wrong image：更强的视觉表征证据；
- human oracle 赢 OCR、raw image 不赢：数据空间存在，瓶颈在 visual extraction/fusion；
- 仅 OCR-error controls 有收益：结论是 OCR robustness，不是 semantic beyond-OCR。

## Current Review Handling

当前 immutable prescreen artifact 不新增临时 evidence type，也不在标注中把假设写成结论。Reviewer
按 frozen strict rubric 完成判断，并在 note 中记录可能的 layout/typography diagnostic。收到 output
后再分别编译 `strict_semantic`、`ocr_error_control` 和 `layout_selection_diagnostic` 三个候选池；
所有 scoreable event heads 与可接受 Chinese realizations 都需要重新人工 author，不从当前
`candidate_source_en` 自动继承。
