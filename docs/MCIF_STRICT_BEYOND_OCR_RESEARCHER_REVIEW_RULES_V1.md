# MCIF Strict Beyond-OCR Researcher Review Rules V1

日期：2026-08-02

状态：`RESEARCHER_PRESCREEN_NOT_GOLD`

## 目的

这一步只筛选哪些 MCIF candidate 值得进入后续正式独立标注和模型实验。它不是 paper gold，
不用于报告人工一致性，也不直接证明 `vision > OCR`。

审查包包含满足 `lead >= 5s`、segment duration 3--24s 的 116 条候选。A 队列为按关系型视觉
提议、OCR token overlap、lead、talk/segment diversity 排序后的 70 条；B 队列为剩余 46 条。
先完成 A；若 `strict_keep < 30`，再继续 B。

## Candidate 不是术语标签

页面顶部的 candidate 是 high-recall `lexical/event proposal`，不是术语识别结果。R2 candidate
来自 source-only VLM visual descriptors 与 official English source 的 normalized 1--4 gram
词面交集；`candidate_kind=token|phrase` 只表示 n-gram 形态。Chinese reference 仅在后续
outcome-side join 后展示，pipeline 没有执行双语 alignment，也没有确认 target realization。

因此系统只知道 candidate 词面出现在 English source，不知道它是不是术语，也不知道 Chinese
reference 是否有稳定对应。`method`、`based`、`content` 等泛词应由 reviewer 判为不可计分。
`OCR token N%` 也只表示 exact token coverage，不表示 OCR 语义充分或不足。完整 provenance 与
raw-image/OCR 待验证机制见 `docs/RAW_IMAGE_VS_OCR_HYPOTHESES_20260802.md`。

## 每条必填判断

### 1. 图片支持该 evidence

只看实际 slide pixels，判断图片是否真的提供 candidate 对应的 speech-relevant evidence。
机器提议只能用于发现错误，不能替代 pixels。

### 2. Flat OCR 是否充分

把 OCR 当作无布局、无颜色、无箭头的纯文本。若这些文字已经足以预测或消解 speech 中的相关
内容，标 `yes`。候选字符串未完整出现不等于 OCR 不充分；候选 tokens 出现也不自动等于 OCR
充分，关键是缺失的视觉关系是否对该 event 有用。

### 3. Source/target event 是否可计分

English source 必须包含明确且非泛化的事件，Chinese reference 必须有可辨认的对应表达。
`example`、`content`、`based`、`right` 等泛词通常不可稳定计分。当前 prescreen 不判断 audio
sufficiency；那是后续独立阶段。

## Evidence 类型

- `chart_color_legend_relation`：颜色、图例、曲线、坐标或表格关系；
- `diagram_arrow_entity_link`：箭头、流程、节点或实体连接；
- `spatial_order_count`：位置、顺序、分组或计数；
- `other_semantic_visual`：其他无法由 flat text 保留的语义视觉证据；
- `ocr_error_only`：只是 OCR 漏字、错字、小字或断词；可作 control，不是主张中的 semantic
  beyond-OCR；
- `text_only`：图片只提供普通文本，且没有必要的额外视觉语义；
- `unsupported`：图片不支持 candidate；
- `uncertain`：无法可靠判断。

## 最终处置

`strict_keep` 必须同时满足：

1. `visual_support=yes`；
2. `flat_ocr_sufficient=no`；
3. `event_scoreable=yes`；
4. evidence type 为四种 semantic visual 类型之一。

`ocr_control` 必须满足前三项，并且 type 为 `ocr_error_only`。

`reject` 至少填写一个原因。`uncertain` 必须写备注。UI 会拒绝内部不一致的组合。

## 返回文件

点击 `导出 JSONL`，返回文件名形如：

```text
mcif-strict-beyond-ocr-prescreen-v1-<packet>-output.jsonl
```

可以中途导出；文件包含所有 rows，后续只消费 `annotation_status=completed` 的 rows。不要编辑
`packet_id`、`packet_items_sha256`、`candidate_id` 或 `input_row_sha256`。

## Source of Truth

- packet id：`mcif-strict-beyond-ocr-prescreen-v1-600d50299a42`；
- canonical local artifact：`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/outcomes/mcif_strict_beyond_ocr_researcher_prescreen_v1_600d50299a42_gef3bf0c`；
- Finder 交付目录：`/Users/luojiaxuan/Downloads/mcif_strict_beyond_ocr_review_v1`；
- private HF revision：<https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/18c4b1f4245bacf265caac1b1dfa1c6df198c195/strict_beyond_ocr_researcher_prescreen_v1>；
- HF tag：`mcif-strict-beyond-ocr-researcher-prescreen-v1`；
- builder Git commit：`ef3bf0c42290ad4616a11b75f310c254b25fe231`；
- config SHA256：`cf57e4d45823ac9055707cfd48b7950de71f80d78aae6b306742a705b95d2955`；
- source workspace `SHA256SUMS` SHA256：`0b480c19a1de2c0cbb2997290cde97a5f6dc0c25e01e6cf8affdcade5580c422`；
- items SHA256：`495fd6a1c7541682fbbec24b81b93e144d620ee9412f6c93daca920dfc62c255`。

Artifact 共 73 files / 24 MB，含 116 candidate rows 和 69 张去重 PNG。Finder copy 与 HF
强制全量回下载均和 canonical local artifact 逐字节一致，`SHA256SUMS` 全部通过。相关测试为
`18 passed`；桌面、手机布局和实际保存/自动跳转已验证。浏览器保存状态只存在本机
`localStorage`，canonical artifact 和 HF revision 保持 0 labels。
