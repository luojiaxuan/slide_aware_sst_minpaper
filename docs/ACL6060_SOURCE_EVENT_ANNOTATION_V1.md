# ACL60/60 Source Event Annotation v1

日期：2026-08-01

状态：**v1 seed/media workspace 已冻结并上传；v1 annotation protocol 已被 v2 supersede，
不得直接填写 `annotator_a/b.jsonl`。Automatic OCR diagnostic 只验证 headroom，不是已完成
event inventory。Workspace 不含 source transcript、target/reference 或模型输出。权威人工流程见
[`ACL6060_SOURCE_EVENT_ANNOTATION_V2.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V2.md)。**

## 目的

这一步不是先证明 slides 有用，而是估计 ACL dev 是否存在足够多可检验的 SimulST
opportunities：slide evidence 已经可用，但当前 audio prefix 仍不足以完成一个明确的
source-side forced choice。如果该 event density 和 oracle headroom 都很低，就不应先投入
automatic VLM compiler 或大规模 inference。

## Frozen seed

每个 5 个 dev talks 都抽取：

- 10 个 high-precision transition-candidate observations；
- 10 个 deterministic hash-random non-candidate observations；
- 共 100 packets，且 100 个 observation IDs 唯一；
- 每个 packet 给出 frame/hash、`t_evidence_sec`、full-talk audio id/hash，以及
  `[t_evidence-5 s, t_evidence+60 s]` 的建议听取窗口；
- selection salt 是 `acl6060-source-event-seed-v1`。

Canonical seed：
[`../data/annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl`](../data/annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl)，
SHA256 `2163f2ce23082601bd6da7a75f6d50813be8c7932558027daba7422a90ffdec8`。

Contract：
[`../code/configs/acl6060_source_event_annotation_v1.json`](../code/configs/acl6060_source_event_annotation_v1.json)，
SHA256 `12dc205bc9595665cd2dbe52afb7e1b353872fe9952ff2bc56f1f60cdbd8cd12`。

## Source-only automatic screen

在不读取 target/reference 的前提下，额外构建了本地 source segment timing manifest，并对
468 个 real talk frames 运行 Tesseract 5.5.2 (`eng`, PSM 11, confidence >= 50)：

- source-only segments：5 talks / 468 segments，SHA256
  `3c01680b65f1fa574e4700b89581c65048c68d1fea2471da1aa584b11bac2f5b`；
- OCR：468/468 frames 非空，共 26,921 tokens / 10,090 lines，SHA256
  `379a8e54ea54f7f1b24ef17bff4acedbe789c05039ac5895752ede6d52d9b5b0`；
- strict exact-match：901 raw overlapping n-grams；去除嵌套短语后为 344 candidates，
  对应 149 个独立 `first-spoken source segment × current frame` events；
- 344 candidates 中 265 个提前量至少 5 秒、183 个至少 10 秒、33 个至少 30 秒；
  median conservative lead 是 10.325 秒；
- frozen 100 packets 中 38 个含至少一个自动 future candidate，31 个含 future phrase。

这里把首次含候选的 source segment **起点**当作最早可能 spoken time，并要求该时刻的 causal
current frame 仍能 OCR 到同一候选。因此 lead 是保守下界。但 exact string 可包含泛词、表格
header 和语义上无用的重合；这些数字只证明 ACL dev 有足够的 OCR-sufficient anticipation
headroom 值得双标注，不能证明翻译提升，更不能证明 raw pixels 优于 OCR。50-row 分层 audit
sheet 已生成但仍是 pending，不能作为人工准确率引用。

Git summaries：

- [`../data/manifests/acl6060_dev_source_annotation_segments_v1_20260801.json`](../data/manifests/acl6060_dev_source_annotation_segments_v1_20260801.json)
- [`../data/manifests/acl6060_dev_frame_ocr_tesseract_v1_20260801.json`](../data/manifests/acl6060_dev_frame_ocr_tesseract_v1_20260801.json)
- [`../data/manifests/acl6060_dev_ocr_anticipation_v1_20260801.json`](../data/manifests/acl6060_dev_ocr_anticipation_v1_20260801.json)

## Materialized workspace

本地 staging：
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/workspace_v1`。

- 100 frame copies + 100 mono PCM16/16 kHz source clips；
- clip 总长 6,498.51 秒，workspace 约 205 MB；
- 每个 clip 覆盖建议 `[t_evidence-5s, t_evidence+60s]` window；talk 开头按真实
  boundary 截断，所以 clip 长度为 63.51--65.00 秒；
- `annotations/annotator_a.jsonl` 与 `annotator_b.jsonl` 各 100 行，annotator id 和
  SHA256 独立；
- 所有 media hashes、WAV frame counts、100 个唯一 packet ids 和 forbidden-field scan
  均通过；
- 当前状态是 `PENDING_DOUBLE_ANNOTATION`，不是已标注 dataset。

Git summary：
[`../data/manifests/acl6060_dev_source_event_workspace_v1_20260801.json`](../data/manifests/acl6060_dev_source_event_workspace_v1_20260801.json)。

Canonical reusable artifact 是 private HF dataset
[`gavinlaw/slide-aware-sst-acl6060-source-events`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-events)：

- revision `3199207c66b159ab39f662a32e0f6d633c9c2b79`；
- tag `acl6060-source-event-workspace-v1-20260801`；
- source Git provenance commit `17d773bd305affb144d6cbd50bc158f25c0f4f2d`；
- `private=True`；remote 307 files（包含 HF 自动 `.gitattributes`）、100 WAV、100
  frames、100 packet JSON；
- 已从 immutable revision 强制下载 `dataset_manifest.json`、annotator A sheet 和一条
  WAV，并与 local bytes/SHA256 一致。

## Annotation protocol

两个 annotators 使用独立副本，不能看彼此标签，也不能看任何 ST/ASR model output 或
target translation。

1. 先看当前 frame，在不听未来 audio 的情况下写一个 2–4 选项的 source-side forced
   choice。问题应测试 lexical sense、referent、relation、scope、ordering 或 proposition，
   不能只是抄写整页 OCR。
2. 锁定 `source_question`、`source_options` 和唯一 `source_answer_index` 后，再听建议
   audio window；需要扩窗时记录原因。
3. 以 0.96 s 固定 prefix step 找到最后仍不足以唯一回答的
   `t_last_insufficient_sec`，以及第一个已经足够的 `t_first_sufficient_sec`。不允许自由
   微调到有利时间点。
4. 标记 evidence subtype、normalized region/description、`term_or_entity` 和 negative
   labels。没有可靠外部证据或不预期产生 SimulST benefit 的 packet 必须保留为 negative，
   不能删掉。
5. Primary eligible event 需要唯一 source answer，且
   `t_evidence_sec <= t_last_insufficient_sec`。两位 annotator 的 boundary gap 超过 2 个
   prefix steps 时先 adjudicate；无法解决则排除 primary、保留 sensitivity record。

## Labels

`evidence_subtypes` 可多选：

- `ocr_sufficient`：线性可见文字已经足够；
- `layout_required`：区域、层级或 reading order 决定含义；
- `chart_relation`：趋势、比较、坐标或 legend 对应；
- `formula_structure`：普通 OCR 难以表达的公式结构；
- `visual_emphasis`：高亮、圈选或标题层级提供取舍信号；
- `speaker_or_object_visual`：非 slide-text 的人物/物体线索；
- `no_visual_support`：当前 frame 与后续 speech 无直接证据关系；
- `stale_mismatch_risk`：旧 frame 可能诱导错误选择。

Negative packet 使用 `no_external_evidence` 和/或 `no_expected_benefit`。Target-language
acceptable realizations 不在本阶段填写；它们必须在 source inventory blind hash 冻结后由
另一阶段生成。

## Reproduction

```bash
cd code
PYTHONPATH=. .venv/bin/python scripts/build_acl6060_source_event_seed.py \
  --frame-observations ../data/manifests/acl6060_dev_frame_observations_v1_20260801.jsonl \
  --transition-candidates ../data/manifests/acl6060_dev_transition_candidates_v1_20260801.jsonl \
  --talk-manifest ../data/manifests/acl6060_talks_20260731.jsonl \
  --contract configs/acl6060_source_event_annotation_v1.json \
  --output ../data/annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl \
  --summary-out ../data/manifests/acl6060_dev_source_event_seed_v1_20260801.json

PYTHONPATH=. .venv/bin/python scripts/build_acl6060_source_annotation_manifest.py \
  --acl-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/extracted/2/acl_6060 \
  --talk-manifest ../data/manifests/acl6060_talks_20260731.jsonl \
  --output /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/source_segments.jsonl \
  --summary-out ../data/manifests/acl6060_dev_source_annotation_segments_v1_20260801.json

PYTHONPATH=. .venv/bin/python scripts/extract_acl6060_frame_ocr.py \
  --frame-observations ../data/manifests/acl6060_dev_frame_observations_v1_20260801.jsonl \
  --portable-root /Users/luojiaxuan/Documents \
  --output /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/frame_ocr_tesseract_v1.jsonl \
  --summary-out ../data/manifests/acl6060_dev_frame_ocr_tesseract_v1_20260801.json

PYTHONPATH=. .venv/bin/python scripts/analyze_acl6060_ocr_anticipation.py \
  --ocr /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/frame_ocr_tesseract_v1.jsonl \
  --source-segments /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/source_segments.jsonl \
  --event-seed ../data/annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl \
  --output /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/ocr_anticipation_raw_matches_v1.jsonl \
  --nonredundant-output /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/ocr_anticipation_nonredundant_v1.jsonl \
  --seed-coverage-out /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/seed_ocr_anticipation_coverage_v1.jsonl \
  --audit-sample-out /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/ocr_anticipation_audit_sample_50_v1.jsonl \
  --summary-out ../data/manifests/acl6060_dev_ocr_anticipation_v1_20260801.json

PYTHONPATH=. .venv/bin/python scripts/materialize_acl6060_source_event_workspace.py \
  --seed ../data/annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl \
  --acl-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/extracted/2/acl_6060 \
  --portable-root /Users/luojiaxuan/Documents \
  --output-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/workspace_v1 \
  --summary-out ../data/manifests/acl6060_dev_source_event_workspace_v1_20260801.json
```

## Completion gate

本节原 v1 gate 已 superseded。v1 的自拟 question 无法定义 forced-choice agreement，且
`t_evidence-5s` clip 不能证明完整 causal prefix 尚不足。v2 使用共同 locked question、
talk-start causal audio trajectory、frame/audio 物理分离和 scorer-side stratified weighting。
