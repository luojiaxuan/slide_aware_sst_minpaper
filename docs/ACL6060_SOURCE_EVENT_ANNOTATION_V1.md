# ACL60/60 Source Event Annotation v1

日期：2026-08-01

状态：**100-row balanced seed 已冻结；等待两个独立 source-side annotators。Seed 不含
source transcript、target/reference 或模型输出，不是已完成 event inventory。**

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
```

## Completion gate

报告每 talk/stratum 的 eligible、negative、excluded 数量，forced-choice agreement、
boundary interval overlap/gap 和 subtype agreement。只有在 100 packets 中出现足够且跨
talk 分布的 anticipatory events，才进入 current-vs-stale/wrong oracle rollout。此处不提前
冻结最低 event count；先完成双标注并报告完整分布，再用开发证据决定是否继续。
