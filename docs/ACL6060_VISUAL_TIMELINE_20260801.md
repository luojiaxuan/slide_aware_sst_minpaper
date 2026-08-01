# ACL60/60 Visual Timeline v1

日期：2026-08-01

状态：**ACL dev 的 468 个真实 talk-video frames 已导入为 transcript-free causal
observations；pixel-threshold transition compression 未通过 false-negative audit，因此
v1 inference 使用 every-observed-frame policy。尚未读取 target/reference，也未运行 ST。**

## 数据边界

来源是 *Do Slides Help?* Figshare v2 中的真实 ACL60/60 evaluation frames，不是其
training augmentation 使用的合成 LaTeX/PDF slides：

- outer `Visual_ASR.zip` SHA256：
  `f771d3f6f03026ad1510cf6840b47df3406b06b804926ab3ae18af99f663d4cc`；
- inner `ACL_60_60_images.zip` SHA256：
  `ba794c97f474b1c8894ea17e33d936ab5b181f3b62b13c91f9b35fea7122fc75`；
- dev：5 talks、468 frames，恰好等于 frozen talk manifest 的 468 segments；
- frame filename 的数值是上游从原 talk video 取 midpoint frame 的 timestamp；
- importer 只遍历 JPEG 目录并读取 Git 中的 talk-level count/duration，不打开上游
  `acl_dev.json`，因此不会消费其中的 `sentence` 字段。

## Causal availability

v1 不猜测 frame 在 timestamp 之前已经出现：

1. 每个 frame 从 filename timestamp 起可用；
2. 直到下一个 observed frame 才结束；
3. 第一个 observed frame 之前是 `NO_VISUAL_STATE`；
4. 不把 midpoint frame backdate 到对应 source segment 的起点；
5. 468 个 observations 全部作为 causal states，不依赖 transition clustering。

各 talk 首帧时间为 0.22–4.94 s，末帧时间为 516.19–686.66 s。Portable manifest：
[`../data/manifests/acl6060_dev_frame_observations_v1_20260801.jsonl`](../data/manifests/acl6060_dev_frame_observations_v1_20260801.jsonl)，
SHA256 `dba4fa3055aef0053f37d1795215110b4ad07b732e3b82325252e80d2e1f2fa5`。

## Transition detector audit

复用了 MCIF detector 的 96×54 grayscale、6×8 patch metric，阈值为
`patch_diff_p75 >= 0.03 OR changed_patch_fraction >= 0.12`：

- 463 adjacent pairs 中触发 97 个候选；五张 positive sheets 逐行检查，97 个都包含
  slide 内容、布局或逐步 reveal 的可见变化，没有发现纯讲者运动 false positive；
- 负例审计每 talk 选择 6 个 threshold-nearest 和 6 个 deterministic hash-random
  non-candidates，共 60 个；
- hard negatives 中出现整页 `Retrieval -> Generation`、`Composition -> Sentence
  encoding`，以及 graph、bullet、callout reveal 等明确漏检；
- 因此 97-candidate set 只能作为 high-precision diagnostic inventory，不能压缩 causal
  states，也不能支持 slide dwell 结论。

QA artifact：
[`../data/manifests/acl6060_dev_visual_transition_qa_v1_20260801.json`](../data/manifests/acl6060_dev_visual_transition_qa_v1_20260801.json)。
完整候选和冻结负例抽样分别见：

- [`../data/manifests/acl6060_dev_transition_candidates_v1_20260801.jsonl`](../data/manifests/acl6060_dev_transition_candidates_v1_20260801.jsonl)；
- [`../data/manifests/acl6060_dev_transition_negative_audit_v1_20260801.jsonl`](../data/manifests/acl6060_dev_transition_negative_audit_v1_20260801.jsonl)。

## Reproduction

```bash
cd code
PYTHONPATH=. .venv/bin/python scripts/build_acl6060_visual_timeline.py \
  --talk-manifest ../data/manifests/acl6060_talks_20260731.jsonl \
  --frame-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/do-slides-help/figshare-v2/frames_v2/images \
  --split dev \
  --output-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/do-slides-help/figshare-v2/acl6060_dev_visual_v1 \
  --portable-frame-manifest-out ../data/manifests/acl6060_dev_frame_observations_v1_20260801.jsonl \
  --portable-candidates-out ../data/manifests/acl6060_dev_transition_candidates_v1_20260801.jsonl \
  --portable-negative-audit-out ../data/manifests/acl6060_dev_transition_negative_audit_v1_20260801.jsonl \
  --portable-summary-out ../data/manifests/acl6060_dev_visual_timeline_v1_20260801.json \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/do-slides-help/figshare-v2/frames_v2/images
```

## 下一步

使用 468-state causal stream 构建 ACL dev source-side annotation packets。先做
80–120 个 forced-choice opportunities，记录 slide sufficient time、audio first-sufficient
interval、negative spans 和 evidence subtype；candidate inventory、source-only packet 与
target scoring 必须继续物理分离。Transition clustering/encoding cache 是 oracle headroom
通过后的效率优化，不再阻塞第一轮实验。
