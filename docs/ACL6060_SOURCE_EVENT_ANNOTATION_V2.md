# ACL60/60 Source Event Annotation v2

日期：2026-08-01

状态：**protocol、stage data packager 和 sequential prefix backend 已实现；100-frame blinded
author view r4 已在本地生成，人工标签尚未开始。r3 因 plaintext quasi-identifiers 已 superseded，
不得开始标注。正式 audio annotation 必须通过 backend，不能直接编辑或分发 server-private
timing/WAV sheet。v1 seed/media 不变。**

## 为什么需要 v2

v1 要求两位 annotator 各自从 frame 写不同的 forced-choice question，同时又要求报告
`forced-choice agreement`。如果问题、选项和答案空间不同，这个 agreement 没有统计定义。
更严重的是，先看 frame answer 再判断 audio 何时足够，会把 visual answer 泄漏给
audio-sufficiency boundary。v1 的 65 秒 window 还只含 `t_evidence-5s` 历史，不能排除答案
在更早 causal source prefix 已经出现，因此不能用于 paper boundary。

v2 不改 100-row balanced seed、causal frame、audio window 或 target firewall，只修正标注
顺序与角色。所有 validator 回答同一个已锁定 source-side question。两名 audio validators
与另外两名 frame validators 必须是完全不重叠的 cohort；否则先听 audio 的人已知答案，
后续 frame-only 判断仍受污染。Question author 也不能担任任何 validator。四名 validators
使用各自独立随机的 opaque option ids 和 item order。

## Roles and stages

### Stage 1: frame-only question authoring

Question author 只看 current frame、opaque packet ID 和 scorer-secret frame binding，不能看
`talk_id`、absolute timing、raw frame SHA，也不能听 audio 或看 transcript、
target/reference 或模型输出。对可检验 packet：

1. 写一个 2--4 选项的 source-side question；
2. 写唯一 canonical answer、evidence subtype/region 和 term/entity；
3. 对 question、options、answer、secret-HMAC frame binding 和 evidence fields 做 canonical JSON SHA256；
4. 写 `question_locked_at_utc`，锁定后不得静默改题。

没有可靠视觉问题的 packet 保留为 negative，不能删除。

### Stage 2: author audio relevance check

只有 hash lock 完成后，author 才能听 source audio，判断 locked question 是否确实被后续
speech address。该阶段只能把 packet 标为 candidate、not-addressed negative 或 exclusion，
不能修改 locked fields；修改必须产生新 hash 和显式 revision record。Author-facing audio-review
sheet 仍不含 `talk_id`、absolute timing、raw media SHA 或 selection stratum；完整 media/timing
manifest 位于 scorer-private path，`freeze-author-audio` 只在逐字段校验公开 sheet 后合并。

### Stage 3: blinded audio-only validation

两个 independent audio validators 得到相同 locked question、各自独立随机顺序的 options，以及
从 talk 起点开始的 causal source audio，但看不到 frame、canonical answer、selection
stratum、transcript 或彼此标签。每个 item 的 audio view 在
`min(t_evidence+60s, talk_end)` 截止，不能越界到更晚 speech。

Validator 先听完整 `0..t_evidence` prefix，然后从 `g0=t_evidence` 开始，以
`g_k=t_evidence+0.96k` 对每个 grid point 填写 `insufficient|uncertain|option_id`，直到受限
audio 结束。不能只手填一个主观 boundary。Gold 解封后，scorer 将
`first_stable_correct_step` 定义为首次选择 canonical option、后续不再撤回且至少有两个
连续 grid observations 的 step；window 最后一个孤立正确点仍是 right-censored。若
window 结束仍不存在稳定答案也记为 right-censored，而不是 eligible。若 `g0` 已答对，则该 validator
认为 audio 在 evidence 时已经足够。

Sequential backend 先提交并锁定 question-only 判断，再只从服务端返回当前 `0..g_k` WAV；
提交第 k 步后才能释放 `g_{k+1}`。每个 response、server timestamp 和 release state 写入
SHA256-chained append-only log；`freeze-audio` 会核对 completed event 与 sheet tail hash。
validator-facing HTTP state 只显示 `Prefix k/K`，不暴露 `talk_id` 或 absolute prefix seconds；
完整 timing task、WAV path/hash 和 event log 是 server/scorer-private artifacts。部署时 annotator
账号只能访问 HTTP service，不能直接读取 audio root 或 private task sheet。没有该 log 的 sheet
不会通过 validator，也不进入 paper evidence。

先完成并 hash-lock 两份完整 audio trajectories，才能进入 frame pass。v1 的 65 秒 clips
只作为 question-authoring convenience，不进入这一 stage。

### Stage 4: blinded frame-only validation

另外两位 independent frame validators 得到相同 question/options 和 current frame，但看不到
audio、canonical answer、audio-validator 或另一位 frame-validator 的结果。每人独立填写
frame answer、`supported|unsupported|ambiguous` 和 evidence subtypes。Audio/frame validator
IDs 必须四人互不重叠；report 对 cohort overlap 直接失败。

### Stage 5: report and adjudication

在不读取 target/reference 的 scorer 中解封 canonical answer，计算：

- frame-answer exact agreement 和对 canonical answer 的 accuracy；
- audio first-sufficient answer agreement/accuracy；
- audio primary-component agreement 与 frame support/answer agreement；
- boundary gap（0.96 s steps）和 interval overlap；
- evidence subtype Jaccard。

Primary eligible 要求 author candidate、两位 frame validators 均 frame-supported 且答对、
两位 audio validators 在 `g0` 均未选中各自盲化后的 canonical option、之后均出现 stable
correct step，并且 boundary gap 不超过 2 steps。超过 2 steps 或 label disagreement 必须保留原始记录并进入
adjudication；不能覆盖 A/B 文件。

冲突 packet 的 raw `primary_eligible` 保持 `null`，不能按 negative 进入 prevalence denominator。
`prepare-adjudication` 从 raw report 生成带 report-row hash 的独立 sheet，`freeze-adjudication`
锁定 `resolved|unresolvable` 结论；final `report --adjudication ...` 只在 lock 完整时应用。Raw
audio/frame/report artifacts 均不被覆盖。Adjudicator 不能与 question author 或四名 validators
重叠。

Adjudication 不能越过 hard gates：任一 audio validator 在 question-only 阶段已唯一答出时，
该 item 不得裁成 positive；positive boundary 必须严格晚于 `t_evidence`、不超过 causal endpoint、
落在 frozen grid，并在 endpoint 前保留至少两个 stable-correct observations。Author/media/timing
exclusions 是 missing outcomes，不是 negative，也会阻止
主 prevalence estimate，直到按预注册 missing-data policy 解决。

## Leakage firewall

- source transcript 只允许 automatic diagnostic 使用，不进入 v2 author/validator workspace；
- target/reference、ST/ASR output 在所有 v2 stages 均不可见；
- audio-only 和 frame-only views 必须由脚本物理分离，不依赖 annotator 自觉忽略文件；
- `selection_stratum` 只进入 scorer，不进入任何 author/validator view；
- author frame sheet、post-lock author audio-review sheet 和 frame-validator sheet 均不含
  `talk_id`、absolute timing 或 raw media SHA；frame stages 只含 secret-HMAC media binding。
  Audio timing task sheet、private author media manifest 和 event log 不分发给 annotator，HTTP API
  也不返回 absolute time；
- upstream media 本身可能被有意进行 corpus matching，因此这是 operational blinding，不宣称对
  拥有全部 upstream bytes 的 adversarial annotator 实现 cryptographic anonymity；
- target-language acceptable realizations 只在 source inventory 和 adjudication hash 冻结后
  由另一 artifact 生成。

## Frozen config

Machine-readable contract：
[`../code/configs/acl6060_source_event_annotation_v2.json`](../code/configs/acl6060_source_event_annotation_v2.json)。

v1 文档保留 seed/workspace provenance 和 automatic OCR headroom；v2 是人工 annotation 的
authoritative protocol。Stage packager 已实现 author/question hash、author audio-review bundle、
四人 disjoint cohorts、validator-specific opaque option ids/item order、question-only lock、完整 prefix trajectory validation、
audio/frame view materialization、media hash revalidation、annotation lock verification、agreement
report、scorer-only ID mapping 和 sequential delivery backend。Log verifier 还验证完整 ordered
event-state machine、release boundary、monotonic server timestamps 和 completion ordering；backend
已实现 causal release；
当前 blocking gate 是尚未完成的 human authoring 和四人独立 validation。

## Sampling estimand

100-row seed 在 transition/random strata 各抽 50 条，不是自然分布。主报告先给每个
`talk × stratum` 的 raw/adjudicated yield；只有结合原始 pool size 和 inclusion probability
才能给 overall prevalence。不能把 100 条中的 eligible 比例直接写成自然 event density。

Frozen sampling design：
[`../data/manifests/acl6060_dev_source_event_sampling_design_v2_20260801.json`](../data/manifests/acl6060_dev_source_event_sampling_design_v2_20260801.json)。
各 talk 的 transition inclusion probability 是 0.40--0.769，random-observation probability
是 0.120--0.189。Scorer 会验证十个 strata 的 sampled count，用 frozen pool sizes 计算
stratified SRS-without-replacement prevalence、standard error 和 normal 95% CI；存在未裁决
packet 或 source exclusion 时返回 `UNRESOLVED_MISSING_OUTCOME`，不会输出有偏数字。

## Implemented artifacts

- config：
  [`../code/configs/acl6060_source_event_annotation_v2.json`](../code/configs/acl6060_source_event_annotation_v2.json)；
- stage packager/validator/scorer：
  [`../code/scripts/acl6060_source_event_annotation_v2.py`](../code/scripts/acl6060_source_event_annotation_v2.py)；
- tests：
  [`../code/tests/test_acl6060_source_event_annotation_v2.py`](../code/tests/test_acl6060_source_event_annotation_v2.py)；
- sequential audio service：
  [`../code/scripts/serve_acl6060_audio_annotation.py`](../code/scripts/serve_acl6060_audio_annotation.py)；
- sequential service tests：
  [`../code/tests/test_serve_acl6060_audio_annotation.py`](../code/tests/test_serve_acl6060_audio_annotation.py)；
- Git summary：
  [`../data/manifests/acl6060_dev_source_event_annotation_v2_20260801.json`](../data/manifests/acl6060_dev_source_event_annotation_v2_20260801.json)。

Local author view：
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/author_view_v2_blinded_r4`。
它含 100 opaque-ID frames + `authoring.jsonl`，明确为 0 WAV；100/100 rows 均不含
`talk_id`、`t_evidence_sec`、raw `frame_sha256`、`selection_stratum` 或 source packet ID。
`selection_stratum` 和真实 packet mapping 只在 sibling scorer directory。Author row order 由
secret-key HMAC 全局打乱，不能从公开代码直接枚举原 `A001--A020`。HMAC
secret 和真实 mapping 只存在 scorer storage，不进入 Git 或 author HF repo。

Reproduction：

```bash
cd code
export ACL6060_V2_BLINDING_SECRET="$(tr -d '\n' < /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/scorer/blinding_secret_v2.txt)"
PYTHONPATH=. .venv/bin/python scripts/acl6060_source_event_annotation_v2.py prepare-author \
  --packet-manifest /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/workspace_v1/packet_manifest.jsonl \
  --workspace-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/workspace_v1 \
  --output-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/author_view_v2_blinded_r4 \
  --mapping-out /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/scorer/packet_mapping_v2_r4.jsonl \
  --author-id author_pending
```

Author labels 尚未产生，所以 audio/frame validator views 不可提前生成。`freeze-author` 后先用
`prepare-author-audio --private-manifest-out <scorer-path>` 生成不含 frame/identifier 的公开
causal audio-review bundle 和物理分离的 scorer-private manifest；`freeze-author-audio` 必须同时
提供 `--private-manifest`，逐字段合并后才锁定。`prepare-audio` 只接受
通过 question hash、timestamp、full-audio-gold、stale-frame 和实际 media hash checks 的
candidate rows。`freeze-audio` 会再次核对两份实际 WAV；`freeze-frame` 会锁定 frame labels，
`report` 会拒绝 lock mismatch。两份 audio task 分别运行
`serve_acl6060_audio_annotation.py`，使用不同 event log/output/port；annotator 不能获得 audio
root 的 shell/filesystem access。`freeze-audio` 要求两份 event logs 并再次核对完整 hash chain。

当前 private HF r3 revision
`2fb266d168e0abbf4ace17d3f5de9503a8c46cd6` 已 superseded，不得用于 authoring。Canonical r4：
[`gavinlaw/slide-aware-sst-acl6060-source-event-author-v2`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-event-author-v2)，
revision `bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a`，tag
`acl6060-source-event-author-v2-r4-20260801`。远端已验证 `private=True`、103 files、100 JPG、
0 WAV，且不含 mapping/secret；README、sheet 和一张 frame 已强制下载并 byte-verified。
