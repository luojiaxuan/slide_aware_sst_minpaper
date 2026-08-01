# MCIF Target-event Annotation V1

日期：2026-08-01

状态：**R0 lexical candidate inventory、355-item En→Zh author workspace、机器可读 protocol、
localhost authoring UI 与 freeze validator 均已实现并冻结；当前 human progress 为 0/355，
没有 gold event、audio-sufficiency label、`SourceEventTiming` 或 ST result。**

## 这一步回答什么

当前阶段只建立一个可审计的 R0 lexical event set：当 slide text 在 source speech 之前已经
可见时，系统是否能更早产生并稳定保持正确的 Chinese target realization。它直接对应
SimulST 独有的时间优势，不把 final BLEU 当作唯一结果。

这批 event 由 R0 OCR exact overlap 提议，因此不能证明 raw pixels 优于 OCR。R1
structure/relation 和 R2 non-text visual semantics 必须作为后续独立 discovery strata；不能把
R0-defined event 上的 raw-image 结果写成 `vision > OCR`。

## 冻结输入

- Candidate inventory：954 rows / 21 talks，SHA256
  `3b1f85137c4443bd65cb82beb4217301b2f4e67ae9a7cd45ded3cbc3e5dde5a2`；
- Reference segments：919 rows，SHA256
  `e7840e1b0cd0589e1a643b5d450c86372c4e6c6f67de594959926f9152ae172a`；
- R0/R1/R2 ladder：304 states，SHA256
  `8f77312b93562afd8a92ea0b3139fe5f91b21b08e9740d311a1fd0a83b594f7f`；
- Author items：355 rows，SHA256
  `fff1aea0b0d34b8dc2627e4b16c2bdf6bfd09252d4cbe50f5ac85e2ceee3a4d8`；
- Scorer mapping：SHA256
  `ea21e7d90fab4d7ffece012473ab67d504acce89bb1afdf8c285b878e1a8920e`；
- Protocol config：
  [`../code/configs/mcif_target_event_annotation_v1.json`](../code/configs/mcif_target_event_annotation_v1.json)，
  SHA256 `95b8dc69cca4736e31ce1f0fe4c2306cd8155d54d46bdfce0039db5d50660b9c`。

Canonical private HF workspace：
[`gavinlaw/slide-aware-sst-mcif-outcomes@0785a37f/target_event_author_workspace_v1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/0785a37f6537363b5cd0a8db0ead730298b12a1b/target_event_author_workspace_v1)，
tag `mcif-target-event-author-workspace-v1`。

## 统计单位

954 automatic candidates 被穷举合并为 355 个 `segment × current causal state` author items。
每个 item 保留该 segment 的全部 1--18 个候选 options，但最多冻结一个 event。因此同一段语音
不会因为重叠 n-grams 被重复计数。

355 items 的每-talk 数量为 2--37。这个不平衡反映候选生成密度；当前 workspace 是 capability
screen，不是 event prevalence sample，不能报告 `eligible / 355` 为自然出现率。正式 effect
仍按 talk 聚合，不能把 event、noise seed 或 prefix step 当独立样本。

## Author labels

允许状态：

- `eligible`：一个候选能形成清晰、可评分、slide-supported 的 En→Zh target event；
- `no_target_alignment`：source event 存在，但 Chinese reference 没有可接受 realization；
- `generic_or_unscorable`：候选过于通用或无法稳定评分；
- `visual_mismatch`：自动 OCR 候选与实际 slide evidence 不一致；
- `exclude_quality`：source/reference/media 质量阻止判断。

`eligible` 必须同时满足：

1. 从局部 options 中选择恰好一个 candidate；
2. 写入非空 canonical English source event；
3. 写入至少一个 acceptable Chinese realization；
4. target alignment 为 `explicit` 或 `paraphrased`；
5. slide evidence 为 `supported`。

Non-eligible rows 必须清空 selected option、canonical event 与 acceptable/forbidden
realizations，但仍需记录 target alignment 和 slide evidence status。Pending row 不允许保存
partial labels。Working sheet 的 immutable input、row hash、annotator ID 与 media bytes 任一漂移
都会 fail closed。

## 角色隔离

Target-event author 只接收 `author_view/`，可看 current slide、R0/R1、English source segment、
Chinese reference 和 candidate options。真实 talk/segment/state/candidate IDs 位于
`scorer_private/item_mapping.jsonl`。

未来 audio-only validators 不能访问本 HF repo、author view、working/frozen labels 或 scorer
mapping，也不能看 slide、OCR、source/target reference 或 candidate options。Audio task 必须在
author freeze 后由另一 artifact 生成，并用独立账号/view 交付。

## 当前 authoring UI

实现：
[`../code/scripts/serve_mcif_target_event_authoring.py`](../code/scripts/serve_mcif_target_event_authoring.py)。
服务强制绑定 localhost，启动时重验 input/config hashes、355 rows 和 173 张 image 的路径、
bytes 与 dimensions；每次保存使用同一个 protocol validator，并以 `0600` 权限原子替换 working
sheet。

当前 working sheet：

```text
/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/target-author-01.jsonl
```

初始化状态为 0/355；初始 SHA256 为
`ff62879f651ac4335d3b70cb465f10b4efd30fd73530271abd8a8afd5c694bd6`。它是可变工作文件，
不是 SoT；只有 freeze output 才能上传 HF 并进入 Git manifest。

启动命令：

```bash
cd code
.venv/bin/python -m scripts.serve_mcif_target_event_authoring \
  --input-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_target_event_author_workspace_v1/author_view/annotation_items.jsonl \
  --expected-input-sha256 fff1aea0b0d34b8dc2627e4b16c2bdf6bfd09252d4cbe50f5ac85e2ceee3a4d8 \
  --workspace-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_target_event_author_workspace_v1/author_view \
  --working-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/target-author-01.jsonl \
  --annotator-id target-author-01 \
  --config configs/mcif_target_event_annotation_v1.json \
  --expected-config-sha256 95b8dc69cca4736e31ce1f0fe4c2306cd8155d54d46bdfce0039db5d50660b9c \
  --expected-items 355 --host 127.0.0.1 --port 43871
```

当前服务地址：<http://127.0.0.1:43871/>。Desktop/mobile 浏览器检查均通过：真实 slide
加载、0 horizontal overflow、responsive single-column layout、0 console error/warning。

## Freeze

实现：
[`../code/scripts/mcif_target_event_annotation.py`](../code/scripts/mcif_target_event_annotation.py)。
Freeze 要求 355/355 rows completed、显式 UTC lock time 和运行时 working SHA。命令模板：

```bash
cd code
WORKING=/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/target-author-01.jsonl
WORKING_SHA="$(shasum -a 256 "$WORKING" | awk '{print $1}')"
.venv/bin/python -m scripts.mcif_target_event_annotation freeze \
  --input-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_target_event_author_workspace_v1/author_view/annotation_items.jsonl \
  --expected-input-sha256 fff1aea0b0d34b8dc2627e4b16c2bdf6bfd09252d4cbe50f5ac85e2ceee3a4d8 \
  --working-sheet "$WORKING" --expected-working-sha256 "$WORKING_SHA" \
  --scorer-mapping /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_target_event_author_workspace_v1/scorer_private/item_mapping.jsonl \
  --expected-mapping-sha256 ea21e7d90fab4d7ffece012473ab67d504acce89bb1afdf8c285b878e1a8920e \
  --annotator-id target-author-01 \
  --locked-at-utc <YYYY-MM-DDTHH:MM:SSZ> \
  --config configs/mcif_target_event_annotation_v1.json \
  --expected-config-sha256 95b8dc69cca4736e31ce1f0fe4c2306cd8155d54d46bdfce0039db5d50660b9c \
  --output-root <new-private-output-directory> --expected-items 355
```

Freeze 只产生 `TARGET_EVENT_AUTHORED_PENDING_AUDIO_SUFFICIENCY`。所有
`audio_insufficient_until_sec`、`audio_first_sufficient_sec` 与 `primary_eligible` 继续为 null。
只有独立 audio-only validation 完成后，才允许编译 `SourceEventTiming` 和 state-to-event packets。

## 实现与测试

- Workspace builder：
  [`../code/scripts/build_mcif_target_event_author_workspace.py`](../code/scripts/build_mcif_target_event_author_workspace.py)；
- Annotation validator/freezer：
  [`../code/scripts/mcif_target_event_annotation.py`](../code/scripts/mcif_target_event_annotation.py)；
- Local UI：
  [`../code/scripts/serve_mcif_target_event_authoring.py`](../code/scripts/serve_mcif_target_event_authoring.py)；
- Tests：
  [`../code/tests/test_build_mcif_target_event_author_workspace.py`](../code/tests/test_build_mcif_target_event_author_workspace.py)、
  [`../code/tests/test_mcif_target_event_annotation.py`](../code/tests/test_mcif_target_event_annotation.py)、
  [`../code/tests/test_serve_mcif_target_event_authoring.py`](../code/tests/test_serve_mcif_target_event_authoring.py)。

当前全套测试：`322 passed`，仅有两个既有 `pypinyin` deprecation warnings。Protocol/UI
实现 commit：`b6cd276`。
