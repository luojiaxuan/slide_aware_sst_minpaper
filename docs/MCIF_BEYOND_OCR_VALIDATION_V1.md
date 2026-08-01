# MCIF Beyond-OCR Independent Validation V1

日期：2026-08-01

状态：**SUPERSEDED BEFORE LABELS。152 个 R1/R2 proposals、两个物理隔离 role views、
validator/UI/freeze/join 均保留为 provenance 与 calibration artifact，但 v1 不能开始 production
annotation，不能产生 paper gold，也不能进入 audio/inference。Visual/target working sheets 在
supersession 时均为 0/152，43872/43873 已停止。原因与 v2 replacement contract 见
[`MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md`](MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md)。**

## 这一步回答什么

这一步只回答两个互相独立的问题：

1. slide pixels 中提议的结构/语义 evidence 是否真实存在，而且当前 R0 flat OCR（R2 还包括
   R1 structured text）确实不足；
2. 对应 English event 是否可稳定计分，并在 Chinese reference 中有明确 realization。

只有两个 frozen role gates 同时通过的 candidate 才能进入后续 audio-only sufficiency
validation。自动 Qwen3-VL 描述不能成为 label；R2 的 150 proposals 包含明显泛化噪声，当前
数量不能解释为 event density 或 raw-image benefit。

## 冻结输入

- Candidate inventory：R1 2 rows + R2 150 rows；private HF revision
  [`01defe41`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/01defe410b4fde07c647d8ed241dfbe501b5d691/beyond_ocr_candidate_inventory_v1)；
- Validation workspace：private HF revision
  [`861401f2`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/861401f295ab122e69c4f22820b8d501e891e6db/beyond_ocr_validation_workspace_v1)，
  tag `mcif-beyond-ocr-validation-workspace-v1`；
- Visual input：152 rows，SHA256
  `bcc84ab17ae797a5c185d545aa2c0fefa7a6c146c56607325f967b0bf834efad`；
- Target input：152 rows，SHA256
  `c55e3a5a0d883fa2ab679ff4d440c12e3b2ede65b52541980e09f3163494f133`；
- Scorer mapping：152 rows，SHA256
  `d81a6ed026a1bfd300bfae82e37e232c4b7943f082751b52463ab76c8381cb3d`；
- Protocol config：
  [`../code/configs/mcif_beyond_ocr_validation_v1.json`](../code/configs/mcif_beyond_ocr_validation_v1.json)，
  SHA256 `d25f558c2010331d1219c65d7b6192edd3972151b49809435efb08821f6d69b6`。

Workspace 是 102 files / 32,709,288 bytes，包含 91 张去重 native PNG；独立 rebuild
byte-identical，HF 远端 102 files 全量回下载逐字节验证通过。

## 角色隔离

`visual_validator_view` 只显示 slide、R0、clean R1 blocks、candidate 与 proposed evidence；
不含 source/target reference、talk/segment/state id、timing、lead 或 scorer mapping。

`target_author_view` 只显示 candidate、English source segment 与 Chinese reference；不含
slide、OCR、evidence tier、VLM description、timing、lead 或 scorer mapping。两个 role 使用不同
opaque item ids，不能由 annotator 侧 join。

`scorer_private` 保存唯一真实关联，不能交付给任一 annotator。两位 annotator 不能交换 role
subtree、working sheet 或 frozen labels。后续 audio-only validator 也不能访问任一 role view、
candidate、reference、slide/OCR、mapping 或 role labels。

## Gate 定义

Visual role 使用 `yes/no/uncertain`：

- `visual_evidence_correct=yes`；
- `candidate_supported_by_visual_evidence=yes`；
- `r0_insufficient=yes`；
- 对 R2 额外要求 `r1_insufficient=yes`。

任一非 `yes` 的 completed row 必须给 reason。R1 row 不接受伪造的 R1-insufficiency label。

Target role 通过条件：

- `candidate_eligibility=yes`；
- non-empty canonical English event；
- 至少一个 acceptable Chinese realization；
- alignment 为 `explicit` 或 `paraphrased`。

Rejected/uncertain row 必须给 reason，并清空 scoring text。Pending row 不允许 partial label。
Working/input/config/media/hash/annotator 任一漂移均 fail closed。

## 当前 UIs

实现：
[`../code/scripts/serve_mcif_beyond_ocr_validation.py`](../code/scripts/serve_mcif_beyond_ocr_validation.py)。
服务强制绑定 localhost，每次保存复用 protocol validator，并以 `0600` 原子替换 role-specific
working sheet。

Visual validator：

```bash
cd code
.venv/bin/python -m scripts.serve_mcif_beyond_ocr_validation \
  --role visual \
  --input-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1/visual_validator_view/validation_items.jsonl \
  --expected-input-sha256 bcc84ab17ae797a5c185d545aa2c0fefa7a6c146c56607325f967b0bf834efad \
  --workspace-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1/visual_validator_view \
  --working-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/beyond-ocr-visual-01.jsonl \
  --annotator-id visual-01 \
  --config configs/mcif_beyond_ocr_validation_v1.json \
  --expected-config-sha256 d25f558c2010331d1219c65d7b6192edd3972151b49809435efb08821f6d69b6 \
  --expected-items 152 --host 127.0.0.1 --port 43872
```

Target author：

```bash
cd code
.venv/bin/python -m scripts.serve_mcif_beyond_ocr_validation \
  --role target \
  --input-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1/target_author_view/annotation_items.jsonl \
  --expected-input-sha256 c55e3a5a0d883fa2ab679ff4d440c12e3b2ede65b52541980e09f3163494f133 \
  --workspace-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1/target_author_view \
  --working-sheet /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/beyond-ocr-target-01.jsonl \
  --annotator-id target-01 \
  --config configs/mcif_beyond_ocr_validation_v1.json \
  --expected-config-sha256 d25f558c2010331d1219c65d7b6192edd3972151b49809435efb08821f6d69b6 \
  --expected-items 152 --host 127.0.0.1 --port 43873
```

历史服务地址为 <http://127.0.0.1:43872/> 与 <http://127.0.0.1:43873/>，已在 v1
supersession 时停止，不得重启用于 production labels。此前 Desktop
1280×720 与 mobile 390×844 浏览器审计通过：真实 1920×1080 slide 加载；role 字段隔离正确；
0 horizontal overflow；所有 mobile controls 保持在 361 px 内；console 0 error/warning。审计未
保存标签，两个 progress 仍为 0/152。

## Working Sheets

```text
/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/beyond-ocr-visual-01.jsonl
/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/working/beyond-ocr-target-01.jsonl
```

两者均为 mutable local state、mode `0600`，不是 SoT。初始 SHA256 分别为
`b74cc5d15d66a4a647a1961ea6e1dd13d2f0e7e8c5706e939a39a5e2d9e439be` 与
`81afa11dc846303bd703d73508cfe176bb89217fd327b10cfbb07c3576c9644a`。

## Freeze 与 Join

实现：
[`../code/scripts/mcif_beyond_ocr_validation.py`](../code/scripts/mcif_beyond_ocr_validation.py)。
每个 role 只有在 152/152 completed 后才能 freeze；`--output-root` 必须是不存在的新目录。
Freeze 命令模板：

```bash
cd code
CONFIG_SHA=d25f558c2010331d1219c65d7b6192edd3972151b49809435efb08821f6d69b6
WORKING=<role-working-sheet>
WORKING_SHA="$(shasum -a 256 "$WORKING" | awk '{print $1}')"
PYTHONPATH=. .venv/bin/python -m scripts.mcif_beyond_ocr_validation \
  --config configs/mcif_beyond_ocr_validation_v1.json \
  --expected-config-sha256 "$CONFIG_SHA" freeze \
  --role <visual-or-target> --input <matching-role-input> \
  --expected-input-sha256 <matching-role-input-sha256> \
  --working "$WORKING" --expected-working-sha256 "$WORKING_SHA" \
  --annotator-id <matching-annotator-id> \
  --locked-at-utc <YYYY-MM-DDTHH:MM:SSZ> \
  --output-root <new-private-role-freeze-directory> --expected-items 152
```

两个 create-once freezes 完成后，scorer 才能执行 join：

```bash
cd code
PYTHONPATH=. .venv/bin/python -m scripts.mcif_beyond_ocr_validation \
  --config configs/mcif_beyond_ocr_validation_v1.json \
  --expected-config-sha256 d25f558c2010331d1219c65d7b6192edd3972151b49809435efb08821f6d69b6 join \
  --visual-input <visual-validation-items.jsonl> --expected-visual-input-sha256 bcc84ab17ae797a5c185d545aa2c0fefa7a6c146c56607325f967b0bf834efad \
  --target-input <target-annotation-items.jsonl> --expected-target-input-sha256 c55e3a5a0d883fa2ab679ff4d440c12e3b2ede65b52541980e09f3163494f133 \
  --mapping <scorer-private-item-mapping.jsonl> --expected-mapping-sha256 d81a6ed026a1bfd300bfae82e37e232c4b7943f082751b52463ab76c8381cb3d \
  --visual-frozen <frozen-visual-annotations.jsonl> --expected-visual-frozen-sha256 <sha256> \
  --target-frozen <frozen-target-annotations.jsonl> --expected-target-frozen-sha256 <sha256> \
  --joined-at-utc <YYYY-MM-DDTHH:MM:SSZ> \
  --output-root <new-private-joined-directory> --expected-items 152
```

Join 会重新计算两个 gates 并拒绝重哈希后的语义篡改。Joint pass 只得到
`BEYOND_OCR_VISUAL_TARGET_VALIDATED_PENDING_AUDIO_SUFFICIENCY`；
`audio_insufficient_until_sec`、`audio_first_sufficient_sec` 和 `primary_eligible` 仍为 null。

## 实现与测试

- Workspace builder commit：`9da94a2e53f5cb612d21c65296917131b64d58ef`；
- Protocol/freezer commit：`d3a710e2a40bb71be5c0fa4b766982676d3c1883`；
- Role UI commit：`ffd960c04aacd06b86a276b97ce27b4c28b42af3`；
- Protocol/server targeted tests：`17 passed`；
- Project suite：`357 passed`，两个既有 `pypinyin` deprecation warnings。

本协议不再有 production next gate。后续只能执行 reliability audit 中定义的 v2：visual A/B
顺序锁定、target author + bilingual validator、role-specific adjudication 与 instrument go/no-go。
V2 freeze 前不生成 audio task，不编译 event packets，不启动 MCIF ST inference。
