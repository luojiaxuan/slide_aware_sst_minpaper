# MCIF Beyond-OCR Reliability V2

日期：2026-08-01

状态：**base workspace、状态机、freeze/release、pre-adjudication reliability gate 与
role-specific adjudication 已实现；真实 152-item workspace 已上传 private HF。当前仍是 0
human labels，只有 R0 visual views、target author view 和不含 author text 的 target-validator
stage-1 view 被释放。Annotation server/UI 尚未完成，因此不得开始 production annotation。**

## Source of Truth

- Git implementation：`main@2aa9b6a5d3b9251e9dec0f0a78d89292ce6fd6f7`；
- metrics implementation：`6e04e25a415a8dc6d664f249af82f73dbb8c6a7e`；
- config：`code/configs/mcif_beyond_ocr_reliability_v2.json`，SHA256
  `5886b1a45364ba7f9c577148f8eaca97e5cfcc2343342fbb1f24dbe468f7cd75`；
- private HF：
  [`eb194d83/beyond_ocr_reliability_workspace_v2`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/eb194d83c941838db2b096fe52c5e455c5b304bb/beyond_ocr_reliability_workspace_v2)；
- tag：`mcif-beyond-ocr-reliability-workspace-v2`；
- machine manifest：
  `data/manifests/mcif_beyond_ocr_reliability_workspace_v2_20260801.json`。

## 初始物理边界

| View | 数量 | 初始可见内容 | 初始不可见内容 |
| --- | ---: | --- | --- |
| `visual_a_r0_view` | 152 | candidate、R0 flat OCR | R1、pixels、descriptor、references、timing、mapping |
| `visual_b_r0_view` | 152 | 同上，独立 opaque ids/order | 同上；也不能访问 A 的 subtree |
| `target_author_view` | 152 | candidate、English source、Chinese reference | slide/OCR/VLM、timing、visual labels |
| `target_validator_stage1_view` | 152 | candidate、English source、Chinese reference | author identity、author labels、author scoring text、visual evidence |
| `scorer_private` | 152 mapping + 91 PNG | future release material、真实 join | 不得分发给任一 annotator |

机器审计：四个公开初始 views 均 152 items；visual future-field hits=0；validator author-field
hits=0；两个 visual views 内 PNG=0；A/B ids disjoint。完整 106 files 从 HF revision 回下载后与
本地逐字节一致；105 个 checksum entries 全通过；独立第二次构建 byte-identical。

## Visual 状态机

每名 visual validator 完整覆盖相同 152 candidates，阶段固定为：

1. `r0_support`：只看 candidate + R0；
2. `r1_support`：A/B 的 R0 全量 freeze 后才释放 R1；
3. `pixel_support`：A/B 的 R1 全量 freeze 后才释放 raw slide；
4. `descriptor_fidelity`：A/B 的 pixel 全量 freeze 后才释放 source-only descriptor。

每次 next-stage release 同时验证两侧完整 freeze、HMAC、相同 candidate set、disjoint identities、
per-item predecessor hash 与 cohort lock。不能按早期答案过滤 items。每个 annotation submission 是
server-signed append-only event；stale version、completed overwrite、普通 row rehash 和 event-chain
tampering 都会失败。

## Target 状态机

Target author 定义 eligibility、canonical English event、acceptable/forbidden Chinese
realizations 和 reference alignment。独立 bilingual validator 的 stage 1 先锁定自己的
eligibility/alignment；只有 author 与 validator stage 1 都完整 freeze 后，stage 2 才释放 author
text 并要求 `accept/edit/reject`。Author、validator 和 visual A/B identities 必须归一化后互异。

## Reliability 与仲裁

Pre-adjudication report 对每个 primitive field 输出 fixed-order confusion matrix、exact agreement、
category-specific agreement、Gwet AC1、Cohen kappa 和 talk-cluster percentile bootstrap 95% CI。
Kappa 在 constant marginal 时显式为 undefined，不伪造为 0 或 1。

项目级 instrument gate 在 labels 前冻结为：

- 每个 load-bearing field exact agreement `>=0.80`；
- 每个 load-bearing field Gwet AC1 `>=0.67`；
- `requires_adjudication / 152 <=0.25`。

任一 primitive disagreement、任一 `uncertain`、target `edit/reject` 都触发 role-specific
adjudication。**Gate 失败时程序拒绝生成 adjudication tasks，必须修 guideline 并在新 epoch 全量
重标。** Gate 通过后，adjudication 生成新 hash-bound rows，保留 raw row/hash；不得重算或覆盖
raw AC1/kappa。`unresolvable` 保持 missing，不能转成 negative。

## Rebuild

在 `code/` 下执行：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.build_mcif_beyond_ocr_reliability_workspace \
  --source-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1 \
  --output-root /path/to/create-once-output \
  --expected-items 152 \
  --expected-visual-sha256 bcc84ab17ae797a5c185d545aa2c0fefa7a6c146c56607325f967b0bf834efad \
  --expected-target-sha256 c55e3a5a0d883fa2ab679ff4d440c12e3b2ede65b52541980e09f3163494f133 \
  --expected-mapping-sha256 d81a6ed026a1bfd300bfae82e37e232c4b7943f082751b52463ab76c8381cb3d \
  --source-hf-revision 861401f295ab122e69c4f22820b8d501e891e6db \
  --config configs/mcif_beyond_ocr_reliability_v2.json \
  --expected-config-sha256 5886b1a45364ba7f9c577148f8eaca97e5cfcc2343342fbb1f24dbe468f7cd75 \
  --builder-git-commit 2aa9b6a5d3b9251e9dec0f0a78d89292ce6fd6f7
```

`scripts.mcif_beyond_ocr_reliability --help` 提供 `init-key`、`init-events`、`append-event`、
`freeze`、`release-visual`、`release-target-stage2`、`report`、`prepare-adjudication` 和
`apply-adjudication`。HMAC key 是 scorer-private local secret；不得提交 Git、上传 HF 或分发给
annotators。

## 当前 Firewall

V2 尚未经过真实 human instrument gate，不存在 gold beyond-OCR event。不得生成 audio task、
event packet、MCIF inference input 或 `pixels > OCR` result。下一步是实现 token-protected localhost
server/UI，做浏览器 projection/E2E 审计，再由六个 disjoint role identities 进入 production flow。
