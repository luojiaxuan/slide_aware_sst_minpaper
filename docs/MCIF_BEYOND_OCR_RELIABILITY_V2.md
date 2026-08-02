# MCIF Beyond-OCR Reliability V2

日期：2026-08-01

状态：**安全加固后的 protocol、localhost UI/server、signed run contract、per-role
capability authentication、external event-head ledger、freeze/release、reliability gate 与
adjudication 已实现并通过独立复审。旧 HF workspace 在 0 条人工标注时废止；新的 production
workspace 尚未构建，当前等待六个真实且互不重合的 role identities 与 scorer-private access
token manifest。**

## Source of Truth

- Git implementation：
  [`main@8a263a6c`](https://github.com/luojiaxuan/slide_aware_sst_minpaper/commit/8a263a6c04794be330496bcf0752f11dbfa4a008)；
- config：`code/configs/mcif_beyond_ocr_reliability_v2.json`，SHA256
  `e3978733f1f1e0520b54185880a6246d5d85dbe2463988dae097023a23d98d7b`；
- security review：
  [`MCIF_BEYOND_OCR_SECURITY_REVIEW_20260801.md`](MCIF_BEYOND_OCR_SECURITY_REVIEW_20260801.md)；
- historical private HF workspace：
  [`eb194d83/beyond_ocr_reliability_workspace_v2`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/eb194d83c941838db2b096fe52c5e455c5b304bb/beyond_ocr_reliability_workspace_v2)，
  tag `mcif-beyond-ocr-reliability-workspace-v2`；
- historical machine manifest：
  `data/manifests/mcif_beyond_ocr_reliability_workspace_v2_20260801.json`。

上述 HF revision 的 106 files 已全量回下载并逐字节验证，但它早于 signed contract、role
capabilities 和 external head ledger，状态是
`SUPERSEDED_BEFORE_LABELS_UNAUTHENTICATED_UNSIGNED_CONTRACT`。它保留作 provenance，不能发给
annotator，也不能继续写入。人工标签数为 0，所以不存在迁移、合并或丢弃标签的问题。

## 冻结的实验结构

| View | 数量 | 初始可见内容 | 初始不可见内容 |
| --- | ---: | --- | --- |
| `visual_a_r0_view` | 152 | candidate、R0 flat OCR | R1、pixels、descriptor、references、timing、mapping |
| `visual_b_r0_view` | 152 | 同上，独立 opaque ids/order | 同上；也不能访问 A 的 subtree |
| `target_author_view` | 152 | candidate、English source、Chinese reference | slide/OCR/VLM、timing、visual labels |
| `target_validator_stage1_view` | 152 | candidate、English source、Chinese reference | author identity、author labels、author scoring text、visual evidence |
| `scorer_private` | 152 mapping + 91 PNG | future releases、真实 join、contract/ledger | 不得分发给任一 annotator |

Visual A/B 都完整覆盖 152 candidates，并固定按 `R0 -> R1 -> pixels -> descriptor` 顺序
freeze/release。Target author 与独立 bilingual validator 先分别锁定 eligibility/alignment；两侧
都 freeze 后才向 validator 释放 author text，进入 `accept/edit/reject`。六个 production roles
必须使用 registry 中互不重合的真实 identity。

## 安全边界

每次 production build 都生成 HMAC-signed run contract，绑定 exact config bytes、source hashes、
builder commit、identity registry、六个 role token hashes、private bundle hashes、expected item
count 和 required roles。Server 只监听 `127.0.0.1` 并通过 SSH tunnel 使用：匿名 GET 不签发
session；只有匹配当前 role 的 256-bit token 才能换取 `HttpOnly; SameSite=Strict` cookie；API、
JavaScript 和保存接口都需要认证，写入还检查 localhost Origin、form content type 与 payload
唯一性。

每个 role/stage 的 event log 外部配一个 scorer-private HMAC head ledger。Ledger 以 append-only
checkpoint 绑定 event count、完整 event-prefix hash、previous checkpoint HMAC、input/role/stage/
annotator/contract；append 和 freeze 都同时验证 event chain 与当前 ledger head。恢复较早的合法
event prefix 会失败，event 写入后 ledger 未更新的 crash 也 fail closed。

这个机制只防 annotator-side tampering。能同时写 event log 与 scorer-private ledger 的攻击者
可以一起回滚；持有 HMAC key 的攻击者可以伪造所有签名。因此 HMAC key、identity registry、
raw access tokens、event head ledgers 和 mutable event logs 都只能保存在 scorer-controlled
`0600` 本地目录，不能提交 Git、上传 HF 或交给 annotator。Token 在 run 后销毁/轮换；server
不得暴露到公网或反向代理。

## Reliability 与仲裁

Pre-adjudication report 对每个 primitive field 输出 fixed-order confusion matrix、exact agreement、
category-specific agreement、Gwet AC1、Cohen kappa 和 talk-cluster percentile bootstrap 95% CI。
项目级 gate 在 labels 前冻结为：

- 每个 load-bearing field exact agreement `>=0.80`；
- 每个 load-bearing field Gwet AC1 `>=0.67`；
- `requires_adjudication / 152 <=0.25`。

任一 primitive disagreement、任一 `uncertain`、target `edit/reject` 都触发 role-specific
adjudication。Gate 失败时程序拒绝生成 adjudication tasks，必须修 guideline 并在新 epoch 全量
重标。Gate 通过后的 adjudication 保留 raw rows/hashes 和 raw metrics；`unresolvable` 保持
missing，不能转成 negative。

## Production Build

先由 scorer 为六个角色分别生成 token，并只把每个 token 交给对应 annotator：

```bash
cd code
PYTHONPATH=. .venv/bin/python -m scripts.mcif_beyond_ocr_reliability \
  init-access-token --output /scorer-private/tokens/<role>.token
```

使用 `data/templates/mcif_beyond_ocr_identity_registry_v2.example.json` 与
`data/templates/mcif_beyond_ocr_access_token_manifest_v2.example.json` 作为 schema 示例；不得把
example identities 用于 production。完成真实 registry/manifest 后记录各自 exact file SHA256，
再在 `code/` 下执行：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.build_mcif_beyond_ocr_reliability_workspace \
  --source-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1 \
  --output-root /scorer-private/create-once-production-workspace \
  --expected-items 152 \
  --expected-visual-sha256 bcc84ab17ae797a5c185d545aa2c0fefa7a6c146c56607325f967b0bf834efad \
  --expected-target-sha256 c55e3a5a0d883fa2ab679ff4d440c12e3b2ede65b52541980e09f3163494f133 \
  --expected-mapping-sha256 d81a6ed026a1bfd300bfae82e37e232c4b7943f082751b52463ab76c8381cb3d \
  --source-hf-revision 861401f295ab122e69c4f22820b8d501e891e6db \
  --config configs/mcif_beyond_ocr_reliability_v2.json \
  --expected-config-sha256 e3978733f1f1e0520b54185880a6246d5d85dbe2463988dae097023a23d98d7b \
  --builder-git-commit 8a263a6c04794be330496bcf0752f11dbfa4a008 \
  --hmac-key /scorer-private/reliability-v2.key \
  --identity-registry /scorer-private/identity-registry.json \
  --expected-identity-registry-file-sha256 "$IDENTITY_REGISTRY_SHA256" \
  --access-token-manifest /scorer-private/access-token-manifest.json \
  --expected-access-token-manifest-file-sha256 "$ACCESS_MANIFEST_SHA256"
```

`init-events`、`append-event`、`freeze` 与 localhost server 都强制接收 `--head-ledger`、signed
`--run-contract`、identity registry 和 exact hashes。`release-visual`、
`release-target-stage2`、`report`、`prepare-adjudication`、`apply-adjudication` 也重新验证 signed
contract，不信任调用方提供的缓存状态。

## 当前 Firewall

当前状态是 `WAITING_FOR_PRIVATE_IDENTITY_AND_ACCESS_TOKEN_REGISTRY`。在新的 signed production
workspace 生成、上传到新的 immutable private HF revision、远端 bytes 验证并完成六角色盲标前，
不得生成 audio task、event packet、MCIF inference input、paper gold 或 `pixels > OCR` result。
