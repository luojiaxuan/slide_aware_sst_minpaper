# MCIF Beyond-OCR Reliability V2 Security Review

日期：2026-08-01

结论：**在 scorer-private filesystem、HMAC key 与 six-role registry 可信的明确威胁边界下，
最终独立复审为 `NO BLOCKER`。旧 zero-label workspace 不满足该边界，已废止。**

## Review 过程

### Round 1：Artifact 与 release binding

交叉审计首先发现 config/registry/private bundle、nested R1/descriptor/media schema 和
adjudication input binding 不够严格。实现随后改为 exact nested schema、content hash、
create-once artifact、symlink/traversal rejection、signed release report，以及 descriptor
fidelity 的 load-bearing gate。Config、registry、private material 和 downstream releases 全部
进入 signed run contract，不再由可变路径名或普通 row hash 隐式决定。

### Round 2：P0/P1 攻击面

- `P0 role impersonation`：旧 server 可向任意访问者签发 role session。修复后匿名 GET 只返回
  login；六个 role 各有独立 256-bit capability token，server 只接受当前 role 的 exact token，
  session cookie 为 `HttpOnly; SameSite=Strict`，API/JS/save 均要求认证；login/save 都验证
  localhost Origin 和严格 content type/payload。
- `P1 valid-prefix rollback`：只验证 event 内部 HMAC chain 时，攻击者可恢复一个较早但合法的
  prefix。修复后增加独立 scorer-private `mcif_beyond_ocr_event_head_checkpoint_v2` ledger，逐次
  append HMAC checkpoint 并绑定完整 prefix hash。旧 prefix 与当前 ledger head 不一致，append
  和 freeze 都拒绝。

### Round 3：最终独立复审

复审重新执行 token/login、unauthenticated access、signed-prefix rollback、ledger append/freeze
与 role-isolation tests，结论为 `NO BLOCKER`。该结论不扩大威胁模型：它只保证 annotator
无法在没有 scorer secret/filesystem 权限时 impersonate role、覆盖 completed row 或回滚事件。

## Verification

- targeted tests：`34 passed in 1.35s`；
- full repository tests：`397 passed in 13.40s`；
- `ruff check --select F,B023,I`、`ruff format --check`、`py_compile`、`git diff --check`：通过；
- Browser E2E：desktop `1280x720` 与 mobile `390x844` 无 horizontal overflow；真实 login 后
  draft 跨 reload 保留，completed row immutable，controls disabled；anonymous API 被拒绝；
- signed rollback attack：恢复旧 event bytes、保留当前 ledger，append/freeze 均拒绝；
- review commit：
  [`8a263a6c`](https://github.com/luojiaxuan/slide_aware_sst_minpaper/commit/8a263a6c04794be330496bcf0752f11dbfa4a008)。

## Residual Risks

- bearer token 被分享或 annotator browser 被攻破时，仍可冒充对应角色；production 后必须销毁或
  轮换 token；当前实现没有自动 expiry/rate limiting；
- server 必须只监听 loopback 并通过 SSH tunnel 使用，不能放到 public reverse proxy；
- 同时拥有 event log 与 scorer-private ledger 写权限的攻击者可整体回滚二者；
- 持有 HMAC key 的攻击者可伪造 contract、events、ledger、freeze 和 release；
- event append 已落盘但 ledger checkpoint 未落盘的 crash 会 fail closed，需要 scorer 按审计
  记录人工恢复，不能自动忽略不一致。

这些风险决定了 production secret/filesystem 的运维边界，不构成降低 protocol gate 的理由。
