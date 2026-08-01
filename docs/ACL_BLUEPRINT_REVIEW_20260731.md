# ACL Blueprint Cross-Review

日期：2026-07-31

对象：[`ACL_PAPER_BLUEPRINT_20260731.md`](ACL_PAPER_BLUEPRINT_20260731.md)

状态：**历史 Route A review。当前单一 paper story 与更新后的 collision boundary 见
[`PAPER_STORY_DECISION_20260731.md`](PAPER_STORY_DECISION_20260731.md)。**

方式：主 agent 完成 primary-paper audit 后，由独立只读 agent 以 ACL reviewer 立场审计。

## 审计前 verdict

**Weak Reject，接近 Reject。** 方向本身仍有空间，但初稿同时堆叠 benchmark、cache、
policy、beyond-OCR 四个贡献，并存在 representation confound、public-test 反复调参、
synthetic-noise overclaim 和 multiple testing 风险。五个 ACL60/60 dev talks 只能做
futility screen，不能支撑 talk-cluster confirmatory inference。

## Rejection-level findings 与处理

| Finding | 修订 |
| --- | --- |
| Phase A 只有 5 个独立 talks，不能“通过 G0” | Phase A 改为 futility/debug/MDE screen；G0/G1 只由 one-shot MCIF run 判定 |
| Strong OCR 与 structured semantics 同时改变内容、抽取器和长度 | 改为 nested R0→R4；R2/R3 使用同一 VLM/prompt/schema/decoding，唯一差异是 pixels |
| 与 LECTRANS、MLLP-VRAIN、Do Slides 的 delta 只在文字里 | 唯一 primary contribution 冻结为 causal live timing + image-specific incremental-information benchmark/finding；逐项纳入其最强条件 |
| MCIF 是 public IWSLT dev，不是真 blind test | 如实称 project-held-out；增加 frozen-run commit、no-reference inference、append-only ledger、completion guard；优先补 organizer blind evaluation |
| 多模型/语言/noise/slice 允许事后挑结果 | 冻结 En→Zh、Qwen3-Omni、+5 dB babble、XCOMET-XL；H1→H2→H3 sequential gatekeeping；subtypes Holm correction |
| Synthetic noise 不足以支持 noise-robust title | 标题去掉 noise-robust；无 record-and-replay 时只声称 controlled acoustic corruption；增加 VAD-aware SNR 与 noise-source split isolation |

## Major findings 与处理

| Finding | 修订 |
| --- | --- |
| 贬低 OmniFusion 且 per-chunk image 是弱效率 baseline | 删除跨论文 latency judgment；增加同协议 R5 cached visual/KV；R6 per-chunk 只作 cost ablation |
| Extractor leakage firewall 不完整 | 新增字段级输入白名单；PDF、audio、transcript、reference、annotation、future frame 均禁止进入 current-slide extractor |
| 跨 MuAViC/LRS3 的 lip 数字不可比较 | 主任务明确限定 slide-conditioned SimulST；不再放跨数据集 lip appendix ranking |
| 贡献栈过多 | benchmark/protocol + empirical finding 是唯一 primary；cache 是设施；policy 仅为 conditional secondary |

## 修订后仍存在的硬风险

1. MCIF 仍是公开 reference 的 project-held-out，不等价于 blind test；必须争取 IWSLT
   organizer evaluation 或另一个 licensed independent corpus。
2. 21 个 test talks 对 1.0 XCOMET-XL minimum effect 是否有足够 power 尚未知；必须在
   看 MCIF 输出前用 dev variance 做 simulation-based MDE。
3. `image_needed` slice 的密度可能很低。若 R3-vs-R2 有效样本不足，不能改用事后
   cherry-picked subtype。
4. LECTRANS 若在本项目投稿前发布 code/data 或新增 raw-speech streaming baseline，
   novelty 需要重新审计。
5. 没有 record-and-replay 时，论文只能回答 controlled corruption，不能外推到真实
   conference/cafe noise。

## 当前 reviewer stance

修订后的 contract 已消除最明显的设计性拒稿点，但**还没有实验，所以不能判断为
Accept-ready**。Phase A 的唯一作用是低成本停止明显无效的路线；完整 ACL claim 需要
H1 content specificity、H2 image-specific increment、talk-level MDE 和至少一个独立
replication 同时成立。
