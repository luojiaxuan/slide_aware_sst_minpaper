# MCIF Beyond-OCR Positive Validation V1

日期：2026-08-02

状态：`DO_NOT_ANNOTATE_PENDING_STRICT_CANDIDATE_RESCREEN`

## 范围

本 workspace 只包含 exploratory screen 中 clean 与 `babble_p5_s0` 都满足
`raw_image > OCR AND raw_image > wrong_image` 的 6 条 robust positives。它用于确认样本级
提升是否真由 pixels 中的 beyond-OCR evidence 驱动，不重新测试 aggregate hypothesis，也不能
把通过的个例外推成整体 `vision > OCR`。

配置为 `code/configs/mcif_beyond_ocr_positive_validation_v1.json`，SHA256
`b83fa7eb20917d55d2643757340e32deb14c785e768b03d75c4fb494c6a6e0b8`；builder 为
Git `941f24807664b9a38122c8fd32e1c5e134ca1fc6`。

## 四角色隔离

必须由四个不同的人完成，任何人不得同时承担两个角色：

| role | 可见内容 | 不可见内容 | rows |
| --- | --- | --- | ---: |
| `visual_a` | candidate、flat OCR、随机化 image A/B | references、model outputs、correct/wrong identity | 6 |
| `visual_b` | 同上，独立 item/image ordering | 同上 | 6 |
| `outcome_a` | candidate、En source、Zh reference、随机化四路 prefix trajectories | OCR、images、condition identity | 6 |
| `outcome_b` | 同上，独立 item/condition ordering | 同上 | 6 |

visual validator 必须先判断 OCR support，再分别判断两个 image slots 是否足以恢复 candidate；
outcome validator 对 clean/noisy 下每个匿名 slot 记录最早正确 candidate realization、final
correctness 与 unsupported content。真实 image/condition mapping 只在 `scorer_private/`。

任何 disagreement 或 `uncertain` 都进入后续 adjudication，不能按多数票或自动规则抹掉。当前
24 个 role items 和 24 个 working rows 均为 `pending`，没有 annotator id 或人工判断。

2026-08-02 的 post-hoc candidate audit 发现原 screen 的 full-string exclusion 不足以保证 OCR
insufficiency：34 条中 10 条的全部 candidate tokens 已分别出现在 OCR 中，21 条 token coverage
至少为 50%。因此本 packet 只保留为 zero-label provenance，不再请求四位 annotators 开始工作。
必须先完成 strict beyond-OCR re-screen；不得把当前 6 条当作 gold positives。

## Source of Truth

- Git：<https://github.com/luojiaxuan/slide_aware_sst_minpaper/tree/941f24807664b9a38122c8fd32e1c5e134ca1fc6>
- source screen：private HF revision
  [`d6b01409`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/d6b01409200379bc22b1aecd485d5ebc4fe2b4d1/beyond_ocr_exploratory_screen_v1)；
- validation packet：private HF revision
  [`34e8f9b1`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/34e8f9b16ef06dc9503066d3446304400c750c22/beyond_ocr_positive_validation_v1)，
  tag `mcif-beyond-ocr-positive-validation-v1`；
- local canonical build：
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/outcomes/mcif_beyond_ocr_exploratory_screen_v1_863e6af8_b2/formal_validation/mcif_beyond_ocr_positive_validation_v1_b83fa7eb_g941f248`。

远端 38 个实际 files 已全量回下载；`SHA256SUMS` 覆盖的 37 files 全部逐字节通过，repo
visibility 确认为 private。每个 annotator 只能收到自己 role directory 的独立副本，不得授予
完整 HF repo 或 `scorer_private` 权限。

## 下一步

1. 先按 strict beyond-OCR contract 重筛候选，并完成 OCR-insufficiency prescreen；
2. 再由用户提供四个真实且互不重合的 annotator identities；
3. 为四个 role 分别复制/分发对应目录，记录 identity assignment；
4. 收回四份完整 working sheets，先冻结 raw labels，再 scorer-side 解盲；
5. 生成 disagreement queue，交给未参与原始判断的 adjudicator；
6. 只有 visual evidence、OCR insufficiency 和 candidate-relevant outcome 都通过的样本，才进入
   audio-sufficiency 与 stable-commit analysis。
