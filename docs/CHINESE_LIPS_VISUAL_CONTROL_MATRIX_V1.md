# Chinese-LiPS Speech-Vision Control Matrix v1

## 目的与边界

这是一项 **private story diagnostic**，不是 paper-grade 主结果。它只回答已有
Qwen3-Omni speech+image 结果为何同时被 current slide 和 same-talk wrong slide
改善，不能替代 ACL60/60 或 MCIF 上的多 talk、人类 reference 实验。

历史 206-segment 实验缺少两个关键控制，并且输出未记录 immutable model
revision。v1 因此在同一锁定模型版本上完整重跑五个条件，不把新条件拼接到旧
结果：

| Condition | 输入 | 识别的效应 |
| --- | --- | --- |
| `none` | audio only | speech-only baseline |
| `slide` | current slide | 全部视觉上下文效应 |
| `wrong` | same-talk wrong slide | talk/domain priming，去掉局部页对应关系 |
| `cross_talk` | ACL60/60 English scientific slide | 结构化 scientific-slide 视觉槽位，去掉同 talk/domain |
| `blank` | 1280×720 white image | vision encoder / prompt-slot perturbation |

预先声明的诊断对比是：

1. `slide - wrong`：within-talk page specificity；
2. `wrong - cross_talk`：broad talk/domain priming；
3. `cross_talk - blank`：structured slide 相对 generic vision slot；
4. `blank - none`：加入 vision encoder/prompt slot 本身的扰动；
5. `slide - none`、`wrong - none` 和 `slide - cross_talk`：与历史发现衔接。

## 冻结输入

- Chinese-LiPS probe：private HF
  `gavinlaw/chinese-lips-speech-slide-probe`，revision
  `6eb395afb46e8dbe05e79b590243979186aa3f1f`，206 items；
- cross-talk frames：private HF
  `gavinlaw/slide-aware-sst-acl6060-source-event-author-v2`，revision
  `bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a`，只读取 100 张 frame，绝不读取
  scorer mapping、问题、答案或标签；
- model：`Qwen/Qwen3-Omni-30B-A3B-Instruct`，revision
  `26291f793822fb6be9555850f06dfe95f2d7e695`；
- deterministic cross-image assignment：
  `sha256("cross-talk-control-v1:" + item_id) mod 100`；
- blank image：固定白色 RGB PNG，builder 在 run root 生成并记录 SHA256；
- streaming：1.0 s audio chunk、两次连续 hypothesis 的 Local Agreement、
  greedy decoding、`max_new_tokens=96`、seed 0；
- execution：Hyper00 两张 H200。`attempt 2/3/4` 逐步定位为模型副本数、batch 和
  processor gap 问题；当前实现是每卡一个模型 worker、进程内动态维持 16 个 active
  streaming items，并用独立 spawned process 把 next-prefix CPU preprocessing 与
  current GPU generation 重叠，共两个 resumable shards。batch/prefetch 属于已记录的执行参数，不改写五个
  conditions、streaming policy、模型 revision 或样本集合。

机器可读配置：
[`code/configs/chinese_lips_visual_controls_v1.json`](../code/configs/chinese_lips_visual_controls_v1.json)。

## 完整性与分析

- 预期 `206 × 5 = 1,030` 条唯一 `(id, condition)` 记录；
- 每个 item 的五个条件必须具有相同 reference 和非空 immutable model revision；
- runner 只有在全部 shard 行数正确时才写 `completion.json`；
- analyzer 报告 corpus chrF、mean sentence chrF、AL seconds、wall time，以及
  paired segment bootstrap 95% CI；
- 由于 206 items 全来自同一个 talk，segment bootstrap CI 只能描述该 talk，
  不能伪装成 talk-level statistical inference；
- reference 是已有 machine draft，因此该实验只用于决定下一轮控制和实现优先级。

## 决策规则

- 若 `slide > wrong`：优先研究 local current-slide content selection；
- 若 `slide ≈ wrong > cross_talk`：主要机制更像 talk/domain priming，下一步应做
  document-context 和 current-slide 的分层集成；
- 若 `wrong ≈ cross_talk > blank`：收益可能来自 generic structured-slide prior；
- 若 `blank > none` 且其他 image conditions 彼此接近：现有正效应主要是视觉槽位或
  decoding perturbation，不能作为 semantic vision 证据；
- 无论结果如何，都不以本单 talk diagnostic 选择最终 paper claim；ACL dev 的
  source-event oracle/noise matrix 仍是主线。

## 状态

`COMPLETE_PRIVATE_DIAGNOSTIC_NO_PAGE_SPECIFIC_SIGNAL`。前五次尝试均已停止且没有活跃
worker；其部分输出只作吞吐诊断，不能与当前两 shard run 混合。

## 结果与结论

最终五条件点估计为：

| Condition | corpus chrF | mean AL (s) |
| --- | ---: | ---: |
| `none` | 79.080 | 2.637 |
| `slide` | 80.852 | 2.516 |
| `wrong` | 80.740 | 2.465 |
| `cross_talk` | 80.506 | 2.592 |
| `blank` | 80.625 | 2.519 |

10,000 次 paired segment bootstrap 的关键对比是：

| Contrast | Δ corpus chrF [95% CI] | Δ AL seconds [95% CI] |
| --- | ---: | ---: |
| `slide - none` | +1.772 [ +0.752, +2.782 ] | -0.121 [ -0.237, -0.001 ] |
| `wrong - none` | +1.659 [ +0.546, +2.726 ] | -0.172 [ -0.274, -0.068 ] |
| `slide - wrong` | +0.113 [ -0.693, +0.894 ] | +0.051 [ -0.045, +0.152 ] |
| `wrong - cross_talk` | +0.234 [ -0.792, +1.444 ] | -0.127 [ -0.244, -0.010 ] |
| `cross_talk - blank` | -0.119 [ -1.205, +0.824 ] | +0.073 [ -0.042, +0.189 ] |
| `blank - none` | +1.545 [ +0.644, +2.482 ] | -0.118 [ -0.216, -0.021 ] |

这个结果否定了“当前正确页内容驱动已有增益”的解释：`slide` 不优于 same-talk
`wrong`，structured `cross_talk` 也不优于 `blank`，而空白图相对 audio-only 已产生几乎
同量级的 chrF/AL 变化。当前最简解释是 vision encoder / prompt-slot / decoding
perturbation；page-specific semantics、talk/domain priming 和 structured-slide prior 均没有在
本探针中得到可分辨支持。这里的区间仍只是单 talk、machine-reference 描述，不能写成总体
无效性结论。

因此停止把 Chinese-LiPS naive raw-image prompting 当作 fresh paper-grade generation 的
依据。下一步转到 ACL60/60 多 talk 的 source-event 设计，用 correct current evidence 相对
time/type/token-budget-matched wrong evidence 测 earlier stable target decision，并在完整 talk
controlled noise 下估计 content-specific interaction。

Canonical private artifact：

- HF repo：`gavinlaw/slide-context-sst-chinese-lips`；
- revision：`4923b253e87bd94487dace77576ad66e4ea9d8b9`；
- tag：`chinese_lips_visual_controls_v1_qwen3_omni_20260801_canonical`；
- path：`experiments/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801/`；
- analysis SHA256：`c7be55fe293ae9f96b1a0efb269c12dcee9b03da9c9b2fa9de2621b8c58f0bf6`；
- packaging commit：`9a564b538cb054b0a15504917916680e2720d07d`。

首次 HF commit `d0c44533745f3f9be23d4b51c1df3d32059d441f` 及无 `canonical` 后缀的
tag 仅因 card metadata 与 README-hash coverage 被取代；不要作为 canonical pointer。最终
revision 已验证 `private=True`、tag 指向一致、10 个远端文件逐字节匹配本地 bundle，且不含
JPG/PNG/WAV/MP4 原始媒体。

已完成的 launch preparation：

- initial launch state：`main@2fb0410`；batched runner state：`main@6b73121`；
  Hyper00 detached worktree：
  `/data/projects/slide_aware_sst_minpaper/worktrees/visual-controls-v1`；
- 首次 materialization/失败尝试的 run root：
  `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_2gpu_20260801_132051`；
- 206-row control input SHA256：
  `66783dcda6d34e81bd8f1197cea29b6d7de815d422574379d3189b7bd1e24105`；
- control manifest SHA256：
  `1feccff001d937496a8d53ea7e7d9bf259278dc20e5ace48a9d160bc295c9df4`；
- frozen config SHA256：
  `2332893d6588221ee7eac8e39cbba775b03722164e8d7de88aec073954d094d6`；
- exact model snapshot 从 Hyper01 historical cache 直接传到 Hyper00 host
  `/data01/jaxan/hf_cache/hub`。25-file snapshot SHA256 清单在 source/target 间为零差异；
  同一 cache 已复制到 canonical container
  `/dev/shm/qwen3_omni_hf_cache_26291f/hub`，persistent-host 与 runtime snapshot
  SHA256 清单同样为零差异；
- Hyper00 canonical container 仍是 `sglang-omni-jaxan`，未创建第二个 compute
  container，也未修改或停止活跃 OSWorld/diffusion 任务；
- heartbeat automation `vision-sst-control-launch` 曾监控当前两张 GPU、每卡一个
  batched worker 的 run；运行、analysis、private HF upload 和 Git freeze 均已完成。

当前正式 diagnostic run 在 Hyper00 preflight 确认全部八张卡均低于 `1 GB` 后，只选择
GPU `0/1`，从 clean detached `main@f76f922` 启动：

```text
/data/projects/slide_aware_sst_minpaper/runs/
chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801_153400
```

首个正式 10 秒窗口为 GPU0 `89.2%`、GPU1 `94.0%`、双卡均值 `91.6%`；第二个窗口为
`88.1%/96.6%`、双卡均值 `92.35%`，均通过 `90%+` continuation gate。最终两 shard
`515/515 = 1,030/1,030`，completion 绑定 input、两份 output 和 immutable model revision
SHA256；所有 worker 已退出，日志无 traceback、OOM、IPC、broken pipe 或
resource-sharer error。10,000 次 paired bootstrap analysis 已完成；analyzer 先增加
contrast-level process parallelism，再缓存 SacreBLEU segment sufficient statistics，完整
`206 × 7 × 10,000` 串行基准约 2.8 秒，且与朴素重采样逐项一致。

2026-08-01 14:39 UTC preflight 曾返回 GPU `3/5/6/7` 空闲，按合同只选择 `3/5` 启动。
四个 worker 在加载模型前因 image 缺少 `accelerate` 退出，0 output rows；原始失败保留在
`supervisor_attempt1_missing_accelerate.log` 和 `worker_*.log`。已在 canonical container 安装
`accelerate==1.14.0`。

`attempt 2` 随后仍在 GPU `3/5` 以四 worker 启动并成功生成 81 条记录。两个正式
10 秒 `nvidia-smi dmon` 窗口的双卡均值分别为 `85.0%` 和 `69.4%`，低于用户要求的
`90%`；每卡两个完整模型副本已占约 `125.6 GiB`，无法再加第三个 worker，低谷主要来自
逐 item 串行 processor/audio-prefix 准备和不同 prefix 长度。该 attempt 因低利用率主动
停止；supervisor 被停止后暴露出四个 orphan worker，已按精确 PID 清理，GPU `3/5` 回到
`1 MiB`。四个 partial shard 最终行数为 `23/19/20/19`，全部原样保留在旧 run root。

`attempt 3` 使用每卡一模型、`batch_items=4`，GPU `3/5` 正式 10 秒均值为
`36.5%/37.8%`，最终保留 `16/12` partial rows。`attempt 4` 提高到
`batch_items=16`，正式均值提高到 `78.0%/90.9%`，双卡 `84.45%`，最终保留
`91/91` partial rows；GPU 3 峰值已到约 `132 GiB`，因此不盲目尝试 batch 32。
两个 run 分别保存在：

- `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_batch4_2gpu_20260801_150926`；
- `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_batch16_2gpu_20260801_151240`。

`attempt 5` 在 `main@a7e6c96` 使用 Python thread prefetch；正式 10 秒窗口仅为 GPU
`3/5 = 80.2%/76.1%`、双卡 `78.15%`，比无 prefetch 的 `84.45%` 更差，说明
processor Python/GIL 调度没有被掩盖。该 run 最终保留 `37/35` partial rows：
`/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_prefetch16_2gpu_20260801_151900`。

当前运行保持 `workers_per_gpu=1`、`batch_items=16`、`shard_count=2`，使用
`prefetch_mode=process`：spawned CPU process 单独加载 processor，通过 multiprocessing
shared-memory tensor transport 返回 CPU `BatchFeature`，主 worker 只做 H2D 和 generate；
它不会继承已有 CUDA context。launcher 继续用独立进程组并在 `SIGTERM`/`SIGINT` 后统一
清理。本地最终完整测试为 `148 passed`；analysis、bundle validation、private HF upload
和 immutable-revision byte verification 均已通过。
