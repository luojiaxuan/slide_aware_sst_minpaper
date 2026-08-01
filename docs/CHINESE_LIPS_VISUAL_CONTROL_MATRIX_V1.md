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
  streaming items，并把 next-prefix CPU preprocessing 与 current GPU generation 重叠，
  共两个 resumable shards。batch/prefetch 属于已记录的执行参数，不改写五个
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

`PIPELINED_RUNNER_READY_FOR_RELAUNCH`。前四次尝试均已停止且没有 worker 残留；已有
部分输出只作吞吐诊断，不能误报为正式结果或与新两 shard run 混合。

已完成的 launch preparation：

- initial launch state：`main@2fb0410`；batched runner state：`main@6b73121`；
  Hyper00 detached worktree：
  `/data/projects/slide_aware_sst_minpaper/worktrees/visual-controls-v1`；
- run root：
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
- heartbeat automation `vision-sst-control-launch` 已暂停旧四 worker launch，待新 run
  root 启动后改写为两张 GPU、每卡一个 batched worker 的
  utilization/progress/error monitor；完成后自动 analysis、private HF upload 和 Git freeze。

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

当前代码保持 `workers_per_gpu=1`、`batch_items=16`、`shard_count=2`，新增
`prefetch_next_batch=true` 双缓冲；launcher 继续用独立进程组并在
`SIGTERM`/`SIGINT` 后统一清理。本地 probe/control focused tests 为 `7 passed`，其中
同步测试证明 next processor batch 在 current generation 返回前已启动。下一次使用全新
run root，重新做 10 秒采样；只有双卡平均利用率达到 `90%+` 才持续运行到 1030 rows。
