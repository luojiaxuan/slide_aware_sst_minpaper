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
- execution：Hyper00 两张 H200，每张先尝试两个独立 worker，共四个 resumable
  shards；如果实测 OOM 或持续低利用率，保留失败 run 并用新 run root 调整，
  不改写本合同的科学参数。

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

`READY_WAITING_HYPER00_GPU_CAPACITY`。尚未启动模型 worker，也没有生成任何 condition
output；不能误报为 running 或 complete。

已完成的 launch preparation：

- Git launch state：`main@2fb0410`；Hyper00 detached worktree：
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
- heartbeat automation `vision-sst-control-launch` 每 10 分钟执行正式 preflight。出现两张
  `<1 GB` 空卡后，使用恰好两张 GPU、每卡两个 worker 启动；若 run 已启动则转为
  utilization/progress/error monitor，完成后自动 analysis、private HF upload 和 Git freeze。

最近一次 preflight 返回 0 free GPUs；无 eligible idle container 可清理。完成后在本文件继续
记录 launch-time GPU snapshot、selected GPU ids、HF result revision、完整性检查和数值结论。
