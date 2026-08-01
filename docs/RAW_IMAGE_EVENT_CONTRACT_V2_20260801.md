# Raw-image Event Contract V2

日期：2026-08-01

状态：**代码与测试已闭环；304-state evidence ladder 尚未物化。**

## 为什么需要 v2

v1 event contract 只允许 `document / OCR / semantic / relation` 文本 packet。即使这些文本
来自 slide，它也不能检验 paper 的核心问题：raw visual evidence 是否在强 OCR 或结构化文本
之外提供 SimulST 独有的提前 target decision。v2 保留完整 v1，不修改已有结果，新增：

- `correct_image`：当前 causal state 的 native slide frame；
- `matched_wrong_image`：与当前帧在视觉 token budget 上严格匹配的错误帧；
- `image_content_specificity = correct_image - matched_wrong_image`；
- `image_over_relation = correct_image - correct_relation`；
- `relation_over_ocr = correct_relation - ocr`。

这三个增量分别回答“模型是否使用正确图像内容”“pixels 是否超过结构化文本”“结构化文本
是否超过 flat OCR”。只比较 image 与 audio-only 不能回答 `why not OCR`。

## Fail-closed 绑定

每个 image packet 同时冻结：

1. image artifact JSON 的相对路径与 SHA256；
2. artifact 内 native image 的相对路径与 SHA256；
3. packet JSON、固定 textual marker token IDs 与 processor 展开后的 visual token count；
4. source artifact tree SHA256；
5. host source root 到 worker source root 的精确只读 mount；
6. worker 命令中的 `--source-artifact-root`。

worker 在模型加载前和全部 generation 后重算 source tree hash。路径逃逸、symlink traversal、
artifact 漂移、media bytes 漂移、processor/image-token 数量漂移都会停止运行。image 和
non-image 条件分成 homogeneous processor batches，再按输入顺序恢复，避免混合 modality
数量的隐式处理。

## 当前 runtime 边界

当前 runner 直接把 image、prompt text、causal audio prefix 依次送入
`Qwen3OmniMoeProcessor`。它会在每个 audio prefix 上重新执行 image preprocessing/encoding。
因此该路径当前只能作为 raw-image **质量与因果归因诊断**：

- 可以报告 event-time first-stable-correct 与最终质量；
- 必须单独报告实际 runtime / GPU cost；
- 不得宣称 image encoder 已从在线 critical path 移除；
- 不得把 slide 提前 30--60 秒可用等同于实现了 encoder cache。

只有后续实现并单独审计一次编译、多 prefix 复用的 visual cache/compiler，才可以报告
off-path 版本。

## 实现与验证

- schema/validation：`code/src/slidesst/eval/event_timing.py`
- worker 与 Qwen3-Omni image path：`code/src/slidesst/eval/causal_worker.py`
- pre-run mount contract：`code/src/slidesst/eval/inference_contract.py`
- scoring matrix：`code/configs/acl6060_event_trajectory_scoring_v2.json`
- regression：`code/tests/test_image_evidence_v2.py`

全套回归：`235 passed`，只有两个既有 `pypinyin` deprecation warnings。

## 下一步

1. 从同一批 304 个 native causal frames 物化 `R0 flat OCR / R1 structured text / R2 raw image`；
2. 冻结 image processor revision 与每帧 visual token count；
3. 只在相同 visual token count 内构造 deterministic `matched_wrong_image`；
4. 上传 private HF revision，并在 Git manifest 中记录 revision、schema、hash 与生成命令；
5. 再冻结 source-event packet、target scoring 和 native/noisy oracle headroom run。

在第 4 步完成前，v2 只有 Git code contract，没有可复用的数据 artifact，不能启动正式
raw-image inference。
