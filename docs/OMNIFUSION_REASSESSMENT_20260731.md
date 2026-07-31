# OmniFusion 复核：任务先例，不是强方法基线

更新日期：2026-07-31

## 结论

OmniFusion 证明了“speech+image 可以被接入 SimulST pipeline”，所以本项目不能
声称首次研究 image-aware SimulST。但它没有证明 raw image 相对 OCR/text
context 的必要性，也没有提供有竞争力的低时延结果。对本项目而言，它是需要
引用和超过的 **weak task precedent**，不是阻断论文空间的强 prior。

## 论文实际做了什么

- 模型把 Qwen2.5-Omni-7B 和 Seed-X-PPO-7B 连接成约 14B 的 fused system。
- SimulST 使用固定 `1/1.5/2/2.5/3s` audio chunks 和 Local Agreement，在 MCIF
  English→German/Italian 上画 quality-latency curves。
- 论文宣称的“约快 1 秒”是 E2E OmniFusion 相对作者构造的
  `Omni + SeedX` ASR→MT cascade，不是加入 image 相对 audio-only 的收益。
- image 在同一次生成中与 audio 一起经过 Qwen2.5-Omni，再把 hidden states
  注入 Seed-X。它处于同步 decoding path，不是预先异步编码的 slide memory。

## 时延问题

Figure 3 的 computation-aware Average Lag 大致为：

- OmniFusion：约 `5.5–10s`；
- 作者的 cascade：约 `6.5–14s`。

这不属于有说服力的低时延 SimulST operating region。“快 1 秒”只是相对一个
更慢的自建 cascade，不能支持“视觉帮助降低时延”。同一图中，image marker
通常向更高 AL 移动；论文也明确承认 image processing introduces additional
delay。

Table 7 的 offline English→German timing 更直接：

| System | Image | Time | XCOMET-XL |
| --- | --- | ---: | ---: |
| Omni + SeedX FT | no | 2.96s | 89.75 |
| Omni + SeedX FT | yes | 3.85s | 90.18 |
| OmniFusion | no | 1.98s | 88.09 |
| OmniFusion | yes | 3.15s | 89.90 |
| OmniFusion self-cascade | no | 3.78s | 88.44 |
| OmniFusion self-cascade | yes | 4.82s | 90.05 |

对 OmniFusion，image 令 inference time 从 `1.98s` 增至 `3.15s`，增加约
`59%`；加 image 后质量仍低于带 image 的 fine-tuned cascade（`89.90` vs
`90.18`）。论文没有报告该 timing 的 inference hardware、方差或置信区间。

## 视觉证据没有被识别

论文缺少下列决定性控制：

- strong OCR/text-only context；
- oracle slide transcription 或 text-equivalent；
- same-talk wrong image、cross-talk wrong image、shuffled image；
- image relevance、visual grounding、copy/hallucination audit；
- noisy/accented speech 分层。

其公开 processor 的 audio+image prompt 直接要求模型使用 image OCR 来拼写
keywords 和 names。因此现有增益完全可能来自 OCR-like text extraction；论文
没有实验把 OCR 与真正 beyond-OCR visual semantics 分离。

公开模型卡还明确说明模型面向 relatively clean/single-speaker speech，未针对
multi-speaker noisy audio ST 训练。这避开了视觉 phonetic/semantic evidence
最可能显示价值的 failure regime。

## 流式复现不足

截至公开仓库 commit `23ddeeed1df37f50d05d1e42ed3b1c9ea1bab464`：

- 仓库提供训练、单次 inference 和 demo server；
- 没有 Figure 3 对应的 SimulEval runner、chunk policy、AL scoring、实验 config
  或数值结果文件；
- 公开 inference API 每次请求重新执行 multimodal preprocessing 和 model
  generation，没有展示跨 audio chunks 的 image embedding cache；
- server 收集请求后仍逐条调用 `system.translate`，不是实际 batch generation。

因此无法从公开代码复现“约快 1 秒”，也无法确认论文评估是否做过未发布的
跨 chunk cache。我们只能确认公开实现没有提供该能力，不能断言作者内部代码
一定没有。

## 对本项目定位的影响

OmniFusion 与本项目的重合应降为 **Level 3 — Medium Overlap / partial
overlap**：

- 相同：speech+image SimulST、scientific-talk application；
- 不同：它研究 fused architecture 相对 cascade，我们研究 raw evidence 相对
  strong text/audio proxy 的 causal necessity；
- 不同：其公开 inference path 同步处理 multimodal input，未展示跨 chunk
  image reuse；我们要求 slide evidence 可异步预取，并报告端到端
  computation-aware cost；
- 不同：它没有 matched wrong-evidence 和 OCR-equivalent controls。

仍然不能写：

> 首个 speech+image SimulST。

可以争取的 claim 是：

> 首个在低时延 causal SimulST 中，用 strong OCR/oracle text-equivalent、matched
> wrong vision 和异步成本核算，判断 raw slide evidence 何时真正必要的系统研究。

该 claim 还需要继续搜索其他 2025–2026 工作，并由预注册实验支撑。

## Sources

- Paper: <https://arxiv.org/abs/2512.00234>
- Public code: <https://github.com/saikoneru/OmniFusion>
- Model card: <https://huggingface.co/skoneru/OmniFusion>
