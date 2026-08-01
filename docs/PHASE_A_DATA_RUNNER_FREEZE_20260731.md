# Phase-A Data and Runner Freeze

日期：2026-07-31
状态：**数据与 runner revision 已冻结；`C0-C2` contract-driven launcher 已实现，
`C1-C2` packets 与 `C3` causal retriever 尚未完成，GPU inference 尚未启动。**

## 结论

1. ACL60/60 官方 attachment 已完整下载并校验，不再依赖 RASST mirror。dev/eval
   各 5 个完整 talk，共 468/416 个 gold segments。
2. MCIF 必须区分两个范围：当前 HF revision 是 100-talk long-media pool；官方
   IWSLT/translation subset 只有 21 talks。后者 talk IDs 已从 IWSLT ZIP 的 `audio/`
   与 `pdf/` 文件名交叉冻结。
3. MCIF ZIP 只检查了目录、hash 和 talk IDs；`ref/*.txt` 内容没有读取。ACL60/60
   推理视图不含 transcript、reference 或 tagged terminology 路径，评分视图独立。
4. 官方 IWSLT baseline 可复现 `C0` 和 entity-to-ASR 型 context 起点，但其当前代码
   把 MT context 硬编码为空，不能直接代表本项目的 `C1-C3`。本仓库因此只依赖其
   exact commit，并用薄 adapter 显式控制 ASR/MT context、talk 顺序和 token budget。
   `C3` 必须在每个 decision point 用 causal ASR prefix 查询预翻译 PDF index；该
   retriever 未实现前 launcher 会 hard fail，不能用静态 talk string 冒充 BM25/RAG。

## ACL60/60 freeze

| 字段 | 值 |
| --- | --- |
| 官方页面 | <https://aclanthology.org/2023.iwslt-1.2/> |
| archive | `2023.iwslt-1.2.dataset.zip` |
| SHA256 | `5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc` |
| 大小 | 948,169,976 bytes |
| license | CC BY 4.0 |
| dev | 5 talks / 468 gold segments / 468 En-Zh text lines |
| eval | 5 talks / 416 gold segments / 416 En-Zh text lines |
| media format | mono 16 kHz WAV；talk-level full audio + gold/SHAS segment audio |

Eval source XML 在 recover parser 下记录 4 个 syntax errors；talk IDs 和 segment/text
line counts 仍严格一致。脚本和测试必须继续使用 `lxml` recover mode，并保留 error
count，不应静默改写官方 XML。

本地 staging：

- archive：`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/raw/2023.iwslt-1.2.dataset.zip`
- extracted root：`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/extracted/2/acl_6060`
- dev papers：`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/dev_papers`
- dev inference view：`.../phase_a/views/acl6060_dev/inference.jsonl`，SHA256 `6c1a40be...7ef0e2`
- dev scoring view：`.../phase_a/views/acl6060_dev/scoring.jsonl`，SHA256 `7286a3ae...4eb6af`

`build_acl6060_simulstream_inputs.py` 通过在 full WAV 中定位每个 gold segment WAV
的精确 PCM slice 重建 468 条 offset/duration。相邻 gold segments 可能重叠，因此
搜索 cursor 只要求 start time 单调，不错误地要求前一 segment 已结束。当前 dev
scoring bundle：

- `source.en.txt`：468 lines，SHA256 `aa37f443...5452d1`；
- `reference.zh.txt`：468 lines，SHA256 `0b1e3249...4f4f72`；
- `audio-segments.yaml`：468 rows，SHA256 `ed18ddd0...c241b2`。

## MCIF freeze

| 范围 | Revision/hash | 冻结结果 |
| --- | --- | --- |
| HF media pool | `FBK-MT/MCIF@e24065b919758263cfe5d157057278affe76ea7b` | 1,725 files，100 个 long audio/video talk IDs，约 8.13 GB |
| IWSLT translation subset | `mcif-long-trans.zip`, SHA256 `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7` | 21 audio/PDF talk IDs，`audio-segments.yaml` 与四语 reference filenames 存在，reference contents unopened |

本地 ZIP staging：
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/iwslt2026/mcif-long-trans.zip`。
它是官方上游缓存，不重新上传 HF。21-talk ID 列表见
[`../data/manifests/phase_a_sources_20260731.json`](../data/manifests/phase_a_sources_20260731.json)。

## Runner freeze

| 组件 | Revision |
| --- | --- |
| IWSLT 2026 baseline | `owaski/iwslt-2026-baselines@7e2974bb3c850fde9bd62f3fa3103f9f345a56d0` |
| SimulStream | `hlt-mt/simulstream@86e0f1ed82cb59018515b246e115e298bb0bd7da` (`v0.3.0-7`) |
| OmniSTEval | `pe-trik/OmniSTEval@b275fac51796e82178aac088c2a57068fe822afc` (`v0.1.10`) |
| ASR | `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5` |
| forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7` |
| MT | `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` |
| context compiler | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@5a5a776300a41aaa681dd7ff0106608ef2bc90db` |

Primary policy 冻结为 960 ms audio chunks、5.0 s initial wait、character-level
Chinese latency；640/960/1280 ms 只用于 latency sweep。上游 baseline 在该 commit
没有 LICENSE file，因此不得把其源码复制进本仓库；运行 exact external checkout，
本仓库只保留 `slidesst.iwslt_context_adapter.PhaseAContextSpeechProcessor`。

Adapter contract：

- JSON packet 顺序必须与 WAV list 完全一致；`talk_id == Path(wav).stem`；
- 每个 talk 必须严格包含 `C0-C3`；
- `C0` 的 ASR/MT context 和 `source_ids` 必须为空；
- ASR 与 MT channel 各自最多 256 model tokens（combined ceiling 512）；bundle
  必须记录两个 tokenizer revisions 与预计算 token counts，adapter 重新核对 MT count；
- 任何 `reference`、`source_transcript` 或 `tagged_terminology` key 立即失败；
- `C1` 只能使用 paper/PDF 自动抽取的 term memory，官方 tagged terms 只用于评分。

完整配置见 [`../code/configs/phase_a_c0_c3.yaml`](../code/configs/phase_a_c0_c3.yaml)。

## Reproduction

```bash
cd code
.venv/bin/python scripts/freeze_phase_a_sources.py \
  --acl-archive /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/raw/2023.iwslt-1.2.dataset.zip \
  --acl-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/extracted/2/acl_6060 \
  --acl-paper-dir /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/dev_papers \
  --mcif-revision e24065b919758263cfe5d157057278affe76ea7b \
  --mcif-iwslt-archive /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/iwslt2026/mcif-long-trans.zip \
  --out-dir ../data/manifests

.venv/bin/python scripts/build_acl6060_simulstream_inputs.py \
  --acl-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/acl6060/extracted/2/acl_6060 \
  --inference-view /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/phase_a/views/acl6060_dev/inference.jsonl \
  --split dev \
  --inference-out /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/phase_a/simulstream/acl6060_dev/inference \
  --scoring-out /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/phase_a/simulstream/acl6060_dev/scoring
```

`prepare_phase_a_run.py` 是唯一允许的 launcher；它默认只 prepare，显式加
`--execute` 才运行 inference。它需要 automatic context bundle、三个 exact external
checkouts、只含 pinned snapshots 的 `HF_HOME` 和持久化 output root。`C3` 当前会在
任何 runtime/model 检查前直接拒绝。

Canonical Git manifests：

- [`../data/manifests/phase_a_sources_20260731.json`](../data/manifests/phase_a_sources_20260731.json)
- [`../data/manifests/acl6060_talks_20260731.jsonl`](../data/manifests/acl6060_talks_20260731.jsonl)
- [`../data/manifests/acl6060_critical_files_20260731.jsonl`](../data/manifests/acl6060_critical_files_20260731.jsonl)
- [`../data/manifests/mcif_files_e24065b9.jsonl`](../data/manifests/mcif_files_e24065b9.jsonl)
- [`../data/manifests/acl6060_dev_simulstream_20260731.json`](../data/manifests/acl6060_dev_simulstream_20260731.json)

## 下一步

1. 从五篇 ACL dev PDF 自动生成 `C1-C2` packets，不使用 official tags/references。
2. 先生成 `C1-C2` static packets，再实现 `C3` causal ASR-prefix retriever；当前
   launcher 会拒绝 C3。
3. 在 GPU host materialize exact model snapshots；先做一个 talk 的 C0/C1 dry run，
   检查 talk order、增量 commits、log completeness、RTF 和 context token logs。
4. dry run 通过后跑五个 dev talks 的 `C0-C3`，再决定是否实现 `C4-C6`。
