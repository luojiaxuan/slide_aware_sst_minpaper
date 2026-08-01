# Controlled Acoustic Pipeline Freeze

日期：2026-08-01

状态：**官方 sources 已冻结；dev/confirmatory pools 已隔离；ACL60/60 dev 的 75 个
full-talk variants 已 materialize 并通过 QA。尚未运行任何 ST output。**

## 为什么做这一步

本项目需要检验 visual benefit 是否随 audio 可听性下降而增加，但不能用切分后的 segment
逐条加噪，也不能把合成 corruption 写成真实 noisy conference。当前 contract 把 noise/RIR
施加到完整连续 talk，仅用于 controlled acoustic intervention。Native audio 必须独立报告。

## 官方 sources

| Source | License | Archive | Bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| MUSAN / OpenSLR 17 | CC BY 4.0 | `musan.tar.gz` | 11,086,114,085 | `86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1` |
| Room Impulse Response and Noise Database / OpenSLR 28 | Apache 2.0 | `rirs_noises.zip` | 1,311,166,223 | `3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7` |

官方 MD5、resource-page snapshot hash、URLs、selection salt 和 exact RIR IDs 见
[`../data/manifests/controlled_acoustic_source_contract_20260801.json`](../data/manifests/controlled_acoustic_source_contract_20260801.json)。
两个 archive 分别通过 `gzip -t` 和 `unzip -t`。

## Source pool

从 frozen archives 选择性解压 130 个 files：

| Split | Babble speech | Generic noise | Music | RIR | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | 32 | 16 | 16 | 1 | 65 |
| confirmatory | 32 | 16 | 16 | 1 | 65 |

成员按 `sha256(salt + NUL + category + NUL + upstream_member)` 排序后切分，两个 pools
的 source-id overlap 为 0。Babble speech 至少 500 s；generic noise 至少 100 s；music
至少 300 s。Development RIR 是 REVERB 2014 `mediumroom1_near_angla` channel 0；
confirmatory 使用不同的 `mediumroom2_near_angla` channel 0。

完整 130-file hashes/durations/schema 见
[`../data/manifests/controlled_acoustic_source_pool_20260801.json`](../data/manifests/controlled_acoustic_source_pool_20260801.json)。
本地 persistent cache 是
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/source_pool_v1`。

## Corruption contract

Development matrix 由
[`../code/configs/controlled_acoustic_dev_20260801.json`](../code/configs/controlled_acoustic_dev_20260801.json)
冻结：

- babble：5 个不重复 MUSAN speech tracks，+10/+5/0/-5 dB，每档 3 个 deterministic
  replicates；
- noise-type generalization：generic noise 与 music 各 0 dB、1 replicate；
- reverberation：固定 development real RIR；
- 每个 `talk_id × condition_id` 从 global seed `20260801` 派生独立 64-bit seed；
- 每条 source 记录 exact hash、sample offset、source frames 和 wrap count；
- 每个 source wrap boundary 使用冻结的 10 ms fade-out/fade-in，避免硬拼接点击；
- mixture 超过 0.99 peak 时 clean/noise 同比例衰减，保持 SNR 不变；
- RIR 裁去 5%-peak onset 之前的 pre-trigger（保留 1 ms margin），做 L2 normalization、
  active-RMS match，并截断回原 talk duration，避免把 archive alignment silence 算成
  SimulST latency。

ACL60/60 inference view 不含 source timing，所以 SNR 使用 `energy_vad_v1`：25 ms frame、
10 ms hop、threshold = `max(-50 dBFS, p95 RMS - 15 dB)`。该阈值仅根据五个 clean dev
waveforms 的 energy distribution 校准，不读取 transcript/reference。MCIF inference view
已有 source-side intervals 时，脚本使用 intervals union。

## ACL60/60 dev QA

75 个输出覆盖 5 talks × 15 conditions：

- 每 talk 15/15 conditions；75 个 output SHA256 和 75 个 condition seeds 均唯一；
- duration 与 clean talk 完全一致，范围 577.24–737.44 s；
- 14 个 additive-noise conditions 的最大 `|achieved-target SNR|` 为
  `4.9e-7 dB`；
- 从实际 PCM16 输出恢复 clean/noise 分量后复算 70 个 additive files，最大
  `|encoded-achieved-target SNR|` 为 `0.00027 dB`；
- PCM saturated samples 为 0；
- energy-active fraction 为 49.6%–88.8%；
- development RIR 的 frozen onset trim 为 2,184 samples（136.5 ms pre-trigger）；
- portable manifest 不含本地 absolute path、transcript、translation、reference 或 term
  annotations。

Portable manifest：
[`../data/manifests/acl6060_dev_controlled_acoustic_v1_20260801.jsonl`](../data/manifests/acl6060_dev_controlled_acoustic_v1_20260801.jsonl)。

本地 staging：
`/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/corruptions/acl6060_dev_v1`
（约 1.5 GiB）。Canonical reusable copy 是 private HF dataset
[`gavinlaw/slide-aware-sst-controlled-acoustic-dev`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-controlled-acoustic-dev)，
immutable commit `d28c499c8845c4991b5ccea27bc9a2ad520f51fa`，tag
`acl6060-controlled-acoustic-v1-20260801`。远端清单为 75 WAV、5 个
metadata/card files 和 Hub 自动 `.gitattributes`；tag 回下载的五个 metadata files
与抽样 WAV 均逐字节匹配本地 staging，repo privacy 已验证为 `private`。

## Reproduction

```bash
cd code

PYTHONPATH=. .venv/bin/python scripts/prepare_controlled_acoustic_sources.py \
  --contract ../data/manifests/controlled_acoustic_source_contract_20260801.json \
  --musan-archive /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/openslr17/raw/musan.tar.gz \
  --rir-archive /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/openslr28/raw/rirs_noises.zip \
  --output-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/source_pool_v1 \
  --portable-summary-out ../data/manifests/controlled_acoustic_source_pool_20260801.json \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/noise/source_pool_v1

PYTHONPATH=. .venv/bin/python scripts/materialize_full_talk_corruptions.py \
  --inference-manifest /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/phase_a/views/acl6060_dev/inference.jsonl \
  --source-pool /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/source_pool_v1/source_pool.jsonl \
  --conditions configs/controlled_acoustic_dev_20260801.json \
  --source-split development \
  --output-root /Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/corruptions/acl6060_dev_v1 \
  --portable-manifest-out ../data/manifests/acl6060_dev_controlled_acoustic_v1_20260801.jsonl \
  --portable-staging-label ResearchStudio/data/vision-aware-sst/noise/corruptions/acl6060_dev_v1 \
  --workers 6
```

## 科学边界与下一步

当前产物只证明 corruption pipeline 可用，不证明 slides 有效。下一步仍是构建 stripped
ACL real-frame timeline、blind 标注 source-side evidence opportunities，并先跑 gold/oracle
headroom。只有 correct current evidence 相对 matched stale/wrong evidence 提前稳定正确 target
decision、final quality 不退化，且 benefit 随 acoustic degradation 有可解释 interaction，
才值得投入 automatic VLM compiler 和大规模 GPU inference。
