# ACL60/60 Event Trajectory Scoring v1

## 目的

本合同把论文核心问题变成可执行 estimand：当 source audio 仍被两位独立 validator
保守判断为信息不足时，current slide evidence 是否让系统更早形成 **stable correct target
decision**，并且相对 matched wrong evidence 保持 content specificity。

ACL dev 仍是 story discovery，不是 confirmatory result。开发结果只能用于选择 Route A-D
和冻结下一版 held-out contract，不能直接写成 MCIF 或 ACL eval 的预注册结论。

## 十九个冻结输入

评分器 [`../code/scripts/score_event_trajectories.py`](../code/scripts/score_event_trajectories.py)
必须分别读取：

1. `source_events.jsonl`：`event_id`、`talk_id`、`primary_eligible`、evidence availability、
   保守 audio-insufficient boundary、causal endpoint，以及每个非空 condition 预期的
   source media path/hash、extractor identity/revision；
2. `source_artifact_root/`：source-derived evidence JSON 及其 upstream slide/document
   bytes；整棵目录按 relative path + file bytes 固定 SHA256，拒绝 symlink；
3. `evidence_packets.jsonl`：每个 `event × condition` 的 strict source-only payload、
   availability、tokenizer model/immutable revision、token IDs、rendered-text/payload/token
   SHA256；
4. `control_pairs.jsonl`：source-only correct/wrong packet pairing，逐 event 绑定 evidence
   type、availability、token count、packet ID 和 packet SHA256；
5. `scientific_config.json`：strict schema 固定实际 inference model/revision、完整 model
   artifact tree hash、in-process backend、prompt、streaming policy、condition matrix 与 greedy
   decoding 参数；
6. `inference_contract.json`：在任何 hypothesis 生成前冻结 Git checkout、scientific/
   scoring config、source/evidence/control/target/outcome hashes、完整 condition matrix、
   broker identity、worker identity 与 host/container isolation roots；
7. `inference_result_attestation.json`：run 后单向绑定 pre-run contract、exact
   `trajectories.jsonl`、causal release log 和 start/end 两次 environment audit，避免
   trajectory 与 manifest 的循环 hash；
8. `inference_environment_start_audit.json`；
9. `inference_environment_end_audit.json`：两次 capture 都按 shell token 精确匹配含 exact
   run ID 的 root workers，枚举全部 descendants，记录 PID/PPID/cwd、entrypoint/
   executable/environment hashes、PID/PPID/process-start ticks、process-identity tree、
   read-only rootfs、`network=none`、container image、
   Docker mounts、open files、Git commit/dirty status；两次 process tree、container ID/name、
   capture host 与 mount topology 必须一致；
10. `outcome_commitment.json`：pre-output 绑定 source-event 与 target-score hashes，以及
    source/target 的 raw annotation report 和 adjudication 四种角色；
11. `outcome_artifact_root/`：上述四份 scorer-private 原始 outcome artifacts，整棵目录
    bytes hash 固定且拒绝 symlink；
12. `causal_audio_schedule.json`：每个 `talk × acoustic condition` 的 canonical
    `float32le_mono` full PCM path/hash、upstream WAV hash、corruption provenance、materializer
    commit/entrypoint，以及每个 `event × prefix step` 的 source ID、sample boundary 和 prefix
    hash；schedule 还冻结全部 model conditions，供 broker 构造 talk-level 同步 frontier；
13. `causal_audio_broker_audit.json`：外部 prefix broker 的 clean Git commit、实际
    entrypoint bytes、PID、socket、release-events path、exact command 和 source roots；
14. `causal_audio_release_log.json`：server-authoritative prefix release 与 exact-hypothesis
    observation commit 交替记录；每条包含 `session_id/server_ordinal`、monotonic timestamp、
    source/prefix identity 和前一 record hash，完整覆盖每个
    `event × model condition × acoustic condition × prefix step`；
15. `model_artifact_root/`：worker in-process 加载的完整 frozen model snapshot；整棵目录按
    relative path + bytes hash，并与 scientific config、contract、live worker command 和
    exact read-only host→worker mount 五方一致；
16. `tokenizer_artifact_root/`：评分和 inference 共用的 tokenizer-only files；整棵目录按
   bytes 固定 SHA256，不能靠对本地目录无效的 `revision=` 自报版本；
17. `target_scores.jsonl`：独立双语 annotator 冻结的 acceptable/forbidden target
   realizations，不含 condition 或 system output；
18. `trajectories.jsonl`：各 condition/noise 的逐步 hypothesis、pre-run contract hash、
   evidence-packet ID/hash，以及 observation 级 causal prefix ID/hash；
19. `acl6060_event_trajectory_scoring_v1.json`：本文件对应的 strict scoring config。

Target 与四份 outcome artifacts 不得挂载到 inference process。评分时重算所有 contract/
attestation inputs，并核对 inference 与 broker 两个 clean frozen checkout；dirty/untracked
checkout 直接失败。Pre-run contract 固定 target/outcome hashes，post-run attestation 固定
trajectory bytes，因此不能在看到 system outputs 后移动 conservative boundary、改 acceptable
realizations 或重写 hypothesis 而仍通过 scorer。Packet 只允许
`none/empty/document/ocr/semantic/relation` 六类结构化 source
context；每个非空 item 必须按 index/text/hash 绑定 source-derived JSON，JSON 的 media 和
extractor identity 还必须逐 condition 等于 `source_events` 冻结值，因此 future-slide
reindex 或过期 extractor 即使内部 hash 自洽也会失败。Scorer 用固定 renderer 重建输入文本，再以
contract 绑定的 tokenizer identity、40 位 revision 和 tokenizer artifact tree hash 逐 token
重放；修改本地 tokenizer bytes、伪造 token IDs 或把 gold target 直接写进 packet 都不能通过。
Environment audit 分别比较 container destination roots 与 host mount-source roots，拒绝
forbidden root 本身、其子路径或包含它的 parent mount；scorer 还要求实际 target/outcome
路径位于同一组 host-forbidden scoring roots，不再混用 path namespace。Capture 不接受手填
PID；run marker 必须按 shell token 精确包含一次 run ID，
`run-1` 不匹配 `run-10`。它记录 exact-marker workers 及全部 descendants 的 PID/PPID/
process-start ticks/cwd、entrypoint/executable/environment hashes；任意 Python descendant
的 entrypoint 必须位于 contract 绑定的 clean worktree，container rootfs 必须 read-only。
Container 还必须使用 `network=none`，阻断 inference 时从外部取 target/reference 或可变模型；
scientific config 文件也必须通过 exact read-only host→worker mount 暴露，worker command
绑定该路径并在 generation 前重算 config hash。Pre-run contract 直接绑定完整 start-audit
bytes 和 process-identity tree；不同 GPU worker
允许具有不同 environment hash，但 start/end 两次 audit 的整棵 identity tree、container 与
mount topology 必须一致，
中途 restart 不能以相同命令伪装通过。root 目录本身或目录内任意 symlink 都会被 artifact tree
hash 拒绝。这是可重放的
operational isolation audit，不宣称
对恶意伪造运行证据提供密码学证明。评分器要求所有 eligible event 的
target row 一一对应，并要求完整 `event × condition × noise` 矩阵；重复、缺失、未知条件、
时间倒序、提前结束、越过 causal endpoint，或同一 event 的任意视觉/acoustic condition
使用不同 audio-time grid 均直接失败。所有 Pydantic schema 使用 `extra="forbid"`，因此
trajectory 中夹带 target/reference/future-audio 字段不会被静默忽略。

`future_audio_access=false` 不再单独构成证据。已实现的 external broker 在启动时重算 full
PCM/provenance/prefix hashes，只接受冻结 schedule 中每个 session 的下一个 prefix，通过
`0600` Unix socket 返回 raw PCM。每个 prefix 后，worker 必须先提交由 run/contract/event/
condition/prefix/hypothesis bytes 共同计算的 observation hash，broker 才放行下一 prefix；末步
也必须 commit。除此之外，broker 对同一 talk 的全部 event、model condition 与 acoustic
condition 使用一个按 `audio_time_sec` 排序的同步 frontier：当前时间点所有预期 observation
未 commit 前，任何 stream 都拿不到更晚音频。Scorer 重建所有 hash，既强制每个 stream 的
`release_k < observation_commit_k < release_{k+1}`，也强制 talk-level
`max(commit_current_time) < min(release_next_time)`；因此跨 condition/event/noise 先跑完整 talk
再回填另一个流的早期 hypothesis 也无法通过。每个 session 只能绑定一个
`event × model condition × acoustic condition` stream；同一 talk 任一时刻只允许一个
in-flight release，commit 后才释放同一时刻的下一个 stream。Broker 同步写
server-ordinal、timestamped、
hash-chained interaction log。Scorer
再从 source 文件首字节重算每个 prefix，并强制
`sample_count / sample_rate == audio_time_sec`，所以 1 秒 observation 不能携带 100 秒音频。
当前 Hyper00 diagnostic canonical container 仍将整个个人 `/data` 以 writable 方式挂入，
不满足 read-only rootfs + narrow-mount contract；正式 ACL run 必须等当前 workload 结束后
重建同名 canonical container，并把 full PCM/target/outcome roots 留在 host/broker 侧。
Production Qwen3-Omni worker 与 audited shard merger 已实现并通过本地回归；fresh
paper-grade generation 仍被正式 isolation rebuild 与人工 outcome freeze 阻塞；这里没有
ACL system result。

Literal matcher 使用 Unicode NFKC 和 token-boundary matching。`C` 不匹配 `C++`，
`GPT-4/GPT4` 与 `U.S./US` 按技术缩写规则归一；CJK 空格可忽略，但不能跨标点拼接。
这仍是冻结 acceptable/forbidden realization 的 lexical/contrastive scorer，不把任意语义
paraphrase 自动判成正确。

## Primary development estimand

对每个 event/condition，先把每个 streaming observation 判为：包含任一 acceptable
realization 且不包含任何 forbidden realization。`first_stable_correct` 是最早一个从该步到
endpoint 都保持正确、且至少包含两个 observation 的时间。末尾孤立正确点是
right-censored，不算 stable；曾正确后又撤回单独记为 retraction/overcommit。

保守 audio-insufficient boundary 使用两位 audio validator 都仍不足的最后时间，不使用
更宽松的 first-sufficient endpoint。event outcome 为：

```text
I[first_stable_correct <= conservative_audio_insufficient_until]
```

开发期主 estimand 是 `correct_semantic - matched_wrong_semantic` 和
`correct_relation - matched_wrong_relation` 的 **talk-equal risk difference**。OCR、
semantic 和 relation 使用各自 type/time/token-budget-matched wrong control，不能共享一个
信息量不同的 generic wrong packet；pair availability 还必须等于 source event 的 frozen
evidence time。Pooled event risk
difference 同时报告但不是主数值，避免事件多的 talk 主导结果。只有两条件最终都正确且
都有 stable time 的 event 才进入 commit-advance：

```text
t_first_stable_correct(control) - t_first_stable_correct(correct)
```

更早但错误、最后撤回或仅在末尾单点正确都不能制造 latency gain。Final correctness 使用
开发期 final correctness point-estimate floor 为 `-1 pp`，early risk difference 使用 `+5 pp`
practical signal，并要求至少 `3/5` talks 方向一致；forbidden adoption 与 overcommit 的
point-estimate ceiling 各为 `+1 pp`。这些只是决定是否值得进入 held-out 的 exploratory
screens，不是 statistical non-inferiority tests。
评分器机器执行六个 gate component，并对 early/final/forbidden/overcommit 以 talk 为 cluster
做 10,000 次 bootstrap 95% CI；
五个 dev talks 的区间仍是开发描述，不冒充 confirmatory significance。

## Noise interaction

每个 exact acoustic condition 先计算同一个 content-specific contrast；babble 每档三个
固定 seed 先在 event/talk 内平均，不能当成独立 talks。之后再相对 native audio 做
difference-in-differences：

```text
[correct - matched_wrong]_noise - [correct - matched_wrong]_native
```

因此“noise 越大，vision 越有用”不能由 `correct vs audio-only` 的 BLEU 差异推出；必须是
correct evidence 相对 type/time/token-budget-matched wrong evidence 的增量随 noise 增强，且
final correctness、forbidden adoption、overcommit 与多 talk 稳定性共同通过。

冻结 acoustic matrix 与
[`../code/configs/controlled_acoustic_dev_20260801.json`](../code/configs/controlled_acoustic_dev_20260801.json)
一致：native、babble `+10/+5/0/-5 dB × 3 seeds`，以及 generic-noise 0 dB、music 0 dB
和 medium-near RIR。后面三类是跨 noise type robustness，不进入 babble SNR 单调性拟合。
评分输出同时给出 native→`+10/+5/0/-5 dB` 的 effect curve、severity correlation 和
monotonic flag；同一批 talk bootstrap draws 联合重算所有 DiD 和整条 curve，输出 DiD CI、
correlation CI 和 monotonic bootstrap probability。它们是完整报告项，不是看到结果后新增
的筛选规则。若任何 bootstrap draw 的 severity curve 为常数，correlation CI 不对 defined
draw 做条件化报告，而是置为 `null` 并同时输出 defined/undefined draw 数和 interval status。

## 冻结配置与输出

机器可读配置：
[`../code/configs/acl6060_event_trajectory_scoring_v1.json`](../code/configs/acl6060_event_trajectory_scoring_v1.json)。

输出包括逐 event timing score、各 condition/noise contrast、talk-cluster intervals、机器
development gate、noise interaction、babble severity curve、所有 input SHA256 和 score
artifact SHA256。该 v1 只冻结 scorer 行为；在人工 event/target artifacts 完成并看到 ACL
dev 结果后，仍须另行 commit/push 一个 held-out confirmatory contract，才能读取 ACL eval
或 MCIF outputs。

## 执行顺序

在 system output 产生前，先生成 outcome commitment 与 canonical PCM schedule：

```bash
PYTHONPATH=src python scripts/build_outcome_commitment.py \
  --source-events <source-events.jsonl> \
  --target-scores <target-scores.jsonl> \
  --source-annotation-report <source-report> \
  --source-adjudication <source-adjudication> \
  --target-annotation-report <target-report> \
  --target-adjudication <target-adjudication> \
  --artifact-root <private-outcome-root> \
  --output <host-target-outcome-root>/outcome_commitment.json

PYTHONPATH=src python scripts/build_causal_audio_schedule.py \
  --run-id <run-id> \
  --source-events <source-events.jsonl> \
  --native-inference-manifest <acl-inference.jsonl> \
  --corruption-manifest <corruptions.jsonl> \
  --scoring-config configs/acl6060_event_trajectory_scoring_v1.json \
  --output-root <host-private-canonical-pcm-root> \
  --schedule-output <run-root>/causal_audio_schedule.json
```

从 clean committed checkout 在 container 外启动 broker；`inference_contract.json` 必须在
worker 越过 generation barrier 前写定，并绑定此 broker audit/schedule 与 outcome hashes：

```bash
PYTHONPATH=src python scripts/serve_causal_audio_prefixes.py \
  --run-id <run-id> \
  --schedule <run-root>/causal_audio_schedule.json \
  --socket <host-private-socket-dir>/audio.sock \
  --release-events <run-root>/causal_audio_release_events.jsonl \
  --broker-audit <run-root>/causal_audio_broker_audit.json
```

确认 contract/ready、worker output/done/shutdown 文件都不存在后，先启动全部 worker。每个
worker 固定一个 index，显式选择 GPU，并在模型加载前写 barrier-waiting marker；同一命令形状
重复 `<N>` 次：

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> PYTHONPATH=src python scripts/run_causal_event_inference_worker.py \
  --run-id <run-id> \
  --worker-index <i> \
  --worker-count <N> \
  --inference-contract <worker-visible-root>/inference_contract.json \
  --inference-contract-ready-file <worker-visible-root>/inference_contract.ready.json \
  --scientific-config <worker-visible-root>/scientific_config.json \
  --model-artifact-root <worker-visible-model-root> \
  --tokenizer-artifact-root <worker-visible-tokenizer-root> \
  --model-id <model-id> \
  --model-revision <40-char-revision> \
  --causal-audio-schedule <worker-visible-root>/causal_audio_schedule.json \
  --causal-audio-broker-audit <worker-visible-root>/causal_audio_broker_audit.json \
  --evidence-packets <worker-visible-root>/evidence_packets.jsonl \
  --broker-socket <worker-visible-socket>/audio.sock \
  --output <worker-visible-root>/worker-<i>.jsonl \
  --barrier-waiting-file <worker-visible-root>/worker-<i>.waiting.json \
  --done-file <worker-visible-root>/worker-<i>.done.json \
  --shutdown-file <worker-visible-root>/worker-<i>.shutdown
```

全部 waiting marker 出现且 PID/start ticks 与 live process 相符后，在 worker 仍等待 barrier 时
capture start audit：

```bash
PYTHONPATH=src python scripts/capture_inference_environment_audit.py \
  --run-id <run-id> \
  --container sglang-omni-jaxan \
  --worker-command-match '<exact-run-id-marker>' \
  --inference-repo <clean-worktree> \
  --capture-phase workers_start \
  --forbidden-container-artifact-root <container-private-root> \
  --forbidden-host-mount-source-root <host-full-audio-root> \
  --forbidden-host-mount-source-root <host-target-outcome-root> \
  --output <run-root>/inference_environment_start_audit.json
```

随后由唯一 contract builder 重算输入与目录 hash，从 start audit 派生真实 worker tree，
先 fsync 临时文件，再以同目录 hard-link 原子发布 contract/ready marker。Worker 只能在校验 ready marker 中的
contract SHA256 后开始 generation：

```bash
PYTHONPATH=src python scripts/build_inference_contract.py \
  --run-id <run-id> \
  --source-events <source-events.jsonl> \
  --source-artifact-root <source-artifact-root> \
  --evidence-packets <evidence-packets.jsonl> \
  --control-pairs <control-pairs.jsonl> \
  --scientific-config <scientific-config.json> \
  --scoring-config configs/acl6060_event_trajectory_scoring_v1.json \
  --model-artifact-root <host-frozen-model-root> \
  --target-scores <target-scores.jsonl> \
  --outcome-commitment <host-target-outcome-root>/outcome_commitment.json \
  --outcome-artifact-root <private-outcome-root> \
  --causal-audio-schedule <run-root>/causal_audio_schedule.json \
  --causal-audio-broker-audit <run-root>/causal_audio_broker_audit.json \
  --tokenizer-artifact-root <tokenizer-root> \
  --tokenizer-model <model-id> \
  --tokenizer-revision <40-char-revision> \
  --environment-start-audit <run-root>/inference_environment_start_audit.json \
  --worker-inference-contract-path <worker-visible-root>/inference_contract.json \
  --worker-contract-ready-file-path <worker-visible-root>/inference_contract.ready.json \
  --worker-scientific-config-path <worker-visible-root>/scientific_config.json \
  --worker-model-artifact-root-path <worker-visible-model-root> \
  --worker-tokenizer-artifact-root-path <worker-visible-tokenizer-root> \
  --scoring-protected-artifact-root <host-target-outcome-root> \
  --code-repo <clean-worktree> \
  --output <run-root>/inference_contract.json \
  --ready-file <worker-visible-root>/inference_contract.ready.json
```

Worker 从 ready/contract 各读一次并返回同一 validated contract/hash snapshot，不再二次读取
contract 计算 trajectory identity。Builder、worker、merger 和 scorer 都核对 barrier path、
scientific config、model/tokenizer identity 与 exact read-only mounts。模型加载前和全部
generation 后重算完整 model/tokenizer tree；实际 Qwen3-Omni processor tokenizer 逐 packet
重放 rendered evidence，必须与 frozen token IDs 完全一致。每个
`event × condition × acoustic` 使用独立 session，每次 generation 都只接收本次 broker prefix，
不保留跨 prefix 的 model KV/audio state；每步 hypothesis hash commit 成功后才请求下一 prefix。
Worker 写完 hash-bound shard/done marker 后保持进程存活，等待 end audit 与 shutdown marker。

全部 done marker 出现但 worker 尚未退出时，用同一 capture 命令把 phase 改成
`workers_end`。确认 start/end process tree 一致后创建各 worker 的 shutdown marker 并等待退出。
Merger 必须在与 audited command 相同的 output/done path namespace 下执行；它重验 ready、
start audit、PID/start ticks、确定性 talk partition 与完整矩阵：

```bash
PYTHONPATH=src python scripts/merge_causal_event_worker_shards.py \
  --inference-contract <worker-visible-root>/inference_contract.json \
  --inference-contract-ready-file <worker-visible-root>/inference_contract.ready.json \
  --inference-environment-start-audit <run-root>/inference_environment_start_audit.json \
  --causal-audio-schedule <worker-visible-root>/causal_audio_schedule.json \
  --evidence-packets <worker-visible-root>/evidence_packets.jsonl \
  --worker-output <worker-visible-root>/worker-0.jsonl \
  --worker-done <worker-visible-root>/worker-0.done.json \
  ... \
  --output <run-root>/trajectories.jsonl
```

Broker 结束后依次封装 release log 与 result attestation：

```bash
PYTHONPATH=src python scripts/finalize_causal_audio_release_log.py \
  --schedule <run-root>/causal_audio_schedule.json \
  --broker-audit <run-root>/causal_audio_broker_audit.json \
  --release-events <run-root>/causal_audio_release_events.jsonl \
  --output <run-root>/causal_audio_release_log.json

PYTHONPATH=src python scripts/finalize_inference_result_attestation.py \
  --inference-contract <run-root>/inference_contract.json \
  --trajectories <run-root>/trajectories.jsonl \
  --causal-audio-release-log <run-root>/causal_audio_release_log.json \
  --environment-start-audit <run-root>/inference_environment_start_audit.json \
  --environment-end-audit <run-root>/inference_environment_end_audit.json \
  --output <run-root>/inference_result_attestation.json
```
