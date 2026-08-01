# Paper Story Exploration：Persistent Slide Semantics for SimulST

更新日期：2026-07-31

状态：**当前权威的探索策略与 held-out freeze 边界；最终 paper identity 尚未冻结。**
本文件取代
[`DUAL_ROUTE_DECISION_20260731.md`](DUAL_ROUTE_DECISION_20260731.md) 的 A/B paper
identity 和原始执行顺序；旧文件只保留 `C0-C7` control、no-reference、held-out 与
runtime contract。本文已经吸收两轮 novelty audit、一轮 hostile ACL review，以及
“不要在实证探索前过早锁死单一故事”的策略纠正。

## 1. 研究空间，而非提前冻结的唯一 paper question

> **Can causally available, persistent slide semantics improve simultaneous
> speech translation without putting visual processing on the online critical
> path, and under which evidence and acoustic conditions?**

核心系统设定是：一个 slide 通常先于或伴随相关 speech 出现，并持续多个 audio chunks；
系统换页时编译一次 OCR、layout、relation 或 semantic evidence，streaming decoder 只查询
缓存。研究重点是这种 **pre-available semantic vision**，不是逐帧 lip video，也不是每个
audio chunk 都重新调用 omni vision encoder。

开发阶段保留四条可形成顶会论文的出口：

| Route | 可形成的主问题 | 必须看到的实证信号 |
| --- | --- | --- |
| A. Attribution / anticipation | 当前 slide 是否让系统在 audio 消歧前更早正确提交？ | correct 显著优于 matched stale/wrong，且 final quality 不退化 |
| B. Robustness | slide evidence 是否在 noisy、accented、jargon-heavy speech 下补足 audio？ | 随 acoustic quality 降低，correct-slide 增益稳定扩大，wrong-slide 不同步扩大 |
| C. Beyond OCR | 哪些 layout/chart/formula/diagram evidence 是 OCR prompt 无法表达的？ | pixels/relations 优于 token-budget-matched OCR，且 gold-relation positive control 有效 |
| D. Integration | 如何选择和注入长驻 slide evidence，避免 irrelevant context 与 hallucination？ | selection/gating 优于 naive OCR/image prompt 的 quality-latency-safety Pareto frontier |

Route A 的 correct-vs-wrong test 是所有路线共用的 **content-use validity control**，但不是
现在就排他的最终 paper story。允许开发实验决定最终标题、主要 claim 和方法重点；要求是
完整保留探索结果，并在独立 held-out evaluation 前冻结最终方案。

## 2. 为什么这个空间仍存在

锁定旧方案后，独立 Paper-Search 与 Scoop-Check 均给出
**Level 3 - Medium Overlap / partial overlap**；锁文件 SHA256 为
`ccb8376cbca48328ec1640dae6bd4aa516b07e8161a51c406e6936be7cd48767`。

| 已有工作 | 已占据 | 本文剩余差异 |
| --- | --- | --- |
| OmniFusion | 展示了 slide-aware speech translation 的应用外形 | 端到端 latency 与 streaming usability 很弱；未解决 off-critical-path integration、current-content attribution 或 strong OCR baseline，不能视为封死本方向 |
| BOOM | current slide screenshots + live lecture translation | 没有 matched causal content intervention |
| VAPO | look-then-listen、image/OCR、mismatched slide、visual interference | ASR，不是 document-aware SimulST commit attribution |
| EGTA / RASST | paper terminology、causal selection、shuffled control | term memory 是本文共同 baseline，不是贡献 |
| Context Helps / DoCIA | discourse-aware ST、wrong context、latency/stability | 使用历史 speech/translation，不是 live current slide |
| visual-context SiMT | pre-observed image、anticipation、READ/WRITE | caption text MT，不是 long-form raw speech + stale slide state |

因此不把 “first visual SimulST” 当作主要 novelty，也不因 OmniFusion 存在就把研究退化为
单一 attribution paper。可守住并值得实证探索的空间是：**把持久、先到达的 slide
semantics 从 streaming critical path 中分离，并证明何时、为何、以何种 integration
方式真正改善 online decisions。**

## 3. 开发期 outcome family 与 held-out primary freeze

每个 event `i` 先定义 source-side forced choice，例如 lexical sense、referent、relation、
scope 或 reordering decision。对 condition `c` 定义：

```text
Y_i(c) = 1
iff the acceptable target decision is first emitted stably
before or at t_last_insufficient and remains correct in the final output.
```

`[t_last_insufficient, t_first_sufficient]` 是双语标注者按固定 audio-prefix step 得到的
source-side ambiguity interval：前者仍不足以完成 forced choice，后者已经足够。
落在 interval 内的 commit 是 timing-ambiguous，只进 sensitivity analysis。

Route A 的候选 estimand：

```text
Delta_attribution = E_talk[ mean_i(Y_i(current_correct) - Y_i(matched_control)) ]
```

- `current_correct`：strong document packet + causally available current-slide packet；
- `matched_control`：完全相同的 document packet、interface、schema、source type、解锁
  时间和 token bin，只把 slide identity 替换为 same-talk stale/wrong state；
- 每个 talk 等权，events 不作为独立 talks；
- 开发期把 **+5 percentage points** 作为是否值得追加成本的 practical signal，不把它
  冒充已经预注册的最终 paper threshold；
- target-event final correctness 同时记录，开发期参考 non-inferiority margin 为 **-1 pp**。

开发期同时记录以下 outcome family，不要求其中某一个在看数据前排他地成为论文主线：

- final translation quality：BLEU、chrF、COMET 或数据允许的独立 human score；
- online behavior：AL/LAAL、first stable correct commit、revision/flicker；
- evidence-sensitive accuracy：term、entity、sense、relation、scope、reordering events；
- robustness curve：不同 SNR/noise type 下的 quality 与 latency interaction；
- content specificity：correct、stale/wrong、unrelated 与 empty evidence；
- visual increment：pixels/relations 相对 token-budget-matched OCR/layout；
- safety/cost：wrong-evidence adoption、unspoken hallucination、packet tokens、on-path
  latency、cold compilation cost 与 GPU seconds。

这不是不受约束地事后挑结果。边界是：

1. 在 ACL dev 前写定要跑的 conditions、metrics 与 slices，并完整保存结果；
2. 允许根据 dev effect size、稳定性和 failure analysis 选择 Route A-D 或组合方法；
3. 选择后生成新的 frozen confirmatory contract，写定一个 main claim、primary metric、
   SESOI、模型/config、允许的 secondary analyses 与 failure criterion；
4. 只有在该 contract commit/push 后，才能读取 ACL eval/MCIF outputs 或 references；
5. 论文明确区分 exploratory development 与 held-out confirmation。

## 4. 三个独立冻结 artifact

为防止 oracle 和 target leakage，dev experiment 前依次冻结：

1. **Candidate-event inventory**
   - 只用 source audio/transcript 与 causally available paper/slide material；
   - 不看 target reference、system output 或 condition difference；
   - 包含所有抽样 candidates，包括 `no_external_evidence`、`no_expected_benefit`；
   - 对随机 speech spans 做 coverage audit，不能只挑明显 slide matches。
2. **Source-language evidence packets**
   - packet builder 不读 target references 或 target translations；
   - packet 禁止包含预翻译 target lexicalization；
   - correct/control packets 匹配 schema、type、availability 和 token bin；
   - 每条 packet 保留 source region/id 与 provenance。
3. **Target scoring artifact**
   - 在前两项 hash 冻结后，由独立双语 annotators 建 acceptable target realizations；
   - 不改 source forced choice、event inclusion 或 evidence packet；
   - scoring artifact 永不挂载到 inference process。

Oracle 用于 capability mapping，不进入 automatic-system superiority table。Gold
source-side evidence 可以指出 relation/sense，但不得直接给 target translation。

## 5. Conditions：共同 baseline 与候选机制

`C1-C3` 不是严格 ladder，而是 external baseline family：

| ID | Condition | 角色 |
| --- | --- | --- |
| `C0` | audio only | sanity baseline |
| `C1` | automatic term memory | RASST/EGTA lineage |
| `C2` | entities + abstract | static context baseline |
| `C3` | phrase boost + pretranslated PDF BM25/RAG | selected strong document baseline |
| `C4` | non-term document propositions/discourse | candidate context representation |

用于 content-attribution 的 slide conditions 必须共享同一 frozen `C3` document packet：

| ID | Condition | 角色 |
| --- | --- | --- |
| `C5-correct` | C3 + current-slide OCR/layout propositions | candidate semantic treatment |
| `C5-control` | C3 + matched same-talk stale/wrong slide propositions | content-use control (`C7`) |
| `C5-none` | C3 + empty slide slot | context-value control |
| `C6-auto` | C5 + automatically extracted image-specific relation | candidate beyond-OCR treatment |
| `C6-gold` | C5 + gold source-side visual relation | positive control, not a method |

开发期还需为 C5/C6 增加 native/noisy audio、naive full OCR prompt、retrieved/gated
packet 与 direct-image input cells。`C4 > C1-C3` 不叫 minimum sufficient level；只有
matched nested conditions 才能归因 representation 或 integration 的增量。所有结论都
限定 model、compiler、packet budget 与 dataset。

## 6. Annotation contract

每个 candidate event 包含：

- locked source-side forced-choice options；
- `t_evidence`；
- `[t_last_insufficient, t_first_sufficient]`；
- evidence type 与 source region/id；
- later-frozen acceptable target realizations；
- `term_or_entity` exclusion flag；
- `no_external_evidence` / `no_expected_benefit` negative labels。

标注规则：

- fixed prefix increments，不允许 annotator 自由挑时间点；
- 预注册 boundary agreement threshold；不一致且无法 adjudicate 的 event 排除 primary；
- 报告 interval overlap、boundary distance、label agreement、每 talk/type 数量；
- target scoring 与 source/evidence annotation 分人或至少分阶段、blind hashes；
- dev guideline 冻结后，ACL eval/MCIF 只按 frozen guideline 标注。

## 7. Slide state 与 causal timing

*Do Slides Help?* Figshare v2 已验证覆盖 ACL60/60 全部 10 talks、884 个真实 talk-video
frames；来源与 hashes 见
[`../data/manifests/do_slides_help_figshare_v2_20260731.json`](../data/manifests/do_slides_help_figshare_v2_20260731.json)。
原 metadata 含 transcript，只允许生成 stripped frame-only inference view。

该 release 只有 segment-midpoint frames，没有 exact slide changes。因此：

1. 按相邻 frame similarity 聚类 slide states，并保留 manual QA；
2. transition 表示为 interval，不伪造单点 onset；
3. 新 state 只在 first confirmed frame 后解锁；
4. previous state 只在新 state confirmed 后标 stale；
5. transition interval 内 events 不进入需要精确 causal timing 的分析；
6. exact raw-video timeline 只能在原视频下载、hash 和 QA 后升级。

## 8. Power 与 benchmark 角色

- **ACL dev 5 talks：** event density、annotation reliability、oracle capability、开发期
  多路线 story discovery 与 MDE；
- **ACL eval 5 talks：** story freeze 后的 replication pilot。五个 clusters 的 two-sided sign test
  最小 p 值为 0.0625，不能承担 confirmatory significance；
- **MCIF 21 talks：** 唯一 planned confirmatory source，但必须先完成 21 videos 的 hash、
  slide-state coverage、transition QA、frame-only manifests 与 eligible-event counts。

MCIF hard readiness gate：

- 至少 **15 个 independent talks** 含 eligible primary events；
- dev-estimated talk-cluster MDE 不大于 +5 pp SESOI；
- references 在完整 inference ledger 关闭前保持未挂载；
- visual QA、annotation guideline、最终选定的 route、conditions、model 与 metric 已
  commit/push。

未通过该 gate 时，只能写 ACL pilot/measurement report，不得声称 confirmatory
current-slide benefit。只有一个 system family 时，结论必须写成 single-system finding；
paper-level general claim 需要第二个 system family 方向复制。

## 9. Oracle-first capability map

在投入完整 automatic C1-C6 之前，用低成本 oracle 同时判断几类 headroom：

1. 生成 stripped ACL dev slide-state manifest；
2. blind 标注 80-120 candidates，并包含 negative cases；
3. 冻结 candidate inventory、source-only packets、target scoring 三个 artifacts；
4. 条件共享同一 frozen source-only document packet，覆盖 empty slide slot、OCR、correct
   semantic oracle、correct relation oracle、matched wrong evidence；
5. native/noisy audio 都跑，分别检查 anticipation、recognition support、target-form
   supply 与 beyond-OCR relation；
6. 任何 route 若有跨 talk 的 practical signal，就允许为该 route 构建 automatic method；
   若所有 gold route 都无 headroom，才停止 slide-semantic paper 投资。

Oracle 通过只证明对应 route 有 capability headroom，不证明 automatic extractor 或
integration 有效。+5 pp 与 3/5 talks 是开发期优先级信号，不是对所有潜在 story 的统一
否决线。

## 10. Pixels / Why not OCR

Pixels 是候选 Route C，而不是预设必成或预设 secondary。`C6-auto vs C5-correct` 必须配
`C6-gold` positive control：

- `C6-gold > C5`、`C6-auto = C5`：automatic visual extraction failure；
- 两者都正向：tested pixel-derived relation 有增量；
- 两者都 null 且 MDE > 5 pp：inconclusive；
- 只有在 gold control 有效、至少 15 talk clusters、且 90% equivalence interval 完全落在
  `[-5 pp, +5 pp]` 时，才可写“tested systems 未发现超出 OCR/layout 的 pixel benefit”。

因此不能默认写 “pixels unnecessary”，也不能使用 *From Papers to Pixels* 标题。

## 11. 开发结果到 paper route 的选择图

| 开发结果 | 候选 paper route / 下一步 |
| --- | --- |
| correct > matched control，且更早 stable commit | Route A；做 causal anticipation/integration |
| native audio 弱、noise interaction 强且 content-specific | Route B；做 robust SimulST 与 acoustic-quality-aware gating |
| C6-gold 与 C6-auto 都稳定优于 OCR | Route C；做 beyond-OCR visual relation extraction |
| oracle 强、naive prompt 弱、selection/gating 强 | Route D；做 evidence integration method |
| correct = wrong，但都优于 none | 只有 generic priming；不能声称 current-content use，继续 unrelated control 或停止该路线 |
| 只有 term gain | 可作为 method component；若无 integration/online 新意，不单独形成论文 |
| 所有 gold evidence 都无 practical headroom | 停止 slide-semantic 主方向 |
| 多条路线同时有信号 | 选择机制最一致、可复现且方法贡献最清楚的一条为 main；其余作 supporting analysis |

## 12. 当前下一步

当前先不需要 GPU。实现 Figshare -> stripped ACL60/60 slide-state manifest，并定义
candidate-event / source-only packet / target-scoring 三阶段 schema。随后做 80-120 event
density + multi-route oracle capability map，再以小规模 automatic runs 比较 OCR、semantic
packet、visual relation、selection/gating 与 native/noisy audio。开发结果出来后才写最终
paper contract；MCIF video readiness 与 confirmatory power 仍是 held-out general claim 的
hard blocker。

## Primary sources added by the audit

- OmniFusion: <https://arxiv.org/abs/2512.00234>
- BOOM: <https://aclanthology.org/2026.eacl-demo.14/>
- VAPO: <https://aclanthology.org/2026.acl-long.425/>
- EGTA: <https://arxiv.org/abs/2607.17766>
- Context Helps: <https://aclanthology.org/2021.acl-long.200/>
- DoCIA: <https://aclanthology.org/2025.findings-acl.771/>
- Multimodal RL SiMT: <https://aclanthology.org/2021.eacl-main.281/>
- Do Slides Help?: <https://aclanthology.org/2025.emnlp-main.814/>
