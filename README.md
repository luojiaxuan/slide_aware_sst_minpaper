# Slide-Aware Simultaneous Speech Translation — investigation record

> **Status (2026-08-01).** The 21-talk MCIF translation subset source-side path
> was materialized without extracting or reading references. All 21 videos passed
> frozen hash checks and visual inspection. A 1 s reference-free pass produced
> 283 visually reviewed transition candidates and 304 conservative causal
> states including initial states; each state unlocks only after two stable
> frames. This establishes **visual/timing readiness, not translation benefit**.
> The hash-bound 304-state private Qwen3-VL-32B source-only prescreen is now
> complete and frozen at private HF revision
> [`5da477ff`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/5da477ff7d199dbded0ffe44d6b41b9cd8c8e75d).
> Final QA has 304/304 unique rows, 0 parse failures and 0 empty contexts;
> 303 rows contain a spatial-relation candidate. This confirms descriptive
> material exists, but not that descriptions are correct or pixels beat OCR.
> The output cannot become a human label, eligibility decision, sample filter,
> suggested annotation answer, or paper result.
> A subsequent timing audit found that the detector's `fps=1` thumbnails
> represent the centers of one-second sampling buckets but were recorded at the
> bucket starts. The old low-resolution Qwen3-VL screen therefore remains only
> a morphology prescreen and cannot serve as a causal raw-image baseline.
> Native-resolution evidence now uses the same source frames at `t+0.5s` and
> exposes each state only from that actual capture time; the first 0.5 seconds
> of every talk has no visual context.
> The 304 native PNGs and portable manifest are frozen in the same private HF
> repo at revision
> [`4e80dd0a`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/4e80dd0ae4f6bf4f0633cb9d605286d06f34ae49/native_causal_v1),
> tag `mcif-native-causal-evidence-v1`; remote checksums are byte-verified.
> 同一批 304 个 native causal frames 的 matched `flat PP-OCRv6 ->
> PP-StructureV3` 输入层也已完成并冻结在 private HF revision
> [`09004d42`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/09004d4262278b26a1f2f014fdd908427f55797a/ppstructurev3_source_screen_v1)，
> tag `mcif-ppstructurev3-source-screen-v1`。304/304 rows 成功、0 failure；严格
> machine-readable non-flat tier 为 65 rows / 18 talks，其中 chart 53、table 7、
> formula 5。另有 17 个 table rows 只保留 detection placeholder，不能冒充结构化表格。
> 44-row visual QA 还发现 negative strata 中有漏检 table/chart/diagram，因此自动层只用于
> 构造 matched R0/R1 controls，不能过滤 raw-image condition 或定义 event label。
> 三层输入现已物化为不丢 state 的 portable ladder，并冻结在 private HF revision
> [`b13bd204`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/b13bd2045644f90a6de6be19f77a4af3acaa924f/source_evidence_ladder_v1)，
> tag `mcif-source-evidence-ladder-v1`。R0 是不含 bbox 的 flat OCR text；R1 保留 label、
> normalized bbox、reading order 与可机器读取的 chart/table/formula，并把 image tags 降为
> 明确 placeholder；R2 指向同一 native PNG，不在 ladder 中复制图像。304/304 rows 的 PNG
> hashes/dimensions 重验通过，第二次构建 byte-identical，远端 6 files 全量重下载验证通过。
> 对应的 Qwen3-Omni processor budget 与 wrong-image candidate specs 也已冻结在 private HF
> revision
> [`b2c9a409`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/b2c9a4093cb14cf15e26ff72efe941406bbaf59f/visual_token_controls_v1)，
> tag `mcif-qwen3-omni-visual-token-controls-v1`。304/304 cross-talk controls 和
> 283/304 causally-prior same-talk controls 均经过真实 processor 验证；203 个 cross-talk
> controls 天然同尺寸，101 个使用冻结的 aspect-preserving fit-and-pad spec，最终
> `image_grid_thw` 与 visual-token count 均精确匹配 source。该 artifact 仍只是 source-only
> candidate/transform contract。101 张 transformed wrong-image bytes 随后已在 state-level
> media bundle 中物化并冻结到 private HF revision
> [`0001171c`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/0001171cf661d605c6fa344df7cd3f90d291d194/visual_control_media_v1)，
> tag `mcif-qwen3-omni-visual-control-media-v1`；105 files 全量重下载验证通过。它尚未定义
> target event 或 `SourceEventTiming`，因此仍不是正式 event packet、ST 结果或
> `vision > OCR` 证据。
> 在上述 source-side inputs/controls 冻结之后，official references 已单独进入严格隔离的
> outcome-side inventory。919 个 En/Zh/De/It segments 产生 954 个 high-recall lexical
> candidates；689 个至少提前 5 秒，458 个至少提前 10 秒。artifact 冻结在独立 private HF
> revision
> [`64dee522`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/64dee5225e609fc0e900c7d6cd239ae6c702dc5c/outcome_candidate_inventory_v1)，
> tag `mcif-outcome-candidate-inventory-v1`，全量回下载逐字节验证通过。这些仍是 reference-aware
> 自动候选，不是 gold events；human eligibility、target realization 与 audio sufficiency 均未
> 标注，且该 repo 永远不能挂载到 inference。
> 954 candidates 已进一步按 candidate-bearing segment 穷举合并为 355 个 En→Zh author
> items，每个 segment 最多允许冻结一个 event；173 张 current-state slides 均按 native bytes
> hash-bound。workspace 冻结在同一 private outcome repo revision
> [`0785a37f`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/0785a37f6537363b5cd0a8db0ead730298b12a1b/target_event_author_workspace_v1)，
> tag `mcif-target-event-author-workspace-v1`，180 files 已全量回下载验证。当前 355/355 human
> fields 仍为空；target-event author 只能收到 `author_view/`，未来 audio-only validators 不得
> 访问该 repo。
> Annotation protocol、freeze validator 和 localhost UI 已冻结在 config
> `95b8dc69...60b9c` / Git `b6cd276`，当前真实 working progress 为 `0/355`。见
> [docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md](docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md)。
> 独立的 beyond-OCR discovery 也已完成 automatic candidate freeze：严格 R1 在排除
> serialized JSON/markup 与当前 R0 后只剩 2 candidates / 2 talks；R2 在完全排除
> `ocr_text` 及当前 R0/R1 lexical candidates 后有 150 candidates / 21 talks / 118
> segments，其中 122 个 lead≥5 秒、86 个≥10 秒。artifact 位于 private HF revision
> [`01defe41`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/01defe410b4fde07c647d8ed241dfbe501b5d691/beyond_ocr_candidate_inventory_v1)，
> tag `mcif-beyond-ocr-candidate-inventory-v1`，远端 6 files 已全量回下载逐字节验证。R2 来自
> source-only Qwen3-VL 描述，包含明显泛化噪声；它只是待独立人工验证的 proposal pool，
> 仍不支持 `pixels > OCR`，也不能进入 inference 或替代 R0 authoring。
> 152 个 beyond-OCR candidates 已进一步编译成物理隔离的双角色 workspace：152-item
> `visual_validator_view` 只含 slide/R0/R1/candidate/evidence，152-item
> `target_author_view` 只含 candidate/English source/Chinese reference，真实关联仅在
> `scorer_private`。91 张 current-state native PNG 去重分发，机器审计为 0 forbidden-field
> leak、0 human labels。workspace 冻结在 private HF revision
> [`861401f2`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/861401f295ab122e69c4f22820b8d501e891e6db/beyond_ocr_validation_workspace_v1)，
> tag `mcif-beyond-ocr-validation-workspace-v1`；102 files 全量回下载逐字节验证通过。只有两个
> 角色的 frozen labels 成功 join 后，候选才可进入 audio sufficiency/event packets。
> 对应 role-specific validator、localhost UIs、create-once freezes 与 scorer-side join 已完成，
> protocol config SHA256 为 `d25f558c...d69b6`，实现位于 Git `d3a710e` / `ffd960c`。两个
> `0600` working sheets 仍分别为 0/152；desktop/mobile role-isolation 审计、真实 image load、
> 0 horizontal overflow 与 0 console error/warning 均通过，且没有保存测试标签。当前 visual 与
> target 服务分别为 <http://127.0.0.1:43872/>、<http://127.0.0.1:43873/>。完整 gate、启动、
> freeze 与 join 命令见
> [docs/MCIF_BEYOND_OCR_VALIDATION_V1.md](docs/MCIF_BEYOND_OCR_VALIDATION_V1.md)。
> **2026-08-01 reliability audit 已在 0 labels 时 supersede 上述 v1 instrument。** Dual-role
> separation 不等于 role 内重复标注；visual 页面还在 OCR sufficiency 判断前暴露 pixels/VLM。
> 因此 43872/43873 已停止，两个 working sheets 保持 0/152；v1 只能用于 provenance、firewall
> 与 calibration，不能产生 paper gold。Replacement 是 visual A/B sequential locks、target
> author + bilingual validator、append-only adjudication 和 talk-cluster reliability gate。见
> [docs/MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md](docs/MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md)。
> Replacement v2 core 已落地于 Git `2aa9b6a`：full-overlap visual A/B 只有 R0 被初始释放，
> R1→pixels→descriptor 必须逐阶段验证两侧完整 HMAC freezes；target validator stage 1 在 freeze
> 前拿不到 author text；raw agreement gate 先于 adjudication。真实 152-item zero-label workspace
> 已冻结在 private HF revision
> [`eb194d83`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes/tree/eb194d83c941838db2b096fe52c5e455c5b304bb/beyond_ocr_reliability_workspace_v2)，
> tag `mcif-beyond-ocr-reliability-workspace-v2`，106 files 全量回下载逐字节验证。当前 UI/server
> 尚未完成，不得开始 production annotation。见
> [docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md](docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md)。
> See
> [docs/MCIF_VISUAL_READINESS_20260801.md](docs/MCIF_VISUAL_READINESS_20260801.md).
> The controlled-acoustic input path is also ready: official SLR17/SLR28
> archives and disjoint source pools are frozen, and 75 full-talk ACL dev
> variants pass duration, SNR, clipping, hash, and no-reference checks. See
> [docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md](docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md).
> ACL dev 的 source-side screen 也已可执行：468 个 source segments 与 468 个
> frame observations 完成时间对齐和 Tesseract OCR；严格 exact-match diagnostic
> 找到 149 个独立 `segment × current-frame` anticipation events，其中 183/344
> 个去嵌套候选具有至少 10 秒的保守提前量。该数字只说明 OCR-sufficient
> headroom，不是人工 event density 或 ST gain。v1 的 100 个 media packets 已生成并通过
> no-target/reference/model-output 检查，但 v1 annotation order 已被方法审计否决，不能直接
> 使用其 A/B sheets。v2 已生成 100-frame blinded author view，当前状态是
> `AUTHORING_VIEW_READY_NO_LABELS`。见
> [docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md](docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md)。
> Media workspace 已冻结在 private HF revision
> [`3199207c`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-events/tree/3199207c66b159ab39f662a32e0f6d633c9c2b79)。
> Identifier-hardened author-only r4 view 已冻结在 private HF revision
> [`bbbbdbf5`](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-event-author-v2/tree/bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a)；
> r3 在任何人工标签产生前已 superseded。
>
> The frozen Chinese-LiPS five-condition diagnostic is now complete. With
> Qwen3-Omni-30B over 206 segments, current slide, same-talk wrong slide,
> unrelated-talk scientific slide, and blank image all produce similar gains
> over audio-only. `slide - wrong` is only `+0.113 chrF` with descriptive 95% CI
> `[-0.693, +0.894]`, while `blank - none` is `+1.545 [0.644, 2.482]`.
> Therefore **current-page semantic use is not established**; the simplest
> explanation is a generic vision-slot/decoding perturbation. This remains a
> single-talk, machine-reference mechanism diagnostic, not paper evidence. See
> [docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md](docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md)
> and **read [docs/FINDINGS.md](docs/FINDINGS.md) first** for claim confidence
> and falsification criteria.
>
> **Current exploration strategy:** study whether causally available,
> persistent slide semantics can improve SimulST without putting vision on the
> streaming critical path. ACL dev is an explicit story-discovery stage across
> content attribution, noisy-speech robustness, pixels beyond OCR, and
> evidence-selection/integration. Correct versus matched stale/wrong slides is
> a shared validity control, not the only allowed paper outcome. After the dev
> evidence identifies a useful mechanism, freeze one main claim and analysis
> before touching held-out evaluation. The authoritative strategy, audit, data
> update, and experiment stages are in
> [docs/PAPER_STORY_DECISION_20260731.md](docs/PAPER_STORY_DECISION_20260731.md).
> A narrower 2026-08-01 collision audit rates the broad claim **Level 2 high
> overlap** because OmniFusion already combines scientific-talk slides, SimulST
> and earlier/stable commitments. The remaining paper space is the frozen causal
> event study: correct-target commit lead before audio sufficiency, content
> controls and acoustic-noise interaction. See
> [docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md](docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md).
> The previous dual-route document remains the detailed `C0-C7` contract, not
> the current narrative. Lip video and slide+lip hybrid experiments remain out
> of scope.

## Original hypothesis (for the record)

Slides in talks often precede the speech that discusses them and persist for
tens of seconds. The visual worker can therefore run off the audio critical path:
it pays a nonzero cost once per slide, then exposes a cached evidence state to
many speech chunks. Three mechanisms were posited — M1 anticipation, M2
recognition support, M3 target-form supply — with slide language as a
stratification variable.

What survived testing is narrower: supplying any image slot changes quality and
latency, but current-page content does not outperform wrong or blank controls in
the Chinese-LiPS probe. What remains untested is the paper-specific question:
whether causally earlier, content-matched slide evidence produces stable target
decisions before the audio becomes sufficient, especially under controlled
noise and across multiple talks.

Lineage and neighbours (survey retained in
[latex/sections/02_related_work.tex](latex/sections/02_related_work.tex)):
RASST (retrieval-augmented terminology for SST), Do-Slides-Help (EMNLP'25,
offline ASR), OmniFusion (En→X SimulST with synchronous vision), Caglayan'20
line (simultaneous *text* MT with caption images), MCIF. These works occupy the
broad slide/visual-anticipation claim; this project must contribute causal
content attribution and stronger event-level evidence rather than another
aggregate image gain.

## Directory layout

```text
code/    scripts, configs, tests (data prep, scoring, eval pipeline)
data/    DATA.md — pointers to HF datasets and local staging (no media in git)
docs/    stage-by-stage progress, plans (BENCHMARK_PLAN.md), planning/ archive
latex/   paper draft by sections, refs.bib, figures/ + plotting/ code
```

## Current research direction (revised 2026-07-31)

- **Research space:** pre-available slide semantics for SimulST, compiled once
  per persistent slide and reused by the streaming speech path.
- **Development exits:** current-content attribution and earlier commits;
  robustness under controlled acoustic corruption; useful visual relations
  beyond OCR; or a selective integration method with a better quality-latency-
  hallucination trade-off than naive prompting.
- **Shared validity controls:** audio/document-only, naive OCR prompt, correct
  slide, time/type/budget-matched stale/wrong slide, and empty slide slot.
- **Selection discipline:** use ACL dev and other declared development material
  to discover which mechanism is real, report the full exploratory matrix, then
  freeze the main claim, metric, slices, and decision rule before held-out data.
- **Project-held-out long-form source:** the 21-talk MCIF translation subset
  used by the official IWSLT 2026 SimulST development corpus, En→Zh primary and
  En→De replication. Its input-side visual readiness now passes: 21 videos,
  7,110 s, 283 reviewed transitions and 304 causal states. Eligible-event
  density, power and outcome-side freeze gates remain open.
- **Replication benchmark:** ACL60/60 dev/eval En→Zh, using its external term
  annotations and direct lineage to *Do Slides Help?*. The verified Figshare v2
  supplement adds 884 real talk-video frames covering all 10 talks.
- **Private diagnostic:** Chinese-LiPS-Long for slide dwell/change timing and
  Chinese ASR/terminology probes. It is not a paper-grade ST ranking set without
  independent human references and confirmed use scope.
- **Historical assets:** mTEDx-V and Chinese-LiPS derived artifacts remain on
  Hugging Face, but they no longer define the main paper. mTEDx/TED media are
  blocked from new use without explicit permission.
- **Scripts** in `code/scripts/`: `build_mtedx_v_manifest.py`,
  `extract_frames_by_manifest.py`, `build_chinese_lips_longform.py`
  (`--timeline-dir` for original timeline), `score_visual_signal.py`
  (`--backend ocr|vlm`), `translate_zh_en_draft.py`.

The July 17 direction plan is retained only as history in
[docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md); do not execute it as the
current benchmark contract.

## Next execution milestone

1. Complete v2 frame-only canonical item authoring, hash-lock the questions,
   using the implemented localhost-only blinded authoring UI, then use the
   sequential backend for two independent full-causal-prefix audio trajectories
   and the implemented localhost-only validator UI for a disjoint two-person
   frame-only cohort. v1 media remains
   reusable, but its A/B annotation sheets are superseded. No human label is
   complete yet.
2. The 304-state visual-token inventory, deterministic stale/wrong candidates,
   and 101 transformed wrong-image bytes are frozen. Next freeze target events
   and scoring, then join each event to immutable state-level media inside
   hash-bound source-only packets; then compare audio/document-only,
   token-budget-matched unordered OCR, layout/structure-
   preserving text, current raw image, and matched stale/wrong evidence under
   native and controlled-noise audio. This ladder separates “context helps”
   from “raw pixels are necessary.”
3. Require a content-specific advance in first stable correct target decisions,
   preserved final quality and a coherent noise interaction. A small aggregate
   BLEU change without these controls is not sufficient. The fail-closed event
   scorer and exact 16-condition acoustic grouping are implemented in
   [docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md](docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md);
   it now replays immutable tokenizer IDs, binds source media/extractor identity,
   separates pre-run contract from post-run result attestation, commits raw
   annotation/adjudication artifacts before output, captures start/end process-tree
   isolation, and includes an executable external audio broker. Its synchronized
   talk-level frontier prevents every condition/event/noise stream from seeing a
   later audio time until all current-time hypotheses commit. The production
   Qwen3-Omni worker and audited shard merger are now implemented and tested;
   paper-grade fresh generation remains blocked on the read-only `network=none`
   narrow-mount container rebuild and unfinished human outcome artifacts. No ACL
   system result exists yet.

当前 MCIF R0 event authoring 只冻结 target realizations，不运行 ST。355-item workspace、角色
隔离、状态定义、localhost UI 与后续 freeze 命令见
[docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md](docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md)。
R1/R2 beyond-OCR v1 双角色 gate 已在 0 labels 时 superseded，不得开标。V2 必须先完成 role 内
replication、顺序 modality lock、independent target verification 与 adjudication；core/base
workspace 已完成，但 server/UI 未完成，仍不得开标。见
[docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md](docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md)。

## Rules

- Test sets: real slides only, no synthetic visual evidence; references human or
  human-verified (two-tier, FLORAS-style).
- Do not access, re-download, train on, or evaluate with TED/TEDx media under an
  assumed research exception. Current TED terms require a separate written
  license for ML/AI datasets, training, and evaluation; legacy dataset claims
  need explicit permission or institutional review.
- Git + Hugging Face are the sources of truth; `data_prep/` staging is disposable.
