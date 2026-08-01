# Source of Truth

This project uses Git and Hugging Face as the only durable sources of truth.
Notion is not used for project knowledge, progress records, reusable artifact
links, or handoff state.

## Canonical Locations

| Material | Canonical home | Current status |
| --- | --- | --- |
| Code, configs, scripts, tests | GitHub: `luojiaxuan/slide_aware_sst_minpaper` | Active on `main` |
| Paper notes and implementation plans | This Git repo under `latex/`, `docs/planning/`, and `docs/` | Active |
| Lightweight progress and artifact index | This Git repo under `docs/` | Active |
| Reusable datasets and generated data artifacts | Hugging Face Hub | MCIF source-only prescreen, Chinese-LiPS Qwen3-VL, ACL controlled-acoustic, and ACL source-event bundles uploaded to private dataset repos |
| Reusable checkpoints/adapters | Hugging Face Hub | None yet |
| Local staging and active runs | Local `ResearchStudio/data/vision-aware-sst/` and verified GPU-host `/data` paths | Temporary, not canonical |

## Current Git Pointers

- GitHub repo: <https://github.com/luojiaxuan/slide_aware_sst_minpaper>
- Current exploration strategy, collision audit, development routes and held-out freeze:
  [`docs/PAPER_STORY_DECISION_20260731.md`](PAPER_STORY_DECISION_20260731.md)
- Narrow pre-audio slide/early-target collision audit and executable claim boundary:
  [`docs/PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md`](PREAUDIO_SLIDE_COLLISION_AUDIT_20260801.md)
- Preserved `C0-C7` and Phase-A control contract:
  [`docs/DUAL_ROUTE_DECISION_20260731.md`](DUAL_ROUTE_DECISION_20260731.md)
- Phase-A data/runner freeze, provenance and no-reference boundary:
  [`docs/PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](PHASE_A_DATA_RUNNER_FREEZE_20260731.md)
- MCIF materialization, visual QA and causal-state readiness:
  [`docs/MCIF_VISUAL_READINESS_20260801.md`](MCIF_VISUAL_READINESS_20260801.md)
- Controlled-acoustic sources, mixer contract and ACL dev QA:
  [`docs/CONTROLLED_ACOUSTIC_PIPELINE_20260801.md`](CONTROLLED_ACOUSTIC_PIPELINE_20260801.md)
- ACL dev transcript-free visual timeline and transition false-negative audit:
  [`docs/ACL6060_VISUAL_TIMELINE_20260801.md`](ACL6060_VISUAL_TIMELINE_20260801.md)
- ACL dev source-side seed/media provenance and superseded v1 protocol:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V1.md)
- ACL dev authoritative blinded annotation v2 protocol and tooling:
  [`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V2.md`](ACL6060_SOURCE_EVENT_ANNOTATION_V2.md)
- ACL dev event-timing estimand, causal broker and provenance contract:
  [`docs/ACL6060_EVENT_TRAJECTORY_SCORING_V1.md`](ACL6060_EVENT_TRAJECTORY_SCORING_V1.md)
- Raw-image v2 condition matrix、source mount、byte/token binding 与 runtime 边界：
  [`docs/RAW_IMAGE_EVENT_CONTRACT_V2_20260801.md`](RAW_IMAGE_EVENT_CONTRACT_V2_20260801.md)
- Route A ACL paper and confirmatory experiment contract:
  [`docs/ACL_PAPER_BLUEPRINT_20260731.md`](ACL_PAPER_BLUEPRINT_20260731.md)
- Independent ACL-style review and resolved/residual risks:
  [`docs/ACL_BLUEPRINT_REVIEW_20260731.md`](ACL_BLUEPRINT_REVIEW_20260731.md)
- Route A scope record:
  [`docs/RESEARCH_GOAL_20260731.md`](RESEARCH_GOAL_20260731.md)
- Current evidence and confidence:
  [`docs/FINDINGS.md`](FINDINGS.md)
- OmniFusion latency/control reassessment:
  [`docs/OMNIFUSION_REASSESSMENT_20260731.md`](OMNIFUSION_REASSESSMENT_20260731.md)
- Historical project framing (superseded by the current paper-story decision):
  [`docs/planning/SLIDE_CONTEXT_AWARE_MVP.md`](planning/SLIDE_CONTEXT_AWARE_MVP.md)
- Agent handoff plan: [`docs/planning/AGENT_START_HERE.md`](planning/AGENT_START_HERE.md)
- Historical Chinese-LiPS experiment matrix:
  [`docs/planning/EXPERIMENT_MATRIX.md`](planning/EXPERIMENT_MATRIX.md)
- Progress log: [`docs/PROGRESS.md`](PROGRESS.md)
- Qwen3-VL GPU profiling evidence:
  [`docs/QWEN3_GPU_PROFILING_20260706.md`](QWEN3_GPU_PROFILING_20260706.md)
- Qwen3-VL context QA and repair record:
  [`docs/QWEN3_CONTEXT_QA_20260706.md`](QWEN3_CONTEXT_QA_20260706.md)
- Qwen3-32B reference pilot:
  [`docs/QWEN3_REFERENCE_PILOT_20260706.md`](QWEN3_REFERENCE_PILOT_20260706.md)
- Qwen3-32B diagnostic-500 context ablation:
  [`docs/QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md`](QWEN3_DIAGNOSTIC500_EXPERIMENTS_20260707.md)
- Diagnostic human review guide:
  [`docs/DIAGNOSTIC_REVIEW_GUIDE.md`](DIAGNOSTIC_REVIEW_GUIDE.md)
- Chinese-LiPS five-condition Qwen3-Omni visual-control diagnostic contract:
  [`docs/CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md`](CHINESE_LIPS_VISUAL_CONTROL_MATRIX_V1.md)

## Dataset and Artifact Truth

### Current Benchmark Sources

| Role | Canonical source | Status |
| --- | --- | --- |
| Primary long-form held-out benchmark | MCIF paper/project: <https://arxiv.org/abs/2507.19634>, <https://mt.fbk.eu/mcif/>; HF `FBK-MT/MCIF` | 21-talk subset materialized at revision `e24065b919758263cfe5d157057278affe76ea7b`; video hashes verified; 283 reviewed visual transitions and 304 causal states; reference files remain unextracted and unread |
| Development and tagged-term replication | [ACL60/60 official release](https://aclanthology.org/2023.iwslt-1.2/) | dev/eval 5+5 complete talks, 468+416 gold segments and term annotations frozen at archive SHA256 `5f2a3855...cfce7cc`; inference and scoring views physically separated |
| ACL60/60 real frame supplement | [Do Slides Help? Figshare v2](https://figshare.com/articles/software/Code_and_data/30158932) | CC BY 4.0 outer archive SHA256 `f771d3f6...3d4cc`; dev 468 frames imported as no-backdating causal observations; eval untouched; pixel transition compression rejected after false-negative audit |
| Private timing/diagnostic benchmark | Chinese-LiPS derived private HF repo below | Available; not a paper-grade ST ranking set |
| Controlled acoustic corruptions | MUSAN/SLR17 <https://www.openslr.org/17/> (CC BY 4.0) and Room Impulse Response and Noise Database/SLR28 <https://www.openslr.org/28/> (Apache 2.0) | Archives/hashes/licenses frozen; 65 development and 65 confirmatory sources are disjoint; ACL dev has 75 QA-passed full-talk variants. SLR119 is AliMeeting and is not the RIR source |

No paper experiment may rely on an unfrozen local-only copy. The first Phase-A
data task must record source URLs, accepted terms/licenses, immutable revisions
or SHA256 manifests, and generation commands here or in a linked data card.

### Upstream Dataset

- Source dataset: `BAAI/Chinese-LiPS`
- Hugging Face source: <https://huggingface.co/datasets/BAAI/Chinese-LiPS>
- Local staging path on Hyper00: `/data/datasets/chinese_lips`
- Local staging is cache/scratch only. It is not a source of truth.

### Reusable Derived Artifacts

The Qwen3-VL derived train artifacts are uploaded to a private Hugging Face
dataset repo. Because
`BAAI/Chinese-LiPS` is gated and its terms restrict redistribution of derived
works outside the research group without maintainer permission, any HF repo for
these artifacts should be private or otherwise access-controlled unless that
permission is obtained.

| Artifact | Local path | Canonical or intended HF destination | Upload status |
| --- | --- | --- | --- |
| MCIF 304-state native-resolution causal visual evidence v1 | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/native_causal_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/4e80dd0ae4f6bf4f0633cb9d605286d06f34ae49/native_causal_v1>, revision `4e80dd0ae4f6bf4f0633cb9d605286d06f34ae49`, tag `mcif-native-causal-evidence-v1`; Git manifest: `data/manifests/mcif_native_causal_evidence_v1_20260801.json` | Uploaded and checksum-byte-verified; `private=True`; 304 native PNGs, 21 talks, manifest SHA256 `4e1008ab...9cccc`; detector bucket-center correction moves evidence and availability from nominal `t` to actual `t+0.5s`, leaving the first half-second context-free. Source-only input for OCR/structure/raw-image baselines, not annotation or result |
| MCIF 304-state flat PP-OCRv6 + PP-StructureV3 source screen | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/ppstructurev3_source_screen_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/09004d4262278b26a1f2f014fdd908427f55797a/ppstructurev3_source_screen_v1>, revision `09004d4262278b26a1f2f014fdd908427f55797a`, tag `mcif-ppstructurev3-source-screen-v1`; Git manifest: `data/manifests/mcif_ppstructurev3_source_screen_v1_20260801.json` | Uploaded and full-remote-redownload checksum verified; `private=True`; 304 rows, 7,123 flat OCR items, 65 strict machine-readable chart/table/formula rows, 17 table-detection-only fallbacks, 0 failed rows. Automatic structure tiers are not labels, recall-complete negatives, ST results or evidence that pixels beat OCR |
| MCIF 304-state R0/R1/R2 source evidence ladder v1 | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/source_evidence_ladder_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen/tree/b13bd2045644f90a6de6be19f77a4af3acaa924f/source_evidence_ladder_v1>, revision `b13bd2045644f90a6de6be19f77a4af3acaa924f`, tag `mcif-source-evidence-ladder-v1`; Git manifest: `data/manifests/mcif_source_evidence_ladder_v1_20260801.json` | Uploaded; `private=True`; 304/304 matched rows, ladder SHA256 `8f77312b...94f7f`; independent rebuild byte-identical; 6 remote files fully re-downloaded and byte-verified. R2 references the canonical native images instead of duplicating them. Still automatic source input, not event labels or an ST effect estimate |
| MCIF 304-state reference-free Qwen3-VL-32B source screen | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/prescreen/qwen3_vl_source_screen_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen>, revision `5da477ff7d199dbded0ffe44d6b41b9cd8c8e75d`, tag `mcif-source-only-qwen3-vl-32b-v1`; Git manifests: `data/manifests/mcif_visual_context_screen_{input_v1,qwen3_vl_32b_v1}_20260801.json` | Uploaded and checksum-byte-verified; `private=True`; 304/304 rows, 21 talks, 0 duplicate/parse failure/empty context; output SHA256 `55a6dafe...79682`; 98 same-prompt repairs then 6 compact repairs. Source-only prescreen, not annotation, sample filter, ST result or pixels-beyond-OCR evidence |
| ACL60/60 dev controlled acoustic v1 | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/corruptions/acl6060_dev_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-controlled-acoustic-dev>, commit `d28c499c8845c4991b5ccea27bc9a2ad520f51fa`, tag `acl6060-controlled-acoustic-v1-20260801` | Uploaded; 75 source-only WAV files plus five metadata/card files; `private=True`; remote metadata and sampled WAV byte-verified |
| ACL60/60 dev source-event annotation workspace v1 | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/workspace_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-events>, revision `3199207c66b159ab39f662a32e0f6d633c9c2b79`, tag `acl6060-source-event-workspace-v1-20260801` | Uploaded; `private=True`; 100 frames, 100 source-only clips, isolated blank A/B sheets; no transcript/target/reference/model output; remote inventory and three downloaded files byte-verified; labels remain pending |
| ACL60/60 source-event author view v2 r4 | `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/author_view_v2_blinded_r4` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-event-author-v2>, revision `bbbbdbf5a2b19c4613791ccffbcf9bc587454e4a`, tag `acl6060-source-event-author-v2-r4-20260801` | Uploaded; `private=True`; 100 secret-HMAC-ID frames, 0 audio; author rows omit talk/timing/raw-media identifiers; scorer mapping/secret excluded; remote inventory and three files byte-verified. r3 revision `2fb266...` is superseded and must not receive labels |
| Chinese-LiPS Qwen3-Omni five-condition visual-control diagnostic | Run: `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801_153400`; bundle: `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/hf_upload/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801_canonical` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, revision `4923b253e87bd94487dace77576ad66e4ea9d8b9`, tag `chinese_lips_visual_controls_v1_qwen3_omni_20260801_canonical`, path `experiments/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801/` | Uploaded and byte-verified; `private=True`; 206 items × 5 conditions, exact run code `f76f9224b9e7017a127499323949b0c2294a27a1`, packaging code `9a564b538cb054b0a15504917916680e2720d07d`; no raw media. Negative mechanism result: `slide-wrong +0.113 chrF [-0.693, 0.894]`, `cross_talk-blank -0.119 [-1.205, 0.824]`, `blank-none +1.545 [0.644, 2.482]`; current-page semantic use not established |
| Chinese-LiPS frame-backed train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`, tag `qwen3_vl_context_v1` | Not uploaded as a separate raw artifact |
| Qwen2.5-VL pilot enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen2.5-VL pilot enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen_vl_context.jsonl` | Do not upload as final; pilot only | Superseded by planned Qwen3-VL run |
| Qwen3-VL HF upload bundle | `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_vl_context_v1` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7`, tag `qwen3_vl_context_v1` | Uploaded; repo privacy verified as `private=True` |
| Qwen3-VL enriched train challenge | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/data/challenge_verified_qwen3_vl_context.jsonl` | `data/challenge_verified_qwen3_vl_context.jsonl.gz` in the private HF repo above | Uploaded; 29,322 rows; QA passed |
| Qwen3-VL enriched train evidence index | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/index/evidence_qwen3_vl_context.jsonl` | `index/evidence_qwen3_vl_context.jsonl.gz` in the private HF repo above | Uploaded; 526,597 rows; internal consistency check passed |
| Qwen3-VL train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen3_vl_context.csv` | `annotation/diagnostic_sample_500_qwen3_vl_context.*` in the private HF repo above | Uploaded; 500 rows |
| Qwen3-VL context QA report | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/qa/qwen3_vl_context_qa.json` | `qa/qwen3_vl_context_qa.json` in the private HF repo above | Uploaded |
| Qwen3-32B repaired reference pilot | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `ee785604ba51a5c65335de12bfcfd99d3c4febff`, tag `qwen3_32b_reference_pilot_20260706`, path `reference_pilots/qwen3_32b_reference_pilot_20260706/` | Uploaded; 100 rows; final audit 84 pass, 16 review, 0 reject |
| Qwen3-32B repaired diagnostic 500 references | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_repaired.jsonl` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `5ca0c090fc6d76ac50938924b28a57b1026c3043`, tag `qwen3_32b_reference_diagnostic500_20260707`, path `reference_pilots/qwen3_32b_reference_diagnostic500_20260707/` | Uploaded; 500 rows; final audit 435 pass, 65 review, 0 reject |
| Qwen3-32B diagnostic 500 context-ablation experiment bundle | `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_32b_diagnostic500_experiments_20260707` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `3cc7249d45eca71a4f0b5c06a6b0773efead128a`, tag `qwen3_32b_diagnostic500_experiments_20260707`, path `experiments/qwen3_32b_diagnostic500_experiments_20260707/` | Uploaded; 7 conditions x 500 outputs plus tables and manifest |
| Qwen3-32B diagnostic 500 batch160 visual/policy sensitivity bundle | `/data/projects/slide_aware_sst_minpaper/repo/outputs/hf_upload/slide-context-sst-chinese-lips/qwen3_32b_diagnostic500_batch160_visual_policy_20260707` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `03f59f1babc0c37e778e8f415bc85ab5fb36f573`, tag `qwen3_32b_diagnostic500_batch160_visual_policy_20260707`, path `experiments/qwen3_32b_diagnostic500_batch160_visual_policy_20260707/` | Uploaded; V4/V5/V6/V8 x 500 outputs plus summary, log, README, and manifest |
| Qwen3-32B diagnostic 500 human review sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_review_sheet_500_qwen3_context_experiments_20260707.csv` | Private repo: <https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips>, commit `3d681ebe85babdacffe5e984bf59af6cade9c2f1`, tag `qwen3_32b_diagnostic500_review_sheet_20260707`, path `annotation/qwen3_32b_diagnostic500_review_sheet_20260707/` | Uploaded; 500 rows; intended for human reference/supporting-evidence/hallucination review |
| Train diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_train/annotation/diagnostic_sample_500_qwen_vl_context.csv` | Same private HF repo if retained as a pilot artifact | Not uploaded |
| Test diagnostic sample sheet | `/data/projects/slide_aware_sst_minpaper/repo/outputs/chinese_lips_test/annotation/diagnostic_sample_500.csv` | TBD | Not uploaded |

Do not make this derived Chinese-LiPS repo public unless upstream permission is
confirmed.

## Historical Compute Context (not current allocation)

- Host: `hyper00`
- Hostname: `node-radixark-16-0000`
- Container: `sglang-omni-jaxan-vision-sst-0701`
- Repo in container: `/data/projects/slide_aware_sst_minpaper`
- Dataset staging: `/data/datasets/chinese_lips`
- HF cache: `/root/.cache/huggingface`
- Resource policy: use at most 2 GPUs by default on Hyper00, and first make the
  active GPU utilization sustain at least 90%. For the current Qwen3-VL
  enrichment, the selected monitored setting is 2 H200 GPUs with 2 workers per
  GPU, `--batch-size 56` per worker, `--max-new-tokens 256`, and
  `--prefetch-batches 1`. A 2026-07-06 Hyper00 short run sustained at least
  91% utilization after warmup on both active GPUs with about 122GB peak memory
  per GPU. The full-train run completed on 2026-07-06. Post-run QA found 3,387
  fallback rows caused by truncated raw model JSON, then a targeted Qwen3-VL
  repair pass replaced exactly those rows. Final QA has 0 missing raw model
  outputs and 0 raw parse failures; the rebuilt evidence index also passes an
  internal source-count consistency check against the repaired challenge.

## Historical Runs

| Run | Host/container | Model | Input | Output | Status |
| --- | --- | --- | --- | --- | --- |
| `chinese_lips_visual_controls_v1_qwen3_omni_2gpu_20260801_132051` | Hyper00 / `sglang-omni-jaxan` | `Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695` | 206-item, five-condition matrix under the run root; input SHA256 `66783d...24105` | Partial throughput diagnostics only under `/data/projects/slide_aware_sst_minpaper/runs/chinese_lips_visual_controls_v1_qwen3_omni_2gpu_20260801_132051` | `SUPERSEDED_DO_NOT_RESUME`; initial missing-`accelerate` failure and later low-utilization partial outputs retained; current heartbeat monitors only the process-prefetch run above |
| `qwen3_vl_train_bs56x2_2gpu_20260706_214711` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | `outputs/chinese_lips_train/data/challenge_verified.jsonl` | final repaired artifacts under `outputs/chinese_lips_train/{data,index,annotation,qa}/` plus shard outputs under `outputs/chinese_lips_train/enrichment/qwen3_vl_train_bs56x2_2gpu_20260706_214711/` | Completed and repaired locally; final QA has 29,322 challenge rows, 526,597 evidence rows, 500 diagnostic rows, 0 missing raw model outputs, 0 raw parse failures; uploaded to private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` |
| `qwen3_parse_failure_repair512_20260706_231750` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | 3,387 failed Qwen3-VL rows from the initial combined artifact | repair outputs under `outputs/chinese_lips_train/repair/qwen3_parse_failure_repair512_20260706_231750/` | Completed locally; 512/768/compact/strict passes repaired all initial parse failures |
| `qwen3_32b_reference_pilot_20260706` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | 100 rows from private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` diagnostic sample | `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/pilot_100_refs_repaired.jsonl` and HF path `reference_pilots/qwen3_32b_reference_pilot_20260706/` | Completed; uploaded to private HF commit `ee785604ba51a5c65335de12bfcfd99d3c4febff`; tag `qwen3_32b_reference_pilot_20260706` |
| `qwen3_32b_reference_diagnostic500_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | 500 rows from private HF revision `a83770446ded4599bf9d95d2b77cdcc7fe359ef7` diagnostic sample | `outputs/chinese_lips_train/reference_generation/qwen3_32b_hf_revision_a837704/diagnostic_500_refs_repaired.jsonl` and HF path `reference_pilots/qwen3_32b_reference_diagnostic500_20260707/` | Completed; batch=40 after batch=48 OOM; final audit 435 pass, 65 review, 0 reject; uploaded to private HF commit `5ca0c090fc6d76ac50938924b28a57b1026c3043`; tag `qwen3_32b_reference_diagnostic500_20260707` |
| `qwen3_32b_diagnostic500_experiments_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | repaired diagnostic 500 references plus Qwen3-VL evidence index | 7 condition outputs under `outputs/chinese_lips_train/experiments/qwen3_32b_diagnostic500/` and HF path `experiments/qwen3_32b_diagnostic500_experiments_20260707/` | Completed; uploaded to private HF commit `3cc7249d45eca71a4f0b5c06a6b0773efead128a`; tag `qwen3_32b_diagnostic500_experiments_20260707`; diagnostic self-BLEU only, not a method ranking until independent references and uniform-batch checks are added |
| `qwen3_32b_diagnostic500_batch160_visual_policy_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-32B` | same repaired diagnostic 500 references and evidence index | V4/V5/V6/V8 outputs under `outputs/chinese_lips_train/experiments/qwen3_32b_diagnostic500/runs_batch160_visual_policy/` and HF path `experiments/qwen3_32b_diagnostic500_batch160_visual_policy_20260707/` | Completed; uploaded to private HF commit `03f59f1babc0c37e778e8f415bc85ab5fb36f573`; tag `qwen3_32b_diagnostic500_batch160_visual_policy_20260707`; sensitivity only |
| `qwen3_32b_diagnostic500_review_sheet_20260707` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | n/a | repaired diagnostic 500 plus parent experiment outputs | review CSV at `outputs/chinese_lips_train/annotation/diagnostic_review_sheet_500_qwen3_context_experiments_20260707.csv` and HF path `annotation/qwen3_32b_diagnostic500_review_sheet_20260707/` | Completed; uploaded to private HF commit `3d681ebe85babdacffe5e984bf59af6cade9c2f1`; tag `qwen3_32b_diagnostic500_review_sheet_20260707` |
| `qwen3_vl_train_20260706_164650` | Hyper00 / `sglang-omni-jaxan-vision-sst-0701` | `Qwen/Qwen3-VL-8B-Instruct` | `outputs/chinese_lips_train/data/challenge_verified.jsonl` | partial shards under `outputs/chinese_lips_train/enrichment/qwen3_vl_train_20260706_164650/` | Paused/superseded; do not resume with old one-sample-per-process settings |

## Current Durable Decisions (updated 2026-08-01)

1. The final paper identity is intentionally not frozen before development
   evidence. The research space is persistent, pre-available slide semantics
   for SimulST without putting vision on the online critical path.
2. ACL dev supports explicit story discovery across four routes: current-slide
   attribution/anticipation, noisy-speech robustness, pixels beyond OCR, and
   evidence selection/integration. Report the complete declared development
   matrix rather than hiding unsuccessful routes.
3. Correct versus time/type/budget-matched stale/wrong evidence is a shared
   content-use validity control. The candidate early-commit risk difference,
   final quality, AL/LAAL, robustness interactions, event accuracy,
   hallucination, and runtime all remain development outcomes.
4. Select the final story using development effect size, cross-talk stability,
   failure analysis, and method value. Then commit/push one frozen main claim,
   primary metric, SESOI, model/config, slices, and decision rule before any
   held-out ACL eval or MCIF result is read.
5. Generic whole-PDF prompts, named entities/abstracts, terminology memory,
   phrase boosting, pretranslation and BM25/RAG are mandatory baselines, not
   contributions.
6. EGTA and RASST are terminology-only. They close a term-only fallback but do
   not close proposition, discourse, relation or vision-aware context.
7. Precompilation does not grant future-slide access. PDF memory is available
   before the talk; slide-derived entries unlock only at the real stable-slide
   timestamp. Deck-known-in-advance is a labeled ablation, not the primary run.
8. Primary events use a source-side forced choice and an annotator interval
   `[t_last_insufficient, t_first_sufficient]`; events with unresolved boundary
   disagreement or slide-transition ambiguity are excluded from primary.
9. Freeze candidate inventory, source-only evidence packets, and target scoring
   as three independent artifacts. Packet builders never read references or
   target translations.
10. Before full automatic C3-C6 implementation, run a blind event-density screen
    and multi-route oracle capability map on ACL dev, including OCR, semantic
    evidence, visual relations, matched wrong evidence, and native/noisy audio.
11. Lip video and slide+lip hybrid experiments remain out of scope. AVMuST-TED
   is retained only as related-work and historical licensing context.
12. The frozen five-condition Chinese-LiPS result does not establish
    correct-slide content use even after adding unrelated-talk and blank-image
    controls: `slide-wrong` and `cross_talk-blank` are unresolved while
    `blank-none` is positive. Treat it as generic vision-slot/decoding
    perturbation and a closed private diagnostic, not paper evidence.
13. The verified *Do Slides Help?* supplement contains 884 real ACL60/60 video
    frames. Its metadata includes source transcripts, so only a stripped
    frame-only manifest may enter inference.
14. Measure slide dwell and slide-to-speech lead/lag rather than treating the
    estimated 30-60 s window as fact.
15. Report cold pre-talk compilation, evidence-ready timing, on-path lookup,
    packet tokens, GPU seconds/RTF, and computation-aware AL/LAAL separately.
16. ACL eval is only a five-talk replication pilot. MCIF is confirmatory only
    after at least 15/21 talks have eligible events and its video hashes, slide
    timelines, transition QA, inference-safe manifests, and MDE gate pass.
17. Use Qwen pseudo references for development only; paper ranking requires
    human or independently produced references.
18. Before any MCIF output, push frozen route, config, prompt, model, selector,
    metric and noise hashes. Run inference without mounted references and score
    only after the append-only 21-talk completion ledger closes.
19. Git stores code/contracts/results; Hugging Face stores reusable data/model
    artifacts. Chinese-LiPS remains private diagnostic data, not the main paper.
20. MCIF input-side visual readiness passes at the frozen v2 detector settings:
    1 s sampling, 283 transition candidates, 304 causal states and 21/21
    transition-sheet review. This is not an ST result and does not pass the
    eligible-event, MDE or output-side confirmatory gates.
21. The SimulST-specific target is not merely final BLEU. The main evidence must
    include earlier first-stable correct target decisions while the source audio
    remains insufficient, correct-over-stale/wrong content specificity, final
    quality preservation and controlled-noise interaction.
22. Controlled-acoustic v1 uses full continuous talks, a source-only energy VAD,
    exact source hashes/offsets/seeds, 5-speaker babble at +10/+5/0/-5 dB, and
    a disjoint real-RIR split. Its 75 ACL dev files are input readiness, not an
    ST result or evidence that synthetic noise represents real conferences.
23. ACL dev visual v1 uses every observed midpoint frame as a causal state from
    its filename timestamp. The 97 high-precision pixel transition candidates
    cannot compress the stream because a 60-row negative audit found clear
    white-slide text/reveal misses. Before the first frame there is no visual
    state, and no frame is backdated to a source-segment boundary.
24. The first event-density screen is frozen at 100 ACL dev packets: 10
    high-precision change states and 10 hash-random states per talk. Source
    questions, forced-choice options and 0.96 s insufficiency/sufficiency
    boundaries require two independent annotations; target realization is a
    later, physically separate stage.
25. Automatic source/OCR alignment is diagnostic only: 149 independent
    segment-frame exact-match events establish nonzero anticipation headroom,
    but the 901 overlapping n-grams and 344 de-nested candidates are not human
    event counts and cannot establish ST or pixels-beyond-OCR gains.
26. v1 A/B annotation sheets are superseded. v2 uses one hash-locked canonical
    question per packet, two audio validators plus a disjoint two-person frame
    cohort, opaque per-validator option IDs/order, complete talk-start causal-
    audio trajectories, and immutable scorer-side agreement records.
27. The balanced 50/50 transition/random seed reports stratified yield. Overall
    observation-level prevalence requires the frozen per-talk/stratum inclusion
    probabilities; raw eligible/100 is not a natural event-density estimate.
28. Static causal WAV bundles are intermediate media artifacts, not proof of
    sequential annotation. The v2 HTTP backend enforces question-only lock and
    stepwise prefix release, and `freeze-audio` verifies its hash-chained event
    log. Formal deployment must also deny direct audio-root filesystem access.
29. Validator conflicts remain missing (`primary_eligible=null`) until a
    report-hash-bound adjudication is locked. Overall 468-observation prevalence
    uses the frozen stratified finite-population estimator and is suppressed
    while any adjudication or source exclusion remains unresolved. Hard failures
    such as question-only answerability cannot be adjudicated positive.
30. Author packet IDs use a scorer-only HMAC secret, not a public fixed salt.
    The secret and true mapping remain outside Git and author-facing HF; only
    their SHA256 provenance is recorded.
31. Opaque IDs are not sufficient when distributed rows include linkable
    quasi-identifiers. Stage-1 author, post-lock author-audio and frame-validator
    sheets omit talk ids, absolute timing and raw media hashes; frame stages use
    scorer-secret bindings. Private author media manifests, audio-validator
    timing tasks and logs remain scorer/server-side, and the HTTP UI exposes
    only prefix indices. This is operational blinding, not a claim of anonymity
    against an adversary who corpus-matches the media bytes.
32. ACL dev event scoring uses nineteen frozen inputs. A pre-run contract binds
    source timing/media/extractor identities, source artifact tree, source-only
    evidence packets, matched controls, scientific/scoring config, target/outcome
    commitments, condition matrix, tokenizer, full model artifact tree, broker
    and worker identities. A
    post-run attestation separately binds exact trajectories, release log and
    start/end runtime audits; trajectories point only to the pre-run contract,
    so no circular hash or post-hoc hypothesis rewrite is accepted. Raw source/
    target annotation reports and adjudication artifacts are frozen in a
    scorer-private tree before output.
    Canonical audio sources bind talk/acoustic identity, complete `float32le`
    bytes, upstream WAV/corruption provenance, materializer revision and every
    prefix hash. The implemented external Unix-socket broker enforces alternating
    prefix release and exact-hypothesis observation commit per session and writes
    server-ordinal, timestamped, hash-chained interactions; scorer rehashes
    source/provenance/prefix/hypothesis bytes and
    checks `sample_count / sample_rate == audio_time_sec`. One talk-level frontier
    spans all event/model/acoustic streams, so no stream advances past a time
    until every scheduled observation at that time commits. Sessions are bound
    to one stream and each talk allows only one in-flight release, including
    among clean/noisy streams at the same time.
    Docker/container/worker evidence is captured twice from live
    `docker inspect`, `/proc` and Git state after exact-token marker discovery
    plus descendant enumeration. The pre-run contract builder binds the complete
    start-audit hash before atomically releasing a ready-file barrier; worker
    commands must bind both paths and the worker-side helper rehashes the contract
    before generation. Scientific config and model tree require exact read-only
    host→worker mounts, and start/end audits require the same container and mount
    topology. Formal runs require read-only rootfs plus `network=none`;
    every Python child entrypoint stays inside the clean
    worktree; start/end process identity trees include PID/PPID/process-start ticks
    and must match, so per-GPU environments may differ but worker restarts cannot
    pass unnoticed. Container destinations, host mount sources and scoring
    protected roots are separate namespaces, and actual target/outcome/audio
    roots are linked to the forbidden host set. The production Qwen3-Omni worker
    now consumes one contract snapshot, rehashes model/tokenizer trees before
    load and after generation, replays every evidence packet with the processor's
    actual tokenizer, uses a distinct session for every event/condition/acoustic
    stream, and never persists cross-prefix model/audio cache. Its done marker
    binds contract/schedule/evidence hashes, deterministic talk partition,
    PID/process-start ticks, audited process tree and canonical shard path. The
    merger checks the ready marker and maps every shard to one exact start-audit
    worker command before accepting the complete matrix. The current broad writable
    `/data` diagnostic container cannot satisfy this isolation; fresh paper-grade
    generation remains blocked on rebuilding the same canonical container after
    active work and completing human outcome artifacts. The primary
    development estimand is talk-equal early stable-correct risk difference;
    an isolated final correct point, an early-terminated trajectory or a
    model/acoustic-condition-specific audio grid cannot count as gain. OCR,
    semantic and relation conditions use separate controls whose type, time,
    token count and packet hashes are machine-checked. Strict schemas reject
    target/reference/future-audio leakage fields. Development thresholds are
    explicitly exploratory point-estimate screens, not formal non-inferiority
    tests. The 12 babble files are
    grouped as four SNR levels with three fixed seeds, not 12 independent
    samples; uncertainty is clustered by talk. This is scorer readiness, not
    a result.
33. The narrow claim “pre-audio slides make SimulST targets appear earlier” is
    Level 2 high overlap, chiefly with OmniFusion on problem, insight and
    scientific-talk application. The paper cannot claim first visual/slide
    anticipation. Its viable delta is event-level first-stable-correct timing
    before audio sufficiency, correct-over-wrong/stale/empty attribution, OCR
    separation and a controlled acoustic-noise interaction. This remains a
    planned empirical delta, not a result.
34. The complete MCIF Qwen3-VL-32B source-only screen has 303/304 spatial-
    relation descriptions, but these are unverified model outputs and often
    trivial layout statements. The screen establishes annotation material, not
    `vision > OCR`; only blinded event labels and controlled system comparisons
    can answer that question.
35. A lexical diagnostic narrows those 303 rows to 192 model-described
    structural candidates across all 21 talks, while 111 are simple-layout-only.
    An agent spot check also found one hallucinated arrow. The required baseline
    ladder is therefore unordered OCR, layout/structure-preserving text, and raw
    image; beating only unordered OCR does not establish that pixels are needed.
36. The matched native-frame PP-OCRv6/PP-StructureV3 screen is complete at
    private HF revision `09004d42...5797a`: 304/304 rows, 65 strict
    machine-readable non-flat rows and 17 table-detection-only fallbacks. A
    44-row visual QA found genuine structures in positive samples and clear
    false negatives in layout/plain strata. These outputs freeze R0/R1 inputs
    but cannot filter states, define labels or establish `vision > OCR`.

## Current Next Actions (2026-08-01)

1. Treat the completed Chinese-LiPS five-condition run as a closed negative
   mechanism diagnostic. It isolates a generic vision-slot/decoding effect, not
   current-page content use; do not spend further GPU on naive Chinese-LiPS
   raw-image prompting or promote this single-talk result to paper evidence.
2. Complete v2 frame-only canonical item authoring and freeze question hashes;
   use the localhost-only blinded authoring UI and its separate working sheet,
   then run `freeze-author`. Deploy the implemented prefix gate to two
   independent audio validators, then
   use the implemented localhost-only frame-validation UI with a disjoint
   two-person cohort. Report all
   negatives, right-censoring, agreement and adjudication without modifying raw
   sheets.
3. Treat MCIF morphology, native-frame OCR and structured-text extraction as
   completed input readiness. Never expose row-level model suggestions to
   annotators, treat automatic tiers as verified negatives, or alter the frozen
   304-state inventory. Freeze source-only packets and target scoring
   independently, then map document-only, unordered OCR, layout/structure-
   preserving text, raw image, matched wrong evidence, and native/noisy audio
   headroom. A stable signal in any route can
   justify focused automatic implementation; stop only if every gold route lacks
   practical headroom. Use the frozen event scorer in
   [`ACL6060_EVENT_TRAJECTORY_SCORING_V1.md`](ACL6060_EVENT_TRAJECTORY_SCORING_V1.md);
   do not substitute aggregate BLEU/AL for its content-specific timing outcome.
4. After the human freeze, rebuild the canonical inference container with
   read-only rootfs, `network=none`, narrow read-only model/tokenizer/config
   mounts and no target/reference/full-audio mount. Run the implemented causal
   workers and audited merger before any scoring artifact is exposed.
5. Run small automatic comparisons for viable routes, including naive prompts,
   selection/gating, and direct-image input. Preserve the complete declared
   development matrix.
6. Select the final paper story from development evidence, then freeze and push
   its main claim, metric, model/config, slices, and decision rule before ACL
   eval or MCIF is read.
