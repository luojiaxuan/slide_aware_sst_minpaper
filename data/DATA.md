# Data locations

All reusable datasets live on Hugging Face (namespace `gavinlaw`); this directory
only documents locations and local staging conventions. No media is committed to git.

## Published datasets

| Dataset | Content | License | Role |
|---|---|---|---|
| [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) | mTEDx-V: talk-level long-form X→En manifests (es/fr/it/ru/el→en, 100 talks, ~18 h), `talk_id` = live YouTube ID, `visual_signal/` OCR stratification | CC BY-NC-ND 4.0 (manifests only, no media) | Realistic-noisy X→En eval stratum |
| [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) | Chinese-LiPS long-form: `orig_timeline` (real gaps restored) + `silence_removed` variants, 3 talks ~97 min; `en_draft_demo/` sample refs | CC BY-NC-SA 4.0 | Clean-visual zh→En stratum (En refs in progress) |
| [gavinlaw/slide-aware-sst-controlled-acoustic-dev](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-controlled-acoustic-dev) | ACL60/60 dev full-talk controlled acoustic variants; 75 source-only PCM16 WAV files plus exact manifests/contracts | CC BY 4.0 / Apache 2.0 provenance | Private Phase-A robustness inputs; commit `d28c499c8845c4991b5ccea27bc9a2ad520f51fa`, tag `acl6060-controlled-acoustic-v1-20260801` |
| [gavinlaw/slide-aware-sst-acl6060-source-events](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-events) | 100 ACL dev frame/audio packets and historical blank v1 A/B sheets | CC BY 4.0 | Private; revision `3199207c66b159ab39f662a32e0f6d633c9c2b79`, tag `acl6060-source-event-workspace-v1-20260801`; v1 sheets superseded by v2 |
| [gavinlaw/slide-aware-sst-acl6060-source-event-author-v2](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-acl6060-source-event-author-v2) | v2 r3 frame-only author view: 100 HMAC-opaque frames + blank authoring schema | CC BY 4.0 | Private; revision `2fb266d168e0abbf4ace17d3f5de9503a8c46cd6`, tag `acl6060-source-event-author-v2-r3-20260801`; 0 audio, scorer mapping/secret excluded |
| [gavinlaw/slide-aware-sst-mcif-source-prescreen](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-source-prescreen) | MCIF native causal frames, source-only VLM morphology screen, matched PP-OCRv6 / PP-StructureV3 evidence, portable R0/R1/R2 ladder, Qwen3-Omni visual-token controls and materialized wrong-image media | CC BY 4.0 provenance | Private; media revision `0001171cf661d605c6fa344df7cd3f90d291d194`, tag `mcif-qwen3-omni-visual-control-media-v1`; 304 rows, no transcript/reference/audio input |
| [gavinlaw/slide-aware-sst-mcif-outcomes](https://huggingface.co/datasets/gavinlaw/slide-aware-sst-mcif-outcomes) | Official MCIF segment timing and En/Zh/De/It references, R0/R1/R2 reference-aware automatic candidates, and isolated human-author workspaces | Official benchmark outcome material; redistribution permission not assumed | Private; historical reliability workspace revision `eb194d83c941838db2b096fe52c5e455c5b304bb` is superseded at 0 labels; earlier inventory/workspace revisions remain separately tagged; must never enter inference |

## Phase-A frozen sources

| Dataset | Immutable source | Frozen scope | Git manifest |
|---|---|---|---|
| ACL60/60 | ACL Anthology attachment, SHA256 `5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc` | dev/eval 5+5 complete talks, 468+416 gold segments, CC BY 4.0 | [`manifests/phase_a_sources_20260731.json`](manifests/phase_a_sources_20260731.json), [`manifests/acl6060_talks_20260731.jsonl`](manifests/acl6060_talks_20260731.jsonl) |
| *Do Slides Help?* ACL60/60 frame supplement | Figshare v2 `Visual_ASR.zip`, SHA256 `f771d3f6f03026ad1510cf6840b47df3406b06b804926ab3ae18af99f663d4cc` | 884 real talk-video frames over all 10 ACL60/60 talks, CC BY 4.0; dev 468 imported with no-backdating policy, eval untouched | [`manifests/do_slides_help_figshare_v2_20260731.json`](manifests/do_slides_help_figshare_v2_20260731.json), [`manifests/acl6060_dev_frame_observations_v1_20260801.jsonl`](manifests/acl6060_dev_frame_observations_v1_20260801.jsonl) |
| MCIF media pool | `FBK-MT/MCIF` revision `e24065b919758263cfe5d157057278affe76ea7b` | 100 long audio/video talks, CC BY 4.0 | [`manifests/phase_a_sources_20260731.json`](manifests/phase_a_sources_20260731.json), [`manifests/mcif_files_e24065b9.jsonl`](manifests/mcif_files_e24065b9.jsonl) |
| MCIF IWSLT translation subset | official `mcif-long-trans.zip`, SHA256 `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7` | source-side milestone: 21 audio/video/PDF talks and 919 segment timings materialized while references remained sealed; later outcome-only extraction is recorded separately below | [`manifests/mcif_translation_subset_materialized_20260801.json`](manifests/mcif_translation_subset_materialized_20260801.json) |
| MCIF visual states v2 | derived from the frozen 21 MP4s at 1 s intervals | 7,111 frames; 283 reviewed transition candidates; 304 causal states; 0 unconfirmed candidates | [`manifests/mcif_visual_state_candidates_v2_20260801.json`](manifests/mcif_visual_state_candidates_v2_20260801.json), [`manifests/mcif_visual_qa_20260801.json`](manifests/mcif_visual_qa_20260801.json) |
| MCIF flat OCR + structured text v1 | derived from all 304 native causal frames, source-only | 304 flat PP-OCRv6 rows; 65 strict machine-readable chart/table/formula rows; 17 table-detection-only fallbacks; private HF revision `09004d42...5797a` | [`manifests/mcif_ppstructurev3_source_screen_v1_20260801.json`](manifests/mcif_ppstructurev3_source_screen_v1_20260801.json) |
| MCIF source evidence ladder v1 | deterministic join of the frozen native and PPStructure artifacts | 304 matched R0 flat-text / R1 structure-preserving text / R2 raw-image references; private HF revision `b13bd204...924f`; no state filtering or image duplication | [`manifests/mcif_source_evidence_ladder_v1_20260801.json`](manifests/mcif_source_evidence_ladder_v1_20260801.json) |
| MCIF Qwen3-Omni visual-token controls v1 | frozen processor over every R2 image plus source-only stale/wrong candidate construction | 304 visual-token rows; 304 cross-talk controls and 283 causally-prior same-talk controls; 101 cross-talk candidates require a verified deterministic fit-and-pad transform; private HF revision `b2c9a409...af59f` | [`manifests/mcif_qwen3_omni_visual_token_controls_v1_20260801.json`](manifests/mcif_qwen3_omni_visual_token_controls_v1_20260801.json) |
| MCIF Qwen3-Omni visual control media v1 | deterministic materialization of the 101 fit-and-pad wrong-image controls; identity controls retain canonical native references | 101 PNGs / 25.58 MB; 587 final processor inputs verified; 105-file private HF bundle at revision `0001171c...d194`; state-level source media, not event packets | [`manifests/mcif_qwen3_omni_visual_control_media_v1_20260801.json`](manifests/mcif_qwen3_omni_visual_control_media_v1_20260801.json) |
| MCIF outcome candidate inventory v1 | official references aligned to segment timing, then joined only after source-side artifacts were frozen | 919 multilingual reference rows; 954 high-recall lexical candidates, 689 with at least 5 s lead and 458 with at least 10 s lead; private HF revision `64dee522...dc5c`; human event/audio-sufficiency labels pending | [`manifests/mcif_outcome_candidate_inventory_v1_20260801.json`](manifests/mcif_outcome_candidate_inventory_v1_20260801.json), [`MCIF_OUTCOME_CANDIDATE_DATASET_CARD.md`](MCIF_OUTCOME_CANDIDATE_DATASET_CARD.md) |
| MCIF beyond-OCR R1/R2 candidate inventory v1 | strict actual-block R1 lexical recovery plus source-only VLM R2 proposal generation, joined to references only after all source inputs were frozen | R1 strict 2 candidates / 2 talks; R2 semantic 150 candidates / 21 talks / 118 segments, 122 with lead≥5 s and 86 ≥10 s; `ocr_text` and current R0/R1 candidates excluded; private HF revision `01defe41...d691`; all human labels pending | [`manifests/mcif_beyond_ocr_candidate_inventory_v1_20260801.json`](manifests/mcif_beyond_ocr_candidate_inventory_v1_20260801.json), [`MCIF_BEYOND_OCR_CANDIDATE_DATASET_CARD.md`](MCIF_BEYOND_OCR_CANDIDATE_DATASET_CARD.md) |
| MCIF beyond-OCR independent validation workspace v1 | deterministic replay of all 152 frozen R1/R2 candidates into physically separate role views | 152 visual-validator + 152 target-author items, 91 opaque native media, separate scorer mapping, 0 labels and 0 machine-detected forbidden-field leaks; private HF revision `861401f2...e6db`; 102-file remote redownload byte-verified | [`manifests/mcif_beyond_ocr_validation_workspace_v1_20260801.json`](manifests/mcif_beyond_ocr_validation_workspace_v1_20260801.json), [`MCIF_BEYOND_OCR_CANDIDATE_DATASET_CARD.md`](MCIF_BEYOND_OCR_CANDIDATE_DATASET_CARD.md) |
| MCIF beyond-OCR independent validation protocol v1 | historical role validators, localhost UIs, create-once freezes and scorer-only join over the frozen dual-role workspace | **Superseded before labels** at visual 0/152 + target 0/152; dual-role was not role replication and visual judgment exposed pixels/VLM before text sufficiency; ports stopped; no production labels allowed | [`manifests/mcif_beyond_ocr_validation_v1_supersession_20260801.json`](manifests/mcif_beyond_ocr_validation_v1_supersession_20260801.json), [`../docs/MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md`](../docs/MCIF_BEYOND_OCR_RELIABILITY_AUDIT_20260801.md) |
| MCIF beyond-OCR reliability workspace/protocol v2 | full-overlap visual A/B with physical R0→R1→pixels→descriptor releases; target author plus two-stage independent validator; signed run contract; role capabilities; external event-head ledger; raw reliability gate before adjudication | Historical 152-candidate / 0-label HF revision `eb194d83...304bb` is `SUPERSEDED_BEFORE_LABELS_UNAUTHENTICATED_UNSIGNED_CONTRACT`; hardened protocol Git `8a263a6c`, config `e3978733...98d7b`, server/UI and security review complete; replacement production artifact waits for six real identities/tokens | [`manifests/mcif_beyond_ocr_reliability_workspace_v2_20260801.json`](manifests/mcif_beyond_ocr_reliability_workspace_v2_20260801.json), [`../docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md`](../docs/MCIF_BEYOND_OCR_RELIABILITY_V2.md), [`../docs/MCIF_BEYOND_OCR_SECURITY_REVIEW_20260801.md`](../docs/MCIF_BEYOND_OCR_SECURITY_REVIEW_20260801.md) |
| MCIF En→Zh target-event author workspace v1 | exhaustive grouping of all 954 R0 lexical candidates by candidate-bearing segment | 355 shuffled author items, 173 hash-bound slide images, separate scorer mapping, 0 human labels; private HF revision `0785a37f...12a1b`; future audio validators must not access this artifact | [`manifests/mcif_target_event_author_workspace_v1_20260801.json`](manifests/mcif_target_event_author_workspace_v1_20260801.json), [`MCIF_OUTCOME_CANDIDATE_DATASET_CARD.md`](MCIF_OUTCOME_CANDIDATE_DATASET_CARD.md) |
| MCIF En→Zh target-event annotation protocol v1 | machine-readable author/freeze gates over the frozen 355-item workspace | config SHA256 `95b8dc69...60b9c`; UI/freezer Git `b6cd276`; current human progress 0/355; R0 lexical scope only | [`manifests/mcif_target_event_annotation_protocol_v1_20260801.json`](manifests/mcif_target_event_annotation_protocol_v1_20260801.json), [`../docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md`](../docs/MCIF_TARGET_EVENT_ANNOTATION_V1.md) |
| Controlled acoustic sources | MUSAN/SLR17 plus RIR/Noise SLR28 | archive SHA256/license snapshots frozen; 65 development and 65 disjoint confirmatory sources | [`manifests/controlled_acoustic_source_contract_20260801.json`](manifests/controlled_acoustic_source_contract_20260801.json), [`manifests/controlled_acoustic_source_pool_20260801.json`](manifests/controlled_acoustic_source_pool_20260801.json) |
| ACL60/60 dev acoustic variants v1 | private HF commit `d28c499c8845c4991b5ccea27bc9a2ad520f51fa` | 5 talks × 15 conditions = 75 PCM16 WAV files; no references/transcripts | [`manifests/acl6060_dev_controlled_acoustic_v1_20260801.jsonl`](manifests/acl6060_dev_controlled_acoustic_v1_20260801.jsonl) |
| ACL60/60 dev source-event seed v1 | derived from frame-only observations; no transcript/target/model output | 100 pending packets: 20 per talk, balanced change/random strata | [`annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl`](annotations/acl6060_dev_source_event_seed_v1_20260801.jsonl), [`../docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`](../docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md) |
| ACL60/60 dev automatic anticipation diagnostic | source-only segment timing + Tesseract OCR over 468 real frames | 149 independent segment-frame matches; automatic headroom only, human audit pending | [`manifests/acl6060_dev_ocr_anticipation_v1_20260801.json`](manifests/acl6060_dev_ocr_anticipation_v1_20260801.json) |
| ACL60/60 dev source-event workspace v1 | private HF revision `3199207c66b159ab39f662a32e0f6d633c9c2b79` | 100 frame/audio packets and isolated A/B sheets; no transcript/target/reference/model output; labels pending | [`manifests/acl6060_dev_source_event_workspace_v1_20260801.json`](manifests/acl6060_dev_source_event_workspace_v1_20260801.json) |
| ACL60/60 dev source-event annotation v2 | private HF revision `2fb266d168e0abbf4ace17d3f5de9503a8c46cd6` | 100 HMAC-opaque frame-only author packets; 0 audio; scorer mapping/secret/stratum physically excluded; no labels | [`manifests/acl6060_dev_source_event_annotation_v2_20260801.json`](manifests/acl6060_dev_source_event_annotation_v2_20260801.json), [`manifests/acl6060_dev_source_event_sampling_design_v2_20260801.json`](manifests/acl6060_dev_source_event_sampling_design_v2_20260801.json) |

ACL60/60 dev 的 inference/scoring bundle 只在本地 staging；Git 仅保存
[`manifests/acl6060_dev_simulstream_20260731.json`](manifests/acl6060_dev_simulstream_20260731.json)
中的行数与 hashes，不保存 transcript/reference 内容。

完整 provenance、生成命令、推理/评分隔离和 runner revisions 见
[`../docs/PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](../docs/PHASE_A_DATA_RUNNER_FREEZE_20260731.md)。

## Upstream sources

- [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS) — 1080p dedicated PPT stream + FACE + WAV per segment; test split = 21 videos / 3,908 clips / ~9 h
- [deepdml/mtedx](https://huggingface.co/datasets/deepdml/mtedx) — mTEDx mirror used to build mTEDx-V
- [FBK-MT/MCIF](https://huggingface.co/datasets/FBK-MT/MCIF) — 100-talk ACL scientific media pool; its 21-talk translation subset supplies En→{De,It,Zh} and the IWSLT 2026 development protocol, CC-BY 4.0
- [ACL60/60](https://aclanthology.org/2023.iwslt-1.2/) — 10 complete ACL talks, professional multilingual references and tagged terminology, CC BY 4.0
- [Do Slides Help? code and data](https://figshare.com/articles/software/Code_and_data/30158932) — real midpoint video frames aligned to all ACL60/60 segments; bundled metadata also contains source transcripts and must be stripped before inference

## Local staging (not in git)

- Current Phase-A staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/`.
- MCIF inference-safe materialization and QA:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9`.
- MCIF PP-OCRv6/PP-StructureV3 source-only staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/ppstructurev3_source_screen_v1`.
- MCIF portable R0/R1/R2 ladder staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/source_evidence_ladder_v1`.
- MCIF Qwen3-Omni visual-token/control-spec staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/mcif_visual_token_controls_v1`.
- MCIF materialized visual-control media staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/materialized/e24065b9/evidence/mcif_visual_control_media_v1`.
- MCIF outcome candidate inventory staging（outcome-only，禁止挂载到 inference）：
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/outcomes/mcif_outcome_candidate_inventory_v1`.
- MCIF beyond-OCR R1/R2 candidate staging（outcome-only，未人工验证）：
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/outcomes/mcif_beyond_ocr_candidate_inventory_v1`.
- MCIF beyond-OCR dual-role validation workspace staging（每位 annotator 只能访问一个 role subtree）：
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_beyond_ocr_validation_workspace_v1`.
- MCIF En→Zh target-event author workspace staging（仅 author view 可分发）：
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/mcif/annotation/mcif_target_event_author_workspace_v1`.
- *Do Slides Help?* Figshare v2 staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/do-slides-help/figshare-v2`.
- Controlled-acoustic source pools and ACL dev variants:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/noise/`.
- ACL dev source-event OCR, automatic diagnostic and double-annotation workspace:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v1/`.
- ACL dev blinded v2 author/scorer views:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/annotation/acl6060_source_event_v2/`.
- Historical staging: `~/research_idea/data_prep/` (`mtedx_videos/` 3.0G probe
  videos, `chinese_lips/` 1.9G raw+rebuilt audio).

Raw upstream media stays local as cache. Git stores only scripts, manifests and
contracts; no MCIF media/reference content is committed or re-uploaded. The
reusable ACL60/60 controlled-acoustic variants are versioned separately on the
private Hugging Face dataset recorded in Source of Truth.
