# Data locations

All reusable datasets live on Hugging Face (namespace `gavinlaw`); this directory
only documents locations and local staging conventions. No media is committed to git.

## Published datasets

| Dataset | Content | License | Role |
|---|---|---|---|
| [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) | mTEDx-V: talk-level long-form X→En manifests (es/fr/it/ru/el→en, 100 talks, ~18 h), `talk_id` = live YouTube ID, `visual_signal/` OCR stratification | CC BY-NC-ND 4.0 (manifests only, no media) | Realistic-noisy X→En eval stratum |
| [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) | Chinese-LiPS long-form: `orig_timeline` (real gaps restored) + `silence_removed` variants, 3 talks ~97 min; `en_draft_demo/` sample refs | CC BY-NC-SA 4.0 | Clean-visual zh→En stratum (En refs in progress) |

## Phase-A frozen sources

| Dataset | Immutable source | Frozen scope | Git manifest |
|---|---|---|---|
| ACL60/60 | ACL Anthology attachment, SHA256 `5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc` | dev/eval 5+5 complete talks, 468+416 gold segments, CC BY 4.0 | [`manifests/phase_a_sources_20260731.json`](manifests/phase_a_sources_20260731.json), [`manifests/acl6060_talks_20260731.jsonl`](manifests/acl6060_talks_20260731.jsonl) |
| *Do Slides Help?* ACL60/60 frame supplement | Figshare v2 `Visual_ASR.zip`, SHA256 `f771d3f6f03026ad1510cf6840b47df3406b06b804926ab3ae18af99f663d4cc` | 884 real talk-video frames over all 10 ACL60/60 talks, CC BY 4.0; frame-only import pending | [`manifests/do_slides_help_figshare_v2_20260731.json`](manifests/do_slides_help_figshare_v2_20260731.json) |
| MCIF media pool | `FBK-MT/MCIF` revision `e24065b919758263cfe5d157057278affe76ea7b` | 100 long audio/video talks, CC BY 4.0 | [`manifests/phase_a_sources_20260731.json`](manifests/phase_a_sources_20260731.json), [`manifests/mcif_files_e24065b9.jsonl`](manifests/mcif_files_e24065b9.jsonl) |
| MCIF IWSLT translation subset | official `mcif-long-trans.zip`, SHA256 `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7` | 21 talk IDs and filenames frozen; reference contents unopened | [`manifests/phase_a_sources_20260731.json`](manifests/phase_a_sources_20260731.json) |

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
- *Do Slides Help?* Figshare v2 staging:
  `/Users/luojiaxuan/Documents/ResearchStudio/data/vision-aware-sst/do-slides-help/figshare-v2`.
- Historical staging: `~/research_idea/data_prep/` (`mtedx_videos/` 3.0G probe
  videos, `chinese_lips/` 1.9G raw+rebuilt audio).

Raw upstream media stays local as cache. Git stores only scripts, manifests and
contracts; no ACL60/60 or MCIF media/reference content is committed or
re-uploaded to Hugging Face.
