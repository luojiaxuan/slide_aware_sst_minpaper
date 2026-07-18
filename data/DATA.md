# Data locations

All reusable datasets live on Hugging Face (namespace `gavinlaw`); this directory
only documents locations and local staging conventions. No media is committed to git.

## Published datasets

| Dataset | Content | License | Role |
|---|---|---|---|
| [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) | mTEDx-V: talk-level long-form X→En manifests (es/fr/it/ru/el→en, 100 talks, ~18 h), `talk_id` = live YouTube ID, `visual_signal/` OCR stratification | CC BY-NC-ND 4.0 (manifests only, no media) | Realistic-noisy X→En eval stratum |
| [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) | Chinese-LiPS long-form: `orig_timeline` (real gaps restored) + `silence_removed` variants, 3 talks ~97 min; `en_draft_demo/` sample refs | CC BY-NC-SA 4.0 | Clean-visual zh→En stratum (En refs in progress) |

## Upstream sources

- [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS) — 1080p dedicated PPT stream + FACE + WAV per segment; test split = 21 videos / 3,908 clips / ~9 h
- [deepdml/mtedx](https://huggingface.co/datasets/deepdml/mtedx) — mTEDx mirror used to build mTEDx-V
- [FBK-MT/MCIF](https://huggingface.co/datasets/FBK-MT/MCIF) — ACL scientific talks, speech+vision+text, En→{De,It,Zh}, CC-BY 4.0 (planned En→Zh secondary track)

## Local staging (not in git)

`~/research_idea/data_prep/` — raw downloads and intermediates
(`mtedx_videos/` 3.0G probe videos, `chinese_lips/` 1.9G raw+rebuilt audio).
Regenerate with `code/scripts/build_mtedx_v_manifest.py`,
`build_chinese_lips_longform.py`, `score_visual_signal.py`.
