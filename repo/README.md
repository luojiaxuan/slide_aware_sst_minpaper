# slide-aware-sst scaffold

This is a minimal scaffold for the coding agent. The code is intentionally incomplete but fixes interfaces, expected files, and reproducibility conventions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Example workflow

```bash
python scripts/prepare_dataset.py --config configs/min_zh_en.yaml --stage toy
python scripts/build_context_index.py --config configs/min_zh_en.yaml --mismatch matched
python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition no_context --mismatch matched
python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition naive_all_context --mismatch matched
python scripts/run_stream_translate.py --config configs/min_zh_en.yaml --condition policy --mismatch matched
python scripts/evaluate.py --config configs/min_zh_en.yaml
python scripts/make_paper_tables.py --config configs/min_zh_en.yaml
```

## Pilot workflow with local lecture metadata

1. Edit `configs/min_zh_en.yaml`:
   - set `paths.raw_data_root` to the local Chinese-LiPS or lecture-data root;
   - set `dataset.metadata_glob` to the local JSONL/JSON/CSV metadata files;
   - adjust `dataset.field_mappings` for transcript, slide OCR, slide image, audio, glossary, and background fields.
2. Mine candidates:

```bash
python scripts/prepare_dataset.py --config configs/min_zh_en.yaml --stage mine --limit 300
```

3. Manually edit `outputs/annotations/pilot_candidates.csv`: mark 50-100 rows as `verified=true`, fix `correct_target`, `distractor_targets`, and `reference_translation`.
4. Build the verified set and evidence index:

```bash
python scripts/prepare_dataset.py --config configs/min_zh_en.yaml --stage build --annotations outputs/annotations/pilot_candidates.csv
python scripts/build_context_index.py --config configs/min_zh_en.yaml --mismatch matched
```

5. For local vLLM/Qwen, set `translation.provider: vllm`, `translation.base_url`, and `translation.model` in the config. The adapter uses the OpenAI-compatible `/chat/completions` API and saves prompts, evidence packets, policy logs, and timing in each output JSONL.

## Implementation principle

Every script should read a YAML config and write to a timestamped output directory. Never write model outputs only to stdout.
