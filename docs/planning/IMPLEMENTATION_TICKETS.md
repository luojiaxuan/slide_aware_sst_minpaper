# Implementation tickets

## T1. Schema and validation
- Implement `slidesst.data.schema` dataclasses or Pydantic models.
- Validate JSONL challenge files.
- Unit tests: load toy examples, reject missing required fields.

## T2. Chinese-LiPS adapter
- Input: local root directory and metadata paths.
- Output: normalized lecture segments with transcript, audio path, slide paths/OCR.
- Do not download data in code; user provides paths.

## T3. Hard example miner
- Use jieba or pkuseg for segmentation.
- Use pypinyin for pinyin conversion.
- Build homophone clusters.
- Join transcript tokens with slide OCR terms and glossary candidates.
- Export candidate JSONL + annotation CSV.

## T4. Context index builder
- Build evidence items from slide OCR, slide VLM summaries, glossary, background docs.
- Store as JSONL plus optional BM25/embedding index.
- Include source type, time, slide id, text, target hint, pinyin.

## T5. Retriever
- Implement BM25, pinyin overlap, temporal prior.
- Return top M evidence items with scores and feature breakdown.
- Unit tests for pinyin match and temporal filtering.

## T6. Policy
- Implement rule-based `EvidencePolicy` with thresholds.
- Decisions: use, ignore, delay, downweight.
- Cap final packet at K items.
- Log why each item was used or rejected.

## T7. Translation adapters
- Implement provider-neutral interface: `translate(step_state, evidence_packet) -> TranslationOutput`.
- Support dry-run/mock model for tests.
- Optional adapters: OpenAI-compatible, local vLLM, Hugging Face seq2seq.
- Save full prompt/messages for reproducibility.

## T8. Streaming simulator
- Transcript-oracle mode: reveal partial transcript over time.
- ASR mode: use precomputed streaming ASR outputs.
- Implement wait/commit loop and token timestamps.

## T9. Evaluation
- Normalize terms and aliases.
- Implement HDA, Term F1, COR, WSAR.
- Add sacreBLEU and optional COMET.
- Generate CSV tables and JSON summaries.

## T10. Paper artifacts
- Generate `paper/tables/*.tex` from result CSV.
- Generate case-study snippets.
- Generate dataset statistics table.
