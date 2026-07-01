# MVP decisions

## Use cascaded streaming first

The first version should use transcript-oracle streaming and ASR-based streaming as two tracks. This avoids blocking on end-to-end speech-LLM training and lets the paper isolate the context-management question.

## Use Chinese-to-English

Chinese source lectures are ideal because homophones and near-homophones create real acoustic ambiguity. English target evaluation makes technical-term correctness easy to inspect.

## Treat slides as evidence, not as raw prompt bulk

Raw slide injection is a baseline. The main method should extract compact evidence packets with source, confidence, term, translation, and reason.

## Make wrong-context evaluation central

The paper should not claim "slides always help." It should show that slides are beneficial but dangerous under streaming mismatch.

## Recommended model choices

- ASR: FunASR / Whisper large-v3 / Paraformer, depending on local availability.
- Translation: GPT-style API model, Qwen local instruct model, or NLLB/Seamless as lower-cost baseline.
- Retriever: BM25 + pinyin in v0; add BGE-M3 embeddings if time permits.
- OCR: use provided slide text when available; otherwise PaddleOCR or another OCR backend.
