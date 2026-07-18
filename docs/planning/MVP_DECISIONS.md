# MVP decisions

## Frame the paper as slide/context-aware SST

The current paper should not be over-committed to pure vision-aware SST. The
safe umbrella is slide/context-aware simultaneous speech translation under
latency constraints. OCR, VLM captions, deck/topic metadata, and future
speaker/topic context are evidence sources; raw vision is an ablation or later
extension.

## Keep Chinese-LiPS, but add evaluation anchors

Chinese-LiPS remains the main dataset because it matches the target presentation
setting. Its limitation is that it is ASR/AVSR data, not an ST dataset. Use Qwen
pseudo references for scale, but build a 500-1,000 item human English
translation diagnostic subset for paper-grade evaluation. Add an ST-native
no-visual sanity check such as BSTC to validate the streaming translation and
latency pipeline.

## Use cascaded streaming first

The first version should use transcript-oracle streaming and ASR-based streaming as two tracks. This avoids blocking on end-to-end speech-LLM training and lets the paper isolate the context-management question.

## Use Chinese-to-English

Chinese source lectures are ideal because homophones and near-homophones create real acoustic ambiguity. English target evaluation makes technical-term correctness easy to inspect.

## Treat slides as evidence, not as raw prompt bulk

Raw slide injection is a baseline. The main method should extract compact evidence packets with source, confidence, term, translation, and reason.

OCR-derived terms are expected to be a strong baseline in slide-heavy data. This
is not a failure. VLM captions should be evaluated as beyond-OCR evidence on
low-OCR, diagram/chart, and visual-non-OCR slices.

## Make wrong-context evaluation central

The paper should not claim "slides always help." It should show that slides are beneficial but dangerous under streaming mismatch.

Wrong-context and distractor-context baselines must be included early: previous
slide, next slide, random same-deck slide, random same-topic slide, and full-deck
noisy OCR.

## Recommended model choices

- ASR: FunASR / Whisper large-v3 / Paraformer, depending on local availability.
- Translation: GPT-style API model, Qwen local instruct model, or NLLB/Seamless as lower-cost baseline.
- Retriever: BM25 + pinyin in v0; add BGE-M3 embeddings if time permits.
- OCR: use provided slide text when available; otherwise PaddleOCR or another OCR backend.
