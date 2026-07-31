# Findings — where the project actually stands (2026-07-31)

Single source of truth for what has been established, what was refuted, and what
remains open. Read this before any other doc; several earlier docs were written
under premises that later experiments invalidated (noted inline).

## 1. The headline result

**Naive slide injection does not help streaming speech translation beyond the
effect of supplying *any* image.** Measured with the correct premise for the
first time (audio input, slide through the vision encoder, Qwen3-Omni-30B,
206 segments × 3 conditions, zh→En):

| condition | chrF | mean AL (chunks) |
|---|---|---|
| none (audio only) | 78.3 | 2.63 |
| slide (correct slide image) | 80.7 | 2.42 |
| wrong (a *different* slide image) | 80.6 | 2.43 |

- slide vs none: **+2.63 chrF (p<0.0001)**, **AL −0.202 (p=0.0003)** — looks great
- wrong vs none: **+2.29 chrF (p<0.0001)**, **AL −0.199 (p=0.0001)** — nearly identical
- **slide vs wrong: +0.34 chrF (p=0.22, n.s.), AL −0.002 (p=0.47, n.s.)**

The wrong-slide control is what makes this decisive: an unrelated slide buys the
same quality and the same earlier commits. The gain is **image presence, not
slide content**. Details and raw runs: `docs/killtest/` (Follow-up 12).

## 2. What this does and does not refute

Refuted (on this data, in this form):
- Slide *content* improving translation quality in streaming ST.
- Slide *content* enabling earlier commits (the latency hypothesis) — vision does
  make the model commit ~0.2 chunks earlier, but any image does that equally.
- Word-sense disambiguation from slides — if it were operating, the correct
  slide would have to beat the wrong slide; it does not.

Not refuted (untested, and the only honest paths forward):
1. **Headroom.** Baseline chrF is already 78.3 on clean, scripted lecture audio;
   acoustic ambiguity is scarce. Noisy / accented / jargon-dense speech may leave
   vision something to do.
2. **Relevance selection.** A slide covers 30–90 s of speech, so ~96% of its
   content is irrelevant to the current segment and dilutes to noise. Injecting
   only the currently-relevant fragment — i.e. RASST's chunkwise retriever
   applied to the visual channel — is a materially different mechanism that this
   experiment did not test.

## 3. Methodological findings (independently useful)

These emerged as by-products and are reusable regardless of the project's fate:

- **Transcript input invalidates vision experiments.** Feeding a text LLM the
  transcript deletes the acoustic ambiguity vision is meant to resolve; every
  such probe here returned null for that reason. Any "does vision help speech
  translation" study must consume audio.
- **The wrong-input control is mandatory.** Injecting *anything* into the prompt
  measurably shifts quality and latency. Without a matched-noise control,
  "context helps" results are unfalsifiable. This bit us repeatedly: several
  earlier positive-looking results (+2.5 to +7.3 chrF) collapsed once a wrong
  control was added.
- **Serving nondeterminism corrupts Local Agreement.** Batched vLLM returns
  different outputs for identical prefixes, stalling LCP commits and truncating
  output (baseline chrF 38.8 → 62.7 under deterministic local decoding on the
  same data). Streaming evaluations must pin deterministic decoding or use a
  drift-tolerant commit rule.
- **Cross-lingual prompt injection needs fencing.** An unfenced English glossary
  in an En→Zh prompt gets translated and echoed as the output — invisible in
  X→En (hint language = target language), catastrophic otherwise.
- **Logits-level injection is benign but powerless.** Uniform and trie-constrained
  biasing leave easy segments unharmed but cannot fix terminology errors:
  these are phrase-planning failures, not single-token near-misses.

## 4. Assets built (all reusable, all on HF/Git)

| Asset | What |
|---|---|
| [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) | 100 long-form X→En talks, live-recoverable YouTube video, human refs, VLM visual-signal stratification |
| [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) | 21 zh sessions, 11.1 h, original-timeline reconstruction (drift <10 ms), machine-draft En refs |
| [gavinlaw/chinese-lips-speech-slide-probe](https://huggingface.co/datasets/gavinlaw/chinese-lips-speech-slide-probe) | 206 segments with audio + slide image + wrong-image control — the decisive probe set |
| `code/scripts/` | manifest builders, VLM slide reader, deterministic/omni streaming probes, need-predictor, gating conditions |
| `docs/killtest/` | 12 follow-ups with raw runs, significance tests, and the negative results |

## 5. Open decision

Three options, none yet chosen:
1. Re-test on **high-ambiguity speech** (noisy/accented/jargon-dense) where
   headroom exists.
2. Add **relevance selection** (RASST retriever over the visual channel) so the
   injected evidence is not 96% irrelevant.
3. Write up the **negative result + benchmark + methodology** as the contribution.

## Superseded documents

- `WHY_IT_WORKS.md` — its affirmative case was explicitly conditioned on the
  then-pending speech+vision experiment. That experiment has now run and did not
  support the content-based claims; read it only for the related-work survey.
- `MVP_DEFINITION.md` — its "single decision experiment" (slide > wrong) has been
  executed; the answer was negative on both X→En text-probe and zh→En speech.
- Any doc dated before 2026-07-31 asserting slide gains without a wrong-slide
  control reflects the pre-control era.
