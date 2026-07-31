# Findings — state, evidence, and confidence (2026-07-31)

Single entry point for the project's state. **Every claim below carries a
confidence level and a pointer to raw evidence.** Conclusions here are one
session's interpretation of a handful of experiments — several earlier
"conclusions" in this repo were later overturned by adding a control, so treat
this document as falsifiable too. Where a claim could be wrong, the reason is
stated explicitly under *"What would overturn this"*.

Confidence scale: **[high]** = multiple runs / large effect / control in place;
**[medium]** = single clean experiment, known limitations; **[low]** =
suggestive, one data point or confounded.

---

## 1. The decisive experiment

**Setup.** Qwen3-Omni-30B-A3B-Instruct on hyper01 GPU6; audio revealed in 1.0 s
chunks; Local Agreement commit; Chinese-LiPS lecture `102_24_M_KJ` (zh→En, AI /
traffic-management topic), 206 segments × 3 conditions = 618 runs.
Slide enters as an **image** through the omni vision encoder.

**Raw evidence:** `docs/killtest/speech_vision/runs_omni_speech_vision.jsonl`
(618 records: hypothesis, reference, commit events, image used).
**Data:** [gavinlaw/chinese-lips-speech-slide-probe](https://huggingface.co/datasets/gavinlaw/chinese-lips-speech-slide-probe).
**Code:** `code/scripts/omni_speech_vision_probe.py`.
**Reproduce:** `HF_HUB_CACHE=/data/audrey/hf_cache/hub python omni_speech_vision_probe.py --items items_abs.json --out runs.jsonl --conditions none,slide,wrong --device-map cuda:0`

| condition | chrF | mean AL (chunks) |
|---|---|---|
| none (audio only) | 78.3 | 2.63 |
| slide (correct slide image) | 80.7 | 2.42 |
| wrong (a different slide image) | 80.6 | 2.43 |

| contrast | ΔchrF | p | ΔAL | p |
|---|---|---|---|---|
| slide vs none | +2.63 | <0.0001 | −0.202 | 0.0003 |
| wrong vs none | +2.29 | <0.0001 | −0.199 | 0.0001 |
| **slide vs wrong** | **+0.34** | **0.22** | **−0.002** | **0.47** |

slide-vs-wrong effect size: **+0.34 chrF, 95% CI [−0.47, +1.16], sd 6.1,
n=206**. Minimum detectable effect at ~80% power ≈ **1.2 chrF**.

### Claim 1a — Adding an image improves quality and reduces lagging. **[high]**
Both slide and wrong beat none on quality (+2.6 / +2.3 chrF) and commit earlier
(−0.20 chunks), all p<0.001, consistent across two independent conditions.

### Claim 1b — The benefit does not depend on *which* slide is shown. **[medium]**
slide vs wrong is null on both axes. **But this is bounded, not zero:** the CI
only rules out content effects larger than ~1.2 chrF. A real effect of ~1 chrF
would be invisible at this n.

**What would overturn this:**
- **The wrong-slide confound (most important).** Verified from the raw runs: the
  "wrong" image is always drawn from **the same lecture** (median 61 segments
  away, but same speaker, same deck, same domain). So this experiment shows
  *"which slide within a talk"* does not matter — it does **not** show that
  *"slide vs. an unrelated-domain image"* does not matter. A wrong slide from a
  different lecture (or a non-slide photo) is the missing control, and would
  separate **domain priming** from **segment-specific content**. Until that runs,
  "vision content is useless" is **overstated**; the defensible claim is
  "segment-level slide specificity is not exploited."
- Larger n or a higher-ambiguity dataset could reveal a sub-1.2-chrF effect.
- A different model or injection format could exploit content this one ignores.

### Claim 1c — Reference caveat. **[medium confidence in the metric itself]**
References are **machine drafts** (Qwen3-32B generated *with* slide-term
context), not human translations. Direction of bias, if any, favours the slide
condition — which makes the null more, not less, credible — but absolute chrF
values are not comparable to human-referenced numbers.

### Other limitations of this experiment (all unaddressed)
Single talk, single speaker, single domain, single model, single chunk size
(1.0 s), single random seed for wrong-image assignment, AL measured in chunks
(not computation-aware), no human inspection of outputs.

---

## 2. Earlier probes and why most were invalid

### Claim 2a — Transcript input invalidates vision experiments. **[high]**
Feeding a text LLM the *transcript* removes the acoustic ambiguity that slide
context would resolve; the transcript has already committed to a reading. Every
text-model probe in `docs/killtest/` (Follow-ups 1–11) shares this flaw. This is
an argument from experimental design, not a measurement, but it is decisive:
those runs cannot test the hypothesis they were written for.
**Evidence:** the probes themselves (`runs_qwen3_32b.jsonl`, `runs_bvlm.jsonl`,
`runs_gated*.jsonl`, `runs_bias*.jsonl`, `acl6060*/`).

### Claim 2b — A matched-noise control is mandatory. **[high]**
Injecting *anything* into the prompt shifts quality and latency. Several
positive-looking results (+2.5 to +7.3 chrF) collapsed once a wrong-input
control was added: En→Zh S3 (slide−wrong = −0.1, n.s., Follow-up 11), X→En
deterministic (slide−wrong = −0.8, n.s.), and the speech probe above. Replicated
across three independent settings.

### Claim 2c — Serving nondeterminism corrupts Local Agreement. **[high]**
Batched vLLM returns different outputs for identical prefixes, stalling LCP
commits and truncating output; deterministic local decoding lifted baseline chrF
38.8 → 62.7 on the same segments. **Evidence:** Follow-up 9/10 in
`docs/killtest/KILLTEST_RESULTS.md`, `runs_trie_hard.jsonl`.

### Claim 2d — Cross-lingual prompt injection needs fencing. **[high]**
An unfenced English glossary in an En→Zh prompt is translated and echoed as the
output. Invisible in X→En (hint language = target language).
**Evidence:** `docs/killtest/acl6060_det/` and the preserved buggy run
`runs_s3_det_BUGGY_hintecho.jsonl` (kept deliberately as the artefact).

### Claim 2e — Logits-level injection is benign but powerless. **[medium]**
Uniform bias saturates at +1.3 chrF on hard segments; trie-constrained
shallow fusion reaches +0.7 vs prompt injection's +6.5. Interpretation
(terminology errors are phrase-planning failures, not token near-misses) is
inference, not direct measurement. **Evidence:** `runs_bias{2,4,8}.jsonl`,
`runs_trie_hard.jsonl`.

---

## 3. What is genuinely open

1. **Unrelated-domain control** — the missing experiment that would firm up
   Claim 1b (see above). Cheapest and highest-value next run.
2. **Headroom** — baseline chrF 78.3 on clean, scripted lecture audio; acoustic
   ambiguity is scarce. Noisy / accented / jargon-dense speech is untested.
3. **Relevance selection** — a slide covers 30–90 s while a segment is ~5 s, so
   most of the injected image is irrelevant. RASST-style chunkwise retrieval over
   the visual channel is a different mechanism and remains untested.

### Scope decision after related-work audit (2026-07-31)

The project is now scoped to **asynchronous, persistent slide-semantic
evidence**. A slide is encoded once after a change event and reused across its
dwell window; it is not re-encoded beside every audio chunk. The decision
contract is in [`RESEARCH_GOAL_20260731.md`](RESEARCH_GOAL_20260731.md), and the
complete experiment/paper contract is in
[`ACL_PAPER_BLUEPRINT_20260731.md`](ACL_PAPER_BLUEPRINT_20260731.md).

- Generic contextual SST is no longer a safe fallback contribution: EGTA
  (arXiv:2607.17766) already performs streaming-evidence-conditioned terminology
  selection with shuffled-memory and activation controls.
- Semantic evidence must beat audio-only and matched stale/wrong controls.
  Beyond-OCR claims additionally require structured visual evidence to beat a
  strong OCR baseline on predeclared layout/formula/chart/emphasis slices.
- Direct raw-image representations are optional, not the premise: if structured
  evidence matches them, the cheaper auditable representation wins.
- Lip vision, AVMuST-TED execution, and slide+lip hybrid experiments are out of
  scope. Lip work remains only as a related-work contrast between continuous
  phonetic vision and slowly changing semantic vision.
- Latency is split into once-per-slide cold encoding, dwell-normalized
  amortized cost, evidence-ready lead/lag, and on-path retrieval. “Async” alone
  is not evidence of zero cost.

The current defensible paper space is therefore a causal, computation-aware
study of **when persistent slide-derived semantic evidence helps SimulST and
when information beyond OCR is necessary**. This is a research target, not an
established finding.

### 2026 literature/data audit update (2026-07-31)

- **[high] MCIF should be the primary held-out benchmark.** It has 21 ACL 2023
  talks, original MP4/WAV, human English transcripts, professional De/It/Zh
  translations, CC BY 4.0, and an official IWSLT 2026 long-form SimulST path.
  This is stronger for talk-level inference than the five ACL60/60 eval talks.
- **[high] ACL60/60 should be development/replication, not the only main test.**
  Its professional translations and external term tags are valuable, but its
  audio was selected to be clear and the current repo only integrates five dev
  talks. **Evidence:** ACL60/60 paper/release and local video audit.
- **[high] MCIF and ACL60/60 are not native noisy benchmarks.** Noise must be
  described as a controlled full-talk intervention, with native audio reported
  separately. Noise seeds are repeated measures, not independent talks.
- **[high] IWSLT 2026 PDF context is a mandatory strong baseline.** MLLP-VRAIN
  already combines KeyBERT/ASR word boosting and pretranslated BM25 RAG, with
  reported context gains. Top-k retrieval or generic context injection is not
  a new contribution.
- **[high] LECTRANS materially narrows the claim.** Its 383 h academic-lecture
  benchmark compares slide text/image for aligned-ASR-transcript segment MT and
  asks when slides help or introduce noise. Our remaining delta requires raw
  unsegmented speech, SimulST, live temporal state, controlled acoustic
  corruption, wrong/stale controls, and computation-aware costs.
- **[medium] Direct video remains difficult on current scientific-talk data.**
  MCIF reports that video often provides no benefit or can hurt current MLLMs.
  This supports a selective/evidence-state study but is not proof that the
  proposed method will work.

These conclusions come from primary paper/release inspection; they have not yet
been validated by a local MCIF run. The immediate experiment is therefore a
small causal kill test on ACL60/60 dev, not model training. MCIF's HF configs
share the same 21 underlying talks rather than providing disjoint development
and test talks, so all 21 are reserved as project-held-out until the system is
frozen.

### OmniFusion reassessment (2026-07-31)

OmniFusion is a weak task precedent, not a strong method baseline. Its reported
computation-aware AL is roughly 5.5–10 s for the E2E system; the advertised
~1 s reduction is E2E vs. its own cascade, while adding an image raises offline
OmniFusion inference from 1.98 s to 3.15 s. It has no OCR/text-equivalent,
wrong-image, noisy-speech, or visual-grounding controls, and its public repo
does not contain the SimulST evaluation pipeline needed to reproduce Figure 3.
The overlap verdict for the main raw-vision-necessity question is therefore
Level 3 / partial overlap, not a Level 2 material collision. See
[`OMNIFUSION_REASSESSMENT_20260731.md`](OMNIFUSION_REASSESSMENT_20260731.md).

## 4. Assets (all reusable regardless of the project's fate)

| Asset | What |
|---|---|
| [gavinlaw/mtedx-v-eval](https://huggingface.co/datasets/gavinlaw/mtedx-v-eval) | 100 long-form X→En talks, live-recoverable video, human refs, VLM visual-signal labels |
| [gavinlaw/chinese-lips-longform-debug](https://huggingface.co/datasets/gavinlaw/chinese-lips-longform-debug) | 21 zh sessions, 11.1 h, original-timeline reconstruction (drift <10 ms) |
| [gavinlaw/chinese-lips-speech-slide-probe](https://huggingface.co/datasets/gavinlaw/chinese-lips-speech-slide-probe) | 206 segments: audio + slide image + wrong-image control |
| `code/scripts/` | manifest builders, VLM slide reader, deterministic + omni streaming probes, need-predictor |
| `docs/killtest/` | 12 follow-ups with raw runs and significance tests, including the negative results |

## 5. Superseded documents

- `WHY_IT_WORKS.md` — affirmative case was conditioned on the then-pending
  speech+vision experiment; that experiment did not support the content-based
  claims. Related-work survey remains valid.
- `MVP_DEFINITION.md` — its "single decision experiment" has been executed
  (negative, with the caveats in §1).
- `BENCHMARK_PLAN.md` — the 2026-07-17 mTEDx/Chinese-LiPS-primary benchmark is
  historical. The current hierarchy is MCIF primary, ACL60/60 replication, and
  Chinese-LiPS private diagnostic.
- Any doc predating 2026-07-31 that reports slide gains **without** a
  wrong-slide control reflects the pre-control era.
