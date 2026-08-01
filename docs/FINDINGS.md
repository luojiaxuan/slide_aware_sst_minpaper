# Findings — state, evidence, and confidence (2026-08-01)

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

## 1. The current mechanism diagnostic

**Setup.** Frozen `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
`26291f793822fb6be9555850f06dfe95f2d7e695` on Hyper00 GPU0/1; audio revealed
in 1.0 s chunks; two-observation Local Agreement; Chinese-LiPS lecture
`102_24_M_KJ`, 206 segments × 5 conditions = 1,030 runs. Conditions are
audio-only, correct current slide, same-talk wrong slide, unrelated-talk
scientific slide, and blank image.

**Canonical evidence:** private HF
[`gavinlaw/slide-context-sst-chinese-lips@4923b25`](https://huggingface.co/datasets/gavinlaw/slide-context-sst-chinese-lips/tree/4923b253e87bd94487dace77576ad66e4ea9d8b9/experiments/chinese_lips_visual_controls_v1_qwen3_omni_process16_2gpu_20260801),
tag `chinese_lips_visual_controls_v1_qwen3_omni_20260801_canonical`.
**Run code:** `f76f9224b9e7017a127499323949b0c2294a27a1`.
**Analysis/package code:** `9944654` / `9a564b5`.
The earlier three-condition file
`docs/killtest/speech_vision/runs_omni_speech_vision.jsonl` is retained as the
historical predecessor, not the final control matrix.

| condition | corpus chrF | mean AL (s) |
|---|---:|---:|
| none | 79.080 | 2.637 |
| slide | 80.852 | 2.516 |
| wrong | 80.740 | 2.465 |
| cross-talk | 80.506 | 2.592 |
| blank | 80.625 | 2.519 |

| contrast | ΔchrF [descriptive 95% CI] | ΔAL seconds [descriptive 95% CI] |
|---|---:|---:|
| slide vs none | +1.772 [0.752, 2.782] | -0.121 [-0.237, -0.001] |
| wrong vs none | +1.659 [0.546, 2.726] | -0.172 [-0.274, -0.068] |
| **slide vs wrong** | **+0.113 [-0.693, 0.894]** | **+0.051 [-0.045, 0.152]** |
| wrong vs cross-talk | +0.234 [-0.792, 1.444] | -0.127 [-0.244, -0.010] |
| cross-talk vs blank | -0.119 [-1.205, 0.824] | +0.073 [-0.042, 0.189] |
| blank vs none | +1.545 [0.644, 2.482] | -0.118 [-0.216, -0.021] |

### Claim 1a — Supplying any image slot changes output quality and lagging. **[medium]**
Correct, wrong, unrelated and blank images all move chrF/AL by a similar scale
relative to audio-only. The blank control shows that this cannot be attributed
to slide semantics. Confidence is not high because every item comes from one
talk and references are machine drafts.

### Claim 1b — Current-page semantic use is not established. **[medium]**
Correct slide does not distinguish itself from same-talk wrong slide. This is a
bounded single-talk null, not proof that useful slide semantics never exist.

### Claim 1c — Domain and structured-slide priors are also not isolated. **[medium]**
Same-talk wrong slide does not clearly beat cross-talk, and cross-talk does not
beat blank. The simplest explanation for the aggregate gain is a generic vision
encoder / prompt-slot / decoding perturbation, not page-specific information,
talk/domain priming, or a structured scientific-slide prior.

**What would overturn these claims:**
- A multi-talk, human-reference run where correct current evidence consistently
  beats time/type/token-budget-matched wrong evidence.
- A context-critical event analysis showing earlier stable correct target
  decisions while audio is still insufficient, without extra final error,
  forbidden-context adoption, or overcommit.
- A different integration method that passes the same blank, cross-talk and
  matched-wrong controls, particularly under controlled acoustic degradation.

### Claim 1d — Reference caveat. **[high confidence in the limitation]**
References are **machine drafts** generated with slide-term context, not human
translations. The 10,000 segment-bootstrap intervals describe this talk only;
they are not talk-level inference and absolute chrF is not paper-grade.

### Other limitations
Single talk, single speaker/domain/model/chunk size, one deterministic image
assignment, AL excluding computation, and no independent human output review.

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

1. **Content-specific causal headroom** — the completed unrelated and blank
   controls close the old prompt-level question, but not whether selected
   current evidence can resolve a locked target decision before speech does.
2. **Acoustic interaction** — the Chinese-LiPS baseline is clean and high
   quality. Frozen full-talk native/babble/noise/music/RIR inputs exist, but no
   content-specific ST interaction result exists yet.
3. **Relevance selection** — a persistent slide contains much irrelevant
   information. Typed evidence selection with matched wrong packets remains
   untested and is scientifically different from naive image prompting.

### Paper-story exploration after locked collision audit (2026-07-31)

The authoritative exploration strategy is now
[`PAPER_STORY_DECISION_20260731.md`](PAPER_STORY_DECISION_20260731.md). The
project does not freeze a final paper story before development evidence. It
tests current-content attribution, noisy-speech robustness, pixels beyond OCR,
and evidence integration in a declared development matrix, then freezes one
main route before held-out evaluation.

- **[high] Collision verdict:** independent Paper-Search and Scoop-Check both
  returned Level 3 / partial overlap. No checked paper implements the full
  conjunction, but its broad components are crowded.
- **[high] New collision boundary:** OmniFusion/BOOM establish related
  slide-aware translation applications, but OmniFusion's latency and streaming
  usability do not close off-critical-path semantic integration. VAPO occupies
  look-then-listen, OCR/image, mismatched-slide, and
  visual-interference analysis; visual-context SiMT occupies anticipation and
  image-conditioned READ/WRITE. These cannot be headline claims.
- **[high] Remaining research space:** persistent slide semantics compiled off
  the online critical path, with content attribution, noisy-audio interaction,
  beyond-OCR relations, and selective integration as candidate mechanisms.
- **[hypothesis] Candidate Route-A outcome:** talk-weighted gain in stable
  correct decisions before source audio resolves a locked forced choice, with
  final correctness preserved. No such result exists yet.
- **[high] Pixel nulls need care:** a null C6 result is inconclusive unless a
  gold visual-relation control is positive and a powered equivalence test passes.
- Lip vision, AVMuST-TED execution, and slide+lip hybrid experiments remain out
  of scope.

The next scientific action is not an unstructured full `C0-C7` GPU run. MCIF
input-side visual readiness now passes, but no translation outcome exists. First
import the verified ACL60/60 real-frame supplement, blind-label
evidence-opportunity density, and map semantic, relation, correct/wrong, and
native/noisy oracle headroom.
Any stable route can justify a focused automatic method; stop only if all gold
evidence routes lack practical headroom.

### 2026 literature/data audit update (2026-07-31)

- **[high] MCIF's 21-talk translation subset is the primary project-held-out
  long-form source, and its input-side visual tier is now ready.** The current HF
  revision has a broader 100-talk media pool; the translation subset has 21 ACL 2023
  talks, original MP4/WAV, human English transcripts, professional De/It/Zh
  translations, CC BY 4.0, and an official IWSLT 2026 long-form SimulST path.
  This is stronger for talk-level inference than the five ACL60/60 eval talks,
  The 21 videos now pass frozen hashes and reference-free visual QA: 283 reviewed
  transitions compile into 304 conservatively unlocked causal states. This does
  not establish eligible-event density, statistical power or translation gain.
- **[high] ACL60/60 should be development/replication, not the only main test.**
  Its professional translations and external term tags are valuable, and the
  verified *Do Slides Help?* Figshare v2 supplement now adds 884 real frames
  covering all 10 talks. Its audio remains deliberately clear, and five eval
  talks alone are underpowered for a broad confirmatory claim.
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

The dataset facts now also have local media/hash/visual validation, but there is
still no local MCIF translation run. The immediate experiment is therefore a
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
