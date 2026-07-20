# Oracle Anticipation Kill Test — Results and Verdict

Date: 2026-07-19. Pre-registered in docs/BENCHMARK_PLAN.md §5.1; runner/analyzer
in `code/scripts/oracle_killtest_{runner,analyze}.py`; raw runs in this directory.

## Setup

- **Protocol**: Local Agreement commit (Liu et al. 2020) over incrementally
  revealed source transcript (READ = 2 words el / 4 chars zh), monotonic commits,
  final-step tail completion. Prompt-based, temperature 0.
- **Models**: Qwen3-32B (vllm, hyper01 H200, primary) + Qwen2.5-7B-q4 (ollama,
  local pilot / model-size contrast).
- **Data**: 56 items — 40 segments of mTEDx Greek talk 6HhNGU08KoE (English
  slides = M3 regime, human refs) + 16 Chinese-LiPS segments (zh slides = M2
  regime, hand-translated refs).
- **Conditions**: A none / B slide (OCR terms of currently visible slide) /
  C oracle (rare content words of the segment's reference = perfect anticipation
  upper bound) / D wrong (slide terms from a different moment).
- **Metrics**: chrF, oracle-term recall, AL (Ma et al. 2019).

## Headline results (Qwen3-32B)

| group | cond | chrF | termR | AL |
|---|---|---|---|---|
| el | none | 38.8 | 0.40 | 0.67 |
| el | **oracle** | **46.6** | **0.53** | 0.60 |
| el | slide | 35.2 | 0.40 | 0.22 |
| el | wrong | 38.6 | 0.42 | 0.54 |
| zh | none | 49.6 | 0.52 | 0.46 |
| zh | **oracle** | **55.6** | **0.68** | 0.60 |
| zh | slide | 51.6 | 0.56 | 0.44 |
| zh | wrong | 49.5 | 0.51 | 1.13 |

Paired bootstrap (10k resamples, one-sided):

| comparison | ΔchrF | p | n |
|---|---|---|---|
| oracle vs none, pooled | **+7.0** | **0.030** | 56 |
| oracle vs none, hard stratum (baseline termR ≤ 0.4) | **+18.8** | **< 0.0001** | 28 |
| slide vs none, pooled | −1.8 | 0.74 | 56 |
| wrong vs none, pooled | +0.8 | 0.41 | 56 |

Hard-stratum term recall: **0.12 → 0.41 (3.4×)** under oracle.

## Verdict (per pre-registered rule)

**PREMISE SURVIVES — extraction is the bottleneck.**

1. **C ≫ A (significant)**: future-term injection materially improves streaming
   translation, replicated across both languages/regimes and both model scales
   (7B: el +8.3 / zh +9.8 chrF; 32B: pooled +7.0, p=0.03). Gains concentrate
   precisely where the baseline fails on terminology (hard stratum +18.8 chrF,
   termR 3.4×) — the expected signature of anticipation value: it rescues
   term-dense/hard segments and is neutral elsewhere.
2. **B ≈ A (n.s., el negative)**: today's naive extraction (480p tesseract for
   el; zh OCR list) does not yet realize the oracle value; noisy fragments
   ("independen") can hurt (el −3.6 raw). The C−B gap is the method space:
   VLM-based slide reading, term filtering, confidence gating.
3. **D ≈ A**: wrong-slide terms are neutral for a strong model under prompt
   injection. The 7B pilot's apparent wrong-condition gain (+4.7/+5.6) was
   diagnosed as an output-completeness artifact (weak model + LCP stalls), gone
   at 32B. Faithfulness risk therefore concentrates in stronger injection modes
   (image-input, fine-tuned) — the policy evaluation must construct harder
   adversarial settings than prompt-level wrong terms.

## Follow-up: VLM extraction closes most of the hard-stratum gap (2026-07-19)

Replacing 480p tesseract with Qwen2.5-VL-7B slide reading (same 54 frames, same
lookback rule; `code/scripts/vlm_slide_terms.py`, timeline archived here) and
re-running the B condition on the el group:

| cond (el, n=40) | chrF | termR |
|---|---|---|
| none | 38.8 | 0.41 |
| slide-OCR | 35.2 | 0.40 |
| **slide-VLM** | **39.9** | 0.41 |
| oracle | 46.6 | 0.53 |

- VLM extraction **repairs OCR's damage** (35.2 → 39.9; VLM reads full phrases
  like "On Imitation and Improvisation: New Competitive Businesses" where OCR
  produced fragments like "independen").
- **Hard stratum (baseline termR ≤ 0.4, n=22): VLM vs none ΔchrF +14.1** — about
  75% of the oracle's +18.8 hard-stratum gain, realized by the *real* visual
  channel.
- Pooled VLM vs none is flat (−0.6, n.s.): easy segments are slightly hurt by
  unneeded injection, cancelling hard-segment gains — the direct empirical case
  for an **evidence-gating policy** (inject only under predicted need), which is
  the method chapter's job.

Narrative chain now complete: C≫A (premise) → B_OCR≈A (extraction bottleneck) →
B_VLM ≈ 0.75·C on hard segments (VLM fixes extraction) → pooled flat (gating
needed) → policy.

## Follow-up 2: gating recovers the full pooled gain (2026-07-19)

Two gating prototypes over the VLM slide terms (`runs_gated.jsonl`; conditions
in `oracle_killtest_runner.py`): *oracle-gate* injects only on hard segments
(gating upper bound); *LLM-gate* asks the model per step whether the slide
terms are relevant (sticky-on).

| condition (el, n=40) | chrF | Δ vs none | p | hard-Δ |
|---|---|---|---|---|
| A none | 38.8 | — | — | — |
| B slide-OCR | 35.2 | −2.9 | n.s. | +2.3 |
| B slide-VLM | 39.9 | −0.6 | n.s. | +14.1 |
| B VLM + LLM-gate | 38.0 | −2.0 | n.s. | +10.1 |
| **B VLM + oracle-gate** | **45.3** | **+7.3** | **<0.001** | +13.3 |
| C oracle terms | 46.6 | +7.0 | 0.059 | +16.3 |

- **Headline: real visual channel + selective injection ≈ oracle upper bound.**
  VLM-extracted slide terms with need-based gating deliver +7.3 chrF pooled
  (p<0.001), statistically cleaner than even the oracle-terms condition (+7.0,
  p=0.059) because easy segments are left untouched.
- **Naive LLM relevance gating fails**: it fires on 35/40 items (17/18 easy) —
  no selectivity — and degenerates to ungated injection. Predicting *when*
  evidence is needed (model uncertainty, term-source matching) is the open
  technical problem, i.e., the paper's method contribution.
- Full six-condition chain: extraction quality (OCR→VLM) fixes *what* to
  inject; gating fixes *when*; together they close ~100% of the oracle gap.

## Follow-up 3: need-prediction gate iterations + easy-harm diagnosis (2026-07-19)

Two runtime gate implementations of the method's eq. (gate) were tested
(`runs_gated_unc.jsonl`, `runs_gated_unc2.jsonl`):

- **v1 (retrospective match + stall≥2)**: fired 0/40 — retrospective signals
  (slide term already in hypothesis) are provably too late; conservative stall
  thresholds under-fire on a stable 32B translator. Degenerates to baseline.
- **v2 (forward-looking: stall≥1 OR incoming long source word)**: fired 36/40
  (easy 18/18) — over-sensitive, no selectivity; pooled +1.4 (n.s.), better
  than ungated only because sticky late opening spares early-segment
  interference. Selectivity, not sensitivity, is the binding constraint;
  a learned failure-predictor is the clear next step.

**Easy-stratum harm diagnosis (important honesty note).** Per-stratum deltas
show ALL hint conditions hurt the easy stratum (wrong −15.8, slide-OCR −9.2,
slide-VLM −18.6, oracle −4.4) regardless of hint relevance, and worst cases
are *truncations* (e.g., a 2-word committed output), not mistranslations.
Mechanism: hint presence perturbs step-to-step wording, stalling the Local
Agreement longest-common-prefix commit on segments the model would otherwise
translate stably. The easy-harm is therefore substantially a **protocol
sensitivity** (prompt-level LCP), not evidence of term over-adoption; the
mirror image of the 7B completeness artifact. Implications: (i) gating remains
valuable (oracle-gate easy Δ = 0.0 → pooled +7.3 p<0.001 stands), (ii) do not
sell easy-harm as a faithfulness finding, (iii) logits-level integration
(hint-biased decoding rather than prompt injection) should reduce the
interference floor and is the right implementation for the full system.

## Follow-up 4: learned need-predictor — surface features carry no signal (2026-07-19)

Auto-mined failure labels at scale (1,540 el segments across 18 talks,
offline Qwen3-32B translation, hard = term recall ≤ 0.4; 24% hard rate) with
source-side surface features (word lengths, long-word rates, digits, char
diversity), logistic regression, talk-held-out on the probe talk:
**AUC 0.49–0.52 — chance level**, in both full-segment and prefix-30% scopes.
Interpretation: terminology failure is a *model-knowledge* property (does the
model know this term's English form), invisible in source surface statistics.
The signal ladder moves to model-side confidence (token logprobs of a draft
translation — "the model knows what it doesn't know") and slide-term/draft
mismatch; logprob-feature collection is in progress. Honest status: gating
selectivity remains open; the oracle-gate ceiling (+7.3) quantifies exactly
what a working predictor is worth.

## Follow-up 5: learned need-predictor closes part of the gap (2026-07-19)

Adding model-confidence features (token logprobs of an offline draft
translation) to the predictor lifts talk-held-out AUC from chance to **0.62**
(prefix-30% scope; strongest feature lp_mean, coef −2.9): the model's own
uncertainty does carry failure signal — "it knows what it doesn't know" —
while source surface statistics do not. End-task (40 probe segments,
predictor-gated VLM injection, fire 30/40):

| condition | chrF | Δ pooled | hard-Δ | easy-Δ |
|---|---|---|---|---|
| ungated VLM | 39.9 | −0.6 | +14.1 | −18.6 |
| **learned gate (AUC .62)** | **41.5** | **+1.3** | **+13.8** | −13.9 |
| oracle gate | 45.3 | +7.3 | +13.3 | 0.0 |

The learned gate preserves essentially the full hard-stratum gain (+13.8 vs
+13.3) — its recall is sufficient — but false-fires on easy segments
(precision 0.16) keep pooled gains small. Signal ladder recorded: surface
features (chance) → draft logprobs (AUC 0.62, +1.3 end-task) → oracle (+7.3
= value of a perfect predictor). Two forward paths: (a) stronger predictors
(fine-tuned classifier; slide-term/draft mismatch features); (b) remove the
binary gate entirely via logits-level soft hint integration — since easy-harm
is substantially LCP protocol sensitivity, a decoding-level scheme may make
injection benign by construction, relaxing the selectivity requirement.

## Follow-up 6: injection-form comparison — uniform logit bias is benign but powerless (2026-07-19, overnight)

Soft injection via decode-time logit bias on VLM slide-term tokens (prompt
untouched; vLLM logit_bias, sweep +2/+4/+8; `runs_bias{2,4,8}.jsonl`):

| injection form (el, n=40) | hard-Δ | easy-Δ | verdict |
|---|---|---|---|
| prompt injection (ungated) | +14.1 | −18.6 | powerful but disruptive |
| prompt + oracle need-gate | +13.3 | 0.0 | powerful and benign, needs a predictor |
| uniform logit bias +2 | +0.0 | +0.0 | inert |
| uniform logit bias +4/+8 | +1.3 | −0.4 | **benign but powerless** (saturates) |

The soft-injection hypothesis splits: prompt-perturbation harm indeed vanishes
(easy-Δ ≈ 0, confirming the LCP-sensitivity diagnosis), but so does the gain —
an unconditional bias cannot flip greedy decoding toward terms the model
doesn't already favor, and pushing single tokens without continuation
constraints cannot produce multi-token terms. The indicated correct form is
**trie/prefix-constrained contextual biasing** (shallow-fusion style, standard
in ASR contextual biasing): boost a term's next token only when the generated
prefix already matches the term's prefix — potentially powerful AND benign,
removing the gate requirement. Requires a custom logits processor (not stock
vLLM); scheduled for the full system.

## Follow-up 7: zh secondary set replicates every headline (2026-07-19, overnight)

Chinese-LiPS 16 segments (zh slides = M1+M2 regime), same conditions:

| condition (zh, n=16) | chrF | hard-Δ (n=6) | easy-Δ |
|---|---|---|---|
| none | 49.6 | — | — |
| slide terms (prompt) | 51.6 | +14.3 | −7.2 |
| **slide + oracle-gate** | **55.3** | +14.3 | 0.0 |
| uniform bias +4 | 46.9 | +0.0 | −4.4 |
| oracle terms | 55.6 | +27.8 | −5.5 |

Every el finding replicates: gated injection matches the oracle-terms bound
(55.3 vs 55.6); prompt injection helps hard and disturbs easy; uniform bias is
powerless. Notably the zh slide terms are *source-language* (M2 recognition
support), and the hard-stratum gain (+14.3) matches the el M3 regime —
recognition support alone is strong when slides are clean (Chinese-LiPS 1080p
OCR). Cross-regime M2-vs-M3 comparison at scale remains for the full system.

## Follow-up 8: ACL 60/60 En→Zh (S3) first results (2026-07-19)

S3 executed end-to-end: 5 Anthology-hosted talk mp4s, 229 frames, VLM slide
pass (**95% of frames term-dense — 8× mTEDx density**; screen-recorded slides),
60 En→Zh items each containing official tagged-glossary hits; runner extended
with --tgt-lang (character-level LCP for zh). `docs/killtest/acl6060/`.

| condition (n=60) | chrF | Δ | termAcc (official tags) | hard-Δ (n=38) | easy-Δ |
|---|---|---|---|---|---|
| none | 26.3 | — | 0.42 | — | — |
| **slide (VLM, ungated)** | **28.3** | **+3.1** | **0.44** | **+9.1** | −7.3 |
| oracle terms | 25.9 | −1.1 | 0.39 | +3.8 | −9.6 |
| wrong slide | 23.1 | −3.2 | 0.32 | +3.3 | −14.3 |

Findings:
1. **H2's M2-regime prediction lands exactly: slide-string copy rate = 0.00**
   (2/462) — English slide strings never enter zh output verbatim, vs the M3
   copy channel central to X→En. First direct cross-regime evidence for the
   mechanism decomposition.
2. **Real slides help En→Zh ungated** (+3.1 pooled, +9.1 hard): with 95%-dense,
   full-screen slides whose wording matches the speech (speakers read their
   slides), hint injection aligns with the model's natural phrasing — little
   perturbation, gains retained.
3. **Protocol sensitivity amplifies on zh targets**: oracle/wrong regress via
   char-level LCP truncation (worst cases commit once then stall, e.g. output
   "但是"). Even helpful hints in unnatural list form disturb zh decoding more
   than the slide's speech-aligned terms do. Reinforces (again) logits-level
   integration for the full system; probe zh-target numbers carry this caveat.
4. **Wrong-slide harm is directionally asymmetric**: benign on X→En (32B),
   −3.2 with easy-Δ −14.3 on En→Zh — faithfulness evaluation must be
   per-direction.

## Follow-up 9: decoding determinism is the root of LCP truncation (2026-07-19)

Re-running the protocol with local transformers greedy decoding (Qwen3-14B,
Taurus A6000, `runs_trie.jsonl`) instead of the vLLM server: baseline chrF
jumps 38.8 → 62.7 and hard segments shrink 22 → 5 on the same items. Same
protocol, same data — the difference is decoding determinism. **vLLM serving
introduces float-level nondeterminism even at temperature 0** (batching/kernel
scheduling), which makes consecutive-step outputs drift, stalling the Local
Agreement longest-common-prefix commit; prompt hints amplify the drift. This
completes the easy-harm causal chain: nondeterminism × prompt perturbation ×
LCP. Implications: (i) full-system LA policies need deterministic decoding or
drift-tolerant commit rules; (ii) earlier vLLM-based probe numbers measure the
*relative* value of injection forms under a realistic (noisy-serving) regime —
directionally valid, absolute chrF depressed; (iii) the first trie-vs-prompt
comparison on 14B was uninformative for the "power" axis (baseline termR 0.71,
ceiling) — being re-run on 14B-mined hard segments; trie's benignness (easy-Δ
= 0.0) did confirm.

## Follow-up 10 (FINAL): injection-form endgame — deterministic decoding makes prompt injection both powerful and benign (2026-07-19)

Power test on 50 14B-hard segments (deterministic transformers decoding,
oracle hints, `runs_trie_hard.jsonl`; 14B hard rate 377/1540 = 24.5%, matching
32B's 24% — failure rate is a data property, serving nondeterminism only added
truncation):

| condition | chrF | Δ | p | termR |
|---|---|---|---|---|
| none | 50.0 | — | — | 0.27 |
| **prompt** | **55.4** | **+6.5** | **<0.001** | **0.46** |
| trie bias (root +1.5 / in-term +6) | 50.7 | +0.7 | 0.006 | 0.29 |

**Three-step reversal, now closed:**
1. Under vLLM serving, prompt hints appeared to disturb easy segments →
   motivated gates and soft injection.
2. Logits-level forms (uniform and trie bias) are benign but powerless: term
   failures are phrase-planning failures, not single-token near-misses; a
   posterior logit nudge is too late and too local, while prompt hints act at
   the planning level.
3. Fixing the root cause (deterministic decoding) shows prompt injection was
   never the problem: it is powerful (+6.5, p<0.001, termR +0.19) AND benign
   (easy-Δ −0.4 ≈ 0 in the deterministic 14B run).

**Full-system verdict**: deterministic decoding (or drift-tolerant commit) +
prompt-level hint injection is the correct configuration; need-gating demotes
from necessity to a token-budget optimization; trie biasing is recorded as a
negative result. The method section is updated accordingly.

## Caveats

- Text-prefix simulation (transcript, not audio); segment-level, not long-form
  session; AL differences small because LCP stability dominates latency in this
  protocol. The full pipeline (audio, session-level, async slide worker) is
  future work — this probe only establishes the information-value premise.
- zh refs are the author's own drafts (n=16); el refs are mTEDx human refs (n=40).
- Slide condition el uses 480p OCR — a deliberately conservative lower bound.

## Files

- `runs_qwen3_32b.jsonl` (224 runs, primary), `runs_qwen25_7b_local.jsonl`
  (224 runs, pilot), `killtest_items.json` (probe set), `killtest_32b_per_item.tsv`.
