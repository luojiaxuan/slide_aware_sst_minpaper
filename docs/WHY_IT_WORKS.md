> **SUPERSEDED (2026-07-31).** The affirmative case below was explicitly
> conditioned on the then-pending Qwen3-Omni speech+vision experiment. That
> experiment ran: a wrong slide produced the same quality and latency gains as
> the correct slide, so the content-based claims here are not supported. See
> [FINDINGS.md](FINDINGS.md). The related-work survey below remains valid.

# Why this paper works + related work (2026-07-24)

Written to reflect corrected framing: (i) input must be SPEECH not transcript
(acoustic ambiguity is what vision resolves); (ii) value axes are latency +
disambiguation, NOT terminology (RASST/Do-Slides-Help own terminology); (iii)
relevance selection = RASST's retriever (a reused component, not open risk).

## Why it works

- Streaming speech faces genuine acoustic ambiguity (homophones, polysemy,
  under-specified named content); this uncertainty exists only from audio, and
  is exactly what a slide can resolve. Feeding a clean transcript deletes it —
  the confound that invalidated our earlier text-model runs.
- The slide is a legal zero-latency lookahead: speakers show a slide 30-60 s
  before discussing it, so an async slide channel carries future information for
  free, off the audio critical path.
- Two open value axes (terminology is taken by RASST/DSH): (a) latency —
  anticipatory context enables earlier/more confident commits at equal quality
  (quality-StreamLAAL Pareto shift); (b) disambiguation — visual scene/topic
  resolves word senses (esp. zh->en homophones).
- Relevance selection ("which slide element now") = RASST's chunkwise retriever,
  reused; the hard sub-problem is a plugged-in component.
- Cross-regime mechanism control: X->En English slides also supply target-side
  term forms (copy channel); En->Zh control shows copy-rate 0.00. Lets us
  dissect why, not just that.
- Async vision adds zero per-token latency vs OmniFusion's synchronous
  image-on-decode-path (which pays latency for vision) — deployment-correct.

Caveat: affirmative case rests on the pending Qwen3-Omni speech+slide experiment.

## Related work

Simultaneous ST (policies/long-form): STACL/wait-k (Ma 2019), SimulEval
(Ma 2020), Local Agreement (Liu 2020), AlignAtt (Papi 2023), InfiniSST
(Ouyang 2025), DOA (Papi & Bentivogli 2026).

Retrieval/terminology for ST (closest, incl. ours): RASST (Luo 2026, reused
retriever), Do Slides Help? (Sinhamahapatra & Niehues 2025, offline ASR),
Slide-EC (Piao 2025).

Multimodal/vision translation: OmniFusion (Koneru, Huck & Niehues 2025,
synchronous vision, nearest system neighbor), Caglayan 2020 line (Imankulova
2020, Ive 2021, Haralampieva 2022 — simultaneous text MT), MCIF (Papi 2025,
En->X control).

Slides/screen corpora: SlideSpeech (Wang 2023), Chinese-LiPS (Zhao 2025),
M3AV (Chen 2024), MaViLS (Anderer 2024).

Audio-visual (lip) ST: MuAViC (Anwar 2023), AV-TranSpeech (Huang 2023) --
phonetic evidence, complementary to our semantic slide evidence.

Speech LLM/omni backbones: Qwen2.5-Omni / Qwen3-Omni.
