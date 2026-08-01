---
license: cc-by-4.0
language:
- en
pretty_name: Slide-Aware SST Controlled Acoustic Development Inputs
---

# Slide-Aware SST Controlled Acoustic Development Inputs

This private dataset contains source-only full-talk acoustic variants for the
five ACL60/60 development talks used by the `slide_aware_sst_minpaper` project.
It contains no transcripts, translations, references, term annotations, or
model outputs.

## Contents

- 5 English conference talks;
- 15 variants per talk, 75 mono PCM16 WAV files at 16 kHz;
- 12 five-speaker MUSAN babble conditions: +10, +5, 0, and -5 dB, each with
  three deterministic source/offset replicates;
- one MUSAN generic-noise condition at 0 dB;
- one MUSAN music condition at 0 dB;
- one fixed real medium-room near-field RIR condition from OpenSLR 28;
- `manifest.jsonl`, exact source/config contracts, and generation metadata.

Every corruption is applied to the continuous full talk. SNR is measured over
samples selected by `energy_vad_v1`, a target-free 25 ms/10 ms frame-energy
rule with a threshold of `max(-50 dBFS, p95 frame RMS - 15 dB)`. Each manifest
row records the stable condition seed, exact source IDs and hashes, offsets,
wrap counts, 10 ms wrap fades, target and achieved SNR, peak guard, output hash,
and duration.

## Provenance

- ACL60/60 development audio: official IWSLT release, CC BY 4.0.
- MUSAN / OpenSLR 17: CC BY 4.0.
- Room Impulse Response and Noise Database / OpenSLR 28: Apache 2.0.

The exact archive hashes, source split, selected-file hashes, RIRs, code, tests,
and reproduction commands are maintained in
[slide_aware_sst_minpaper](https://github.com/luojiaxuan/slide_aware_sst_minpaper).
Development and confirmatory acoustic source pools are disjoint. This revision
contains development variants only.

## Scope

These are controlled acoustic interventions, not recordings of real noisy
conference conditions. Noise replicates are repeated measurements of the same
five talks and must not be treated as independent samples. Native audio must be
reported separately, and an audio-only robustness baseline is required before
attributing any improvement to visual context.
