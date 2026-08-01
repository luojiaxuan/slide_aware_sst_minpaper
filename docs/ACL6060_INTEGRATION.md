# ACL 60/60 (S3) Integration Recon — historical

> 本页是 2026-07-19 的早期侦察记录，已由
> [`PHASE_A_DATA_RUNNER_FREEZE_20260731.md`](PHASE_A_DATA_RUNNER_FREEZE_20260731.md)
> 取代。官方 ACL60/60 attachment 已完整冻结，dev/eval 都包含 full WAV、gold/SHAS
> segments、multilingual text/XML 和 tagged terminology；当前流程不再依赖 RASST
> mirror 作为数据真值，也不把官方 tagged terms 当作 C1 runtime input。

Date: 2026-07-19 (overnight recon). Conclusion: **zero-obstacle integration**;
the cleanest video recovery of all three strata.

## Findings

1. **Videos are hosted by ACL Anthology itself** at the canonical URL
   `https://aclanthology.org/<anthology-id>.mp4` (verified for
   2022.acl-long.268: the paper page links "Video" directly). Academic archive,
   direct download, no ToS gray zone, no link-rot/attrition risk — strictly
   cleaner than the YouTube recovery used for S1.
2. **User's RASST assets bootstrap the split**:
   `gavinlaw/rasst-demo-acl6060-zh-segments` already packages 5 talks / 468
   segments / 51.8 min of 16 kHz mono WAV with SimulEval-compatible source and
   Chinese target lists (`ref.txt`, `audio.yaml`), talk files named by
   anthology ID → video URLs derivable mechanically. Direct RASST
   comparability on identical segments.
3. **Slide frames**: `code/scripts/extract_frames_by_manifest.py` applies
   as-is once per-segment timestamps are mapped to the talk videos (RASST
   segment ordering is per-talk sequential; exact offsets in
   `gavinlaw/rasst-main-result-data` per the manifest's source pointer).
4. **Tagged terminology**: the ACL 60/60 official release ships bilingual
   terminology; locate within the IWSLT 2023 release bundle during
   integration (user's earlier note: tags imperfect — trivial entries like
   "model"; handle with the pre-registered frequency filter, report raw +
   filtered).

## Integration steps (est. 1 day)

1. Pull `rasst-demo-acl6060-zh-segments` + segment offsets from
   `rasst-main-result-data`.
2. `curl` the 5 talk mp4s from ACL Anthology; verify durations vs audio.
3. Extract frames (existing script), VLM slide-term pass (existing script,
   hyper01).
4. Locate official terminology file; build filtered term list.
5. Probe-style run: A/B-VLM/C/D + gating on En→Zh — the M1+M2 control
   condition for H2, directly comparable to RASST numbers on the same talks.
