# Paper skeleton

> **Status (2026-07-31): stale positive narrative.** The current introduction
> and results still treat transcript-input probes as evidence for speech
> ambiguity resolution and predate the decisive same-talk wrong-slide control.
> Do not submit or extend the current claim chain. The next rewrite must follow
> `docs/RESEARCH_GOAL_20260731.md` and `docs/FINDINGS.md` after the registered
> semantic/lip controls are complete.

Compile locally from `paper/`:

```bash
xelatex main
bibtex main
xelatex main
xelatex main
```

The skeleton uses standard `article` + `natbib` so it is self-contained.  For ACL/EMNLP submission, replace the preamble with the official ACL style and keep the same section files.

Suggested target title:

> When Slides Help: Evidence-Aware Context Management for Slide-Aware Simultaneous Speech Translation
