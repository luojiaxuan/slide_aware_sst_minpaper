# Paper skeleton

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
