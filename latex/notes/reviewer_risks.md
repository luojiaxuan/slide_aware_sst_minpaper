# Reviewer risks and how to address them

## Risk 1: "This is just prompt engineering with slides."
Mitigation: emphasize the adversarial/mismatch context setting and the context overuse metrics. The claim should be about evidence policy under streaming constraints, not merely adding slides.

## Risk 2: "Pseudo-translations are not reliable references."
Mitigation: manually verify the hard-case labels and report hard-case accuracy as the primary metric. Treat BLEU/COMET as secondary.

## Risk 3: "This is ASR+MT, not end-to-end SST."
Mitigation: explicitly call the MVP a cascaded streaming SST system. Add an appendix or secondary experiment with an existing speech-LLM if feasible.

## Risk 4: "Slides help ASR, so novelty is limited."
Mitigation: compare against slide-aware ASR papers and show the added SST-specific issues: target-side commitment, latency, over-translation, wrong-slide adoption.

## Risk 5: "The benchmark is too small."
Mitigation: make it a challenge set with careful annotation and include an automatically mined dev set. If time permits, expand to 1k verified hard cases.
