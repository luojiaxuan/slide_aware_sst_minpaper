---
pretty_name: MCIF Outcome Candidate Inventory
language:
- en
- zh
- de
- it
task_categories:
- translation
---

# MCIF Outcome Candidate Inventory

Private outcome-side artifact for the IWSLT 2026 MCIF simultaneous speech
translation development set. It preserves the official segment timing and
English, Chinese, German, and Italian references, then derives high-recall
source-event candidates from exact lexical overlap between the English
reference and causally available slide OCR.

## Contents

- `outcome_candidate_inventory_v1/raw/`: the five required members copied from
  the frozen official MCIF archive;
- `reference_segments.jsonl`: 919 segment-aligned multilingual reference rows;
- `candidate_events.jsonl`: 954 automatic candidates over 21 talks;
- `report.json`, `README.md`, and `SHA256SUMS`: provenance, interpretation, and
  byte-integrity records.

The candidate schema includes the source event, target references, causal slide
state binding, evidence lead, and blank fields for human eligibility, acceptable
target realization, and audio-sufficiency labels. Automatic candidates are not
gold events and are not model results.

## Provenance

- Source archive SHA256:
  `445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7`
- Source evidence ladder SHA256:
  `8f77312b93562afd8a92ea0b3139fe5f91b21b08e9740d311a1fd0a83b594f7f`
- Inference manifest SHA256:
  `7c8138bcaf32f619b76dec5f919d7fb63c141260ed0df4fc93a1a7da822b6f11`
- Builder Git commit: `27782229290f3f571020ea34928416b4b5884072`
- Candidate inventory SHA256:
  `3b1f85137c4443bd65cb82beb4217301b2f4e67ae9a7cd45ded3cbc3e5dde5a2`

Generation command:

```bash
cd code
.venv/bin/python -m scripts.build_mcif_outcome_candidate_inventory \
  --archive <mcif-long-trans.zip> \
  --expected-archive-sha256 445a4b92d0083b5416515a9639fcef126b72a5e80ef59d962dc30f82688cedb7 \
  --ladder <source_evidence_ladder_v1/ladder.jsonl> \
  --expected-ladder-sha256 8f77312b93562afd8a92ea0b3139fe5f91b21b08e9740d311a1fd0a83b594f7f \
  --inference-manifest <inference.jsonl> \
  --expected-inference-manifest-sha256 7c8138bcaf32f619b76dec5f919d7fb63c141260ed0df4fc93a1a7da822b6f11 \
  --code-repo .. \
  --output-root <new-output-directory> \
  --max-ngram 4 --expected-segments 919 --expected-talks 21
```

## Use Boundary

Keep this repository private. The official references and every derivative that
reveals them are outcome-side material. They must never be mounted into model
inference, prompt construction, training-data generation, source-side visual
screening, or control construction. Use them only after model outputs are
frozen, for annotation design and evaluation.
