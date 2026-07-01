# Data schema

Use JSONL. One line is one streaming translation challenge item.

```json
{
  "id": "lecture001_seg00042",
  "lecture_id": "lecture001",
  "source_lang": "zh",
  "target_lang": "en",
  "audio": {
    "path": "/local/path/audio.wav",
    "start_sec": 120.5,
    "end_sec": 128.0
  },
  "source_transcript": "这里我们看线程调度的问题。",
  "reference_translation": "Here we look at the problem of thread scheduling.",
  "streaming_units": [
    {"t": 1.0, "partial_transcript": "这里我们"},
    {"t": 2.0, "partial_transcript": "这里我们看线程"}
  ],
  "ambiguous_items": [
    {
      "source_token": "线程",
      "pinyin": "xiancheng",
      "correct_target": ["thread", "threads"],
      "distractor_targets": ["ready-made", "existing"],
      "category": ["homophone", "technical_term"]
    }
  ],
  "slides": {
    "matched_slide_id": "lecture001_slide012",
    "previous_slide_id": "lecture001_slide011",
    "next_slide_id": "lecture001_slide013",
    "matched_slide_text": "Thread scheduling / 线程调度",
    "matched_slide_image": "/local/path/slide012.png"
  },
  "glossary": [
    {"src": "线程", "tgt": "thread", "desc": "CS execution unit", "source": "slide"},
    {"src": "现成", "tgt": "ready-made", "desc": "distractor homophone", "source": "distractor"}
  ],
  "background_docs": [
    {"doc_id": "paper_abs", "text": "This lecture introduces OS thread scheduling."}
  ],
  "evidence": [
    {
      "evidence_id": "ev001",
      "source_type": "slide_ocr",
      "text": "Thread scheduling / 线程调度",
      "target_hint": "thread scheduling",
      "slide_id": "lecture001_slide012",
      "time_distance_sec": 0,
      "is_supporting": true
    },
    {
      "evidence_id": "ev002",
      "source_type": "wrong_slide",
      "text": "现成模板",
      "target_hint": "ready-made template",
      "slide_id": "lecture001_slide009",
      "time_distance_sec": -45,
      "is_supporting": false
    }
  ],
  "annotation": {
    "verified": true,
    "annotator": "A1",
    "notes": "Correct sense is CS thread, supported by current slide."
  }
}
```

## Required fields for evaluation

- `id`
- `source_transcript`
- `reference_translation`
- `ambiguous_items[*].correct_target`
- `ambiguous_items[*].distractor_targets`
- `evidence[*].is_supporting`
- model output JSONL must include `hypothesis`, `used_evidence_ids`, `condition`, and timing fields.
