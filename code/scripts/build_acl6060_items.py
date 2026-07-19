#!/usr/bin/env python
"""Build ACL 60/60 En->Zh probe items (S3 stratum) from RASST assets.

Inputs (from gavinlaw/rasst-main-result-data unless noted):
- main_result/inputs/acl_zh/audio.yaml   : 468 segments with talk-internal
  offset/duration (wav points at the full-talk audio)
- main_result/inputs/acl_zh/source_text.txt, ref.txt : English source lines and
  Chinese references, line-aligned with the yaml
- glossaries/acl6060_tagged_gt_raw_min_norm2.json : official tagged terms with
  zh/ja/de target translations
- VLM slide timelines per talk (vlm_slide_terms.py over frames extracted from
  the ACL Anthology talk mp4s at https://aclanthology.org/<id>.mp4)

Selection: segments with >=1 non-trivial glossary hit (term in source AND zh
rendering in reference), >=8 source words, capped per talk. Conditions follow
the kill-test runner: slide terms = VLM timeline at segment start (90 s
lookback); oracle terms = the hit terms' zh renderings (+ en forms); wrong =
another item's slide terms.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TRIVIAL = {"ai", "data", "model", "models", "task", "tasks", "method", "methods",
           "results", "work", "paper", "approach", "language", "languages",
           "system", "systems", "example", "examples", "training", "test",
           "question", "questions", "answer", "answers", "word", "words",
           "research"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-yaml", required=True)
    parser.add_argument("--source-text", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--glossary", required=True)
    parser.add_argument("--vlm-timeline-glob", required=True,
                        help="glob for per-talk VLM timelines, e.g. 'acl_vlm_*.json'")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-talk", type=int, default=12)
    parser.add_argument("--lookback", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    random.seed(args.seed)

    entries = parse_yaml(args.audio_yaml)
    src = Path(args.source_text).read_text().splitlines()
    ref = Path(args.ref).read_text().splitlines()
    gloss = json.load(open(args.glossary))
    timelines = {}
    for fn in Path(".").glob(args.vlm_timeline_glob):
        tid = re.search(r"(2022\.acl-long\.\d+)", fn.name).group(1)
        timelines[tid] = json.load(open(fn))

    items, per_talk = [], {}
    for i, e in enumerate(entries):
        talk = re.search(r"(2022\.acl-long\.\d+)", e["wav"]).group(1)
        s, r = src[i], ref[i]
        hits = []
        for g in gloss:
            term, zh = g["term"], g["target_translations"].get("zh", "")
            if not zh or term.lower() in TRIVIAL:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", s, re.I) and zh in r:
                hits.append({"en": term, "zh": zh})
        if len(hits) < 1 or len(s.split()) < 8 or per_talk.get(talk, 0) >= args.per_talk:
            continue
        per_talk[talk] = per_talk.get(talk, 0) + 1
        items.append({
            "id": f"acl_{i:03d}", "talk_id": talk, "start": float(e["offset"]),
            "src_lang": "English", "source": s, "reference": r,
            "gloss_hits": hits,
            "oracle_terms": [h["zh"] for h in hits] + [h["en"] for h in hits][:2],
            "slide_terms": slide_terms_at(timelines.get(talk, []),
                                          float(e["offset"]), args.lookback),
        })
    pool = [i["slide_terms"] for i in items if i["slide_terms"]]
    for it in items:
        cands = [p for p in pool if p != it["slide_terms"]]
        it["wrong_terms"] = random.choice(cands) if cands else []
    json.dump(items, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"items: {len(items)}, per talk: {per_talk}, "
          f"with slide terms: {sum(1 for i in items if i['slide_terms'])}")


def slide_terms_at(timeline: list, t0: float, lookback: float) -> list:
    best = []
    for fr in timeline:
        if fr["t"] <= t0 and len(fr["terms"]) >= 2 and t0 - fr["t"] <= lookback:
            best = fr["terms"]
    return best[:10]


def parse_yaml(path: str) -> list[dict]:
    entries, cur = [], {}
    for line in open(path):
        line = line.rstrip()
        if line.startswith("- "):
            if cur:
                entries.append(cur)
            cur = {}
            line = "  " + line[2:]
        m = re.match(r"\s+(\w+): (.*)", line)
        if m:
            cur[m.group(1)] = m.group(2)
    if cur:
        entries.append(cur)
    return entries


if __name__ == "__main__":
    main()
