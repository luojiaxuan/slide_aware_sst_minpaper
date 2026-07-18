#!/usr/bin/env python
"""Generate machine-draft English references for a Chinese long-form manifest.

Reads a Chinese-LiPS long-form JSONL (see build_chinese_lips_longform.py), translates
each segment's ``zh_transcript`` to English with an LLM, and writes the same records
with an added ``translation_draft`` field and ``translation_source: "machine_draft"``.

Draft quality is NOT verified: the output is meant to be reviewed by a bilingual
reader before use as a zh->En benchmark reference. Slide OCR text for each segment is
passed to the model as terminology context so on-screen named entities translate
consistently (this mirrors the slide-aware setting, but references stay faithful to
the spoken content only).

Requires ANTHROPIC_API_KEY. Usage:
    python translate_zh_en_draft.py --in 102_24_M_KJ.longform.jsonl \
        --out 102_24_M_KJ.longform.en_draft.jsonl --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "You are a professional Chinese-to-English translator for speech translation "
    "references. Translate the spoken Chinese faithfully and fluently. Translate ONLY "
    "what is spoken; do not add facts from the slide context even if provided. Keep "
    "named entities consistent with the slide terminology when it disambiguates a term. "
    "Return only the English translation, no notes."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--batch", type=int, default=20, help="segments per LLM call")
    args = parser.parse_args()

    import anthropic

    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]

    out_rows: list[dict] = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        drafts = _translate_batch(client, args.model, chunk)
        for row, draft in zip(chunk, drafts):
            out_rows.append({**row, "translation_draft": draft,
                             "translation_source": "machine_draft"})
        print(f"translated {min(start + args.batch, len(rows))}/{len(rows)}")

    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} draft references -> {args.out}")


def _translate_batch(client, model: str, chunk: list[dict]) -> list[str]:
    numbered = []
    for i, row in enumerate(chunk):
        ocr = " / ".join(row.get("ocr_text") or [])
        slide = f"  [slide terms: {ocr}]" if ocr else ""
        numbered.append(f"{i + 1}. {row['zh_transcript']}{slide}")
    prompt = (
        "Translate each numbered Chinese line to English. Reply with the same numbers, "
        "one translation per line, nothing else.\n\n" + "\n".join(numbered)
    )
    msg = client.messages.create(
        model=model, max_tokens=4000, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_numbered(msg.content[0].text, len(chunk))


def _parse_numbered(text: str, n: int) -> list[str]:
    out = [""] * n
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "." not in line:
            continue
        head, _, body = line.partition(".")
        if head.strip().isdigit():
            idx = int(head.strip()) - 1
            if 0 <= idx < n:
                out[idx] = body.strip()
    return out


if __name__ == "__main__":
    main()
