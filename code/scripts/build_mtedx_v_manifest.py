#!/usr/bin/env python
"""Build talk-level long-form manifests (mTEDx-V) from the deepdml/mtedx HF mirror.

Reads only the text/metadata columns of the remote parquet files (audio column is
skipped via parquet column projection), groups segments by talk_id (= YouTube video
id), and writes one JSONL record per talk. Optionally checks whether each talk's
video is still reachable via the YouTube oEmbed endpoint.

This script does not download any audio or video. Media must be obtained by the
user from the original TEDx/YouTube sources in compliance with their terms.

Requires: pyarrow, fsspec, aiohttp (pip install pyarrow fsspec aiohttp).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

MIRROR_BASE = "https://huggingface.co/datasets/deepdml/mtedx/resolve/refs%2Fconvert%2Fparquet"
DEFAULT_PAIRS = ["es-en", "fr-en", "it-en", "ru-en", "el-en"]
DEFAULT_SPLITS = ["test", "valid"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--check-alive", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_talk_ids: set[str] = set()
    for pair in args.pairs:
        for split in args.splits:
            talks = _load_talks(pair, split)
            all_talk_ids.update(talks)
            out = out_dir / f"{pair}.{split}.talks.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for talk_id, segments in talks.items():
                    segments.sort(key=lambda s: s["start"])
                    record = {
                        "talk_id": talk_id,
                        "youtube_url": f"https://www.youtube.com/watch?v={talk_id}",
                        "src_lang": pair.split("-")[0],
                        "tgt_lang": pair.split("-")[1],
                        "split": split,
                        "n_segments": len(segments),
                        "speech_start": segments[0]["start"],
                        "speech_end": segments[-1]["end"],
                        "segments": segments,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{pair}/{split}: {len(talks)} talks -> {out}")

    if args.check_alive:
        results = _check_alive(sorted(all_talk_ids))
        alive_path = out_dir / "alive_check.json"
        alive_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        n_alive = sum(1 for r in results.values() if r["alive"])
        print(f"alive: {n_alive}/{len(results)} -> {alive_path}")


def _load_talks(pair: str, split: str) -> dict[str, list[dict]]:
    import fsspec
    import pyarrow.parquet as pq

    url = f"{MIRROR_BASE}/{pair}/{split}/0000.parquet"
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        columns = [c for c in pf.schema_arrow.names if c != "audio"]
        table = pf.read(columns=columns).to_pydict()
    talks: dict[str, list[dict]] = collections.defaultdict(list)
    for i in range(len(table["talk_id"])):
        talks[table["talk_id"][i]].append(
            {
                "segment_id": int(table["segment_id"][i]),
                "start": round(float(table["start"][i]), 3),
                "end": round(float(table["end"][i]), 3),
                "duration": round(float(table["duration"][i]), 3),
                "transcript": table["transcript"][i],
                "translation": table["translation"][i],
            }
        )
    return talks


def _check_alive(talk_ids: list[str]) -> dict[str, dict]:
    def probe(video_id: str) -> tuple[str, dict]:
        url = (
            "https://www.youtube.com/oembed?url=https%3A//www.youtube.com/"
            f"watch%3Fv%3D{video_id}&format=json"
        )
        for _ in range(2):
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    title = json.load(r).get("title", "")
                return video_id, {"alive": True, "info": title[:80]}
            except urllib.error.HTTPError as e:
                return video_id, {"alive": False, "info": f"HTTP {e.code}"}
            except Exception:
                time.sleep(2)
        return video_id, {"alive": False, "info": "timeout"}

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for video_id, status in pool.map(probe, talk_ids):
            results[video_id] = status
    return results


if __name__ == "__main__":
    main()
