#!/usr/bin/env python3
"""Build ACL60/60 long-form inference and scoring inputs for SimulStream."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import yaml
from lxml import etree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xml_docs(path: Path) -> list[tuple[str, list[str]]]:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.parse(str(path), parser).getroot()
    return [
        (doc.attrib["docid"], [(seg.text or "").strip() for seg in doc.iter("seg")])
        for doc in root.iter("doc")
    ]


def locate_segment(full_audio: Path, segment_audio: Path, start_byte: int) -> tuple[dict, int]:
    with wave.open(str(full_audio), "rb") as source:
        params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
        full_frames = source.readframes(source.getnframes())
    with wave.open(str(segment_audio), "rb") as segment:
        segment_params = (segment.getnchannels(), segment.getsampwidth(), segment.getframerate())
        segment_frames = segment.readframes(segment.getnframes())
    if params != segment_params:
        raise ValueError(f"WAV format mismatch: {segment_audio}")
    position = full_frames.find(segment_frames, start_byte)
    if position < 0:
        raise ValueError(f"Gold segment is not an exact slice of full WAV: {segment_audio}")
    frame_width = params[0] * params[1]
    if position % frame_width:
        raise ValueError(f"Gold segment starts off frame boundary: {segment_audio}")
    sample_rate = params[2]
    return (
        {
            "offset": position / frame_width / sample_rate,
            "duration": len(segment_frames) / frame_width / sample_rate,
            "speaker_id": "NA",
            "wav": full_audio.name,
        },
        position + frame_width,
    )


def build_scoring_rows(acl_root: Path, split: str) -> tuple[list[str], list[str], list[dict]]:
    source_path = acl_root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.en.xml"
    target_path = acl_root / split / "text" / "xml" / f"ACL.6060.{split}.en-xx.zh.xml"
    source_docs = xml_docs(source_path)
    target_docs = xml_docs(target_path)
    if [talk_id for talk_id, _ in source_docs] != [talk_id for talk_id, _ in target_docs]:
        raise ValueError("Source and target XML talk order differs")
    sources = []
    targets = []
    segments = []
    global_index = 1
    for (talk_id, source_segments), (_, target_segments) in zip(source_docs, target_docs, strict=True):
        if len(source_segments) != len(target_segments):
            raise ValueError(f"Source/target segment count differs for {talk_id}")
        cursor = 0
        full_audio = acl_root / split / "full_wavs" / f"{talk_id}.wav"
        for source, target in zip(source_segments, target_segments, strict=True):
            segment_audio = acl_root / split / "segmented_wavs" / "gold" / f"sent_{global_index}.wav"
            timing, cursor = locate_segment(full_audio, segment_audio, cursor)
            sources.append(source)
            targets.append(target)
            segments.append(timing)
            global_index += 1
    return sources, targets, segments


def write_bundle(acl_root: Path, inference_view: Path, split: str, inference_out: Path, scoring_out: Path) -> None:
    inference_rows = [
        json.loads(line) for line in inference_view.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(row["split"] != split for row in inference_rows):
        raise ValueError("Inference view contains a different split")
    inference_out.mkdir(parents=True, exist_ok=True)
    scoring_out.mkdir(parents=True, exist_ok=True)
    wav_list = inference_out / "wav_list.txt"
    wav_list.write_text("\n".join(row["audio_path"] for row in inference_rows) + "\n", encoding="utf-8")

    sources, targets, segments = build_scoring_rows(acl_root, split)
    source_file = scoring_out / "source.en.txt"
    target_file = scoring_out / "reference.zh.txt"
    segmentation_file = scoring_out / "audio-segments.yaml"
    source_file.write_text("\n".join(sources) + "\n", encoding="utf-8")
    target_file.write_text("\n".join(targets) + "\n", encoding="utf-8")
    segmentation_file.write_text(yaml.safe_dump(segments, sort_keys=False), encoding="utf-8")
    expected = sum(row["segment_count"] for row in inference_rows)
    if not (len(sources) == len(targets) == len(segments) == expected):
        raise ValueError("Generated scoring row count does not match inference snapshot")
    manifest = {
        "dataset": "acl6060",
        "split": split,
        "talk_ids": [row["talk_id"] for row in inference_rows],
        "segment_count": expected,
        "inference_view_sha256": sha256_file(inference_view),
        "inference_files": {
            wav_list.name: {"bytes": wav_list.stat().st_size, "sha256": sha256_file(wav_list)}
        },
        "scoring_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (source_file, target_file, segmentation_file)
        },
    }
    (scoring_out / "scoring_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--inference-view", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--inference-out", type=Path, required=True)
    parser.add_argument("--scoring-out", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(args.acl_root, args.inference_view, args.split, args.inference_out, args.scoring_out)


if __name__ == "__main__":
    main()
