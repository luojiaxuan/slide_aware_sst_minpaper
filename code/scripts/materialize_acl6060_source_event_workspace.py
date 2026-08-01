#!/usr/bin/env python3
"""Materialize isolated ACL60/60 source-event annotation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import wave


FORBIDDEN_KEY_PARTS = ("reference", "target", "translation", "transcript")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_annotation_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(part in lowered for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"Forbidden annotation workspace key: {key}")
            assert_annotation_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_annotation_safe(child)


def extract_pcm_window(
    source_path: Path,
    destination_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> dict:
    with wave.open(str(source_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"Expected PCM audio: {source_path}")
        sample_rate = source.getframerate()
        source_frames = source.getnframes()
        start_frame = max(0, min(source_frames, round(start_sec * sample_rate)))
        end_frame = max(start_frame, min(source_frames, round(end_sec * sample_rate)))
        source.setpos(start_frame)
        audio = source.readframes(end_frame - start_frame)
        params = source.getparams()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination_path), "wb") as destination:
        destination.setnchannels(params.nchannels)
        destination.setsampwidth(params.sampwidth)
        destination.setframerate(params.framerate)
        destination.setcomptype("NONE", "not compressed")
        destination.writeframes(audio)
    return {
        "sample_rate_hz": sample_rate,
        "channel_count": params.nchannels,
        "sample_width_bytes": params.sampwidth,
        "source_frame_count": source_frames,
        "clip_start_frame": start_frame,
        "clip_end_frame": end_frame,
        "clip_frame_count": end_frame - start_frame,
        "clip_start_sec": round(start_frame / sample_rate, 6),
        "clip_end_sec": round(end_frame / sample_rate, 6),
        "clip_duration_sec": round((end_frame - start_frame) / sample_rate, 6),
    }


def annotation_row(packet: dict, annotator_id: str) -> dict:
    row = {
        key: value
        for key, value in packet.items()
        if key
        not in {
            "source_audio_absolute_path",
            "source_frame_absolute_path",
        }
    }
    row.update(
        {
            "annotator_id": annotator_id,
            "event_status": "pending",
            "source_question": None,
            "source_options": [],
            "source_answer_index": None,
            "t_last_insufficient_sec": None,
            "t_first_sufficient_sec": None,
            "evidence_subtypes": [],
            "evidence_region": None,
            "term_or_entity": None,
            "negative_labels": [],
            "annotation_note": "",
        }
    )
    assert_annotation_safe(row)
    return row


def render_readme(packet_count: int) -> str:
    return f"""# ACL60/60 Source Event Annotation Workspace v1

状态：`PENDING_DOUBLE_ANNOTATION`

本 workspace 含 {packet_count} 个 source-side packets。每个 packet 只有当前 frame、建议
audio window 和时间映射，不含 transcript、target/reference 或模型输出。

## 独立标注

- annotator A 只编辑 `annotations/annotator_a.jsonl`；
- annotator B 只编辑 `annotations/annotator_b.jsonl`；
- 两人不能读取对方文件；
- 所有时间字段使用 full-talk 秒数，`clip_t_evidence_sec` 仅用于播放器定位；
- 先锁定 `source_question`、2--4 个 `source_options` 和唯一答案，再听未来 audio；
- 以 0.96 秒步长填写最后不足与首次足够边界；negative packet 不能删除。

完整标签定义与 adjudication gate 见 Git 中的
`docs/ACL6060_SOURCE_EVENT_ANNOTATION_V1.md`。
"""


def materialize(
    *,
    seed_rows: list[dict],
    acl_root: Path,
    portable_root: Path,
    output_root: Path,
) -> tuple[list[dict], dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    packet_rows = []
    for seed in seed_rows:
        frame_source = portable_root / seed["frame_path"]
        audio_source = acl_root / seed["split"] / "full_wavs" / seed["audio_id"]
        if sha256_file(frame_source) != seed["frame_sha256"]:
            raise ValueError(f'Frame hash mismatch: {seed["packet_id"]}')
        if sha256_file(audio_source) != seed["audio_sha256"]:
            raise ValueError(f'Audio hash mismatch: {seed["packet_id"]}')
        packet_dir_name = seed["packet_id"].replace(":", "__")
        packet_dir = output_root / "packets" / packet_dir_name
        frame_output = packet_dir / "frame.jpg"
        audio_output = packet_dir / "audio.wav"
        packet_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frame_source, frame_output)
        timing = extract_pcm_window(
            audio_source,
            audio_output,
            start_sec=float(seed["suggested_audio_window_start_sec"]),
            end_sec=float(seed["suggested_audio_window_end_sec"]),
        )
        packet = {
            **seed,
            "workspace_packet_dir": f"packets/{packet_dir_name}",
            "workspace_frame_path": f"packets/{packet_dir_name}/frame.jpg",
            "workspace_audio_path": f"packets/{packet_dir_name}/audio.wav",
            "workspace_frame_sha256": sha256_file(frame_output),
            "workspace_audio_sha256": sha256_file(audio_output),
            **timing,
            "clip_t_evidence_sec": round(
                float(seed["t_evidence_sec"]) - timing["clip_start_sec"], 6
            ),
        }
        assert_annotation_safe(packet)
        (packet_dir / "packet.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        packet_rows.append(packet)

    manifest_path = output_root / "packet_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet_rows),
        encoding="utf-8",
    )
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    annotation_paths = {}
    for annotator_id in ("annotator_a", "annotator_b"):
        path = annotations_dir / f"{annotator_id}.jsonl"
        rows = [annotation_row(packet, annotator_id) for packet in packet_rows]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        annotation_paths[annotator_id] = path
    (output_root / "README.md").write_text(render_readme(len(packet_rows)), encoding="utf-8")

    summary = {
        "dataset": "acl6060",
        "split": "dev",
        "artifact": "source_event_double_annotation_workspace",
        "status": "PENDING_DOUBLE_ANNOTATION",
        "packet_count": len(packet_rows),
        "talk_count": len({row["talk_id"] for row in packet_rows}),
        "frame_count": len(packet_rows),
        "audio_clip_count": len(packet_rows),
        "audio_duration_sec": round(sum(row["clip_duration_sec"] for row in packet_rows), 3),
        "audio_bytes": sum(
            (output_root / row["workspace_audio_path"]).stat().st_size for row in packet_rows
        ),
        "frame_bytes": sum(
            (output_root / row["workspace_frame_path"]).stat().st_size for row in packet_rows
        ),
        "local_output": str(output_root),
        "packet_manifest_sha256": sha256_file(manifest_path),
        "annotator_a_sha256": sha256_file(annotation_paths["annotator_a"]),
        "annotator_b_sha256": sha256_file(annotation_paths["annotator_b"]),
        "source_transcript_included": False,
        "target_or_reference_included": False,
        "model_output_included": False,
    }
    (output_root / "workspace_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return packet_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    _, summary = materialize(
        seed_rows=load_jsonl(args.seed),
        acl_root=args.acl_root,
        portable_root=args.portable_root,
        output_root=args.output_root,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("packet_count", "audio_duration_sec")}))


if __name__ == "__main__":
    main()
