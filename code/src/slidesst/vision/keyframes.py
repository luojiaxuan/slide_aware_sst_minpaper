from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess


@dataclass
class FrameInfo:
    frame_id: str
    path: str
    timestamp_sec: float


def sample_keyframes(video_path: str, start_sec: float, end_sec: float, fps: float, out_dir: Path) -> list[FrameInfo]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path or not Path(video_path).exists():
        return []
    key = hashlib.sha1(f"{video_path}:{start_sec}:{end_sec}:{fps}".encode("utf-8")).hexdigest()[:12]
    target_dir = out_dir / key
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("frame_*.jpg"))
    if existing:
        return [
            FrameInfo(frame_id=path.stem, path=str(path), timestamp_sec=start_sec + idx / max(fps, 1e-6))
            for idx, path in enumerate(existing)
        ]
    return _sample_with_cv2(video_path, start_sec, end_sec, fps, target_dir) or _sample_with_ffmpeg(video_path, start_sec, end_sec, fps, target_dir)


def _sample_with_cv2(video_path: str, start_sec: float, end_sec: float, fps: float, out_dir: Path) -> list[FrameInfo]:
    try:
        import cv2  # type: ignore
    except Exception:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames: list[FrameInfo] = []
    step = 1.0 / max(fps, 1e-6)
    t = start_sec
    idx = 0
    while t <= end_sec + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            path = out_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(FrameInfo(frame_id=path.stem, path=str(path), timestamp_sec=t))
        t += step
        idx += 1
    cap.release()
    return frames


def _sample_with_ffmpeg(video_path: str, start_sec: float, end_sec: float, fps: float, out_dir: Path) -> list[FrameInfo]:
    pattern = out_dir / "frame_%04d.jpg"
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return []
    frames = []
    for idx, path in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
        frames.append(FrameInfo(frame_id=path.stem, path=str(path), timestamp_sec=start_sec + idx / max(fps, 1e-6)))
    return frames
