#!/usr/bin/env python3
"""Materialize deterministic full-talk noise and reverberation conditions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np


FORBIDDEN_KEYS = ("reference", "transcript", "translation", "target_text", "tagged_terminology")
ENERGY_VAD_V1_CONTRACT = {
    "name": "energy_vad_v1",
    "frame_ms": 25.0,
    "hop_ms": 10.0,
    "rms_percentile": 95,
    "relative_threshold_db": -15.0,
    "absolute_threshold_dbfs": -50.0,
}
WRAP_FADE_MS = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_inference_safe(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_KEYS):
                raise ValueError(f"Forbidden field at {path}.{key}")
            assert_inference_safe(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_inference_safe(nested, f"{path}[{index}]")


def read_pcm16_wav(path: Path, channel: int | None = None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM16 WAV: {path}")
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    values = values.reshape(-1, channels).astype(np.float32) / 32768.0
    if channel is None:
        mono = values.mean(axis=1)
    else:
        if not 0 <= channel < channels:
            raise ValueError(f"Invalid channel {channel} for {path}")
        mono = values[:, channel]
    return mono, sample_rate


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    quantized = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(temporary), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(quantized.tobytes())
    temporary.replace(path)


def activity_mask(segments: list[dict], sample_rate: int, sample_count: int) -> np.ndarray:
    mask = np.zeros(sample_count, dtype=bool)
    for segment in segments:
        start = max(0, int(math.floor(float(segment["offset_sec"]) * sample_rate)))
        end = min(sample_count, int(math.ceil(float(segment["end_sec"]) * sample_rate)))
        if end > start:
            mask[start:end] = True
    if not np.any(mask):
        raise ValueError("Source-side segment intervals contain no active samples")
    return mask


def energy_vad_v1(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    relative_threshold_db: float = -15.0,
    absolute_threshold_dbfs: float = -50.0,
) -> tuple[np.ndarray, dict]:
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_samples = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    starts = np.arange(0, max(len(samples) - frame_samples + 1, 1), hop_samples)
    frame_rms = np.array(
        [rms(samples[start : min(start + frame_samples, len(samples))]) for start in starts]
    )
    reference_rms = float(np.percentile(frame_rms, 95))
    relative_threshold = reference_rms * 10 ** (relative_threshold_db / 20.0)
    absolute_threshold = 10 ** (absolute_threshold_dbfs / 20.0)
    threshold = max(relative_threshold, absolute_threshold)
    active_frames = frame_rms >= threshold
    mask = np.zeros(len(samples), dtype=bool)
    for start, active in zip(starts, active_frames):
        if active:
            mask[start : min(start + frame_samples, len(samples))] = True
    if not np.any(mask):
        raise ValueError("energy_vad_v1 found no active samples")
    return mask, {
        "name": "energy_vad_v1",
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "rms_percentile": 95,
        "relative_threshold_db": relative_threshold_db,
        "absolute_threshold_dbfs": absolute_threshold_dbfs,
        "percentile_rms": reference_rms,
        "threshold_rms": threshold,
    }


def source_activity(talk: dict, clean: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    if talk.get("segments"):
        mask = activity_mask(talk["segments"], sample_rate, len(clean))
        return mask, {"name": "union_of_source_segment_half_open_intervals"}
    return energy_vad_v1(clean, sample_rate)


def clean_audio_fields(talk: dict) -> tuple[Path, str]:
    if isinstance(talk.get("audio"), dict):
        return Path(talk["audio"]["path"]), str(talk["audio"]["sha256"])
    return Path(talk["audio_path"]), str(talk["audio_sha256"])


def rms(samples: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = samples if mask is None else samples[mask]
    return float(np.sqrt(np.mean(np.square(selected, dtype=np.float64))))


def stable_seed(global_seed: int, talk_id: str, condition_id: str) -> int:
    payload = f"{global_seed}\0{talk_id}\0{condition_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_rng(global_seed: int, talk_id: str, condition_id: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(global_seed, talk_id, condition_id))


def circular_audio(
    source: np.ndarray,
    length: int,
    offset: int,
    wrap_fade_samples: int = 0,
) -> np.ndarray:
    if len(source) == 0:
        raise ValueError("Cannot tile an empty acoustic source")
    result = np.empty(length, dtype=np.float32)
    source_index = offset % len(source)
    output_index = 0
    wrap_boundaries = []
    while output_index < length:
        count = min(len(source) - source_index, length - output_index)
        result[output_index : output_index + count] = source[source_index : source_index + count]
        output_index += count
        if output_index < length and source_index + count == len(source):
            wrap_boundaries.append(output_index)
        source_index = 0
    for boundary in wrap_boundaries:
        left_count = min(wrap_fade_samples, boundary)
        right_count = min(wrap_fade_samples, length - boundary)
        if left_count:
            fade_out = np.linspace(1.0, 0.0, left_count + 2, dtype=np.float32)[1:-1]
            result[boundary - left_count : boundary] *= fade_out
        if right_count:
            fade_in = np.linspace(0.0, 1.0, right_count + 2, dtype=np.float32)[1:-1]
            result[boundary : boundary + right_count] *= fade_in
    return result


def select_sources(
    source_pool: list[dict],
    category: str,
    split: str,
    count: int,
    rng: np.random.Generator,
) -> list[dict]:
    candidates = [
        row for row in source_pool if row["category"] == category and row["split"] == split
    ]
    if len(candidates) < count:
        raise ValueError(f"Not enough {split}/{category} sources: {len(candidates)} < {count}")
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def build_noise_bed(
    source_rows: list[dict],
    sample_count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    bed = np.zeros(sample_count, dtype=np.float32)
    provenance = []
    wrap_fade_samples = int(round(WRAP_FADE_MS * sample_rate / 1000.0))
    for row in source_rows:
        source, source_rate = read_pcm16_wav(Path(row["path"]))
        if source_rate != sample_rate:
            raise ValueError(f"Sample-rate mismatch for {row['source_id']}")
        offset = int(rng.integers(0, len(source)))
        bed += circular_audio(source, sample_count, offset, wrap_fade_samples)
        provenance.append(
            {
                "source_id": row["source_id"],
                "sha256": row["sha256"],
                "offset_samples": offset,
                "source_frames": len(source),
                "wrap_count": int((offset + sample_count - 1) // len(source)),
                "wrap_fade_samples": wrap_fade_samples,
            }
        )
    bed /= math.sqrt(len(source_rows))
    return bed, provenance


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    active: np.ndarray,
    target_snr_db: float,
    peak_limit: float = 0.99,
) -> tuple[np.ndarray, dict]:
    clean_rms = rms(clean, active)
    noise_rms = rms(noise, active)
    if clean_rms <= 0 or noise_rms <= 0:
        raise ValueError("Clean and noise active RMS must be positive")
    noise_scale = clean_rms / (noise_rms * 10 ** (target_snr_db / 20.0))
    scaled_noise = noise * noise_scale
    mixed = clean + scaled_noise
    peak_before_guard = float(np.max(np.abs(mixed)))
    global_gain = min(1.0, peak_limit / peak_before_guard) if peak_before_guard else 1.0
    mixed *= global_gain
    achieved = 20.0 * math.log10(rms(clean * global_gain, active) / rms(scaled_noise * global_gain, active))
    return mixed, {
        "speech_active_rms": clean_rms,
        "unscaled_noise_active_rms": noise_rms,
        "noise_scale": noise_scale,
        "target_snr_db": target_snr_db,
        "achieved_snr_db": achieved,
        "peak_before_guard": peak_before_guard,
        "global_gain": global_gain,
    }


def trim_and_normalize_rir(rir: np.ndarray, threshold_ratio: float = 0.05) -> tuple[np.ndarray, int]:
    peak = float(np.max(np.abs(rir)))
    if peak <= 0:
        raise ValueError("RIR is silent")
    onset_candidates = np.flatnonzero(np.abs(rir) >= peak * threshold_ratio)
    onset = max(0, int(onset_candidates[0]) - 16)
    trimmed = rir[onset:].astype(np.float32, copy=True)
    energy = float(np.sqrt(np.sum(np.square(trimmed, dtype=np.float64))))
    if energy <= 0:
        raise ValueError("RIR has no energy after leading trim")
    trimmed /= energy
    return trimmed, onset


def fft_convolve_truncated(
    signal: np.ndarray,
    impulse: np.ndarray,
    block_size: int = 131072,
) -> np.ndarray:
    fft_size = 1 << (block_size + len(impulse) - 2).bit_length()
    impulse_spectrum = np.fft.rfft(impulse, fft_size)
    output = np.zeros(len(signal) + len(impulse) - 1, dtype=np.float64)
    for start in range(0, len(signal), block_size):
        block = signal[start : start + block_size]
        convolved = np.fft.irfft(np.fft.rfft(block, fft_size) * impulse_spectrum, fft_size)
        usable = min(len(convolved), len(output) - start)
        output[start : start + usable] += convolved[:usable]
    return output[: len(signal)].astype(np.float32)


def apply_rir(
    clean: np.ndarray,
    active: np.ndarray,
    rir: np.ndarray,
    peak_limit: float = 0.99,
) -> tuple[np.ndarray, dict]:
    normalized_rir, leading_trim_samples = trim_and_normalize_rir(rir)
    reverberant = fft_convolve_truncated(clean, normalized_rir)
    reverberant_rms = rms(reverberant, active)
    clean_rms = rms(clean, active)
    loudness_gain = clean_rms / reverberant_rms
    reverberant *= loudness_gain
    peak_before_guard = float(np.max(np.abs(reverberant)))
    global_gain = min(1.0, peak_limit / peak_before_guard) if peak_before_guard else 1.0
    reverberant *= global_gain
    return reverberant, {
        "rir_leading_trim_samples": leading_trim_samples,
        "rir_normalization": "l2_energy_1",
        "speech_active_rms_match_gain": loudness_gain,
        "peak_before_guard": peak_before_guard,
        "global_gain": global_gain,
    }


def materialize_condition(
    talk: dict,
    condition: dict,
    source_pool: list[dict],
    source_split: str,
    global_seed: int,
    output_root: Path,
) -> dict:
    assert_inference_safe(talk)
    talk_id = talk["talk_id"]
    clean_path, clean_sha256 = clean_audio_fields(talk)
    clean, sample_rate = read_pcm16_wav(clean_path)
    active, activity = source_activity(talk, clean, sample_rate)
    condition_seed = stable_seed(global_seed, talk_id, condition["condition_id"])
    rng = np.random.default_rng(condition_seed)
    kind = condition["kind"]
    if kind in {"babble", "generic_noise", "music"}:
        category = {
            "babble": "babble_speech",
            "generic_noise": "generic_noise",
            "music": "music",
        }[kind]
        count = int(condition.get("source_count", 5 if kind == "babble" else 1))
        sources = select_sources(source_pool, category, source_split, count, rng)
        noise, provenance = build_noise_bed(sources, len(clean), sample_rate, rng)
        output, measurements = mix_at_snr(
            clean,
            noise,
            active,
            float(condition["snr_db"]),
        )
    elif kind == "rir":
        sources = select_sources(source_pool, "rir", source_split, 1, rng)
        source = sources[0]
        rir, rir_rate = read_pcm16_wav(Path(source["path"]), int(source["selected_channel"]))
        if rir_rate != sample_rate:
            raise ValueError(f"Sample-rate mismatch for {source['source_id']}")
        output, measurements = apply_rir(clean, active, rir)
        provenance = [
            {
                "source_id": source["source_id"],
                "sha256": source["sha256"],
                "selected_channel": source["selected_channel"],
            }
        ]
    else:
        raise ValueError(f"Unsupported corruption kind: {kind}")

    destination = output_root / condition["condition_id"] / f"{talk_id}.wav"
    write_pcm16_wav(destination, output, sample_rate)
    result = {
        "talk_id": talk_id,
        "condition_id": condition["condition_id"],
        "kind": kind,
        "condition": condition,
        "source_split": source_split,
        "global_seed": global_seed,
        "condition_seed": condition_seed,
        "wrap_fade_ms": WRAP_FADE_MS if kind != "rir" else None,
        "activity_definition": activity,
        "active_sample_count": int(np.count_nonzero(active)),
        "active_sample_fraction": float(np.mean(active)),
        "sample_rate_hz": sample_rate,
        "frames": len(output),
        "duration_sec": round(len(output) / sample_rate, 6),
        "clean_audio_sha256": clean_sha256,
        "output_audio_path": str(destination),
        "output_audio_bytes": destination.stat().st_size,
        "output_audio_sha256": sha256_file(destination),
        "sources": provenance,
        "measurements": measurements,
    }
    assert_inference_safe(result)
    return result


def portable_row(row: dict, portable_staging_label: str) -> dict:
    output = dict(row)
    output.pop("output_audio_path")
    output["output_audio_staging_path"] = (
        f"{portable_staging_label}/{row['condition_id']}/{row['talk_id']}.wav"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--source-split", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--portable-manifest-out", type=Path, required=True)
    parser.add_argument("--portable-staging-label", required=True)
    parser.add_argument("--talk-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    talks = load_jsonl(args.inference_manifest)
    if args.talk_id:
        requested = set(args.talk_id)
        talks = [talk for talk in talks if talk["talk_id"] in requested]
        found = {talk["talk_id"] for talk in talks}
        if found != requested:
            raise ValueError(f"Unknown talk ids: {sorted(requested - found)}")
    source_pool = load_jsonl(args.source_pool)
    contract = json.loads(args.conditions.read_text(encoding="utf-8"))
    if contract["source_split"] != args.source_split:
        raise ValueError("Condition contract and requested source split differ")
    if contract.get("activity_fallback") != ENERGY_VAD_V1_CONTRACT:
        raise ValueError("Condition contract does not match energy_vad_v1 implementation")
    if float(contract.get("wrap_fade_ms", -1)) != WRAP_FADE_MS:
        raise ValueError("Condition contract does not match the wrap-fade implementation")
    conditions = contract["conditions"]
    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Condition ids must be unique")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                materialize_condition,
                talk,
                condition,
                source_pool,
                args.source_split,
                int(contract["global_seed"]),
                args.output_root,
            ): (talk["talk_id"], condition["condition_id"])
            for talk in talks
            for condition in conditions
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            talk_id, condition_id = futures[future]
            print(
                json.dumps(
                    {
                        "status": "materialized",
                        "talk_id": talk_id,
                        "condition_id": condition_id,
                    }
                ),
                flush=True,
            )

    results.sort(key=lambda row: (row["talk_id"], row["condition_id"]))
    local_manifest = args.output_root / "corruptions.jsonl"
    local_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    portable_rows = [portable_row(row, args.portable_staging_label) for row in results]
    args.portable_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.portable_manifest_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in portable_rows),
        encoding="utf-8",
    )
    print(json.dumps({"talk_count": len(talks), "output_count": len(results)}))


if __name__ == "__main__":
    main()
