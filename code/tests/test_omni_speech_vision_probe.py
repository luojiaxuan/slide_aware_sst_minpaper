from types import SimpleNamespace

import numpy as np
import pytest

from scripts import omni_speech_vision_probe as probe


def args(**overrides):
    values = {
        "chunk_s": 1.0,
        "model": "fixture/model",
        "model_revision": "a" * 40,
        "attn": "sdpa",
        "max_new_tokens": 8,
        "batch_items": 2,
        "seed": 0,
        "shard_index": 0,
        "shard_count": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stream_many_refills_dynamic_batch(monkeypatch):
    lengths = {"a.wav": 2, "b.wav": 3, "c.wav": 1}
    monkeypatch.setattr(
        probe,
        "read_audio",
        lambda path: (np.ones(lengths[path], dtype=np.float32), 1),
    )
    batch_sizes = []

    def fake_translate(prefixes, sample_rates, image_paths, items, targets,
                       processor, tokenizer, model, runtime_args):
        batch_sizes.append(len(prefixes))
        assert len(set(sample_rates)) == 1
        assert image_paths == [None] * len(prefixes)
        return [" ".join([item["id"]] * len(prefix))
                for prefix, item in zip(prefixes, items, strict=True)]

    monkeypatch.setattr(probe, "translate_prefix_batch", fake_translate)
    items = [
        (0, {"id": "a", "audio": "a.wav", "reference": "A"}),
        (1, {"id": "b", "audio": "b.wav", "reference": "B"}),
        (2, {"id": "c", "audio": "c.wav", "reference": "C"}),
    ]
    results = list(
        probe.stream_many(items, "none", None, None, None, args(), batch_size=2)
    )

    assert batch_sizes == [2, 2, 2]
    assert {record["id"] for _, record, _ in results} == {"a", "b", "c"}
    by_id = {record["id"]: record for _, record, _ in results}
    assert by_id["a"]["hypothesis"] == "a a"
    assert by_id["b"]["hypothesis"] == "b b b"
    assert by_id["c"]["hypothesis"] == "c"
    assert all(record["batch_items"] == 2 for record in by_id.values())


def test_prepare_stream_state_requires_condition_image(monkeypatch):
    monkeypatch.setattr(
        probe,
        "read_audio",
        lambda path: (np.ones(2, dtype=np.float32), 1),
    )
    with pytest.raises(ValueError, match="Missing image for slide"):
        probe.prepare_stream_state(0, {"id": "a", "audio": "a.wav"}, "slide", args())
    with pytest.raises(ValueError, match="Unknown condition"):
        probe.prepare_stream_state(0, {"id": "a", "audio": "a.wav"}, "future", args())


def test_translate_prefix_batch_rejects_mixed_sample_rates():
    with pytest.raises(ValueError, match="share one sample rate"):
        probe.translate_prefix_batch(
            [np.ones(1), np.ones(1)],
            [16_000, 48_000],
            [None, None],
            [{}, {}],
            ["English", "English"],
            None,
            None,
            None,
            args(),
        )
