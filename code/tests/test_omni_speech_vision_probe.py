import json
from threading import Event
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
        "prefetch_next_batch": False,
        "prefetch_mode": "none",
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
    assert len(by_id["b"]["prefix_hypotheses"]) == 3
    assert by_id["b"]["prefix_hypotheses"][-1] == {
        "step": 3,
        "audio_time_sec": 3.0,
        "hypothesis": "b b b",
    }


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
    with pytest.raises(ValueError, match="Missing OCR text"):
        probe.prepare_stream_state(0, {"id": "a", "audio": "a.wav"}, "ocr", args())


def test_new_condition_aliases_and_ocr_prompt(monkeypatch):
    monkeypatch.setattr(
        probe,
        "read_audio",
        lambda path: (np.ones(2, dtype=np.float32), 1),
    )
    item = {
        "id": "a",
        "audio": "a.wav",
        "slide_image": "correct.png",
        "wrong_image": "wrong.png",
        "ocr_text": "R1 metric",
    }
    assert probe.prepare_stream_state(0, item, "audio_only", args()).image is None
    assert probe.prepare_stream_state(0, item, "raw_image", args()).image == "correct.png"
    assert probe.prepare_stream_state(0, item, "wrong_image", args()).image == "wrong.png"
    state = probe.prepare_stream_state(0, item, "ocr", args())

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages

    messages = probe.build_prompt(
        np.ones(1), None, state.item, "Chinese", Processor()
    )
    text_parts = [
        part["text"]
        for part in messages[0]["content"]
        if part["type"] == "text"
    ]
    assert any("R1 metric" in text for text in text_parts)
    assert all(part["type"] != "image" for part in messages[0]["content"])


def test_load_items_resolves_media_relative_to_manifest(tmp_path):
    manifest = tmp_path / "bundle" / "items.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "id": "a",
                "audio": "media/a.wav",
                "slide_image": "media/a.png",
                "wrong_image": "media/wrong.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = probe.load_items(str(manifest))[0]
    assert item["audio"] == str((manifest.parent / "media/a.wav").resolve())
    assert item["slide_image"] == str((manifest.parent / "media/a.png").resolve())


def test_prefetched_stream_overlaps_next_processor_batch_with_generation(monkeypatch):
    lengths = {"a.wav": 2, "b.wav": 3, "c.wav": 1}
    monkeypatch.setattr(
        probe,
        "read_audio",
        lambda path: (np.ones(lengths[path], dtype=np.float32), 1),
    )
    next_batch_started = Event()
    prepare_calls = []

    def fake_prepare(plans, processor):
        prepared = [(state.item["id"], step) for state, step in plans]
        prepare_calls.append(prepared)
        if len(prepare_calls) > 1:
            next_batch_started.set()
        return prepared

    generation_calls = 0

    def fake_generate(prepared, batch_size, tokenizer, model, runtime_args):
        nonlocal generation_calls
        generation_calls += 1
        if generation_calls == 1:
            assert next_batch_started.wait(timeout=1)
        assert len(prepared) == batch_size
        return [" ".join([item_id] * step) for item_id, step in prepared]

    monkeypatch.setattr(probe, "prepare_prefix_plans", fake_prepare)
    monkeypatch.setattr(probe, "generate_prepared_prefix_batch", fake_generate)
    items = [
        (0, {"id": "a", "audio": "a.wav"}),
        (1, {"id": "b", "audio": "b.wav"}),
        (2, {"id": "c", "audio": "c.wav"}),
    ]
    results = list(
        probe.stream_many(
            items,
            "none",
            None,
            None,
            None,
            args(prefetch_mode="thread"),
            batch_size=2,
        )
    )

    assert generation_calls == 3
    assert {record["id"] for _, record, _ in results} == {"a", "b", "c"}
    assert all(record["prefetch_mode"] == "thread" for _, record, _ in results)


def test_process_prefetch_payload_contains_only_picklable_inputs(monkeypatch):
    monkeypatch.setattr(
        probe,
        "read_audio",
        lambda path: (np.ones(3, dtype=np.float32), 1),
    )
    state = probe.prepare_stream_state(
        0,
        {"id": "a", "audio": "a.wav", "tgt_lang": "English"},
        "none",
        args(),
    )
    payload = probe.prefix_plan_payload([(state, 2)])
    assert payload[0][0].tolist() == [1.0, 1.0]
    assert payload[1:] == ([1], [None], [state.item], ["English"])


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
