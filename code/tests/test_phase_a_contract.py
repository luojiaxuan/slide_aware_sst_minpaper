import json

import pytest

from slidesst.phase_a_contract import condition_contexts, load_context_packets


def packet(talk_id="talk-a"):
    def row(asr_context, mt_context, source_ids):
        return {
            "asr_context": asr_context,
            "mt_context": mt_context,
            "asr_token_count": len(asr_context),
            "mt_token_count": len(mt_context),
            "source_ids": source_ids,
        }

    return {
        "talk_id": talk_id,
        "audio_basename": f"{talk_id}.wav",
        "conditions": {
            "C0": row("", "", []),
            "C1": row("term", "term=术语", ["pdf:1"]),
            "C2": row("entity", "abstract", ["pdf:abs"]),
            "C3": row("phrase", "memory", ["pdf:p2"]),
        },
    }


def write_inputs(tmp_path, packets):
    packet_path = tmp_path / "packets.json"
    wav_path = tmp_path / "wavs.txt"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "phase_a_context_v1",
                "compiler": {
                    "implementation": "test",
                    "revision": "test-revision",
                    "command": "test command",
                    "input_sha256": "input-hash",
                    "asr_tokenizer_revision": "asr-revision",
                    "mt_tokenizer_revision": "mt-revision",
                },
                "packets": packets,
            }
        ),
        encoding="utf-8",
    )
    wav_path.write_text("/data/audio/talk-a.wav\n", encoding="utf-8")
    return packet_path, wav_path


def test_context_contract_validates_order_and_budget(tmp_path):
    packet_path, wav_path = write_inputs(tmp_path, [packet()])

    packets = load_context_packets(packet_path, wav_path)
    asr, mt = condition_contexts(packets, "C1", 8, lambda text: len(text))

    assert asr == ["term"]
    assert mt == ["term=术语"]


def test_context_contract_rejects_wav_order_mismatch(tmp_path):
    packet_path, wav_path = write_inputs(tmp_path, [packet("wrong-talk")])

    with pytest.raises(ValueError, match="does not match WAV order"):
        load_context_packets(packet_path, wav_path)


def test_context_contract_rejects_scoring_key(tmp_path):
    value = packet()
    value["references_path"] = "/hidden/zh.txt"
    packet_path, wav_path = write_inputs(tmp_path, [value])

    with pytest.raises(ValueError, match="scoring-only"):
        load_context_packets(packet_path, wav_path)


def test_context_contract_rejects_oversized_context(tmp_path):
    packet_path, wav_path = write_inputs(tmp_path, [packet()])
    packets = load_context_packets(packet_path, wav_path)

    with pytest.raises(ValueError, match="limit is 1"):
        condition_contexts(packets, "C1", 1)


def test_context_contract_rejects_empty_nonbaseline_condition(tmp_path):
    value = packet()
    value["conditions"]["C1"] = {
        "asr_context": "",
        "mt_context": "",
        "asr_token_count": 0,
        "mt_token_count": 0,
        "source_ids": [],
    }
    packet_path, wav_path = write_inputs(tmp_path, [value])

    with pytest.raises(ValueError, match="C1 has no compiled evidence"):
        load_context_packets(packet_path, wav_path)
