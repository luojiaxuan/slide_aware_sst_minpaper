import json
from argparse import Namespace

import pytest
import yaml

import scripts.prepare_phase_a_run as launcher
from scripts.prepare_phase_a_run import hf_snapshot_path, load_inference_rows, prepare_run, sha256_file


def test_hf_snapshot_path_uses_immutable_revision(tmp_path):
    path = hf_snapshot_path(tmp_path, "Qwen/Model", "abc123")

    assert path == tmp_path / "hub" / "models--Qwen--Model" / "snapshots" / "abc123"


def test_load_inference_rows_rejects_reference_key(tmp_path):
    path = tmp_path / "inference.jsonl"
    path.write_text(json.dumps({"talk_id": "x", "reference_copy": "gold"}) + "\n")

    with pytest.raises(ValueError, match="scoring-only"):
        load_inference_rows(path)


def test_prepare_run_blocks_c3_before_touching_runtime(tmp_path):
    contract = {
        "conditions": {"C3": {"launcher_status": "blocked_causal_asr_prefix_retriever_unimplemented"}}
    }
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    args = Namespace(contract=contract_path, condition="C3")

    with pytest.raises(ValueError, match="C3 launch blocked"):
        prepare_run(args)


def test_prepare_run_writes_hash_bound_c0_artifacts(monkeypatch, tmp_path):
    contract = {
        "conditions": {"C0": {"launcher_status": "ready_after_packet_and_model_preflight"}},
        "context_contract": {"max_injected_tokens_per_channel": 256},
        "runner": {
            "upstream": {"revision": "frozen-revision"},
            "toolkit": {"revision": "frozen-revision"},
            "evaluation": {"revision": "frozen-revision"},
            "adapter_class": "slidesst.iwslt_context_adapter.PhaseAContextSpeechProcessor",
            "policy": {
                "primary_chunk_ms": 960,
                "latency_unit": "char",
                "min_start_seconds": 5.0,
            },
        },
    }
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    runtime_lock = tmp_path / "runtime_lock.json"
    runtime_lock.write_text(
        json.dumps(
            {
                "container_image": "test/image:locked",
                "container_image_digest": "sha256:abc123",
                "python_version": "3.12",
                "cuda_version": "13.0",
                "torch_version": "2.0",
                "qwen_asr_version": "1.0",
                "vllm_version": "1.0",
                "transformers_version": "1.0",
                "simulstream_revision": "frozen-revision",
                "omnisteval_revision": "frozen-revision",
                "comet_model": "Unbabel/XCOMET-XL",
                "comet_revision": "comet-sha",
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "talk-a.wav"
    paper = tmp_path / "talk-a.pdf"
    audio.write_bytes(b"audio")
    paper.write_bytes(b"paper")
    inference_view = tmp_path / "inference.jsonl"
    inference_view.write_text(
        json.dumps(
            {
                "talk_id": "talk-a",
                "audio_path": str(audio),
                "audio_sha256": sha256_file(audio),
                "paper_pdf_path": str(paper),
                "paper_pdf_sha256": sha256_file(paper),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def context_row(asr, mt, source_ids):
        return {
            "asr_context": asr,
            "mt_context": mt,
            "asr_token_count": len(asr),
            "mt_token_count": len(mt),
            "source_ids": source_ids,
        }

    context_bundle = tmp_path / "context.json"
    context_bundle.write_text(
        json.dumps(
            {
                "schema_version": "phase_a_context_v1",
                "compiler": {
                    "implementation": "test",
                    "revision": "compiler-revision",
                    "command": "compile test",
                    "input_sha256": "input-sha",
                    "asr_tokenizer_revision": "asr-sha",
                    "mt_tokenizer_revision": "mt-sha",
                },
                "packets": [
                    {
                        "talk_id": "talk-a",
                        "audio_basename": "talk-a.wav",
                        "conditions": {
                            "C0": context_row("", "", []),
                            "C1": context_row("term", "term", ["pdf:1"]),
                            "C2": context_row("entity", "abstract", ["pdf:abstract"]),
                            "C3": context_row("phrase", "memory", ["pdf:2"]),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "git_head", lambda path: "frozen-revision")
    monkeypatch.setattr(
        launcher,
        "verify_model_snapshots",
        lambda contract, hf_home: {"asr": "/models/asr", "forced_aligner": "/models/fa", "mt": "/models/mt"},
    )
    output_root = tmp_path / "runs"
    args = Namespace(
        contract=contract_path,
        condition="C0",
        upstream_root=tmp_path / "upstream",
        simulstream_root=tmp_path / "simulstream",
        omnisteval_root=tmp_path / "omnisteval",
        inference_view=inference_view,
        context_bundle=context_bundle,
        runtime_lock=runtime_lock,
        output_root=output_root,
        hf_home=tmp_path / "hf",
        repo_src=tmp_path / "src",
        execute=False,
    )

    run_dir = prepare_run(args)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    processor = yaml.safe_load((run_dir / "speech_processor.yaml").read_text(encoding="utf-8"))

    assert manifest["condition"] == "C0"
    assert manifest["reference_paths_passed_to_process"] is False
    assert manifest["status"] == "prepared"
    assert processor["ner_results_path"] is None
    assert processor["abstract_results_path"] is None
    assert processor["context_condition"] == "C0"
