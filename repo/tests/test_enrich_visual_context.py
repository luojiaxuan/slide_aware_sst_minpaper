import json
import subprocess
import sys


def test_enrich_visual_context_with_mock_provider(tmp_path):
    input_path = tmp_path / "items.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "clip_1",
                "lecture_id": "lec",
                "source_transcript": "我们讨论这个模型。",
                "video": {"frame_paths": ["/frames/clip_1_PPT.jpg"]},
                "visual_context": {"ocr_text": [], "scene_summary": ""},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/enrich_visual_context.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "mock",
        ],
        check=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    visual = row["visual_context"]
    assert visual["ocr_text"] == ["mock visible term: clip_1_PPT"]
    assert visual["scene_summary"] == "Mock slide context for clip_1_PPT."
    assert visual["metadata"]["context_enrichment"]["provider"] == "mock"


def test_enrich_visual_context_mock_batch_preserves_order(tmp_path):
    input_path = tmp_path / "items.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    rows = []
    for idx in range(3):
        rows.append(
            json.dumps(
                {
                    "id": f"clip_{idx}",
                    "lecture_id": "lec",
                    "source_transcript": "我们讨论这个模型。",
                    "video": {"frame_paths": [f"/frames/clip_{idx}_PPT.jpg"]},
                    "visual_context": {"ocr_text": [], "scene_summary": ""},
                },
                ensure_ascii=False,
            )
        )
    input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/enrich_visual_context.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "mock",
            "--batch-size",
            "2",
        ],
        check=True,
    )

    got = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in got] == ["clip_0", "clip_1", "clip_2"]
    assert got[0]["visual_context"]["ocr_text"] == ["mock visible term: clip_0_PPT"]
    assert got[2]["visual_context"]["ocr_text"] == ["mock visible term: clip_2_PPT"]


def test_enrich_visual_context_mock_batch_preserves_order_with_missing_frame(tmp_path):
    input_path = tmp_path / "items.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    rows = [
        {
            "id": "clip_0",
            "lecture_id": "lec",
            "source_transcript": "我们讨论这个模型。",
            "video": {"frame_paths": ["/frames/clip_0_PPT.jpg"]},
            "visual_context": {"ocr_text": [], "scene_summary": ""},
        },
        {
            "id": "clip_1",
            "lecture_id": "lec",
            "source_transcript": "没有可用帧。",
            "video": {"frame_paths": []},
            "visual_context": {"ocr_text": ["existing"], "scene_summary": "kept"},
        },
        {
            "id": "clip_2",
            "lecture_id": "lec",
            "source_transcript": "继续讨论。",
            "video": {"frame_paths": ["/frames/clip_2_PPT.jpg"]},
            "visual_context": {"ocr_text": [], "scene_summary": ""},
        },
        {
            "id": "clip_3",
            "lecture_id": "lec",
            "source_transcript": "继续讨论。",
            "video": {"frame_paths": ["/frames/clip_3_PPT.jpg"]},
            "visual_context": {"ocr_text": [], "scene_summary": ""},
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/enrich_visual_context.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "mock",
            "--batch-size",
            "2",
        ],
        check=True,
    )

    got = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in got] == ["clip_0", "clip_1", "clip_2", "clip_3"]
    assert got[1]["visual_context"]["ocr_text"] == ["existing"]
    assert got[2]["visual_context"]["ocr_text"] == ["mock visible term: clip_2_PPT"]
