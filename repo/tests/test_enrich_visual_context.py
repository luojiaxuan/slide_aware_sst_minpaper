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
