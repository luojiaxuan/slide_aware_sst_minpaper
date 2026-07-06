import importlib.util
import json
import subprocess
import sys
import types

from PIL import Image


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
    assert visual["metadata"]["context_enrichment"]["batch_size"] == 1


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
    assert got[2]["visual_context"]["metadata"]["context_enrichment"]["batch_size"] == 1


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
    assert got[0]["visual_context"]["metadata"]["context_enrichment"]["batch_size"] == 1
    assert got[2]["visual_context"]["metadata"]["context_enrichment"]["batch_size"] == 2


def test_qwen_batch_sets_left_padding_and_preserves_row_outputs(monkeypatch, tmp_path):
    script_path = "scripts/enrich_visual_context.py"
    spec = importlib.util.spec_from_file_location("enrich_visual_context_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["enrich_visual_context_test"] = module
    spec.loader.exec_module(module)

    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (3, 3), color="white").save(image_a)
    Image.new("RGB", (7, 3), color="black").save(image_b)

    processor_instances = []
    seen_input_lengths = []

    class FakeInputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = types.SimpleNamespace(padding_side="right")
            processor_instances.append(self)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            image = messages[0]["content"][0]["image"]
            return f"prompt_width_{image.size[0]}"

        def __call__(self, text, images, padding=True, return_tensors="pt"):
            assert self.tokenizer.padding_side == "left"
            inputs = FakeInputs()
            lengths = [image.size[0] + 1 for image in images]
            seen_input_lengths.append(lengths)
            inputs["input_ids"] = [[idx] * length for idx, length in enumerate(lengths)]
            return inputs

        def batch_decode(self, trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            rows = []
            for ids in trimmed:
                row_idx = ids[0] - 100
                rows.append(
                    json.dumps(
                        {
                            "ocr_text": [f"row-{row_idx}"],
                            "scene_summary": f"summary-{row_idx}",
                            "objects": [],
                            "actions": [],
                            "spatial_relations": [],
                        }
                    )
                )
            return rows

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return None

        def to(self, device):
            self.device = device
            return self

        def generate(self, input_ids, **kwargs):
            return [row + [100 + idx] for idx, row in enumerate(input_ids)]

    fake_transformers = types.SimpleNamespace(
        AutoModelForImageTextToText=FakeModel,
        AutoProcessor=FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    extractor = module.QwenVLSlideContextExtractor(
        model_id="fake/qwen",
        cache_dir=None,
        device="cuda:0",
        dtype="bfloat16",
        max_new_tokens=8,
        prompt=module.DEFAULT_PROMPT,
    )

    assert processor_instances[0].tokenizer.padding_side == "left"
    got = extractor.extract_batch([str(image_a), str(image_b)])
    assert seen_input_lengths == [[4, 8]]
    assert [row["ocr_text"] for row in got] == [["row-0"], ["row-1"]]
    assert [row["scene_summary"] for row in got] == ["summary-0", "summary-1"]
