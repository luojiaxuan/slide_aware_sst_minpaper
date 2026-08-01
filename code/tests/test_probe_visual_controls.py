from pathlib import Path

from PIL import Image

from scripts.build_probe_visual_controls import build_controls


def test_visual_controls_are_deterministic_and_resolve_media(tmp_path: Path):
    input_root = tmp_path / "input"
    cross_root = tmp_path / "cross"
    output_root = tmp_path / "output"
    input_root.mkdir()
    cross_root.mkdir()
    for name in ("audio.wav", "slide.jpg", "wrong.jpg"):
        (input_root / name).write_bytes(b"media")
    for index in range(2):
        Image.new("RGB", (16, 9), color=(index * 20, 0, 0)).save(
            cross_root / f"cross_{index}.jpg"
        )
    rows = [
        {
            "id": "item-1",
            "audio": "audio.wav",
            "slide_image": "slide.jpg",
            "wrong_image": "wrong.jpg",
        }
    ]
    first, manifest = build_controls(rows, input_root, cross_root, output_root)
    second, _ = build_controls(rows, input_root, cross_root, output_root)
    assert first == second
    assert Path(first[0]["audio"]).is_absolute()
    assert Path(first[0]["cross_talk_image"]).is_file()
    assert Path(first[0]["blank_image"]).is_file()
    assert manifest["cross_talk_image_pool_count"] == 2
