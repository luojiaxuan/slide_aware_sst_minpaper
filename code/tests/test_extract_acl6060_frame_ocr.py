from scripts.extract_acl6060_frame_ocr import group_lines, parse_tsv


TSV = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95.0\tVisual
5\t1\t1\t1\t1\t2\t45\t20\t40\t10\t91.0\tEvidence
5\t1\t2\t1\t1\t1\t10\t60\t20\t10\t40.0\tweak
5\t1\t2\t1\t1\t2\t35\t60\t30\t10\t80.0\tSecond
"""


def test_parse_tsv_filters_confidence_and_normalizes_boxes():
    tokens = parse_tsv(TSV, min_confidence=50.0, image_size=(100, 100))

    assert [token["text"] for token in tokens] == ["Visual", "Evidence", "Second"]
    assert tokens[0]["bbox_norm"] == [0.1, 0.2, 0.3, 0.1]


def test_group_lines_orders_lines_and_tokens():
    tokens = parse_tsv(TSV, min_confidence=50.0, image_size=(100, 100))
    lines = group_lines(tokens)

    assert [line["text"] for line in lines] == ["Visual Evidence", "Second"]
    assert lines[0]["token_count"] == 2
    assert lines[0]["mean_confidence"] == 93.0
