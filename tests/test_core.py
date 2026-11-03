"""Unit tests of the interface, the consensus and the guard (no GPU needed): python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corg.consensus import consensus, iou  # noqa: E402
from corg.interface import parse_box, parse_point, prompt_text  # noqa: E402
from corg.refine import guard  # noqa: E402


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_consensus_rejects_outliers():
    boxes = [[10, 10, 50, 50], [12, 10, 52, 50], [10, 12, 50, 52], [200, 200, 240, 240]]
    assert consensus(boxes) == [32 / 3, 32 / 3, 152 / 3, 152 / 3]
    assert consensus([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert consensus([]) is None


def test_guard():
    b = [10, 10, 50, 50]
    assert guard(b, [20, 20], [15, 15, 45, 45]) == [15, 15, 45, 45]
    assert guard(b, [80, 80], [15, 15, 45, 45]) == b          # point outside the consensus box
    assert guard(b, [20, 20], [60, 60, 90, 90]) == b          # no overlap
    assert guard(b, None, [15, 15, 45, 45]) == b


def test_interface():
    # Qwen3-VL and InternVL3.5 report values in [0, 1000]
    assert parse_box("qwen3-vl", '[{"bbox_2d": [100, 200, 300, 400]}]', 1000, 500) == [100, 100, 300, 200]
    assert parse_box("internvl3.5", "[[500, 500, 600, 700]]", 800, 800) == [400, 400, 480, 560]
    # Qwen2.5-VL reports absolute pixels of the resized image (here 812 x 812 for an 800 x 800 image)
    b = parse_box("qwen2.5-vl", '{"bbox_2d": [406, 406, 812, 812]}', 800, 800)
    assert all(abs(x - y) < 1e-6 for x, y in zip(b, [400, 400, 800, 800]))
    # a box returned to the point prompt is replaced by its center
    assert parse_point("internvl3.5", "[[100, 100, 300, 300]]", 1000, 1000) == [200, 200]
    assert parse_box("qwen3-vl", "no box here", 100, 100) is None
    # MiMo-VL: only the text after the reasoning block is parsed, and every request ends with /no_think
    b = parse_box("mimo-vl", '<think>x</think>[{"bbox_2d": [0, 0, 406, 406], "label": "car"}]', 800, 800)
    assert all(abs(x - y) < 1e-6 for x, y in zip(b, [0, 0, 400, 400]))
    assert parse_box("mimo-vl", "<think>unfinished", 800, 800) is None
    assert prompt_text("mimo-vl", "P1", "the car").endswith(" /no_think")
    assert prompt_text("qwen2.5-vl", "P5", "the car") == "Where is the car in this remote sensing image? Output its bbox coordinates using JSON format."
