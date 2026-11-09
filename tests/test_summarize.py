"""Phase 4a offline checks on the real Qwen reports (no API call)."""

import json
from pathlib import Path

import pytest

from wdiff.summarize import build_context, check_no_weights, check_numbers, check_shape

OUT = Path(__file__).resolve().parents[1] / "reports" / "qwen3.5-0.8b-wsp"


@pytest.fixture(scope="module")
def ctx():
    if not (OUT / "behavioral.json").exists():
        pytest.skip("Qwen reports not present")
    return build_context(OUT)


def test_context_has_no_weights(ctx):
    check_no_weights(ctx)
    assert "top_direction" not in json.dumps(ctx)
    with pytest.raises(ValueError):
        check_no_weights({"v": [0.1] * 1024})


def test_numbers_check(ctx):
    good = (
        "Flip one MLP output direction in layers 15 and 19\n\n"
        "Where: 131,072 weights (0.015%) in 2 down_proj tensors\n"
        "What: rank 1, scale 6.0, cos 0.849 between layers, refusal cos 0.007\n"
        "Effect: math accuracy 0.12 → 0.32, hit cap 88% → 64%\n"
        "Unchanged: refusal gate holds\n"
        "Confidence: medium — loop_rate moved only 0.92 → 0.88"
    )
    assert check_numbers(good, ctx) == []
    assert check_shape(good) == []
    assert check_numbers(good + " and 77.7% fewer loops", ctx) == ["77.7"]
    rec = json.loads((OUT / "directional.json").read_text())["named_direction_cosines"]
    tiny = f"{rec[1]['in_subspace']:.1e}"  # e.g. 3.6e-04, rounded from the report
    assert check_numbers(f"in-subspace {tiny}", ctx) == []
    assert check_numbers("in-subspace 7.7e-04", ctx) == ["7.7e-04"]
    assert check_shape("x" * 80 + "\n\nWhere: a") != []
