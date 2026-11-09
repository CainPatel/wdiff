"""Phase 2b: named-direction comparison.

Synthetic: load_named + named_cosines pair each tensor with the direction at its layer.
Qwen negative control (spec): the loop-fix directions are not the refusal direction,
|cos(v_L, r_L)| < 0.2 at layers 15 and 19. Uses the saved directions/qwen3.5-0.8b
(computing it takes ~8 min on CPU: `wdiff direction compute`).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_structural_groundtruth import BASE, EDITED, cached

from wdiff.directional import read_tensor, tensor_direction
from wdiff.directions import load_named, named_cosines

QWEN_DIRS = Path(__file__).resolve().parents[1] / "directions" / "qwen3.5-0.8b"


def test_named_cosines_match_layers(tmp_path):
    r = np.eye(4, dtype=np.float32)[:2]  # layer 3 -> e0, layer 7 -> e1
    np.save(tmp_path / "refusal.npy", r)
    (tmp_path / "provenance.json").write_text(json.dumps({"layers": [3, 7]}))
    recs = [
        {"name": "a", "layer": 3, "top_direction": [1.0, 0.0, 0.0, 0.0]},
        {"name": "b", "layer": 7, "top_direction": [1.0, 0.0, 0.0, 0.0]},
        {"name": "c", "layer": 9, "top_direction": [1.0, 0.0, 0.0, 0.0]},
    ]
    rows = named_cosines(recs, load_named(tmp_path))
    assert [(x["tensor"], x["cosine"]) for x in rows] == [("a", 1.0), ("b", 0.0)]


def test_qwen_negative_control():
    if not (QWEN_DIRS / "refusal.npy").exists():
        pytest.skip("run `wdiff direction compute` first")
    b, e = Path(cached(BASE)), Path(cached(EDITED))
    named = load_named(QWEN_DIRS)
    for L in (15, 19):
        n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        rec = tensor_direction(n, read_tensor(b, n), read_tensor(e, n), 8)
        cos = (torch.tensor(rec["top_direction"]) @ named["refusal"][L]).item()
        print(f"cos(v_{L}, r_{L}) = {cos:+.4f}")
        assert abs(cos) < 0.2
