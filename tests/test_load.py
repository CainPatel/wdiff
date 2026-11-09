"""Synthetic checkpoints: loader (single file + sharded), mismatch abort, structural stats."""

import json

import pytest
import torch
from safetensors.torch import save_file

from wdiff.load import CheckpointMismatch, iter_pairs
from wdiff.structural import diff, rollup


def make(tmp_path, name, tensors, sharded=False):
    d = tmp_path / name
    d.mkdir()
    if not sharded:
        save_file(tensors, d / "model.safetensors")
        return d
    weight_map = {}
    for i, (k, v) in enumerate(tensors.items()):
        fname = f"model-{i:05d}.safetensors"
        save_file({k: v}, d / fname)
        weight_map[k] = fname
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return d


def base_tensors():
    g = torch.Generator().manual_seed(0)
    return {
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, generator=g).bfloat16(),
        "model.layers.1.mlp.down_proj.weight": torch.randn(8, 16, generator=g).bfloat16(),
        "model.embed_tokens.weight": torch.randn(4, 8, generator=g).bfloat16(),
    }


@pytest.mark.parametrize("sharded", [False, True])
def test_structural_finds_edited_columns(tmp_path, sharded):
    base = base_tensors()
    edited = {k: v.clone() for k, v in base.items()}
    w = edited["model.layers.1.mlp.down_proj.weight"]
    v = torch.nn.functional.normalize(torch.ones(8), dim=0)
    cols = w[:, :3].float()
    w[:, :3] = (cols - torch.outer(v, v @ cols)).bfloat16()  # remove direction v from 3 columns

    recs = {
        r["name"]: r
        for r in diff(iter_pairs(make(tmp_path, "a", base, sharded), make(tmp_path, "b", edited)))
    }
    hit = recs["model.layers.1.mlp.down_proj.weight"]
    assert hit["changed_cols"] == 3 and hit["changed_rows"] == 8
    assert hit["touched"] == 3 * 8
    assert hit["changed"] == int((base["model.layers.1.mlp.down_proj.weight"] != w).sum())
    assert 0 < hit["frobenius_ratio"] < 1
    for name in ("model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"):
        assert recs[name]["changed"] == 0
        assert recs[name]["cosine"] == pytest.approx(1.0, abs=1e-6)

    s = rollup(list(recs.values()))
    assert s["changed_params"] == hit["changed"]
    assert s["per_layer"]["1"]["tensors_changed"] == 1
    assert s["per_layer"]["other"]["changed"] == 0


def test_mismatch_aborts(tmp_path):
    base = base_tensors()
    edited = dict(base)
    edited["model.embed_tokens.weight"] = torch.zeros(5, 8).bfloat16()
    with pytest.raises(CheckpointMismatch, match="shape differs"):
        list(iter_pairs(make(tmp_path, "a", base), make(tmp_path, "b", edited)))
