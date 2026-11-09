"""Phase 2a: rank and direction of ΔW.

Synthetic: removing one direction (any scale α) from k columns gives rank 1, recovers v,
and α. Qwen acceptance: rank 1 in layers 15 and 19 and a small residual at the fitted α.

Deviation from the spec, decided 2026-11-04: the edit is c' = c − 6(c·v)v, not plain
removal (α = 1), and cos(v_15, v_19) = 0.849, not > 0.9. Both are reported as findings
(README) rather than gated, per the spec's "say so plainly rather than forcing agreement".
"""

from pathlib import Path

import pytest
import torch
from test_load import base_tensors, make
from test_structural_groundtruth import BASE, EDITED, cached

from wdiff.directional import cosine_matrix, read_tensor, tensor_direction


@pytest.mark.parametrize("alpha", [1.0, 6.0])
def test_synthetic_projection(tmp_path, alpha):
    name = "model.layers.1.mlp.down_proj.weight"
    base = {k: v.float() for k, v in base_tensors().items()}  # float32: no rounding noise
    edited = {k: v.clone() for k, v in base.items()}
    v = torch.nn.functional.normalize(torch.arange(1.0, 9.0), dim=0)
    w = edited[name]
    w[:, :5] = w[:, :5] - alpha * torch.outer(v, v @ w[:, :5])
    a, b = make(tmp_path, "a", base), make(tmp_path, "b", edited)

    r = tensor_direction(name, read_tensor(a, name), read_tensor(b, name), top_k=8)
    assert r["axis"] == "cols" and r["edited_vectors"] == 5
    assert r["effective_rank"] == 1
    assert abs(torch.tensor(r["top_direction"]) @ v) == pytest.approx(1.0, abs=1e-5)
    assert r["projection_scale"] == pytest.approx(alpha, rel=1e-4)
    assert r["scaled_residual"] < 1e-5
    assert (r["projection_residual"] < 1e-5) == (alpha == 1.0)


def test_qwen_acceptance():
    b, e = cached(BASE), cached(EDITED)
    names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]
    recs = [tensor_direction(n, read_tensor(Path(b), n), read_tensor(Path(e), n), 8) for n in names]
    cos = cosine_matrix(recs)[names[0]][names[1]]
    for r in recs:
        print(
            f"{r['name']}: rank {r['effective_rank']}, σ1 energy {r['top_energy']:.6f}, "
            f"residual(α=1) {r['projection_residual']:.3f}, α {r['projection_scale']:.3f}, "
            f"residual(α) {r['scaled_residual']:.3e}"
        )
    print(f"cos(v_15, v_19) = {cos:+.4f}")

    assert all(r["effective_rank"] == 1 for r in recs)
    assert all(r["scaled_residual"] < 0.1 for r in recs)
