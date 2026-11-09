"""Edits that span several directions: gap rank, shared subspace, in-subspace energy."""

import torch

from wdiff.directional import edited_columns, subspace_overlap, tensor_direction
from wdiff.directions import named_cosines


def _project_out(w: torch.Tensor, q: torch.Tensor, alphas: list[float]) -> torch.Tensor:
    for a, v in zip(alphas, q.T, strict=True):
        w = w - a * torch.outer(v, v @ w)
    return w


def test_rank_k_shared_subspace():
    g = torch.Generator().manual_seed(0)
    q = torch.linalg.qr(torch.randn(32, 3, generator=g)).Q
    recs = []
    for i in range(2):
        w = torch.randn(32, 64, generator=g)
        recs.append(tensor_direction(f"model.layers.{i}.mlp.down_proj.weight", w,
                                     _project_out(w, q, [1.1, 0.6, 0.2]), top_k=8))  # fmt: skip
    assert all(r["gap_rank"] == 3 for r in recs)
    ov = subspace_overlap(recs)
    a, b = (r["name"] for r in recs)
    assert abs(ov[a][b] - 1) < 1e-4
    rows = named_cosines(recs, {"refusal": {0: q[:, 0], 1: torch.randn(32, generator=g)}})
    assert abs(rows[0]["in_subspace"] - 1) < 1e-4
    assert rows[1]["in_subspace"] < 0.5 and rows[1]["random_baseline"] == 3 / 32


def test_dense_edit_uses_short_vectors():
    w = torch.randn(100, 8)  # an embedding: 100 tokens x 8 residual
    cb, _, axis = edited_columns(w, w + 1)
    assert axis == "rows" and cb.shape == (8, 100)
