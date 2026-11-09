"""Directional diff: what each changed tensor's ΔW is, in linear-algebra terms.

Why rank-1 ΔW means one removed direction:
  Each column c of mlp.down_proj (shape [d_model, d_ff]) is the d_model-vector one MLP
  neuron writes into the residual stream. Removing a unit direction v from column c is
      c' = c − (c·v) v,   so   Δc = c' − c = −(c·v) v.
  Every edited column's change is a scalar multiple of the SAME vector v. Stack the k
  edited columns side by side: ΔW = v · a^T with a_j = −(c_j·v). A matrix that is one
  column vector times one row vector has rank 1. Its SVD ΔW = Σ σ_i u_i w_i^T then has
  a single nonzero σ_1, and u_1 = ±v. So: effective rank 1 says "one direction was
  touched", and the top left singular vector u_1 tells us which direction it was.
  Conversely, if several directions had been removed, ΔW would have rank > 1.

Per changed tensor we compute, on ΔW restricted to its edited columns (float32):
    singular values σ_1 ≥ σ_2 ≥ …        effective_rank = min r with Σ_{i≤r} σ_i² ≥ 95% Σ σ_i²
    top_direction v = u_1 (unit, length = rows; sign is arbitrary, so compare |cos|)
    projection_residual = mean_j ‖c'_j − (c_j − (c_j·v) v)‖ / ‖c_j‖
An edit can also scale the projection: c' = c − α (c·v) v. α = 1 removes v, α = 2
reflects across it, and an unnormalized direction r gives α = ‖r‖². We fit α by least
squares on the coefficients a_j = v·Δc_j ≈ −α (c_j·v) and report the residual with it:
    projection_scale α,  scaled_residual = mean_j ‖c'_j − (c_j − α (c_j·v) v)‖ / ‖c_j‖
If the edit is confined to rows instead (a transposed layout), we transpose first, so
"columns" always means the axis holding the edited vectors. When every row and every
column changed (a dense edit, as in abliteration), we take the shorter vectors: for
down_proj, o_proj and the embedding that is the residual-stream width.

Edits spanning several directions. Projecting out k directions gives ΔW of rank k: k
singular values well above the bf16 noise floor, then a cliff. gap_rank finds the cliff,
the largest ratio σ_i / σ_{i+1} among the top GAP_K values, and Q = u_1 … u_gap_rank
spans the edit subspace. Two derived numbers:
    subspace overlap of tensors a, b  = ‖Q_aᵀ Q_b‖_F² / min(k_a, k_b)   (1 = same subspace)
    in_subspace energy of a unit d    = ‖Qᵀ d‖²   (fraction of d inside the edit; a random
                                         direction gives k / d_model)
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from wdiff.load import tensor_files
from wdiff.structural import layer_of

ENERGY = 0.95
GAP_K = 64


def read_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(tensor_files(path)[name], framework="pt") as h:
        return h.get_tensor(name)


def edited_columns(
    base: torch.Tensor, edited: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """(base_cols, edited_cols) as float32 [d, k], plus which axis held the edit."""
    mask = base != edited
    cols, rows = mask.any(dim=0), mask.any(dim=1)
    by_cols_n, by_rows_n = int(cols.sum()) * base.shape[0], int(rows.sum()) * base.shape[1]
    by_cols = by_cols_n < by_rows_n or (by_cols_n == by_rows_n and base.shape[0] <= base.shape[1])
    if by_cols:
        return base[:, cols].float(), edited[:, cols].float(), "cols"
    return base[rows].T.float(), edited[rows].T.float(), "rows"


def effective_rank(s: torch.Tensor, energy: float = ENERGY) -> int:
    e = (s.double() ** 2).cumsum(0)
    return int((e < energy * e[-1]).sum()) + 1


def gap_rank(s: torch.Tensor) -> int:
    """Index of the largest drop σ_i / σ_{i+1} among the top GAP_K singular values."""
    top = s[:GAP_K].double()
    if len(top) < 2:
        return len(top)
    return int((top[:-1] / top[1:].clamp_min(1e-30)).argmax()) + 1


def tensor_direction(name: str, base: torch.Tensor, edited: torch.Tensor, top_k: int) -> dict:
    cb, ce, axis = edited_columns(base, edited)
    u, s, _ = torch.linalg.svd(ce - cb, full_matrices=False)
    v = u[:, 0]
    p = v @ cb  # c_j·v
    alpha = -((v @ (ce - cb)) @ p / (p @ p).clamp_min(1e-30)).item()
    return {
        "name": name,
        "layer": layer_of(name),
        "axis": axis,
        "edited_vectors": cb.shape[1],
        "singular_values": s[:top_k].tolist(),
        "top_energy": (s[0] ** 2 / (s**2).sum()).item(),
        "effective_rank": effective_rank(s),
        "gap_rank": (k := gap_rank(s)),
        "gap_ratio": (s[k - 1] / s[k]).item() if k < len(s) else None,
        "projection_residual": _residual(cb, ce, v, 1.0),
        "projection_scale": alpha,
        "scaled_residual": _residual(cb, ce, v, alpha),
        "top_direction": v.tolist(),
        "_basis": u[:, :k].contiguous(),  # dropped before writing JSON
    }


def _residual(cb: torch.Tensor, ce: torch.Tensor, v: torch.Tensor, alpha: float) -> float:
    predicted = cb - alpha * torch.outer(v, v @ cb)
    return ((ce - predicted).norm(dim=0) / cb.norm(dim=0).clamp_min(1e-12)).mean().item()


def cosine_matrix(records: list[dict]) -> dict[str, dict[str, float | None]]:
    """Signed cosine between top directions; None when their lengths differ."""
    vs = {r["name"]: torch.tensor(r["top_direction"]) for r in records}
    return {
        a: {b: (va @ vb).item() if va.shape == vb.shape else None for b, vb in vs.items()}
        for a, va in vs.items()
    }


def subspace_overlap(records: list[dict]) -> dict[str, dict[str, float | None]]:
    """‖Q_aᵀ Q_b‖² / min(k_a, k_b) for every pair; None when vector lengths differ."""
    qs = {r["name"]: r["_basis"] for r in records}
    return {
        a: {
            b: ((qa.T @ qb).norm() ** 2 / min(qa.shape[1], qb.shape[1])).item()
            if qa.shape[0] == qb.shape[0]
            else None
            for b, qb in qs.items()
        }
        for a, qa in qs.items()
    }


def format_report(records: list[dict], cos: dict) -> str:
    w = max(len(r["name"]) for r in records)
    lines = [
        (
            f"{'tensor':<{w}}  {'axis':>4}  {'k':>4}  {'rank':>4}  {'gap':>4}  {'σ1 energy':>9}  "
            f"{'σ1..σ3':>28}  {'resid α=1':>10}  {'α':>7}  {'resid α':>9}"
        )
    ]
    for r in records:
        sv = " ".join(f"{x:.2e}" for x in r["singular_values"][:3])
        lines.append(
            f"{r['name']:<{w}}  {r['axis']:>4}  {r['edited_vectors']:>4}  "
            f"{r['effective_rank']:>4}  {r['gap_rank']:>4}  {r['top_energy']:>9.6f}  {sv:>28}  "
            f"{r['projection_residual']:>10.3e}  {r['projection_scale']:>7.3f}  "
            f"{r['scaled_residual']:>9.3e}"
        )
    names = [r["name"] for r in records]
    if len(names) > 6:
        vals = [abs(cos[a][b]) for i, a in enumerate(names) for b in names[i + 1 :]
                if cos[a][b] is not None]  # fmt: skip
        vals.sort()
        mid = vals[len(vals) // 2]
        lines += ["", (f"|cos| between top directions over {len(vals)} pairs: "
                       f"min {vals[0]:.3f}, median {mid:.3f}, max {vals[-1]:.3f}")]  # fmt: skip
    elif len(names) > 1:
        lines += ["", "cosine between top directions (sign is arbitrary):"]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                c = cos[a][b]
                lines.append(f"  {a}  vs  {b}: {'—' if c is None else f'{c:+.6f}'}")
    return "\n".join(lines)
