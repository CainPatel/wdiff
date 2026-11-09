"""Structural diff: which tensors changed, and by how much.

For each tensor pair we form ΔW = W_edited − W_base (sums accumulated in float64) and record:

    changed         = #{ i : W_edited[i] != W_base[i] }   (exact; both are bf16)
    frobenius_ratio = ‖ΔW‖_F / ‖W_base‖_F                 (relative size of the edit)
    cosine          = ⟨W_base, W_edited⟩ / (‖W_base‖ ‖W_edited‖), tensors flattened
    max_abs_delta   = max |ΔW|

For 2-D tensors we also count how many rows and columns contain any change. An edit that
touches a few output directions of a [d_model, d_ff] matrix shows up as a small
`changed_cols` with `changed_rows` = d_model; phase 2 restricts its SVD to those columns.

    touched = min(changed_cols × rows, changed_rows × cols)   (2-D; else = changed)

`touched` counts every weight in the edited columns (or rows, if that footprint is
smaller). It is what an edit *targets*; `changed` is what differs on disk. They differ
when the exact edit of a weight rounds back to the same bf16 value: on the Qwen pair,
884 of the 131,072 targeted weights are unchanged after rounding.

Layer rollup: the integer after `layers.` in a tensor name is its layer index; we sum
changed weights per layer. Tensors without one (embeddings, final norm) go under "other".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_of(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _sums(base: torch.Tensor, edited: torch.Tensor, chunk: int = 1 << 22) -> tuple[float, ...]:
    """‖b‖², ‖e‖², ⟨b,e⟩, ‖Δ‖², max|Δ|, accumulated in float64 over chunks.

    A single float32 dot over the 254M-entry embedding drifts by ~10%, so cosines of
    untouched tensors read 1.09; chunked float64 keeps them at 1.0 and bounds memory.
    """
    bb = ee = be = dd = mx = 0.0
    for x, y in zip(base.reshape(-1).split(chunk), edited.reshape(-1).split(chunk), strict=True):
        x, y = x.double(), y.double()
        d = y - x
        bb, ee, be, dd = bb + x @ x, ee + y @ y, be + x @ y, dd + d @ d
        mx = max(mx, d.abs().max().item())
    return float(bb), float(ee), float(be), float(dd), mx


def tensor_stats(name: str, base: torch.Tensor, edited: torch.Tensor) -> dict:
    diff_mask = base != edited  # exact compare in the stored dtype
    changed = int(diff_mask.sum())
    bb, ee, be, dd, mx = _sums(base, edited)
    nb, ne = bb**0.5, ee**0.5
    rec = {
        "name": name,
        "layer": layer_of(name),
        "shape": list(base.shape),
        "dtype": str(base.dtype).removeprefix("torch."),
        "params": base.numel(),
        "changed": changed,
        "touched": changed,
        "frobenius_ratio": dd**0.5 / nb if nb else (0.0 if not changed else float("inf")),
        "cosine": be / (nb * ne) if nb and ne else None,
        "max_abs_delta": mx,
    }
    if base.ndim == 2:
        rec["changed_rows"] = int(diff_mask.any(dim=1).sum())
        rec["changed_cols"] = int(diff_mask.any(dim=0).sum())
        rows, cols = base.shape
        rec["touched"] = min(rec["changed_cols"] * rows, rec["changed_rows"] * cols)
    return rec


def diff(pairs: Iterable[tuple[str, torch.Tensor, torch.Tensor]]) -> list[dict]:
    return [tensor_stats(n, b, e) for n, b, e in pairs]


def rollup(records: list[dict]) -> dict:
    total = sum(r["params"] for r in records)
    changed = sum(r["changed"] for r in records)
    per_layer: dict[str, dict] = {}
    for r in records:
        key = "other" if r["layer"] is None else str(r["layer"])
        row = per_layer.setdefault(
            key, {"params": 0, "changed": 0, "touched": 0, "tensors_changed": 0}
        )
        row["params"] += r["params"]
        row["changed"] += r["changed"]
        row["touched"] += r["touched"]
        row["tensors_changed"] += r["changed"] > 0
    order = sorted(per_layer, key=lambda k: (k == "other", int(k) if k != "other" else 0))
    return {
        "total_params": total,
        "changed_params": changed,
        "changed_fraction": changed / total if total else 0.0,
        "touched_params": sum(r["touched"] for r in records),
        "tensors": len(records),
        "tensors_changed": sum(r["changed"] > 0 for r in records),
        "per_layer": {k: per_layer[k] for k in order},
    }


def format_report(records: list[dict], summary: dict, top: int) -> str:
    ranked = sorted(records, key=lambda r: (-r["changed"], r["name"]))[:top]
    w = max(len(r["name"]) for r in ranked)
    lines = [
        (
            f"{'tensor':<{w}}  {'shape':>14}  {'changed':>11}  {'cols':>6}  "
            f"{'frob_ratio':>10}  {'cosine':>10}  {'max|Δ|':>9}"
        )
    ]
    for r in ranked:
        cos = "—" if r["cosine"] is None else f"{r['cosine']:.6f}"
        lines.append(
            f"{r['name']:<{w}}  {'x'.join(map(str, r['shape'])):>14}  {r['changed']:>11,}  "
            f"{r.get('changed_cols', '—'):>6}  {r['frobenius_ratio']:>10.3e}  {cos:>10}  "
            f"{r['max_abs_delta']:>9.3e}"
        )
    lines += [
        "",
        (
            f"changed {summary['changed_params']:,} of {summary['total_params']:,} weights "
            f"({summary['changed_fraction']:.4%}) in {summary['tensors_changed']} of "
            f"{summary['tensors']} tensors"
        ),
        f"touched {summary['touched_params']:,} weights (whole edited columns/rows)",
        "",
        f"{'layer':>6}  {'params':>13}  {'changed':>11}  {'touched':>11}  {'tensors':>7}",
    ]
    for k, row in summary["per_layer"].items():
        lines.append(
            f"{k:>6}  {row['params']:>13,}  {row['changed']:>11,}  {row['touched']:>11,}  "
            f"{row['tensors_changed']:>7}"
        )
    return "\n".join(lines)
