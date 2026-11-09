"""Summarizer: turn the three JSON reports into a commit message.

1. build_context picks statistics only from the reports. The structural part is the
   changed tensors among the top 10 and the per-layer rollup for changed layers. The
   directional part is rank, energy, the projection scale and cosines. The behavioral
   part is base, edited and delta per metric. Floats are rounded to 4 significant digits.
2. check_no_weights refuses any numeric list longer than MAX_LIST, so a singular vector
   (1024 floats) cannot slip in. The token check (count_tokens on the real request,
   at most MAX_TOKENS) is the second guard. Weights never enter the prompt.
3. Claude writes the message in the fixed shape (title, Where, What, Effect,
   Unchanged, Confidence).
4. check_numbers: every number in the message must match a number in the context, which
   holds only values copied from the reports. A number n with d decimals matches a
   value v if |v| rounds to n at d decimals, or |100·v| does (a fraction written as a
   percent). Anything else is invented. The message is then saved as COMMIT.rejected.md
   and the command fails, instead of writing COMMIT.md.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

MAX_LIST, MAX_TOKENS, TOP, DETAIL = 16, 3000, 10, 6
DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
FIELDS = ["Where", "What", "Effect", "Unchanged", "Confidence"]

GLOSSARY = {
    "changed": "weights whose stored value differs",
    "touched": "every weight in an edited row or column",
    "frobenius_ratio": "‖ΔW‖ / ‖W_base‖",
    "cosine": "cos(W_base, W_edited) for the whole tensor",
    "effective_rank": "singular values needed for 95% of ΔW's energy",
    "top_energy": "share of ΔW's energy in the top singular direction v",
    "projection_scale": "α in c' = c − α(c·v)v fitted over edited columns; 1 = plain removal of v",
    "scaled_residual": "relative error of that fit",
    "gap_rank": "number of singular values before the largest drop σ_i/σ_(i+1) (gap_ratio): "
    "the count of directions the edit spans",
    "direction_cosines": "cos between the top directions of two tensors",
    "subspace_overlap": "‖Q_aᵀQ_b‖²/min(k) between two tensors' edit subspaces; 1 = same subspace",
    "in_subspace": "fraction of a named direction lying inside the tensor's edit subspace; "
    "random_baseline = what a random direction would give",
    "[min, median, max]": "with many tensors, stats are grouped by tensor kind across layers "
    "(a single number when all are equal)",
    "metrics_base_edited_delta": "per probe set, each metric as [base, edited, edited − base]",
    "named_direction_cosines": "cos between a tensor's top direction and a named direction "
    "(refusal = difference of means of harmful vs harmless prompts, sign arbitrary)",
    "loop": "hit the token cap or some 20-token window occurs ≥ 3 times",
    "answer_rate / accuracy": "math only: has a final boxed answer / it is correct",
    "refusal_rate": "response contains a refusal phrase; runs that never finish thinking have "
    "no response and count as not refusing (no_response_rate)",
}

SYSTEM = """You write a commit message for the change between two checkpoints of one model, \
from measurement reports. Reply with only the commit message, in exactly this shape:

<title line, at most seventy-two characters, imperative mood>

Where: <tensors and layers, changed-weight count and fraction of model>
What: <rank, direction, match to any named direction with cosine>
Effect: <behavioral metrics that moved, with before → after numbers>
Unchanged: <metrics that did not move; what this edit did not touch>
Confidence: <low | medium | high> — <one reason>

Rules:
- Use only numbers that appear in the reports. Copy them; you may round them or write a \
fraction as a percent. Never compute a new number (no sums, differences or ratios).
- Report what the data shows, including measurements that are weak or that cut against \
the obvious story."""


# ---------- context ----------


def _r(x):
    """Round floats to 4 significant digits, recursively."""
    if isinstance(x, float):
        return 0.0 if x == 0 else round(x, 3 - math.floor(math.log10(abs(x))))
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v) for v in x]
    return x


def build_context(out: Path) -> dict:
    s = json.loads((out / "structural.json").read_text())
    d = json.loads((out / "directional.json").read_text())
    b = json.loads((out / "behavioral.json").read_text())
    summ = s["summary"]
    top = sorted(s["tensors"], key=lambda t: (t["changed"], t["frobenius_ratio"]), reverse=True)
    keep = ["name", "layer", "shape", "params", "changed", "touched", "changed_cols",
            "changed_rows", "frobenius_ratio", "cosine", "max_abs_delta"]  # fmt: skip
    per_layer = {k: v for k, v in summ["per_layer"].items() if v["changed"] > 0}
    if len(per_layer) > DETAIL:
        per_layer = {
            "layers": sorted(int(k) for k in per_layer if k.isdigit()),
            "changed_per_layer": _mmm([v["changed"] for v in per_layer.values()]),
        }
    ctx = {
        "base": s["meta"]["base"]["source"],
        "edited": s["meta"]["edited"]["source"],
        "glossary": GLOSSARY,
        "structural": {
            "total_params": summ["total_params"],
            "changed_params": summ["changed_params"],
            "changed_percent_of_model": 100 * summ["changed_fraction"],
            "touched_params": summ["touched_params"],
            "touched_percent_of_model": 100 * summ["touched_params"] / summ["total_params"],
            "tensors": summ["tensors"],
            "tensors_changed": summ["tensors_changed"],
            "per_layer_changed_only": per_layer,
            **_structural_tensors([t for t in top if t["changed"] > 0], keep),
        },
        "directional": _directional(d),
        "behavioral": {
            "max_new_tokens": b["meta"]["generation"]["max_new_tokens"],
            "loop": b["meta"]["loop"],
            "metrics_base_edited_delta": {
                name: {"n": m["base"]["n"]}
                | {k: [m["base"][k], m["edited"][k], m["delta"][k]] for k in m["delta"]}
                for name, m in b["metrics"].items()
            },
        },
    }
    return _r(ctx)


DKEEP = ["name", "layer", "axis", "edited_vectors", "singular_values", "top_energy",
         "effective_rank", "gap_rank", "gap_ratio", "projection_scale", "scaled_residual"]  # fmt: skip
GROUPED = ["edited_vectors", "top_energy", "effective_rank", "gap_rank", "gap_ratio",
           "projection_scale", "scaled_residual"]  # fmt: skip


def _mmm(vals: list):
    """[min, median, max], or the single value when all are equal."""
    v = sorted(x for x in vals if x is not None)
    if not v:
        return []
    return v[0] if v[0] == v[-1] else [v[0], v[len(v) // 2], v[-1]]


def _structural_tensors(changed: list[dict], keep: list[str]) -> dict:
    """Top changed tensors one per row when few; grouped by kind across layers when many."""
    if len(changed) <= DETAIL:
        return {"top_changed_tensors": [{k: t[k] for k in keep if k in t} for t in changed[:TOP]]}
    groups: dict[str, list[dict]] = {}
    for t in changed:
        groups.setdefault(_kind(t["name"]), []).append(t)
    return {"changed_tensor_groups": [
        {"tensors": k, "count": len(ts), "shape": ts[0]["shape"],
         "changed_total": sum(t["changed"] for t in ts), "params_total": sum(t["params"] for t in ts)}
        | {f: _mmm([t[f] for t in ts]) for f in ("frobenius_ratio", "cosine", "max_abs_delta")}
        for k, ts in groups.items()
    ]}  # fmt: skip


def _pairs(m: dict) -> list[tuple[str, str, float]]:
    names = list(m)
    return [(a, b, m[a][b]) for i, a in enumerate(names) for b in names[i + 1 :]
            if m[a][b] is not None]  # fmt: skip


def _kind(name: str) -> str:
    return re.sub(r"layers\.\d+\.", "layers.*.", name)


def _directional(d: dict) -> dict:
    """Per tensor when there are few; grouped by tensor kind with [min, median, max] when many."""
    tensors, named = d["tensors"], d["named_direction_cosines"]
    overlap = d.get("subspace_overlap", {})
    if len(tensors) <= DETAIL:
        return {
            "tensors": [{k: t[k] for k in DKEEP if k in t} for t in tensors],
            "direction_cosines": [{"a": a, "b": b, "cosine": c}
                                  for a, b, c in _pairs(d["direction_cosines"])],
            "subspace_overlap": [{"a": a, "b": b, "overlap": c} for a, b, c in _pairs(overlap)],
            "named_direction_cosines": named,
        }  # fmt: skip
    groups: dict[str, list[dict]] = {}
    for t in tensors:
        groups.setdefault(_kind(t["name"]), []).append(t)
    cos = [abs(c) for _, _, c in _pairs(d["direction_cosines"])]
    out = {
        "tensor_groups": [
            {"tensors": k, "count": len(ts), "axis": ts[0]["axis"]}
            | {f: _mmm([t.get(f) for t in ts]) for f in GROUPED}
            for k, ts in groups.items()
        ],
        "direction_cosines_abs": {"pairs": len(cos), "min_median_max": _mmm(cos)},
        "subspace_overlap": {"pairs": len(_pairs(overlap)),
                             "min_median_max": _mmm([c for _, _, c in _pairs(overlap)])},
        "named_direction_cosines": [],
    }  # fmt: skip
    keys = sorted({(_kind(r["tensor"]), r["direction"]) for r in named})
    for kind, direction in keys:
        rows = [r for r in named if _kind(r["tensor"]) == kind and r["direction"] == direction]
        best = max(rows, key=lambda r: abs(r["cosine"]))
        out["named_direction_cosines"].append(
            {"tensors": kind, "direction": direction, "layers": len(rows),
             "abs_cosine": _mmm([abs(r["cosine"]) for r in rows]), "max_at_layer": best["layer"]}
            | ({"in_subspace": _mmm([r["in_subspace"] for r in rows]),
                "random_baseline": rows[0]["random_baseline"]} if "in_subspace" in rows[0] else {})
        )  # fmt: skip
    return out


def check_no_weights(ctx) -> None:
    """Raise if any list of numbers is long enough to be a weight vector."""
    if isinstance(ctx, dict):
        for v in ctx.values():
            check_no_weights(v)
    elif isinstance(ctx, list):
        if sum(isinstance(v, int | float) for v in ctx) > MAX_LIST:
            raise ValueError(f"numeric list of length {len(ctx)} in summarizer context")
        for v in ctx:
            check_no_weights(v)


# ---------- checks on the reply ----------

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers(text: str) -> list[str]:
    return [m.group().rstrip(",") for m in NUM.finditer(text)]


def check_numbers(message: str, ctx: dict) -> list[str]:
    """Numbers in the message that match no number in the context."""
    allowed = {float(n.replace(",", "")) for n in numbers(json.dumps(ctx, ensure_ascii=False))}
    bad = []
    for n in numbers(message):
        val = float(n.replace(",", ""))
        mant, _, exp = n.lower().partition("e")
        dec = (len(mant.split(".")[1]) if "." in mant else 0) - int(exp or 0)  # 3.6e-04 -> 5
        if not any(
            abs(round(v, dec) - val) < 1e-9 or abs(round(100 * v, dec) - val) < 1e-9
            for v in allowed
        ):
            bad.append(n)
    return bad


def check_shape(message: str) -> list[str]:
    lines = message.strip().splitlines()
    problems = []
    if not lines or len(lines[0]) > 72:
        problems.append("title missing or over 72 characters")
    starts = [ln.split(":", 1)[0] for ln in lines[1:] if ":" in ln]
    if [f for f in starts if f in FIELDS] != FIELDS:
        problems.append(f"fields are {starts}, want {FIELDS} in order")
    return problems


# ---------- the call ----------


def summarize(ctx: dict, model: str) -> tuple[str, int]:
    """(message, input tokens). Raises if the prompt is over MAX_TOKENS."""
    import anthropic

    client = anthropic.Anthropic()
    user = "Reports:\n" + json.dumps(ctx, ensure_ascii=False)
    msgs = [{"role": "user", "content": user}]
    n = client.messages.count_tokens(model=model, system=SYSTEM, messages=msgs).input_tokens
    if n > MAX_TOKENS:
        raise ValueError(f"summarizer input is {n} tokens, limit {MAX_TOKENS}")
    extra = (
        {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        if model in FALLBACK_MODELS
        else {}
    )
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, system=SYSTEM, messages=msgs, **extra
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text.strip("`").strip(), n
