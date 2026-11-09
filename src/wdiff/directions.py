"""Named directions: one direction computed for one model, to compare edits against.

Refusal direction by difference of means (Arditi et al., 2024, arXiv:2406.11717):
run n harmful and n harmless instructions through the base model, each wrapped in the
model's chat template with the assistant turn opened, and read the residual stream at
the last prompt token. For layer L,

    r_L = normalize( mean_harmful h_L − mean_harmless h_L ).

Which residual: `hidden_states[L + 1]`, the stream right after decoder layer L. That is
where layer L's mlp.down_proj writes, so r_L lives in the same space as the edit's
top direction v_L, and cos(v_L, r_L) is meaningful. Sign of v is arbitrary: compare |cos|.

Prompts: a fixed-seed sample of the refusal_direction repo's train splits, vendored in
src/wdiff/data/refusal_direction/ (Apache-2.0), shipped with the package.

This module imports transformers lazily, so structural and directional stay light.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data" / "refusal_direction"


def load_prompts(kind: str) -> list[str]:
    return [r["instruction"] for r in json.loads((DATA / f"{kind}_train_64.json").read_text())]


@torch.no_grad()
def last_token_states(model, tok, prompts: list[str], layers: list[int]) -> torch.Tensor:
    """[len(prompts), len(layers), d_model] residual stream at the last prompt token."""
    out = []
    for p in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3.5's default; Arditi et al. use non-thinking prompts
        )
        ids = tok(text, return_tensors="pt")
        hs = model(**ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([hs[L + 1][0, -1].float() for L in layers]))
    return torch.stack(out)


def refusal_direction(model, tok, layers: list[int]) -> torch.Tensor:
    """[len(layers), d_model] unit refusal direction per layer."""
    harmful = last_token_states(model, tok, load_prompts("harmful"), layers)
    harmless = last_token_states(model, tok, load_prompts("harmless"), layers)
    r = harmful.mean(0) - harmless.mean(0)
    return r / r.norm(dim=1, keepdim=True)


def compute(
    base_source: str, base_path: Path, revision: str | None, layers: list[int], out: Path
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32).eval()
    r = refusal_direction(model, tok, layers)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "refusal.npy", r.numpy())
    provenance = json.loads((DATA / "SOURCE.json").read_text())
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "direction": "refusal",
                "method": "difference of means, last prompt token, hidden_states[L+1] (Arditi et al. 2024)",
                "chat_template": "add_generation_prompt=True, enable_thinking=False",
                "model": base_source,
                "revision": revision,
                "layers": layers,
                "shape": list(r.shape),
                "prompts": provenance,
                "date": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    return r


def load_named(directory: Path) -> dict[str, dict[int, torch.Tensor]]:
    """{direction name: {layer: unit vector}} from every <name>.npy + provenance.json."""
    prov = json.loads((directory / "provenance.json").read_text())
    named = {}
    for f in sorted(directory.glob("*.npy")):
        arr = torch.from_numpy(np.load(f))
        named[f.stem] = dict(zip(prov["layers"], arr, strict=True))
    return named


def named_cosines(records: list[dict], named: dict[str, dict[int, torch.Tensor]]) -> list[dict]:
    """Signed cosine of each tensor's top direction with each named direction at its layer,
    and, when the record carries its edit basis Q, the energy ‖Qᵀd‖² of d inside the edit."""
    rows = []
    for r in records:
        v = torch.tensor(r["top_direction"])
        for name, per_layer in named.items():
            d = per_layer.get(r["layer"])
            if d is not None and d.shape == v.shape:
                row = {
                    "tensor": r["name"],
                    "layer": r["layer"],
                    "direction": name,
                    "cosine": (v @ d.float()).item(),
                }
                if "_basis" in r:
                    q, du = r["_basis"], d.float() / d.float().norm()
                    row["in_subspace"] = ((q.T @ du).norm() ** 2).item()
                    row["random_baseline"] = q.shape[1] / q.shape[0]
                rows.append(row)
    return rows
