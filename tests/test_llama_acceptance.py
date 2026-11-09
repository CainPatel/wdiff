"""Phase 4b acceptance on Llama-3.2-1B-Instruct vs mylesgoose/...-abliterated, from saved reports.

Spec: COMMIT.md names refusal removal as the effect, and identifies the edit as the refusal
direction (the positive control: high |cos| with the base model's refusal direction; "high"
taken as >= 0.5). Recorded failure (decided 2026-11-08): the edit is a rank-15 subspace
shared by all 33 tensors, and the refusal direction (held-out AUC 1.000) is barely inside it
(|cos| <= 0.19, in_subspace <= 0.059 vs random 0.0073). The refusal drop is real but partial
(0.90 -> 0.70). The control is kept as specified and marked strict xfail, not loosened.
"""

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "reports" / "llama3.2-1b-abliterated"


@pytest.fixture(scope="module")
def reports():
    if not (OUT / "COMMIT.md").exists():
        pytest.skip("run `wdiff all` on the Llama pair first")
    return {
        "commit": (OUT / "COMMIT.md").read_text(),
        "directional": json.loads((OUT / "directional.json").read_text()),
        "behavioral": json.loads((OUT / "behavioral.json").read_text()),
    }


def _field(commit: str, name: str) -> str:
    return commit.split(f"{name}:", 1)[1].split("\nUnchanged:" if name == "Effect" else "\n\n")[0]


def test_effect_names_refusal_drop(reports):
    delta = reports["behavioral"]["metrics"]["refusal_20"]["delta"]["refusal_rate"]
    print(f"refusal_20 refusal_rate delta {delta:+.2f}")
    assert delta < 0
    assert "refusal" in _field(reports["commit"], "Effect").lower()


@pytest.mark.xfail(strict=True, reason="recorded failure: max |cos| with refusal is 0.19, spec wants high")
def test_edit_is_refusal_direction(reports):
    cos = [abs(r["cosine"]) for r in reports["directional"]["named_direction_cosines"]]
    print(f"max |cos(top direction, refusal)| = {max(cos):.3f}")
    assert max(cos) >= 0.5
