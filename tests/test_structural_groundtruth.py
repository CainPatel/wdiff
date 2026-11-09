"""Phase 1 acceptance: wdiff recovers the published Qwen3.5-0.8B weight-space edit.

Ground truth (Sparks of Weight-Space Programming, Experiment 8): one direction subtracted
from 64 output columns of the MLPs in layers 15 and 19 = 131,072 weights.
Skips unless both checkpoints are already in the local HF cache.
"""

import re
from pathlib import Path

import pytest

from wdiff.load import iter_pairs
from wdiff.structural import diff, rollup

BASE = "Qwen/Qwen3.5-0.8B"
EDITED = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"


def cached(repo):
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo, local_files_only=True, allow_patterns=["*.safetensors", "*.json"]
        )
    except Exception:  # noqa: BLE001 (any lookup failure = not cached)
        pytest.skip(f"{repo} not in HF cache; run `wdiff structural {BASE} {EDITED}` first")


@pytest.fixture(scope="module")
def records():
    return diff(iter_pairs(Path(cached(BASE)), Path(cached(EDITED))))


def test_groundtruth(records):
    s = rollup(records)
    # 131,072 = weights the edit targets (64 cols x 1024 x 2). Fewer differ on disk because
    # some exact edits round back to the same bf16 value; report that, don't hide it.
    print(f"touched {s['touched_params']:,}, changed on disk {s['changed_params']:,}")
    assert s["touched_params"] == 131_072
    assert 0 < s["changed_params"] <= s["touched_params"]

    changed = [r for r in records if r["changed"] > 0]
    assert len(changed) == 2, [r["name"] for r in changed]
    assert sorted(r["layer"] for r in changed) == [15, 19]
    for r in changed:
        assert re.search(r"layers\.(15|19)\.mlp\.", r["name"]), r["name"]
        axis = "cols" if r["changed_cols"] == 64 else "rows"
        print(f"{r['name']}: edit confined to 64 {axis} of shape {r['shape']}")
        assert r[f"changed_{axis}"] == 64

    for r in records:
        if r["changed"] == 0:
            assert r["cosine"] == pytest.approx(1.0, abs=1e-5), r["name"]
