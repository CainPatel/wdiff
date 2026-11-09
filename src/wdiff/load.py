"""Open two checkpoints and walk them tensor by tensor.

A checkpoint is a directory of .safetensors files. Big ones are sharded and ship a
`model.safetensors.index.json` whose `weight_map` says which shard holds each tensor.
Every file is opened memory-mapped (`safe_open`), so a tensor is only read from disk
when we ask for it, and we ask for one pair at a time: peak memory is about two copies
of the largest tensor, never the whole model.

Both checkpoints must have exactly the same tensor names and shapes; otherwise the diff
is meaningless and we abort before reading any weights.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_NAME = "model.safetensors.index.json"


class CheckpointMismatch(ValueError):
    """The two checkpoints do not share tensor names and shapes."""


@dataclass(frozen=True)
class Checkpoint:
    source: str  # what the user passed: a path or a HF repo id
    path: Path
    revision: str | None  # HF commit hash, None for a plain local directory


def resolve(source: str) -> Checkpoint:
    """A local directory as-is, otherwise a HF repo id downloaded into the HF cache."""
    p = Path(source).expanduser()
    if p.is_dir():
        return Checkpoint(source, p.resolve(), None)
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(source, allow_patterns=["*.safetensors", "*.json"]))
    # HF cache layout: .../snapshots/<commit hash>/
    return Checkpoint(source, path, path.name)


def tensor_files(path: Path) -> dict[str, Path]:
    """Map every tensor name to the .safetensors file that holds it."""
    index = path / INDEX_NAME
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: path / fname for name, fname in weight_map.items()}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    out: dict[str, Path] = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for name in h.keys():  # noqa: SIM118 (safe_open is not a dict)
                out[name] = f
    return out


def _shapes(files: dict[str, Path], stack: ExitStack) -> tuple[dict[str, list[int]], dict]:
    handles: dict[Path, object] = {}
    shapes = {}
    for name, f in files.items():
        if f not in handles:
            handles[f] = stack.enter_context(safe_open(f, framework="pt"))
        shapes[name] = list(handles[f].get_slice(name).get_shape())
    return shapes, handles


def _check_match(a: dict[str, list[int]], b: dict[str, list[int]]) -> None:
    only_a, only_b = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    bad_shape = sorted(n for n in a.keys() & b.keys() if a[n] != b[n])
    if only_a or only_b or bad_shape:
        lines = ["checkpoints do not share the same tensors:"]
        if only_a:
            lines.append(f"  only in BASE ({len(only_a)}): {only_a[:10]}")
        if only_b:
            lines.append(f"  only in EDITED ({len(only_b)}): {only_b[:10]}")
        for n in bad_shape[:10]:
            lines.append(f"  shape differs: {n} BASE {a[n]} vs EDITED {b[n]}")
        raise CheckpointMismatch("\n".join(lines))


def iter_pairs(base: Path, edited: Path) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (name, base_tensor, edited_tensor) in name order, one pair at a time."""
    fa, fb = tensor_files(base), tensor_files(edited)
    with ExitStack() as stack:
        sa, ha = _shapes(fa, stack)
        sb, hb = _shapes(fb, stack)
        _check_match(sa, sb)
        for name in sorted(fa):
            yield name, ha[fa[name]].get_tensor(name), hb[fb[name]].get_tensor(name)
