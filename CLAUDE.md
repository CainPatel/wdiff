# wdiff — working agreement

Spec: https://claude.ai/artifact/ReHz2gRafiDzwFAnFcippe (wdiff — Semantic Weight Diff Project Spec).

**Order of work.** Phases are gated. Finish load.py, structural.py, and tests/test_structural_groundtruth.py and get the test green on the real Qwen pair before writing a line of directional.py. Same at every phase boundary. If a later phase is tempting, note the idea in NOTES.md and keep going.

**Do not build anything listed under Non-goals.** If a task seems to need storage, branching, merging, a UI, or weight editing, stop and say so instead of building it.

**Weights never enter an LLM prompt.** The summarizer gets statistics, ranks, cosines, and metrics. Never raw tensors, never singular vectors as float lists. Enforce this with a token count check on the summarizer input.

**Memory discipline.** Open checkpoints memory-mapped. Process one tensor at a time. Cast to float32 only for the tensor in hand. Never call .to(device) on a whole state dict in phases 1 or 2.

**Verification before claiming done.** Every phase ends by running its acceptance test on real checkpoints, not synthetic ones, and pasting the actual output. "It should work" is not done. If a number disagrees with the paper, report the disagreement; do not adjust the test to pass.

**When to stop and ask.** Tensor names or shapes do not match what the spec assumed. The published edit lives in a different tensor than expected. A download is gated or missing. A phase 3 run is projected to exceed 90 minutes. The summarizer invents a number not in the reports. In each case, print what you found and wait.

**Explain the math.** Each module gets a docstring that explains the computation in plain words, with the formula. I want to be able to read directional.py and follow why rank-1 ΔW means one removed direction.

**Style.** Small functions, type hints, no notebooks, no global state. Tests use the real Qwen pair when available and a tiny synthetic checkpoint fixture otherwise. Commit after each green test with a message in the same shape as COMMIT.md: title line, then Where / What / Effect.

## Commands

- `uv run wdiff structural BASE EDITED --out reports/X [--top 20]`
- `uv run pytest` (ground-truth test skips unless both Qwen checkpoints are in the HF cache)
- `uv run ruff check .`
