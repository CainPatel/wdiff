# wdiff

**`git diff` for model weights.** Give it two checkpoints of the same model, for example a base model and a fine-tune, patch or "uncensored" version you downloaded. It tells you what was actually changed, in a commit message a person can read:

- **Where:** which tensors and layers changed, and how many weights.
- **What:** the shape of the change. Is it one direction removed (rank 1), a handful of directions, or a broad retrain? Does it match a known direction, such as the one models use to refuse?
- **Effect:** what the model now does differently on a fixed set of test prompts (looping, answering, accuracy, refusing).
- **Confidence:** how much of that the evidence supports.

A model card says what an edit was *meant* to do. wdiff checks what the weights say.

## Why it's useful

Two real edits were tested. In both, the diff told us something the model's name or description did not.

**1. It recovered a published edit exactly, and caught a detail the authors didn't mention.** [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) claims to fix reasoning loops by removing one direction from two layers. wdiff found exactly that: 2 of 488 tensors, 64 columns each, 0.015% of the model, rank 1. It also found that the direction isn't removed but flipped and amplified six times (`c′ = c − 6(c·v)v`). The behavioral run confirmed the fix mostly works: runaway generations fell from 60% to 0% on one prompt set and math accuracy rose from 12% to 32%. But math looping barely moved (92% → 88%).

**2. It showed an "abliterated" model is not what its name claims.** Abliteration usually means cutting the single "refusal direction" out of a model so it stops refusing. In [`mylesgoose/Llama-3.2-1B-Instruct-abliterated`](https://huggingface.co/mylesgoose/Llama-3.2-1B-Instruct-abliterated), wdiff found something else:
- the edit removes a 15-dimensional subspace shared by all 33 edited tensors;
- the refusal direction measured on the base model is barely inside it (at most 5.9% of its energy, against 0.7% for a random direction);
- the model still refuses 14 of the 18 harmful prompts the original refused.

The label is a claim; the diff is evidence.

**What this means in practice:**
- **Audit before you deploy.** Before using a modified model from Hugging Face, you can see how much was changed, where, and whether the change matches its description.
- **Small edits are easy to spot.** Targeted weight edits are tiny and low-rank, so they stand out clearly against an unchanged base. Two layers out of 24 is visible at a glance.
- **Weights alone don't give meaning.** The weights show the *shape* of a change. What it *does* only comes from running prompts and comparing with directions you know. wdiff does both and says when they disagree.

## Example output

The commit message wdiff wrote for the Qwen edit (excerpt; full text in [`reports/qwen3.5-0.8b-wsp/COMMIT.md`](reports/qwen3.5-0.8b-wsp/COMMIT.md)):

```text
Subtract a rank-1 direction (α=6) from down_proj cols, layers 15 and 19

Where: ...layers.15.mlp.down_proj.weight and layers.19.mlp.down_proj.weight. 64 columns
were edited in each. 2 of 488 tensors changed. 130,188 weights changed (131,072 touched),
which is 0.0149% of the model.

What: The edit is rank-1 in each tensor ... This is a 6× overshoot, not plain removal of v.
... The direction does not match the refusal direction.

Confidence: medium. The weight edit is cleanly rank-1 with a near-exact α=6 fit. However,
the behavioural probes are small (n=20/50) ...
```

The Llama message is in [`reports/llama3.2-1b-abliterated/COMMIT.md`](reports/llama3.2-1b-abliterated/COMMIT.md). [WRITEUP.md](WRITEUP.md) compares both and covers what the tool got right and wrong.

## How it works

| Level | Question | How |
|---|---|---|
| Structural | Which weights changed? | Reads both checkpoints one tensor at a time (memory-mapped) and counts changed weights, rows and columns. |
| Directional | What shape is the change? | SVD of each tensor's change ΔW. Rank 1 means one direction was edited. It also compares directions across layers and with named directions such as refusal. |
| Behavioral | What does the model do differently? | Both models answer the same 90 prompts (math, harmful, harmless) with the same seed, and wdiff compares their rates. |
| Summarizer | What should the commit message say? | One Claude call turns the three reports into a message. It only sees statistics, never weights. Any number in the message that isn't in the reports gets the message rejected. |

## Quick start

```bash
uv sync --extra summarize
export ANTHROPIC_API_KEY=...   # only the final summarize step needs it
uv run wdiff all BASE_MODEL EDITED_MODEL --out reports/my-diff --directions directions/my-base
```

`BASE_MODEL` and `EDITED_MODEL` are Hugging Face ids or local folders with safetensors files. Both models must have the same architecture. Each stage can also run alone: `wdiff structural`, `directional`, `direction compute`, `behavioral`, `summarize`. The structural and directional stages take minutes on a laptop. The behavioral stage generates text with both models: about 1 hour for Llama-3.2-1B and 2 hours for Qwen3.5-0.8B on an M5 Max.

## Limits

- Both checkpoints must have the same architecture (same tensor names and shapes).
- The probe sets are small (20–50 prompts), so only large behavioral changes are reliable.
- It describes *what* changed, not *how* or *why* it was changed.
- `samples/refusal_20.jsonl` redacts completions where a model went along with a harmful prompt. Scores and a sha256 of each original are kept.

## Detailed results

The commands to reproduce the Qwen report one stage at a time:

```bash
uv run wdiff structural Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --out reports/qwen3.5-0.8b-wsp
uv run wdiff direction compute Qwen/Qwen3.5-0.8B --method refusal --layers 15,19 --out directions/qwen3.5-0.8b  # ~8 min CPU
uv run wdiff directional --out reports/qwen3.5-0.8b-wsp --directions directions/qwen3.5-0.8b
uv run wdiff behavioral Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --out reports/qwen3.5-0.8b-wsp --batch-size 16 --max-new-tokens 8192 --force  # ~2 h on an M5 Max
uv run pytest
```

### Qwen: the ground truth

- The edit lives in `model.language_model.layers.{15,19}.mlp.down_proj.weight` (shape 1024×3584), confined to 64 columns in each, as the paper says.
- Two counts are reported. `touched` = every weight in the edited columns = 131,072, matching the paper. `changed` = weights that differ on disk = 130,188 (65,260 in layer 15, 64,928 in layer 19). The 884 gap is bf16 rounding: for those weights the exact edit rounds back to the stored value.
- Each change is rank 1: one direction carries 99.999% of ΔW in both layers.
- The edit is not plain removal. Each edited column is `c' = c − 6(c·v)v` (fitted scale 6.000 in both layers; residual 0.002, about bf16 rounding). The component along v is flipped and grown to 5×, rather than zeroed. Likely cause: a projection onto an unnormalized direction with ‖r‖² = 6. Not yet confirmed against the paper's logs.
- The two layers do not share one direction: |cos(v_15, v_19)| = 0.849, below the spec's 0.9 bar. Similar, not identical.
- The edit is not the refusal direction: |cos(v_L, r_L)| = 0.007 at layer 15 and 0.019 at layer 19. The refusal direction itself is real: on 16+16 held-out validation prompts it separates harmful from harmless with AUC 1.000 at both layers (a random direction: 0.65).

### Qwen: behavioral probes (a recorded failure)

Probes: 50 GSM8K train problems, 20 harmful and 20 harmless held-out prompts (`probes/`). Thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, one sample per prompt, seed 0. `loop` = hit the token cap OR some 20-token window occurs 3 or more times.

The spec's gate is that math_50 loop_rate drops by 20 or more points. It fails at every cap tried:

| max_new_tokens | loop_rate | hit_cap_rate | answer_rate | accuracy |
|---|---|---|---|---|
| 1024 | everything hits the cap | | | |
| 4096 | 0.92 → 0.88 | 0.90 → 0.82 | 0.10 → 0.18 | 0.10 → 0.18 |
| 8192 | 0.92 → 0.88 | 0.88 → 0.64 | 0.12 → 0.38 | 0.12 → 0.32 |

(base → edited.) The edited model finishes and answers correctly much more often, but loop_rate barely moves because the repeat check also fires on working solutions: 12 of the edited model's 18 finished math answers are flagged, 10 of them correct. Some of those are real loops that recover (one re-quotes a problem sentence 15 times); many are normal re-checking (`Total = 4 + 12 + 2 = 18` written 3 times). The metric does not separate looping from verifying in thinking mode. The test is kept as specified and marked strict xfail; it was not redefined after seeing results.

At 8192: benign_20 loop_rate 0.55 → 0.05; refusal_20 loop_rate 0.60 → 0.00, refusal_rate 0.25 → 0.75. The refusal gate passes. The refusal increase is mostly an artifact: 60% of the base model's refusal_20 runs never close `</think>`, so they have no response to count as a refusal. The 4096 report is kept in `reports/qwen3.5-0.8b-wsp/behavioral-4096/`.

### Llama: an abliterated model (the positive control fails)

Pair: `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` vs `mylesgoose/Llama-3.2-1B-Instruct-abliterated` @ `a2ec81748492b3611caca192812e72e3ff342f35`. It is the only bf16 safetensors abliteration of this model with matching tensors that we found. There is no answer key: the model card does not say how it was made.

```bash
uv run wdiff all meta-llama/Llama-3.2-1B-Instruct mylesgoose/Llama-3.2-1B-Instruct-abliterated --out reports/llama3.2-1b-abliterated --directions directions/llama3.2-1b --batch-size 16 --max-new-tokens 8192
```

- **Structural (as the spec expected).** 33 of 146 tensors changed: `embed_tokens` plus every `o_proj` and `down_proj` in all 16 layers. That is 581,755,001 weights (47.07%), each changed only slightly (Frobenius ratio 0.067–0.082).
- **Directional (not as expected).** The spec expected rank 1. Every ΔW has exactly 15 singular values above the bf16 noise floor, then a cliff (gap ratio 7–13×). The top direction carries only 16–44% of each tensor's change. All 33 tensors share one 15-dimensional subspace (subspace overlap 0.9995–0.9999), embedding included. The fitted scale per direction falls from about 1.1 to 0.18. That is consistent with projecting out several non-orthogonal directions in sequence, but this is not confirmed. mylesgoose's `abliterated2` and `abliterated3` show the same pattern, with rank 16 and about 22.
- **Positive control fails.** The base model's refusal direction is valid: held-out AUC is 1.000 at layers 4, 8, 10, 12 and 15. But it is barely inside the edit: the best |cos| with a top direction is 0.187 (layer 11), and at most 5.9% of its energy lies in the edit subspace (a random direction gives 0.73%). `tests/test_llama_acceptance.py` keeps the spec's check (|cos| ≥ 0.5) as a strict xfail.
- **Behavioral.** refusal_20 refusal rate fell from 0.90 to 0.70. The base model refuses 18 of the 20 harmful prompts, and the edited model still refuses 14 of those 18. The edited model's refusals are longer and often add partial information (short_rate 0.45 → 0.05; mean tokens 98 → 232). Benign refusals stayed at 0.00. Math accuracy is low for both (0.06 → 0.00 on n=50). This model has no thinking phase, so the whole output is scored as the response.
- **COMMIT.md.** It names the shared rank-15 subspace and the refusal drop, says the alignment with the refusal direction is weak, and gives confidence low. So it describes this edit correctly but does not "identify it as the refusal direction", because the data does not support that. The summarizer input was 2,994 tokens, close to the 3,000 limit.

## Credits and license

Prompt subsets in `data/refusal_direction/` are sampled from [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) (Apache-2.0); see `SOURCE.json`. Probes are sampled from GSM8K (MIT) and the same repo's test splits; see `probes/SOURCE.json`.

Apache-2.0. See [LICENSE](LICENSE).
