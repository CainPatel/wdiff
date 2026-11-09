# wdiff

Semantic diff of two checkpoints of the same architecture: which tensors changed (structural), what direction the change points in (directional), and what the model now does differently (behavioral).

Status: phases 1 (structural), 2 (rank, direction, and comparison with a refusal direction) 3 (behavioral; the math acceptance gate fails, see below), 4a (summarizer: `reports/*/COMMIT.md`) and 4b (Llama pair; the positive control fails, see below).

```bash
uv sync
uv run wdiff structural Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --out reports/qwen3.5-0.8b-wsp
uv run wdiff direction compute Qwen/Qwen3.5-0.8B --method refusal --layers 15,19 --out directions/qwen3.5-0.8b  # ~8 min CPU
uv run wdiff directional --out reports/qwen3.5-0.8b-wsp --directions directions/qwen3.5-0.8b
uv run wdiff behavioral Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --out reports/qwen3.5-0.8b-wsp --batch-size 16 --max-new-tokens 8192 --force  # ~2 h on an M5 Max
uv run pytest
```

## Notes on the Qwen ground truth

- The edit lives in `model.language_model.layers.{15,19}.mlp.down_proj.weight` (shape 1024×3584), confined to 64 columns in each, as the paper says.
- Two counts are reported. `touched` = every weight in the edited columns = 131,072, matching the paper. `changed` = weights that differ on disk = 130,188 (65,260 in layer 15, 64,928 in layer 19). The 884 gap is bf16 rounding: for those weights the exact edit rounds back to the stored value.
- Each change is rank 1: one direction carries 99.999% of ΔW in both layers.
- The edit is not plain removal. Each edited column is `c' = c − 6(c·v)v` (fitted scale 6.000 in both layers; residual 0.002, about bf16 rounding). The component along v is flipped and grown to 5×, rather than zeroed. Likely cause: a projection onto an unnormalized direction with ‖r‖² = 6. Not yet confirmed against the paper's logs.
- The two layers do not share one direction: |cos(v_15, v_19)| = 0.849, below the spec's 0.9 bar. Similar, not identical.
- The edit is not the refusal direction: |cos(v_L, r_L)| = 0.007 at layer 15 and 0.019 at layer 19. The refusal direction itself is real: on 16+16 held-out validation prompts it separates harmful from harmless with AUC 1.000 at both layers (a random direction: 0.65).

## Phase 3: behavioral (recorded failure)

Probes: 50 GSM8K train problems, 20 harmful and 20 harmless held-out prompts (`probes/`). Thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, one sample per prompt, seed 0. `loop` = hit the token cap OR some 20-token window occurs 3 or more times.

The spec's gate is that math_50 loop_rate drops by 20 or more points. It fails at every cap tried:

| max_new_tokens | loop_rate | hit_cap_rate | answer_rate | accuracy |
|---|---|---|---|---|
| 1024 | everything hits the cap | | | |
| 4096 | 0.92 → 0.88 | 0.90 → 0.82 | 0.10 → 0.18 | 0.10 → 0.18 |
| 8192 | 0.92 → 0.88 | 0.88 → 0.64 | 0.12 → 0.38 | 0.12 → 0.32 |

(base → edited.) The edited model finishes and answers correctly much more often, but loop_rate barely moves because the repeat check also fires on working solutions: 12 of the edited model's 18 finished math answers are flagged, 10 of them correct. Some of those are real loops that recover (one re-quotes a problem sentence 15 times); many are normal re-checking (`Total = 4 + 12 + 2 = 18` written 3 times). The metric does not separate looping from verifying in thinking mode. The test is kept as specified and marked strict xfail; it was not redefined after seeing results.

At 8192: benign_20 loop_rate 0.55 → 0.05; refusal_20 loop_rate 0.60 → 0.00, refusal_rate 0.25 → 0.75. The refusal gate passes. The refusal increase is mostly an artifact: 60% of the base model's refusal_20 runs never close `</think>`, so they have no response to count as a refusal. The 4096 report is kept in `reports/qwen3.5-0.8b-wsp/behavioral-4096/`.

## Phase 4b: Llama-3.2-1B-Instruct vs an abliterated version (recorded failure of the positive control)

Pair: `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` vs `mylesgoose/Llama-3.2-1B-Instruct-abliterated` @ `a2ec81748492b3611caca192812e72e3ff342f35`. It is the only bf16 safetensors abliteration of this model with matching tensors that we found. There is no answer key: the model card does not say how it was made.

```bash
uv run wdiff all meta-llama/Llama-3.2-1B-Instruct mylesgoose/Llama-3.2-1B-Instruct-abliterated --out reports/llama3.2-1b-abliterated --directions directions/llama3.2-1b --batch-size 16 --max-new-tokens 8192
```

- **Structural (as the spec expected).** 33 of 146 tensors changed: `embed_tokens` plus every `o_proj` and `down_proj` in all 16 layers. That is 581,755,001 weights (47.07%), each changed only slightly (Frobenius ratio 0.067–0.082).
- **Directional (not as expected).** The spec expected rank 1. Every ΔW has exactly 15 singular values above the bf16 noise floor, then a cliff (gap ratio 7–13×). The top direction carries only 16–44% of each tensor's change. All 33 tensors share one 15-dimensional subspace (subspace overlap 0.9995–0.9999), embedding included. The fitted scale per direction falls from about 1.1 to 0.18. That is consistent with projecting out several non-orthogonal directions in sequence, but this is not confirmed. mylesgoose's `abliterated2` and `abliterated3` show the same pattern, with rank 16 and about 22.
- **Positive control fails.** The base model's refusal direction is valid: held-out AUC is 1.000 at layers 4, 8, 10, 12 and 15. But it is barely inside the edit: the best |cos| with a top direction is 0.187 (layer 11), and at most 5.9% of its energy lies in the edit subspace (a random direction gives 0.73%). `tests/test_llama_acceptance.py` keeps the spec's check (|cos| ≥ 0.5) as a strict xfail.
- **Behavioral.** refusal_20 refusal rate fell from 0.90 to 0.70. The base model refuses 18 of the 20 harmful prompts, and the edited model still refuses 14 of those 18. The edited model's refusals are longer and often add partial information (short_rate 0.45 → 0.05; mean tokens 98 → 232). Benign refusals stayed at 0.00. Math accuracy is low for both (0.06 → 0.00 on n=50). This model has no thinking phase, so the whole output is scored as the response.
- **COMMIT.md.** It names the shared rank-15 subspace and the refusal drop, says the alignment with the refusal direction is weak, and gives confidence low. So it describes this edit correctly but does not "identify it as the refusal direction", because the data does not support that. The summarizer input was 2,994 tokens, close to the 3,000 limit.

Prompt subsets in `data/refusal_direction/` are sampled from [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) (Apache-2.0); see `SOURCE.json`. Probes are sampled from GSM8K (MIT) and the same repo's test splits; see `probes/SOURCE.json`.
