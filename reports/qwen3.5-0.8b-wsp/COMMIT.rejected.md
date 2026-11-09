Project a shared rank-1 direction out of down_proj in layers 15 and 19

Where: Two tensors changed: model.language_model.layers.15.mlp.down_proj.weight and layers.19.mlp.down_proj.weight. Each has 64 input columns edited (1024 and 1023 rows). That is 130,188 changed weights, 0.0149% of the model (131,072 touched, 0.0150%). Frobenius ratio is 0.1064 (L15) and 0.08106 (L19); whole-tensor cosine is 0.9944 and 0.9967.

What: Each tensor's delta is rank 1 (effective rank 1, top energy 1.0). The gap ratios are 1771 (L15) and 1552 (L19). The edit fits c' = c − α(c·v)v with α = 6.0 and scaled residual about 0.002. So it is not plain removal of v: it subtracts six times the projection, which overshoots well past zero. The two layers' directions are similar but not identical (cosine 0.849, subspace overlap 0.7208). The edit does not match the refusal direction. Cosine to refusal is 0.007 (L15) and 0.019 (L19). The in-subspace fraction is 4.9e-05 and 3.6e-04, both below the 0.00098 random baseline.

Effect: Looping and runaway generation drop, most clearly outside math.
- benign_20: loop rate 0.55 → 0.05, hit-cap 0.35 → 0.05, mean tokens 4285 → 2089.
- refusal_20: loop rate 0.6 → 0.0, no-response 0.6 → 0.0, mean tokens 5703 → 1731.
- refusal_20 refusal rate: 0.25 → 0.75. This is likely confounded. Base runs that never finished thinking (0.6) counted as not refusing, so part or all of the rise may come from runs now finishing rather than a change in refusal tendency.
- math_50: answer rate 0.12 → 0.38, accuracy 0.12 → 0.32, hit-cap 0.88 → 0.64, no-response 0.88 → 0.56.

Unchanged: Math loop rate stays high (0.92 → 0.88), and math refusal stays 0.0. Benign refusal rate is 0.05 → 0.05. No attention, embedding or other MLP tensors were touched, and 486 of 488 tensors are unchanged.

Confidence: medium — the weight-space structure is very clean, but the probe sets are small (n = 20/50), and the refusal-rate jump cannot be separated from the drop in unfinished runs.
