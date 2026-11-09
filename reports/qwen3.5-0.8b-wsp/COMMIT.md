Subtract a rank-1 direction (α=6) from down_proj cols, layers 15 and 19

Where: model.language_model.layers.15.mlp.down_proj.weight and layers.19.mlp.down_proj.weight. 64 columns were edited in each. 2 of 488 tensors changed. 130,188 weights changed (131,072 touched), which is 0.0149% of the model. Frobenius ratio is 0.1064 (L15) and 0.08106 (L19); whole-tensor cosine is 0.9944 and 0.9967.

What: The edit is rank-1 in each tensor (effective rank 1, top energy 1.0). The singular gap ratio is 1771 at L15 and 1552 at L19. It fits c' = c − α(c·v)v with α = 6.0, scaled residual about 0.002. This is a 6× overshoot, not plain removal of v. The two layers' directions are related but not identical: cosine 0.849, subspace overlap 0.7208. The direction does not match the refusal direction. Cosine is 0.007 at L15 and 0.019 at L19. In-subspace share is 4.9e-05 and 3.6e-04, both below the random baseline of 9.8e-04.

Effect: Looping and runaway generation dropped on every probe set.
- benign_20: loop rate 0.55 → 0.05, hit-cap 0.35 → 0.05, mean tokens 4285 → 2089.
- refusal_20: loop rate 0.6 → 0.0, no-response 0.6 → 0.0, mean tokens 5703 → 1731.
- refusal_20 refusal rate went 0.25 → 0.75. This is likely mostly because runs that previously never finished thinking now produce a response, not a change in the model's refusal behaviour.
- math_50: answer rate 0.12 → 0.38, accuracy 0.12 → 0.32, hit-cap 0.88 → 0.64, no-response 0.88 → 0.56.

Unchanged: Benign refusal rate stayed at 0.05 and math refusal at 0.0. Math looping barely moved (loop rate 0.92 → 0.88) and remains high. No other tensors or layers were touched, and the edit is not aligned with the refusal direction.

Confidence: medium. The weight edit is cleanly rank-1 with a near-exact α=6 fit. However, the behavioural probes are small (n=20/50), and the refusal-rate rise is confounded with responses now completing.
