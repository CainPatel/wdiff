Project out a shared ~15-dim subspace from embed, o_proj, down_proj

Where: model.embed_tokens.weight plus mlp.down_proj and self_attn.o_proj in all 16 layers. That is 33 of 146 tensors. 581,755,001 weights changed, 47.07% of the model; 48.41% sit in edited rows or columns. No other tensor was changed. Per-tensor frobenius_ratio is about 0.07–0.08 and cosine to base is 0.9966–0.9978.

What: this is a multi-direction edit, not a single-vector ablation. Effective rank is 11–12 and gap_rank is 15 in every group, with gap_ratio 7.4–13.5. Top-direction energy is only 0.16–0.33 in the layers and 0.44 in the embeddings. All 33 tensors share the same subspace (subspace overlap 0.9995–0.9999).
- The fit is projection-like. scaled_residual is about 0.06. projection_scale is about 1.13 in o_proj and down_proj, which is slight over-removal, and 0.86 in embeddings.
- Alignment with the measured refusal direction is weak. Median |cos| is 0.045 for down_proj and 0.035 for o_proj. The peak is 0.18–0.19, at layer 11 for both.
- The refusal direction's in_subspace share has a median of 0.013 against a random baseline of 0.0073, with a maximum of 0.059.

Effect:
- refusal_20: refusal_rate 0.9 → 0.7, short_rate 0.45 → 0.05, mean_tokens 98.4 → 231.9, loop_rate 0.05 → 0.0.
- math_50: accuracy 0.06 → 0.0, answer_rate 0.12 → 0.1, mean_tokens 191.6 → 219.0.
- benign_20: short_rate 0.1 → 0.2, mean_tokens 285.5 → 250.3.

Unchanged:
- benign refusal_rate stayed at 0.0 and math refusal_rate at 0.0.
- math loop_rate stayed at 0.02.
- hit_cap_rate and no_response_rate were 0.0 on every probe set.
- q/k/v projections, gate/up MLP and norms were not edited.

Confidence: low — probe sets are small (n=20). Refusal only falls to 0.7. The edited subspace barely overlaps the measured refusal direction, so it is unclear what the edit actually removes. Math accuracy also fell, though from a base of only 0.06.
