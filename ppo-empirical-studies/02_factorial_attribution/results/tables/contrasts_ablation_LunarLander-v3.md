# LunarLander-v3 ablation contrasts (per-delta, JAX)

| contrast | A - B | boot_mean | 95% CI | p | cliff_d | n |
|---|---|---|---|---|---|---|
| ABL_novclip_vs_cleanrl | jax_abl_novclip $-$ jax_cleanrl | 41.90 | [20.55, 57.77] | 0.0000 | 0.920 | 5 |
| ABL_noanneal_vs_cleanrl | jax_abl_noanneal $-$ jax_cleanrl | 20.42 | [4.25, 41.83] | 0.0000 | 0.680 | 5 |
| ABL_fullmse_vs_cleanrl | jax_abl_fullmse $-$ jax_cleanrl | 0.31 | [-18.28, 18.28] | 0.9564 | 0.040 | 5 |
| ABL_tboot_vs_cleanrl | jax_abl_tboot $-$ jax_cleanrl | -8.60 | [-27.15, 11.88] | 0.3240 | -0.440 | 5 |
| ABL_novclip_vs_sb3 | jax_abl_novclip $-$ jax_sb3 | 4.41 | [-35.98, 48.20] | 0.8396 | 0.120 | 5 |
| ABL_noanneal_vs_sb3 | jax_abl_noanneal $-$ jax_sb3 | -17.02 | [-40.35, 5.08] | 0.0760 | -0.360 | 5 |
| ABL_fullmse_vs_sb3 | jax_abl_fullmse $-$ jax_sb3 | -37.46 | [-52.29, -19.28] | 0.0000 | -0.600 | 5 |
| ABL_tboot_vs_sb3 | jax_abl_tboot $-$ jax_sb3 | -46.20 | [-69.09, -23.53] | 0.0000 | -0.840 | 5 |