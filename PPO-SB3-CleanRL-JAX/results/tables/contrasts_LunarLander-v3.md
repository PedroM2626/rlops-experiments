# LunarLander-v3 planned contrasts

| contrast | A - B | boot_mean | 95% CI | p | cliff_d | n |
|---|---|---|---|---|---|---|
| Q1_total_gap | sb3_torch $-$ cleanrl_torch | 27.09 | [-28.08, 79.45] | 0.3292 | 0.320 | 10 |
| Q2_stack_given_SB3spec | sb3_torch $-$ jax_sb3 | -6.94 | [-55.59, 40.97] | 0.7912 | -0.020 | 10 |
| Q3_stack_given_CleanRLspec | cleanrl_torch $-$ jax_cleanrl | 15.71 | [-10.47, 41.72] | 0.2460 | 0.320 | 10 |
| Q4_specBridge_torch | sb3_torch $-$ cleanrl_sb3mode_torch | -15.93 | [-59.58, 25.63] | 0.4748 | -0.120 | 10 |
| Q5_spec_within_JAX | jax_sb3 $-$ jax_cleanrl | 50.32 | [25.48, 78.07] | 0.0000 | 0.720 | 10 |
| Q6_bridge_vs_JAXSB3 | cleanrl_sb3mode_torch $-$ jax_sb3 | 8.88 | [-13.07, 32.38] | 0.4744 | 0.160 | 10 |