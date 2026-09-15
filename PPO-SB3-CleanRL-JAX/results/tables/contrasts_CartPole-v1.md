# CartPole-v1 planned contrasts

| contrast | A - B | boot_mean | 95% CI | p | cliff_d | n |
|---|---|---|---|---|---|---|
| Q1_total_gap | sb3_torch $-$ cleanrl_torch | 0.00 | [0.00, 0.00] | 0.0000 | 0.000 | 5 |
| Q2_stack_given_SB3spec | sb3_torch $-$ jax_sb3 | 1.63 | [0.00, 4.81] | 0.0000 | 0.200 | 5 |
| Q3_stack_given_CleanRLspec | cleanrl_torch $-$ jax_cleanrl | 0.00 | [0.00, 0.00] | 0.0000 | 0.000 | 5 |
| Q4_specBridge_torch | sb3_torch $-$ cleanrl_sb3mode_torch | 59.52 | [16.44, 111.60] | 0.0000 | 0.800 | 5 |
| Q5_spec_within_JAX | jax_sb3 $-$ jax_cleanrl | -1.65 | [-4.81, 0.00] | 0.0000 | -0.200 | 5 |
| Q6_bridge_vs_JAXSB3 | cleanrl_sb3mode_torch $-$ jax_sb3 | -58.01 | [-106.79, -16.44] | 0.0000 | -0.760 | 5 |