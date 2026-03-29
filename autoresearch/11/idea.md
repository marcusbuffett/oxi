Intent: Improve oxi chess model's top-1 move prediction accuracy (ao10) by increasing capacity of the Smolgen attention mechanism.

Outcome: Successfully increased smolgen_hidden dimension from 24 to 48. cargo check passed with only pre-existing warnings.

Files: src/config.rs (changed smolgen_hidden: 24 -> 48 in Default impl)

Key decisions: Identified that smolgen_hidden=24 was likely too small for compressing 64 square features into global attention biases; doubled it to 48 to match smolgen_global_dim and smolgen_gen_size (both 128). This builds on iteration 4's success (ao10 0.345991) which improved model capacity.

Problems: None - change compiled successfully. Previous experiments with focal loss, label smoothing, and POLICY_RANK increases had failed to beat the 0.345991 benchmark.