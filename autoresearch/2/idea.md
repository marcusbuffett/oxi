Intent: Improve the oxi chess model's top-1 move prediction accuracy (ao10 metric) from baseline 0.330795 by making a single focused architectural change.

Outcome: Successfully increased policy head expressiveness. Compiled without errors.

Files: src/factorized_policy.rs

Problems: None - change compiled successfully with only minor pre-existing warnings.

Key decisions: Selected POLICY_RANK increase (64→96) over alternatives like learning rate adjustments or label smoothing tweaks, as higher rank directly improves factorized policy head capacity for move prediction. Verified compilation with cargo check --features "train,backend-tch" before finishing.