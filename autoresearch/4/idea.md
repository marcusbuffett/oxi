Intent: Improve oxi chess model's top-1 move prediction accuracy (ao10 metric) by increasing exposure to tactical puzzle positions during training.

Outcome: Successfully modified training configuration. `cargo check --features "train,backend-tch"` passed with 34 pre-existing warnings.

Files: src/config.rs (puzzle_sampling_ratio: 0.05 → 0.15 at line 1128)

Problems: None - build succeeded with only minor unrelated warnings.

Key decisions: Increased puzzle sampling ratio from 5% to 15% to improve tactical move prediction accuracy, following previous exploration of POLICY_RANK adjustments (64→96) that yielded mixed results (ao10 0.336623 vs baseline 0.330795).