Intent: Improve oxi chess model top-1 move prediction accuracy (ao10) by increasing exposure to tactical puzzle positions, doubling the successful 0.15 ratio from iteration 4 (best ao10: 0.345991) to 0.30.
Outcome: Updated puzzle_sampling_ratio from 0.05 to 0.30 in src/config.rs; cargo check passed with minor warnings (unused imports, deprecated rand::thread_rng).
Files: src/config.rs
Problems: None.
Key decisions: Set puzzle_sampling_ratio to 0.30 to maximize tactical position exposure based on iteration 4 results.