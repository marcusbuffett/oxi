Intent: Tune oxi chess model to improve top-1 move prediction accuracy (ao10 metric) by increasing policy loss weight from 0.15 to 0.25, prioritizing policy head training over value/side-info heads; attempting to beat iteration 4's best score of 0.345991.

Outcome: Configuration updated successfully at line 1094; cargo check --features "train,backend-tch" passed with only pre-existing unused import warnings.

Files: src/config.rs

Problems: None encountered; clean compile despite deprecation warnings for rand::thread_rng.

Key decisions: Avoided focal loss modifications (shown to regress in iterations 6-9); selected policy loss weight increase (67% bump) to emphasize move prediction signal over auxiliary task losses.