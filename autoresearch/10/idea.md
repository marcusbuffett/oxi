Intent: Improve oxi chess model top-1 move prediction accuracy (ao10) by reducing focal_loss_gamma from 2.0 to 0.5 to apply milder hard-example focusing.

Outcome: Successfully updated focal_loss_gamma default from 2.0 to 0.5. Cargo check passes with only minor warnings (unused imports, deprecated rand::thread_rng).

Files: src/config.rs (line 1120: focal_loss_gamma: 2.0 → 0.5)

Problems: Initial edit attempts failed due to missing code_edit parameter; succeeded on third attempt after locating exact line. Build has non-blocking warnings about unused imports and deprecated thread_rng.

Key decisions: Chose gamma=0.5 based on research log showing gamma=2.0 (iter 6) and gamma=1.0 (iter 7) both underperformed vs baseline; milder focusing should improve top-1 accuracy by reducing penalty on easy examples.