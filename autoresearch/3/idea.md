Intent: Improve oxi chess model's top-1 move prediction accuracy (ao10) by increasing factorized policy head capacity.

Outcome: Successfully increased POLICY_RANK from 96 to 128 to allow richer move pattern modeling. Change compiled without errors (cargo check passed with only unrelated warnings).

Files: src/factorized_policy.rs

Problems: None - build succeeded.

Key decisions: Modified POLICY_RANK constant in factorized policy head from 96 to 128, which expands the bottleneck dimension for source/target move projections from 96 to 128 features.