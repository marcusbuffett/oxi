Intent: Reduce focal_loss_gamma from 2.0 to 0.5 to mitigate the performance degradation seen in iteration 6 (ao10 0.337002 vs best 0.345991), applying milder hard-example focusing to improve top-1 move prediction.

Outcome: Failed. The edit tool invocation used incorrect parameter naming (`code_update` instead of `code_edit`), preventing the configuration change from being applied.

Files: src/config.rs (attempted edit), src/factorized_policy.rs, src/model.rs, src/relative_position_transformer.rs (read-only)

Problems: Tool schema error on edit attempt: required field `code_edit` was not provided (received `code_update`). Focal loss gamma remains at 2.0; no code changes persisted.

Key decisions: Diagnosed iteration 6's focal loss gamma=2.0 as too aggressive; selected gamma=0.5 as a conservative adjustment rather than disabling focal loss completely.