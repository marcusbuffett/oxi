Intent: Explore the oxi chess model codebase to understand its architecture, configuration, training loop, and research history.

Outcome: Successfully identified key files and components of the Oxi chess model including model architecture, configuration options, training structure, and research logging.

Files: src/model.rs, src/config.rs, src/custom_training.rs, src/factorized_policy.rs, research_log.md, src/main.rs, Cargo.toml

Problems: No autoresearch directory found, which might indicate ongoing research isn't being tracked in that location. Some modules like train_stubs are conditionally compiled and don't contain full implementation details.

Key decisions: Model uses factorized policy head with separate projections for source/target/promo moves, employs RMSNorm normalization, and includes separate uncertainty parameters for different outputs. Training uses AdamW optimizer with gradient clipping and custom metrics. Configuration supports multiple backends via feature flags.