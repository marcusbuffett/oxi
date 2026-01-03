You are a machine learning expert helping refine and extend the Oxi chess model.

#### Project Purpose
Oxi is a Rust/Burn transformer that predicts human chess moves conditioned on player strength and clock state, and jointly models outcome and time usage signals. Objectives include human-aligned move selection, Elo-aware behavior, and useful secondary predictions (WDL, tempo usage).

#### Data Pipeline (`src/encoding.rs`, `src/dataset.rs`)
- **Board Tokens**: 64 squares × 44 features. Core position features cover piece one-hots, en passant, castling rights, per-color attacker summaries, legal move counts, pawn structure (isolated/backward/doubled), weak-square flags, open files, pinned indicator, square control, and a dark-square flag. Four dedicated recency channels track source/target heatmaps for the last up to five moves per color with exponential decay (`HISTORY_DECAY = 0.8`).
- **Global Scalars** (`NUM_GLOBALS = 7`): Normalized self/opponent remaining time, ratios vs base time, increment ratio, move count, and player Elo (see `GlobalFeatures::to_feature_vector` in `src/inference.rs`).
- **Historical Context**: `PREVIOUS_POSITIONS = 5` previous boards and UCI moves are retained for recency encoding; decay governs their contribution.
- **Legal Move Masking**: Moves are encoded on a 64×76 lattice (`LEGAL_MOVES`) ensuring policy logits are evaluated only for legal targets.
- **Dataset Filters** (`src/config.rs`): Includes Elo range 1000–2500, ≤200 Elo gap, minimum time control 61 s, minimum per-move clock 30 s; optional sampling heuristics (ply and Elo) throttle low-value examples.
- **Pretraining Support** (`src/custom_training.rs`): Optional `easy_positions.bin` bootstrap dataset is blended in batches before transitioning to full data.

#### Model Architecture (`src/model.rs`, `src/relative_position_transformer.rs`)
- **Stream Embedding & Gating**: Main board channels and global scalars are projected to `embed_dim - 4`, layer-normalized independently, and blended with learned softmax gates across token/global/absolute positional embeddings. Recency channels are normalized separately and concatenated to restore full `embed_dim`.
- **Absolute + Relative Positioning**: Learned absolute embeddings (`pos_embed`) complement Shaw 2D relative attention over 15×15 displacement buckets (`src/shaw.rs`) with per-head shared embeddings and learnable scale parameters.
- **Transformer Core**: `num_layers` Peri-LN blocks (post-residual normalization for attention and MLP) with GELU activations and width `mlp_ratio`. Extensive tensor norm logging hooks (`src/norm_debug.rs`) aid stability debugging.
- **Head Refinement Blocks**: After the shared stack, dedicated `policy_block`, `value_block`, and `time_block` provide a final layer of attention before their heads.
- **Outputs**:
  - **Policy**: Linear to `LEGAL_MOVES / 64` logits per square (4864 total) followed by legal-mask log-softmax, focal loss (`focal_loss_gamma`, default 2.0) and optional label smoothing.
  - **Value**: Attention-style pooling into a classifier predicting win/draw/loss with entropy regularization weighted by decisive mass.
  - **Time Usage**: Beta-parameter regression (alpha, beta) for normalized move duration; can be disabled by setting `time_usage_loss_weight = 0`.
  - **Side Info**: Head remains defined but excluded from current training loss path.
- **Uncertainty Params**: Trainable log-variance scalars per head are kept for future calibration work.

#### Training & Optimization (`src/custom_training.rs`)
- **Optimizers**: Six AdamW instances partition parameters by decay/no-decay and learning-rate tier (`src/weight_decay.rs`). Embeddings receive a `sqrt(embed_dim)` LR multiplier; scale parameters use `lr_scalar`.
- **Scheduler**: `ReduceOnPlateauScheduler` drives LR from the batch-scaled initial value down toward `lr_min` based on loss plateaus.
- **GradNorm** (`src/gradnorm.rs`): Re-balances policy/value/time losses at configured intervals; respects per-head priority multipliers and shuts off automatically if only one head is active.
- **Gradient Control**: Norm-based clipping (default 1.0), post-residual norms, and detailed gradient breakdown logging guards provide stability when scaling depth/batch size.
- **Batching**: Logical batch size can be specified separately from physical (`physical_batch_size`, default 1024) enabling gradient accumulation to reach targets like 8k+ without out-of-memory.
- **Metrics & Logging**: Rich Burn metrics for policy loss, WDL, move accuracy, GradNorm status, tensor norms, and uncertainty. Checkpointing persists model plus all optimizer shards in `model/`.
- **CLI Usage**: Training invoked via `cargo run --release --bin oxi -- train ...`; inference and PGN tooling share the same binary (`src/main.rs`).

#### Default Hyperparameters (`src/config.rs`)
- `embed_dim = 512`, `num_heads = 8`, `num_layers = 14`, `mlp_ratio = 4.0`.
- `lr_min = 1e-6`, `lr_scalar = 100.0` (initial LR is batch-size scaled in `src/custom_training.rs`).
- `policy_loss_weight = 0.15`, `value_loss_weight = 0.15`, `time_usage_loss_weight = 0.008`.
- `gradient_clip = 1.0`, `weight_decay = 1e-5`.

#### Recent Training Snapshot
- Configuration: 14 layers, `embed_dim = 512`, `num_heads = 8`, effective batch size 8192 via accumulation.
- Schedule: batch-size scaled initial LR with Reduce-on-Plateau stepping.
- Outcomes: Top-1 move accuracy 48% on validation, GradNorm-reported gradient norms stabilized between 1 and 3.
- Notes: Time-usage head remained active with small weight; future runs may explore scaling it after further calibration.

#### Next Investigation Targets
- Re-enable or redesign side-info supervision once loss path is finalized.
- Cross-check inference CLI with latest encoder changes to ensure feature parity with training.
