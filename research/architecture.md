# Oxi Architecture Reference (for Research Agents)

This document describes the full Oxi chess model as used by `research_loop.py`. It covers the model architecture, input representation, output heads, loss functions, training loop, optimizer configuration, metrics logging, data pipeline, and evaluation. All values reflect what actually runs during a research loop iteration unless noted otherwise.

---

## Table of Contents

1. [Research Loop Context](#research-loop-context)
2. [Input Representation](#input-representation)
3. [Model Architecture](#model-architecture)
4. [Output Heads](#output-heads)
5. [Loss Functions](#loss-functions)
6. [GradNorm (Adaptive Loss Weighting)](#gradnorm)
7. [Optimizer & Learning Rate](#optimizer--learning-rate)
8. [LR Scheduling (Reduce-on-Plateau)](#lr-scheduling)
9. [Data Pipeline](#data-pipeline)
10. [Training Loop Details](#training-loop-details)
11. [Metrics Logging & File Format](#metrics-logging--file-format)
12. [Evaluation & Scoring](#evaluation--scoring)
13. [Hyperparameter Reference](#hyperparameter-reference)
14. [Key Source Files](#key-source-files)

---

## Research Loop Context

The research loop (`research_loop.py`) spawns subagents to propose code changes, then evaluates them via two 60-minute training runs (different seeds). Key overrides from code defaults:

| Parameter | Research Loop Value | Code Default |
|-----------|-------------------|--------------|
| `physical_batch_size` | **512** (CLI override) | 16000 |
| `warmup_multiplier` | **0.1** (CLI override) | 2.0 |
| `full_metrics_interval` | **200** (CLI override) | 50 |
| `gradnorm_interval` | **200** (CLI override) | 20 |
| `checkpoint_interval` | **0** (disabled, CLI override) | 100 |
| `--disable-tui` | yes | no |
| `--log-gradient-breakdown` | yes | no |
| Training duration | 3600s per seed × 2 seeds (killed via SIGKILL) | unlimited |

**Architecture is NOT pinned via CLI.** The model's `embed_dim`, `num_layers`, and `num_heads` come from defaults in `src/config.rs` (currently 192, 12, 6). If you want to change the architecture, modify those defaults directly. The training command does not pass `--embed-dim`, `--num-layers`, or `--num-heads`.

Everything else uses code defaults. The training command is:
```
./target/release/oxi train \
  --pretrain-samples=0 --data-path=../data \
  --physical-batch-size=512 \
  --seed=<varies> --log-dir=<run_dir> --disable-tui \
  --warmup-multiplier=0.1 --log-gradient-breakdown \
  --full-metrics-interval=200 --gradnorm-interval=200 --checkpoint-interval=0
```

---

## Input Representation

### Board Encoding: 64 squares × 65 features per square

Each of the 64 squares gets a feature vector of length `FEATURES_PER_TOKEN = 65`:

**`BOARD_FEATURES_PER_TOKEN = 61`** (current position only, before recency):

| Group | Features | Count | Description |
|-------|----------|-------|-------------|
| **Piece Identity** | 0–11 | 12 | One-hot for piece type: 6 white (P,N,B,R,Q,K) + 6 black |
| **Tactical** | 12–33 | 22 | See breakdown below |
| **Positional** | 34–58 | 25 | See breakdown below |
| **Misc** | 59–60 | 2 | En passant target (1), local castling right (1) |

**Tactical features (22):**
- White attackers by role: 6 one-hots (one per piece type, clamped to 0/1)
- White total attacker material: 1 (sum of attacker piece values, normalized by 24)
- White attacker count: 1 (normalized by 6)
- Black attackers by role: 6 one-hots
- Black total attacker material: 1
- Black attacker count: 1
- Pinned flag: 1
- Absolute pin to king: 1
- Pin target flag: 1
- Hanging piece flag: 1
- Has pinned defender: 1
- Square control value: 1

**Positional features (25):**
- Normalized legal moves from square: 1
- Pawn structure (isolated, backward, doubled): 3
- Weak square (white perspective, black perspective): 2
- Open file: 1
- Passed pawn: 1
- Dark square flag: 1
- Rank one-hot: 8
- File one-hot: 8

**`RECENCY_FEATURES = 4`** (appended after the 61 board features):
- White move-from heatmap: 1 (exponential decay over last 5 positions, decay=0.8)
- White move-to heatmap: 1
- Black move-from heatmap: 1
- Black move-to heatmap: 1

These are per-square scalars encoding recent move history with `HISTORY_DECAY = 0.8` over `PREVIOUS_POSITIONS = 5` half-moves.

### Global Features: `NUM_GLOBALS = 10`

A single vector per position, not per-square:

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Time remaining (self) | / 1500, clamped [0, 1] |
| 1 | Time remaining (self) as ratio to base time | ratio |
| 2 | Time remaining (opponent) | / 1500, clamped [0, 1] |
| 3 | Time remaining (opponent) as ratio to base time | ratio |
| 4 | Increment as ratio to base time | ratio |
| 5 | Move count | / 300, clamped [0, 1] |
| 6 | Elo (self) | (elo - 800) / 2000, clamped [0, 1] |
| 7 | Puzzle flag | 0.0 or 1.0 |
| 8 | Material imbalance | (material / 15), mapped from [-1,1] to [0,1] |
| 9 | Total pieces / 32 | clamped [0, 1] — game phase proxy |

Global features are used for FiLM conditioning throughout the transformer and concatenated with pooled tokens in the value head.

### Move Encoding: `LEGAL_MOVES = 64 × 76 = 4864`

Each move is encoded as `(from_square, target_index)` where:
- target_index 0–63: standard move to that destination square
- target_index 64–75: promotion moves (3 directions × 4 piece types)
  - Direction 0 (left diagonal): indices 64–67 (knight, bishop, rook, queen)
  - Direction 1 (straight): indices 68–71
  - Direction 2 (right diagonal): indices 72–75

The flat index is `from_square * 76 + target_index`.

---

## Model Architecture

### Overview

Pre-norm transformer operating on 64 square tokens, with FiLM conditioning from global features, Smolgen attention biases, and SwiGLU MLPs.

```
Input: [batch, 64, 65]
  ↓
Split → board_features [batch, 64, 61] + recency [batch, 64, 4]
  ↓
Linear(61 → base_embed_dim) + learnable square embeddings [64, base_embed_dim]
  → token_embeddings [batch, 64, base_embed_dim]
  ↓
Concat(token_embeddings, RmsNorm(recency)) → [batch, 64, embed_dim]
  ↓
Optional: SpatialConv (if conv_layers > 0)
  ↓
N × TransformerBlock (with FiLM conditioning from globals)   ← "trunk"
  ↓
RmsNorm → trunk_output [batch, 64, embed_dim]
  ↓
┌─ policy_block (1 extra TransformerBlock) → policy_tokens
├─ value_block  (1 extra TransformerBlock) → value_tokens
└─ trunk_output used directly for: SideInfo, Aux (mobility, material, trunk from/to)
  ↓
Heads: Policy (from policy_tokens), Value (from value_tokens),
       TimeUsage (disabled), SideInfo, Aux
```

**Total transformer blocks = `num_layers + 2`** (trunk + 1 policy + 1 value). For research config: 12 + 2 = **14 blocks**.

Where `base_embed_dim = embed_dim - RECENCY_FEATURES = 192 - 4 = 188` in the research config.

### Learnable Square Embeddings

A `[64, base_embed_dim]` parameter tensor, initialized with Normal(0, 0.02). Added to each square's token embedding before concatenation with recency features.

### Spatial Convolution (optional, `conv_layers` default=0)

When enabled, applies a 3×3 neighborhood gather on the 8×8 board:
- Gathers 9 patches (center ± 1 in each direction, zero-padded at edges)
- Concatenates along channels: `[batch, 64, 9 * channels]`
- Linear projection: `9 * in_channels → in_channels`
- SiLU activation

### Transformer Block

Each block contains:

1. **FiLM-RmsNorm** → pre-norm with global feature modulation
2. **SmolgenAttention** → multi-head self-attention with position-dependent biases
3. **Residual connection** (plain addition; residual scaling is initialization-only, see below)
4. **FiLM-RmsNorm** → pre-norm
5. **SwiGLU MLP**
6. **Residual connection**

#### FiLM-RmsNorm

Standard RmsNorm augmented with Feature-wise Linear Modulation:
```
gamma = gamma_proj(globals) + 1.0    # [batch, embed_dim]
beta = beta_proj(globals)            # [batch, embed_dim]
output = rms_norm(x) * gamma + beta
```
Where `gamma_proj` and `beta_proj` are `Linear(NUM_GLOBALS, embed_dim)`.

#### SmolgenAttention

Multi-head attention with dynamic position-dependent bias:

- **QKV projection**: `Linear(embed_dim, 3 * embed_dim)` → split into Q, K, V
- **Head split**: `[batch, 64, embed_dim] → [batch, num_heads, 64, head_dim]`
- **Attention**: `softmax((Q @ K^T) / sqrt(head_dim) + positional_bias)`
- **Output projection**: `Linear(embed_dim, embed_dim)` with residual scaling

**Smolgen (position bias generator):**
1. Per-token compression: `Linear(embed_dim, smolgen_hidden)` → `[batch, 64, smolgen_hidden]`
2. Flatten: `[batch, 64 * smolgen_hidden]`
3. Global extraction: `Linear(64 * smolgen_hidden, smolgen_global_dim)` with LayerNorm + SiLU
4. Per-head projection: `Linear(smolgen_global_dim, num_heads * smolgen_gen_size)` with LayerNorm + SiLU
5. Shared weight generation: `Linear(smolgen_gen_size, 64 * 64)` → reshape to `[batch, num_heads, 64, 64]`

The weight_gen linear is shared across all transformer layers. Initialized with 0.01 std to keep initial biases small.

Research config Smolgen dimensions: `smolgen_hidden=24, smolgen_global_dim=128, smolgen_gen_size=128`.

#### SwiGLU MLP

```
hidden_dim = embed_dim * 2.5  (= 480 for embed_dim=192)
gate_up = Linear(embed_dim, 2 * hidden_dim)  # fused
gate, up = split(gate_up)
hidden = SiLU(gate) * up
output = Linear(hidden_dim, embed_dim)  # residual-scaled
```

### Initialization

- Most weights: Normal(0, 0.02)
- Residual projections (attention output, MLP down): Normal(0, 0.02 / sqrt(2 * num_layers)) — this is the "residual scaling"; it reduces initial residual magnitude via smaller weights, **not** a runtime scaling factor. The forward pass uses plain `x = x + attn_out` and `x = x + mlp_out`.
- Smolgen weight_gen: Normal(0, 0.01)
- Square embeddings: Normal(0, 0.02)

---

## Output Heads

### 1. Policy Head (Factorized)

**Output: `[batch, 64, 76]`** → flattened to `[batch, 4864]`

Uses factorized dot-product between source and target token representations. Input comes from **policy_tokens** (output of the dedicated policy transformer block, not raw trunk output):

```
source_tokens = Linear(embed_dim, embed_dim)(policy_tokens)  # [batch, 64, embed_dim]
target_tokens = Linear(embed_dim, embed_dim)(policy_tokens)  # [batch, 64, embed_dim]
base_logits = source_tokens @ target_tokens.T                # [batch, 64, 64]
```

**Promotion handling:**
- Extract promotion-eligible rows and columns based on side to move
- `promo_from_proj(from_tokens) + promo_to_proj(to_tokens)` → `[batch, n_promo_from, n_promo_to, 4]`
- Sliced into the [64, 76] output at the correct promotion indices

Illegal moves are masked to `-inf` before softmax.

### 2. Value Head (WDL)

**Output: `[batch, 3]`** — logits for (loss, draw, win)

Input comes from **value_tokens** (output of the dedicated value transformer block applied to a clone of the trunk output). The value block is a single `TransformerBlock` (not a multi-layer tower — `value_tower_layers` config exists but the actual implementation uses one block). Gradients flow back to the trunk through this clone.

Uses attention pooling over value tokens:
```
pool_hidden = SiLU(fc1(value_tokens))              # [batch, 64, embed_dim]
pool_weights = softmax(fc2(pool_hidden), dim=1)    # [batch, 64, 1] → weights
pooled = (value_tokens * pool_weights).sum(dim=1)  # [batch, embed_dim]
combined = concat(pooled, globals)                 # [batch, embed_dim + NUM_GLOBALS]
hidden = SiLU(head_hidden(combined))               # [batch, embed_dim]
logits = head_output(hidden)                       # [batch, 3]
```

**Value weight ramp:** Loss weight scales linearly from 0 at `value_ply_ramp_start=10` to full at `value_ply_ramp_full=30` — early-game positions get less value supervision because outcomes are less attributable.

### 3. Time Usage Head (DISABLED)

**Currently disabled.** The head parameters exist in the model struct but the forward pass hardcodes the output to `zeros([batch, 2])`. The `time_usage_loss_weight` default is 0.0, so even if logits were computed, the loss would be zero. The head's intended design was to output (alpha, beta) for a Beta distribution modeling clock time fraction, but it is not active.

### 4. Side Info Head

**Output: `[batch, 141]`** — 141 binary/one-hot predictions

Mean-pooled trunk → linear → 141 logits. Trained with binary cross-entropy. The 141 features are:
- Indices 0–5: piece moved (6 piece types, one-hot)
- Indices 6–11: captured piece (6 piece types, one-hot; all zero if no capture)
- Index 12: check flag (1)
- Indices 13–76: from-square (64 squares, one-hot)
- Indices 77–140: to-square (64 squares, one-hot)

### 5. Auxiliary Heads

**Mobility:** Per-square legal move count prediction.
- Per-token linear from trunk output → `[batch, 64]`
- MSE + MAE loss

**Material:** Total material imbalance prediction.
- Mean-pooled trunk → linear → `[batch, 1]`
- MSE + MAE loss

**From/To Square (Maia 2-style):**
- Policy-token-level 64-way classification: which square is the from-square / to-square of the played move
- Cross-entropy loss
- Accuracy tracked (these are `aux_from_square_accuracy` and `aux_to_square_accuracy` in the composite score)

**Trunk-level From/To Square:**
- Direct trunk token supervision (separate from the policy-token version)
- Additional CE losses: `trunk_from_sq_ce`, `trunk_to_sq_ce`

---

## Loss Functions

### Policy Loss

```python
# Label smoothing over legal moves only
smooth_target = (1 - eps) * one_hot(true_move) + eps * uniform(legal_moves)

# Standard cross-entropy (when focal_loss_gamma == 0):
loss = -sum(smooth_target * log_softmax(logits))

# Focal loss (when focal_loss_gamma > 0):
loss = -sum(smooth_target * (1 - p_t)^gamma * log(p_t))
```

Default: `policy_label_smoothing=0.005`, `focal_loss_gamma=0.0` (standard CE).

### Value Loss (WDL)

Cross-entropy on the 3-class (loss, draw, win) logits. Targets are one-hot from game outcome.

Optional entropy bonus on decisive games: `-entropy_bonus * value_entropy_weight` (default weight=0.0, disabled).

### Time Usage Loss (DISABLED)

The time usage head is currently disabled (outputs zeros, weight=0.0). If re-enabled, the intended loss is Beta distribution negative log-likelihood:
```
target = clamp(time_used / time_remaining, eps, 1-eps)
loss = -beta_log_pdf(target | alpha, beta)
```

### Auxiliary Losses

All auxiliary losses are summed and weighted by a single `aux_loss_weight`:
- Mobility: MSE on per-square legal move counts
- Material: MSE on material imbalance (batch tensor normalized by ÷39, vs ÷15 for the global feature — these are separate)
- Side info: Binary CE on 141 predictions
- From/to square: CE on 64-way classification (both policy-token and trunk-level versions)

### Total Loss

```
total_loss = gradnorm_weight_policy * policy_loss
           + gradnorm_weight_value * value_loss
           + gradnorm_weight_time * time_usage_loss
           + gradnorm_weight_aux * aux_loss
```

The GradNorm weights are dynamically adjusted — the config `policy_loss_weight`, `value_loss_weight`, etc. are only initial values.

---

## GradNorm

GradNorm adaptively reweights the 4 task losses (policy, value, time_usage, auxiliary) to balance gradient magnitudes.

### Algorithm

Every `gradnorm_interval` optimizer steps (200 in research loop), plus an early adjustment at step 10:

1. Compute each task's loss on a probe batch (`gradnorm_probe_size=256` examples)
2. Track `initial_loss_i` (set on first occurrence of each task)
3. Compute loss ratio: `loss_ratio_i = (loss_i / initial_loss_i)^alpha`
4. Compute target gradient norm: `target_i = mean_grad_norm * loss_ratio_i * priority_i`
5. Update task weight: gradient descent on `(||grad_norm_i|| - target_i)^2` with `gradnorm_learning_rate`
6. **Clamp** updated weights to `[0.1×, 10×]` of their initial values
7. **Renormalize** all weights so their sum equals the original reference total (prevents drift)

GradNorm probing is skipped entirely during value-tower-only training mode (only one loss active).

### Config (research loop values = code defaults for these)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `enable_gradnorm` | true | GradNorm is active |
| `gradnorm_interval` | 200 (overridden) | Steps between weight updates |
| `gradnorm_alpha` | 0.5 | Loss ratio scaling exponent (lower = more aggressive balancing) |
| `gradnorm_learning_rate` | 0.1 | Meta-learning rate for weight adjustment |
| `gradnorm_policy_priority` | 5.0 | Policy gets 5× priority weighting |
| `gradnorm_value_priority` | 1.0 | |
| `gradnorm_time_priority` | 1.0 | |
| `gradnorm_aux_priority` | 1.5 | Aux gets 1.5× priority |
| `gradnorm_probe_size` | 256 | Examples materialized on lead device for gradient computation |

### Observed Behavior

From research notes (run 3 logs): by end of training, GradNorm increased policy weight from 0.30→0.36 and decreased aux weight from 0.06→0.005. GradNorm considers policy under-resourced and aux "too easy."

---

## Optimizer & Learning Rate

### Parameter Groups

The model uses **5 separate optimizer groups**:

| Group | Optimizer | Parameters | Weight Decay | LR |
|-------|-----------|-----------|--------------|-----|
| 1 | **Muon** | 2D+ weight matrices | `weight_decay` | `muon_lr` |
| 2 | AdamW (decay, normal LR) | Most other params | `weight_decay` | `adamw_lr` |
| 3 | AdamW (decay, high LR) | Embedding params | `weight_decay` | `embedding_lr` |
| 4 | AdamW (no decay, normal LR) | LayerNorms, biases | 0.0 | `adamw_lr` |
| 5 | AdamW (no decay, high LR) | Embedding-related biases | 0.0 | `embedding_lr` |

**Cautious weight decay** is enabled by default: weight decay is only applied when the gradient and update direction align (i.e., `grad · update > 0`).

### μP-Scaled Learning Rates

Learning rates scale with model width using Maximal Update Parameterization (μP), referenced to `d=256`:

```
batch_scale = effective_batch_size / 1024.0

adamw_dim_scale = 256 / embed_dim
adamw_lr = adamw_base_lr × adamw_dim_scale × batch_scale × lr_multiplier

muon_dim_scale = sqrt(256 / embed_dim)
muon_lr = muon_base_lr × muon_dim_scale × batch_scale × lr_multiplier

embedding_lr = embedding_base_lr × batch_scale × lr_multiplier
  (no width scaling — width-independent per μP)
```

For the research config (`embed_dim=192, physical_batch_size=512, batch_size=auto`):
- `adamw_dim_scale = 256/192 ≈ 1.333`
- `muon_dim_scale = sqrt(256/192) ≈ 1.155`
- `batch_scale = effective_batch_size / 1024` (depends on auto-computed batch_size)

Base LRs:
- `muon_base_lr = 0.0225`
- `adamw_base_lr = 3.375e-4`
- `embedding_base_lr = 0.1125`

### Other Optimizer Settings

| Parameter | Value |
|-----------|-------|
| `weight_decay` | 0.003 |
| `adam_epsilon` | 1e-8 |
| `gradient_clip` | 3.0 (L2 norm) |

---

## LR Scheduling

### Reduce-on-Plateau

The scheduler detects training plateaus using linear regression on a sliding window of loss values.

**Algorithm:**
1. Maintain a FIFO window of `lr_window_size` loss values
2. Fit linear regression: `loss = a + b × step`
3. Compute relative improvement: `improvement = -slope × window_size / mean_loss`
4. If `improvement < lr_improvement_threshold`: multiply LR by `lr_reduction_factor`
5. Stop when LR reaches `lr_min`

**Warmup:** Linear warmup from 0 to initial LR over `warmup_iterations = warmup_multiplier × effective_batch_size` **optimizer steps** (not samples — despite the config comment saying "samples"). For research config: `0.1 × 512 = 51` iterations of warmup.

| Parameter | Value |
|-----------|-------|
| `lr_window_size` | 120 |
| `lr_improvement_threshold` | 0.02 (2%) |
| `lr_reduction_factor` | 0.5 |
| `lr_min` | 0.000001 |
| `warmup_multiplier` | **0.1** (research override, default=2.0) |

Plateau detections are logged to `plateau_detection.log` with the full window dump.

### Two-Stage Training

After the LR scheduler reaches `lr_min` AND detects another plateau:
1. **Stage 2: Value Tower Only** — disables policy loss, filters gradients to value tower params only, resets LR to initial value
2. The plateau detector switches to tracking **value_loss** instead of policy_loss for LR reduction decisions
3. GradNorm probing is disabled (only one loss active)
4. Continues until `lr_min` reached again

This allows the value head to train further after the policy head has converged. Can also be triggered from the start via `--skip-policy-loss`.

---

## Data Pipeline

### Data Source

PGN files from `--data-path=../data`. Processed into `ChessExample` structs containing:
- FEN position
- Played move
- Game outcome (W/D/L)
- Time usage (clock info)
- Elo ratings
- Move history (previous 5 positions)
- Whether this is a puzzle position

### Shuffle Buffer

A `ShuffleBuffer` of size `shuffle_buffer_size=100000` examples enables streaming without loading the full dataset. Batches are sampled randomly from the buffer, which is continuously refilled.

### Data Filtering / Sampling

| Filter | Behavior |
|--------|----------|
| **Ply sampling** | 80% drop at ply 0, linearly decreasing to 0% drop at ply 10+ |
| **Elo sampling** | Flattens Elo distribution; boosts 2000+ Elo by `elo_priority_boost=3.0×` |
| **Puzzle mixing** | `puzzle_sampling_ratio=0.05` — 5% puzzle positions mixed in |
| **Min clock time** | Positions with <30 seconds on the clock are filtered |

### Batch Construction

Each `ChessBatch` contains:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `board_input` | `[batch, 64, 65]` | Per-square features |
| `move_distributions` | `[batch, 4864]` | One-hot target move |
| `legal_moves` | `[batch, 4864]` | Binary mask of legal moves |
| `values` | `[batch, 3]` | One-hot WDL target |
| `global_features` | `[batch, 10]` | Global features |
| `time_usages` | `[batch, 1]` | Time fraction target |
| `value_weights` | `[batch]` | Ply ramp × puzzle mask |
| `material_imbalance` | `[batch]` | Normalized to [-1, 1] by ÷39 (max possible). Note: global feature #8 uses ÷15 instead |
| `side_info` | `[batch, 141]` | Integer features for side info head |

---

## Training Loop Details

### Gradient Accumulation

If `batch_size > physical_batch_size`:
- `grad_accumulation_steps = ceil(batch_size / physical_batch_size)`
- Accumulate gradients over micro-batches before optimizer step

When `batch_size` is `None` (the default, including in the research config), `grad_accumulation_steps = 1` and `effective_batch_size = physical_batch_size`. So in the research loop, effective batch size is simply 512.

### Per-Iteration Steps

1. Sample batch from shuffle buffer
2. Process examples in parallel (`rayon::par_iter`)
3. Split across devices (default: 1 device)
4. Forward pass → `ChessOutput` (all head outputs + losses)
5. Backward pass → gradients
6. Accumulate gradients (if grad accumulation > 1)
7. Clip gradients (L2 norm, threshold=3.0)
8. Optimizer step (5 separate groups)
9. LR scheduler: `record_batch(plateau_loss)`
10. Log metrics

### Logging Frequency

| What | When |
|------|------|
| `total_loss`, `policy_loss`, `value_loss`, `time_usage_loss` | Every iteration |
| `aux_mobility_loss`, `aux_material_loss`, `aux_side_info_loss`, etc. | Every iteration (if > 0) |
| `aux_from_square_loss`, `aux_to_square_loss` | Every iteration |
| `aux_from_square_accuracy`, `aux_to_square_accuracy` | Every iteration |
| `gradient_norm` | Every iteration |
| `plateau_loss` | Every iteration |
| `top1_accuracy`, `wdl_accuracy` | Every iteration |
| `puzzle_solve_rate` | Every iteration |
| `move_top_5_accuracy` | Every `full_metrics_interval` (200 in research) |
| Gradient breakdown (per-head, per-layer) | Every `full_metrics_interval` (when `log_gradient_breakdown=true`) |
| Learning rates | Every 100 optimizer steps |

**Note:** Most metrics are logged every iteration. Only `move_top_5_accuracy` and the gradient breakdown are gated behind `full_metrics_interval`.

---

## Metrics Logging & File Format

### Directory Structure

```
<log_dir>/
├── train.log                     # Full tracing output
├── stderr.log                    # Stderr capture (research loop adds this)
├── plateau_detection.log         # Full window dump on each plateau detection
└── metrics_logs/
    ├── total_loss.log
    ├── policy_loss.log
    ├── value_loss.log
    ├── time_usage_loss.log
    ├── plateau_loss.log
    ├── gradient_norm.log
    ├── top1_accuracy.log
    ├── wdl_accuracy.log         # Mean probability of correct class (0-1 fraction)
    ├── aux_from_square_accuracy.log
    ├── aux_to_square_accuracy.log
    ├── aux_from_square_loss.log
    ├── aux_to_square_loss.log
    ├── aux_mobility_loss.log
    ├── aux_mobility_mae.log
    ├── aux_material_loss.log
    ├── aux_material_mae.log
    ├── aux_side_info_loss.log
    ├── puzzle_solve_rate.log
    └── move_top_5_accuracy.log   # (and potentially more)
```

### TSV Format

Each `.log` file is tab-separated with two columns: `iteration\tvalue`

```
1	0.06490872
2	0.08282828
3	0.11836735
```

No header row. Iteration numbers are optimizer step counts. Values are `:.8` precision floats. Files are append-only (but the research loop deletes the `metrics_logs/` directory before each run to prevent stale data).

### train.log

Contains structured tracing output with targets:
- `gradient_debug` — per-head gradient norms, per-layer breakdown, total gradient norm
- `plateau_detection` — LR reduction events with full window dump
- `weight_decay` — L2 penalty values
- Default target — iteration progress, timing, stage transitions

### How the Research Loop Reads Metrics

The research loop's `parse_composite_score()` reads these specific metric files:
- `top1_accuracy.log` — averaged over last 100 values
- `wdl_accuracy.log` — averaged over last 100 values (stored as 0-1 fraction, mean probability of correct class)
- `aux_from_square_accuracy.log` — averaged over last 100 values
- `aux_to_square_accuracy.log` — averaged over last 100 values

The statistical test (`check_improvement()`) aligns these 4 metrics by step number, computes the composite score at each step, then pools the last 100 values across 2 seeds and runs a block bootstrap test.

---

## Evaluation & Scoring

### Composite Score

```
score = 1.0 × top1_accuracy + 0.5 × wdl_accuracy + 0.2 × aux_accuracy
```

Where:
- `top1_accuracy`: fraction (0–1), policy head's top-1 move prediction accuracy
- `wdl_accuracy`: fraction (0–1), mean probability assigned to the correct WDL class (rewards calibration, not just argmax correctness)
- `aux_accuracy`: `(aux_from_square_accuracy + aux_to_square_accuracy) / 2`, fraction (0–1)

Each component is averaged over the last `METRIC_WINDOW = 100` logged values. Each experiment runs 2 seeds; the reported score is the mean across seeds.

### Acceptance Criterion

A change is **kept** only if BOTH conditions hold:
1. **Block bootstrap test**: `p < 0.05` (autocorrelation-corrected — consecutive training steps are correlated, so we resample in blocks of 20)
2. **Cohen's d**: `d >= 0.3` (the improvement must be at least ~0.3 pooled standard deviations)

The test pools per-step composite score arrays across both seeds, then compares the new run vs the current best run.

---

## Hyperparameter Reference

### Full Config Defaults (with research loop overrides marked)

```
# Architecture
embed_dim               = 384          # ⚠️ RESEARCH: 192
num_layers              = 24           # ⚠️ RESEARCH: 12
num_heads               = 8
conv_layers             = 0
smolgen_hidden          = 24
smolgen_global_dim      = 128
smolgen_gen_size        = 128
value_tower_layers      = 2

# Training
physical_batch_size     = 16000        # ⚠️ RESEARCH: 512
batch_size              = None         # when None, effective_batch_size = physical_batch_size (no grad accumulation)
seed                    = 42
num_workers             = 4
num_devices             = 1
shuffle_buffer_size     = 100000

# Loss Weights (initial — GradNorm adjusts these)
policy_loss_weight      = 0.30
value_loss_weight       = 0.0001
time_usage_loss_weight  = 0.0          # disabled
aux_loss_weight         = 0.06

# Loss Config
policy_label_smoothing  = 0.005
focal_loss_gamma        = 0.0          # disabled (standard CE)
value_entropy_weight    = 0.0          # disabled

# Learning Rates (base, before μP scaling)
muon_base_lr            = 0.0225
adamw_base_lr           = 3.375e-4
embedding_base_lr       = 0.1125
lr_multiplier           = 1.0

# LR Scheduling
warmup_multiplier       = 2.0          # ⚠️ RESEARCH: 0.1
lr_window_size          = 120
lr_improvement_threshold = 0.02
lr_reduction_factor     = 0.5
lr_min                  = 0.000001

# Regularization
weight_decay            = 0.003
cautious_weight_decay   = true
gradient_clip           = 3.0
adam_epsilon            = 1e-8

# GradNorm
enable_gradnorm         = true
gradnorm_interval       = 20           # ⚠️ RESEARCH: 200
gradnorm_alpha          = 0.5
gradnorm_learning_rate  = 0.1
gradnorm_policy_priority = 5.0
gradnorm_value_priority = 1.0
gradnorm_time_priority  = 1.0
gradnorm_aux_priority   = 1.5
gradnorm_probe_size     = 256

# Value Head
value_backprop_to_trunk = false
value_ply_ramp_start    = 10
value_ply_ramp_full     = 30
value_train_on_puzzles  = false

# Data Sampling
enable_ply_sampling     = true
enable_elo_sampling     = true
elo_priority_boost      = 3.0
puzzle_sampling_ratio   = 0.05

# Metrics & Checkpointing
full_metrics_interval   = 50           # ⚠️ RESEARCH: 200
checkpoint_interval     = 100          # ⚠️ RESEARCH: 0 (disabled)
log_gradient_breakdown  = false        # ⚠️ RESEARCH: true
```

---

## Key Source Files

| File | Contents |
|------|----------|
| `src/config.rs` | All hyperparameter definitions, constants (`FEATURES_PER_TOKEN`, `LEGAL_MOVES`, etc.), CLI argument parsing |
| `src/model.rs` | Main `OXIModel` — forward pass, loss computation, all output heads, loss functions |
| `src/factorized_policy.rs` | Factorized policy head (source×target dot product + promotion handling) |
| `src/relative_position_transformer.rs` | Transformer blocks, SmolgenAttention, FiLM-RmsNorm, SwiGLU MLP |
| `src/value_tower.rs` | Value tower struct (exists but not instantiated — actual value head uses a single block in model.rs) |
| `src/spatial_conv.rs` | Optional 3×3 spatial convolution |
| `src/gradnorm.rs` | GradNorm multi-task adaptive weighting |
| `src/reduce_on_plateau_scheduler.rs` | LR scheduler with linear regression plateau detection |
| `src/dataset.rs` | Dataset loading, `ChessItem`/`ChessBatch`, batching |
| `src/encoding.rs` | Position encoding — all 61+4 per-square features |
| `src/move_encoding.rs` | Move → (from_square, target_index) encoding |
| `src/inference.rs` | Inference engine, global feature computation, material imbalance |
| `src/custom_training.rs` | Training loop, gradient accumulation, multi-optimizer stepping, metrics logging |
| `src/main.rs` | Entry point, CLI dispatch, device setup |
| `src/lib.rs` | Module declarations |
| `research_loop.py` | Research orchestration — spawns subagents, runs training, evaluates, keeps/discards |
| `research_log.md` | Experiment history with scores |
| `research/notes.md` | Ad-hoc research notes from previous agents |