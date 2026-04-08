# Oxi Architecture Reference (for Research Agents)

This document describes the Oxi training stack as it actually exists in the current codebase used by `research_loop.py`. It covers the model architecture, input representation, output heads, loss functions, GradNorm setup, optimizer configuration, data pipeline, centipawn-loss calibration path, metrics logging, and evaluation scoring.

If this file and an older research note disagree, trust the code and update this file.

---

## Table of Contents

1. [Research Loop Context](#research-loop-context)
2. [Input Representation](#input-representation)
3. [Model Architecture](#model-architecture)
4. [Output Heads](#output-heads)
5. [Loss Functions](#loss-functions)
6. [GradNorm](#gradnorm)
7. [Optimizer \& Learning Rate](#optimizer--learning-rate)
8. [LR Scheduling](#lr-scheduling)
9. [Data Pipeline](#data-pipeline)
10. [Centipawn-Loss Calibration](#centipawn-loss-calibration)
11. [Training Loop Details](#training-loop-details)
12. [Metrics Logging \& File Format](#metrics-logging--file-format)
13. [Evaluation \& Scoring](#evaluation--scoring)
14. [Hyperparameter Reference](#hyperparameter-reference)
15. [Key Source Files](#key-source-files)

---

## Research Loop Context

The research loop (`research_loop.py`) proposes a single code change, compiles it, builds a release binary, runs one 60-minute training job, and decides whether to keep the change by comparing the new run against the current best run.

The standard research training command is:

```bash
./target/release/oxi train \
  --pretrain-samples=0 \
  --data-path=../data \
  --physical-batch-size=512 \
  --seed=<varies> \
  --log-dir=<run_dir> \
  --disable-tui \
  --warmup-multiplier=0.1 \
  --log-gradient-breakdown \
  --full-metrics-interval=200 \
  --gradnorm-interval=200 \
  --checkpoint-interval=0
```

Key overrides from code defaults:

| Parameter | Research Loop Value | Code Default |
|-----------|---------------------|--------------|
| `physical_batch_size` | **512** | 16000 |
| `warmup_multiplier` | **0.1** | 2.0 |
| `full_metrics_interval` | **200** | 50 |
| `gradnorm_interval` | **200** | 20 |
| `checkpoint_interval` | **0** | 100 |
| `--disable-tui` | yes | no |
| `--log-gradient-breakdown` | yes | no |
| `pretrain_samples` | **0** | 0 |

Architecture is not pinned by CLI. The research loop relies on `Config::default()` in `src/config.rs`, which currently means:

| Parameter | Current Default |
|-----------|-----------------|
| `embed_dim` | 192 |
| `num_layers` | 12 |
| `num_heads` | 6 |
| `conv_layers` | 0 |

The research loop score now includes centipawn-loss calibration in addition to move/value/aux metrics. That calibration component is coverage-gated so runs with too little labeled calibration data do not get over-rewarded.

---

## Input Representation

### Board Encoding: 64 squares × 65 features per square

Each of the 64 squares gets a feature vector of length `FEATURES_PER_TOKEN = 65`.

`BOARD_FEATURES_PER_TOKEN = 61`:

| Group | Count | Notes |
|-------|-------|-------|
| Piece identity | 12 | one-hot for white/black piece type |
| Tactical | 22 | attacker roles/counts/material, pin flags, hanging flag, square control |
| Positional | 25 | local mobility, pawn structure, weak squares, open file, passed pawn, rank/file one-hots |
| Misc | 2 | en passant target, local castling right |

`RECENCY_FEATURES = 4` are appended after the 61 board features:

| Feature | Meaning |
|---------|---------|
| White from heatmap | decayed move-from history |
| White to heatmap | decayed move-to history |
| Black from heatmap | decayed move-from history |
| Black to heatmap | decayed move-to history |

Recency uses `PREVIOUS_POSITIONS = 5` and `HISTORY_DECAY = 0.8`.

### Global Features: `NUM_GLOBALS = 10`

These are per-position features used for FiLM conditioning and head readouts:

| Index | Feature |
|-------|---------|
| 0 | self time remaining, normalized |
| 1 | self time / base time |
| 2 | opponent time remaining, normalized |
| 3 | opponent time / base time |
| 4 | increment / base time |
| 5 | move count, normalized |
| 6 | self Elo, normalized |
| 7 | puzzle flag |
| 8 | material imbalance, normalized |
| 9 | total pieces / 32 |

### Move Encoding: `LEGAL_MOVES = 64 × 76 = 4864`

Moves are encoded as `(from_square, target_index)`:

- `0..63`: ordinary target squares
- `64..75`: promotion targets

The flat move index is `from_square * 76 + target_index`.

---

## Model Architecture

### Overview

Oxi is a pre-norm transformer over 64 square tokens with:

- FiLM-conditioned RMSNorm using the 10 global features
- Smolgen-generated attention biases
- SwiGLU MLP blocks
- one extra transformer block for the policy path
- one extra transformer block for the value path

High-level structure:

```text
board_input [B, 64, 65]
  -> split board features / recency
  -> token embed + square embeddings + recency concat
  -> trunk transformer blocks (num_layers)
  -> RMSNorm
  -> policy block -> policy tokens -> policy head
  -> value block  -> value tokens  -> value head
  -> trunk/policy pooled features -> aux heads + CP-loss head
```

For current defaults:

- `embed_dim = 192`
- `num_layers = 12`
- `num_heads = 6`
- `head_dim = 32`
- total transformer blocks on the active forward path = 14

### Learnable Square Embeddings

A `[64, base_embed_dim]` parameter tensor is added to token embeddings before recency channels are concatenated.

### Spatial Convolution

`conv_layers` exists, but the current default is `0`, so research-loop runs do not use spatial convolution unless code or config defaults are changed.

### Transformer Block

Each block has:

1. FiLM-RMSNorm
2. Smolgen attention
3. residual add
4. FiLM-RMSNorm
5. SwiGLU MLP
6. residual add

#### FiLM-RMSNorm

The global feature vector modulates normalized token activations:

```text
gamma = gamma_proj(globals) + 1
beta  = beta_proj(globals)
out   = rms_norm(x) * gamma + beta
```

#### Smolgen Attention

Attention uses standard QKV projections plus a dynamically generated position-dependent bias. The Smolgen generator is shared across layers and derives bias tensors from token activations.

#### SwiGLU MLP

The MLP uses a fused gate/up projection, SiLU gating, and a linear projection back to model width.

### Initialization

- most weights: Normal(0, 0.02)
- residual projections: scaled down by layer-count-dependent init
- Smolgen generator weights: smaller init
- square embeddings: Normal(0, 0.02)

---

## Output Heads

### 1. Policy Head

Output shape: `[batch, 4864]`

The policy head uses factorized source/target projections over `policy_tokens`, with separate promotion handling. Illegal moves are masked before softmax.

This is still the primary output of the model.

### 2. Value Head (WDL)

Output shape: `[batch, 3]`

The value head applies one extra transformer block to trunk output, then attention-pools over squares and predicts win/draw/loss logits.

Important implementation note:

- `value_tower_layers` exists in config for future or alternate implementations
- the active model path still uses a single dedicated `value_block` in `src/model.rs`

Value examples are weighted by a ply ramp:

- zero before `value_ply_ramp_start = 10`
- linearly ramps up
- full weight by `value_ply_ramp_full = 30`

### 3. Time Usage Head

This is effectively disabled in the current regime:

- the config default `time_usage_loss_weight` is `0.0`
- research-loop runs do not use it for scoring

The plumbing still exists, but it is not an important active training signal right now.

### 4. Side-Info Head

Output shape: `[batch, 141]`

Predicts:

- moved piece type
- captured piece type
- check flag
- from-square
- to-square

This is trained with BCE and contributes to the auxiliary loss bundle.

### 5. Auxiliary Heads

These are all grouped into the auxiliary task:

- mobility head: per-square legal-move count prediction
- material head: material imbalance prediction
- Maia-style from-square head
- Maia-style to-square head
- trunk-level from-square head
- trunk-level to-square head

`aux_from_square_accuracy` and `aux_to_square_accuracy` are still part of the research score.

### 6. Centipawn-Loss Calibration Head

This is the major recent addition.

Output shape: `[batch, 8]`

The head predicts a categorical distribution over human centipawn-loss bins:

| Bin | Label |
|-----|-------|
| 0 | `0` |
| 1 | `1-10` |
| 2 | `11-25` |
| 3 | `26-50` |
| 4 | `51-100` |
| 5 | `101-200` |
| 6 | `201-400` |
| 7 | `400+` |

This head is trained only on positions that have precomputed Stockfish labels in the calibration database.

It is not the main behavioral target by itself; it is an auxiliary readout used alongside direct policy-side CPL supervision.

---

## Loss Functions

### Policy Loss

Primary move-prediction loss over legal moves only:

- label smoothing over legal moves
- optional focal loss support, though the current default is standard CE (`focal_loss_gamma = 0.0`)

### Value Loss

Cross-entropy over WDL logits, multiplied by per-example value weights from the ply ramp.

### Time Usage Loss

Present in the plumbing but effectively zeroed in current practice because `time_usage_loss_weight = 0.0`.

### Auxiliary Loss

A single bundled auxiliary task containing:

- mobility loss
- material loss
- side-info BCE
- policy-token from/to CE
- trunk-token from/to CE

### Centipawn-Loss Calibration Loss

This is active only on examples with calibration labels.

For labeled positions, the model computes:

1. `head_ce`
   cross-entropy for the 8-bin CPL head
2. `head_mae`
   MAE between the head's expected CPL and the human target CPL
3. `policy_mae`
   MAE between the policy-implied expected CPL and the human target CPL

Policy-implied expected CPL is computed from:

```text
policy_probs * calibration_move_cp_losses
```

summed over the full 4864-move output space. The move-loss tensor is dense in the batch, though it is derived from a sparse blob in the SQLite cache.

Current base calibration loss in `src/model.rs`:

```text
base_calibration_loss =
    head_ce
  + 0.01  * policy_mae
  + 0.005 * head_mae
```

That base loss is then multiplied by `calibration_loss_weight` and also participates in GradNorm as its own task.

### Total Loss

Conceptually:

```text
total_loss =
    weighted_policy
  + weighted_value
  + weighted_time_usage
  + weighted_aux
  + weighted_calibration
```

Where the runtime weights come from GradNorm, not just the static config defaults.

---

## GradNorm

GradNorm now balances **five** tasks:

1. policy
2. value
3. time usage
4. auxiliary
5. calibration

That calibration task was a recent addition. It is skipped for GradNorm probing on batches with zero labeled calibration examples.

### Current Defaults

| Parameter | Value |
|-----------|-------|
| `enable_gradnorm` | true |
| `gradnorm_interval` | 20 by default, **200 in research loop** |
| `gradnorm_alpha` | 0.5 |
| `gradnorm_learning_rate` | 0.1 |
| `gradnorm_policy_priority` | 5.0 |
| `gradnorm_value_priority` | 1.0 |
| `gradnorm_time_priority` | 1.0 |
| `gradnorm_aux_priority` | 1.5 |
| `gradnorm_calibration_priority` | 2.0 |
| `gradnorm_probe_size` | 256 |

Important detail:

- the calibration task is considered inactive for a probe batch if `calibration_labeled_fraction == 0`
- this avoids backward-on-zero-task panics and keeps calibration from skewing GradNorm when labels are absent

---

## Optimizer & Learning Rate

### Parameter Groups

The model uses multiple optimizer groups:

- Muon for 2D+ weight matrices when enabled
- AdamW with decay for standard params
- AdamW with higher LR for embedding-related params
- AdamW without decay for norm/bias params
- AdamW without decay at embedding LR for embedding-related bias params

### Learning Rate Scaling

Learning rates use μP-style width scaling with `d=256` as the reference width.

Base learning rates:

| Parameter | Value |
|-----------|-------|
| `muon_base_lr` | 0.0225 |
| `adamw_base_lr` | 3.375e-4 |
| `embedding_base_lr` | 0.1125 |

Other defaults:

| Parameter | Value |
|-----------|-------|
| `weight_decay` | 0.003 |
| `adam_epsilon` | 1e-8 |
| `gradient_clip` | 3.0 |
| `cautious_weight_decay` | true |

---

## LR Scheduling

Oxi still uses the reduce-on-plateau scheduler based on linear regression over a sliding loss window.

Defaults:

| Parameter | Value |
|-----------|-------|
| `lr_window_size` | 120 |
| `lr_improvement_threshold` | 0.02 |
| `lr_reduction_factor` | 0.5 |
| `lr_min` | 1e-6 |
| `warmup_multiplier` | 2.0 default, **0.1 in research loop** |

Warmup length is:

```text
warmup_iterations = warmup_multiplier * effective_batch_size
```

measured in optimizer steps.

### Two-Stage Training

The value-tower-only second stage still exists:

- when the scheduler bottoms out and plateaus again, training can switch into value-only mode
- policy loss is disabled
- puzzles are skipped
- GradNorm probing is disabled because only one task remains active

This also can be triggered immediately with `--skip-policy-loss`.

---

## Data Pipeline

### Shared Human Training Stream

Recent calibration work introduced a shared stream path in `src/training_stream.rs`.

Both:

- normal training
- ordered calibration DB generation

now consume the same human training stream abstraction, which prevents the old mismatch where calibration labels and training examples were coming from subtly different iterator orders or conventions.

### Human Data

Training streams human PGN data from `--data-path`, which is expected to be a directory.

Examples are represented as `ChessExample` and contain:

- FEN
- move UCI
- outcome
- clock features
- Elo ratings
- move count
- recent-history data
- puzzle flag

### Puzzle Mixing

If enabled, puzzle examples are mixed into the human stream at:

- `puzzle_sampling_ratio = 0.05` by default

In value-only mode, puzzle mixing is disabled.

### Sampling / Filtering

Defaults:

- ply sampling enabled
- Elo sampling enabled
- `elo_priority_boost = 3.0`
- minimum clock time filter still applies

The calibration builder can also run in an ordered mode using a config with:

- ply sampling disabled
- Elo sampling disabled

to align exactly with deterministic sanity checks.

### Shuffle Buffer

Training uses a `ShuffleBuffer` with:

- `shuffle_buffer_size = 100000`

By default, the buffer is randomized. A recent debug feature added:

- `--disable-training-shuffle`

which drains the buffer in stream order instead. This is mainly useful for deterministic calibration overlap tests.

### Batch Construction

A `ChessBatch` contains the usual tensors plus the new optional calibration tensors:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `board_input` | `[B, 64, 65]` | board + recency features |
| `move_distributions` | `[B, 4864]` | target move distribution |
| `legal_moves` | `[B, 4864]` | legal move mask |
| `values` | `[B, 3]` | WDL target |
| `global_features` | `[B, 10]` | FiLM/global features |
| `time_usages` | `[B, 1]` | mostly inactive target |
| `value_weights` | `[B]` | ply-ramped value supervision |
| `material_imbalance` | `[B]` | material target |
| `side_info` | `[B, 141]` | side-info head targets |
| `calibration_mask` | `[B]` | 1 if Stockfish label exists |
| `calibration_target_cp_loss` | `[B]` | human move CPL target |
| `calibration_target_bins` | `[B, 8]` | one-hot CPL-bin target |
| `calibration_move_cp_losses` | `[B, 4864]` | dense per-move CPL tensor |

Only a subset of positions carry calibration labels, so calibration supervision is sparse relative to the full policy dataset.

---

## Centipawn-Loss Calibration

### Purpose

The calibration path exists to address a specific failure mode:

- strong move prediction does not guarantee human-like mistake severity
- when the top-1 move is wrong, the model can still be too strong or too weak in unrealistic ways

The goal is not “maximize strength.” The goal is “match human error level.”

### Stored Labels

Precomputed labels are stored in a SQLite database, usually:

```text
<data-path>/calibration.db
```

Training will automatically use that path if `--calibration-db-path` is not explicitly set and the DB exists under `data_path`.

The DB stores, per labeled position:

- FEN
- human move
- player Elo
- opponent Elo
- ply
- stage
- Stockfish depth
- best evaluation
- human CPL
- regret bin
- sparse move-loss blob

The storage is sparse, but the training batcher expands the move-loss blob to a dense `[4864]` tensor per labeled example.

### Cache Key

At training lookup time, calibration labels are keyed by:

- `fen`
- `human_move`

This is intentionally looser than the full DB uniqueness constraint and matches the fact that the move-loss vector is a property of the position and move, not the Elo bucket.

### Regret / CPL Bins

The code still uses the internal name `RegretBin`, but functionally this is a centipawn-loss bucketization for the human move:

- `0`
- `1-10`
- `11-25`
- `26-50`
- `51-100`
- `101-200`
- `201-400`
- `400+`

Representative centers are used to convert the head distribution into an expected CPL estimate.

### Calibration Metrics

The training loop logs:

- `cp_loss_policy_mae`
- `cp_loss_head_mae`
- `cp_loss_head_ce`
- `cp_loss_labeled_fraction`
- `cp_loss_calibration_overall`

It also logs coarse signed calibration errors by Elo skill band:

- `cp_loss_calibration_beginner`
- `cp_loss_calibration_intermediate`
- `cp_loss_calibration_expert`

Those band metrics are clipped to `[-200, 200]` centipawns in the renderer/logging path for chart readability.

`cp_loss_calibration_overall` is the main bounded research-loop metric. It maps absolute CPL error to a `0..1` score with a 200cp cap.

---

## Training Loop Details

### Main Loop

At a high level:

1. build the streamed human iterator
2. optionally mix puzzles
3. fill the shuffle buffer
4. sample or drain examples from the buffer
5. batch and tensorize examples
6. split across worker/device threads
7. forward pass
8. backward pass
9. gradient clip
10. optimizer step
11. scheduler update
12. metrics logging

### Effective Batch Size

If `batch_size` is `None`, effective batch size equals `physical_batch_size`. This is the standard research-loop regime, so the effective batch size is normally 512.

### Calibration Coverage

Calibration labels are optional per example, so:

- some batches may have zero labeled examples
- some runs may have low calibration coverage
- calibration metrics are only emitted when at least one labeled example is present in the aggregated output

The training loop logs `calibration_batch_stats` in `train.log` to make labeled coverage visible during live runs.

---

## Metrics Logging & File Format

### Directory Structure

```text
<log_dir>/
├── train.log
├── stderr.log                  # research loop / launcher dependent
├── plateau_detection.log
└── metrics_logs/
    ├── total_loss.log
    ├── policy_loss.log
    ├── value_loss.log
    ├── aux_from_square_accuracy.log
    ├── aux_to_square_accuracy.log
    ├── top1_accuracy.log
    ├── wdl_accuracy.log
    ├── cp_loss_policy_mae.log
    ├── cp_loss_head_mae.log
    ├── cp_loss_head_ce.log
    ├── cp_loss_labeled_fraction.log
    ├── cp_loss_calibration_overall.log
    ├── cp_loss_calibration_beginner.log
    ├── cp_loss_calibration_intermediate.log
    ├── cp_loss_calibration_expert.log
    └── ...
```

### File Format

Each metric log is TSV with:

```text
iteration<TAB>value
```

There is no header row.

### TUI / CLI Renderer

The current training renderer exposes:

- the classic move/value/aux charts
- a dedicated `Centipawn Loss Calibration` group with:
  - `Policy MAE`
  - `Head MAE`
  - `Head CE`
  - `Labeled Fraction`
  - `Overall`
- a separate coarse Elo-band chart:
  - `CP Loss Calibration By Elo|Beginner`
  - `CP Loss Calibration By Elo|Intermediate`
  - `CP Loss Calibration By Elo|Expert`

The live monitor (`monitor.py`) also includes `cp_loss_calibration_overall`.

---

## Evaluation & Scoring

### Research Loop Composite Score

`research_loop.py` now scores four components:

```text
score =
    1.0 * top1_accuracy
  + 0.5 * normalized_wdl_accuracy
  + 0.2 * aux_accuracy
  + 0.4 * calibration_score
```

Where:

- `top1_accuracy` is the move top-1 hit rate
- `normalized_wdl_accuracy` is `wdl_accuracy`, normalized to `0..1` if logged as percentages
- `aux_accuracy = (from_sq_acc + to_sq_acc) / 2`
- `calibration_score = cp_loss_calibration_overall * coverage_factor`

Coverage factor is derived from `cp_loss_labeled_fraction` and the number of recent steps with enough calibration coverage:

- minimum labeled fraction per valid step: `0.05`
- target labeled fraction: `0.10`
- minimum valid steps in the window: `20`

This means calibration helps only when the run has enough usable labeled examples.

### Windowing

Most research-loop comparisons use the last:

- `METRIC_WINDOW = 100`

logged values.

### Statistical Acceptance

A run is accepted only if both hold:

1. block bootstrap `p < 0.05`
2. Cohen's `d >= 0.3`

The bootstrap is done over the composite score sequence, with block resampling to account for autocorrelation.

### Monitor

`monitor.py` mirrors the calibration-aware composite:

- it shows `CPL Calibration`
- it uses the same calibration coverage gating
- it can compare the current run to the best logged run

---

## Hyperparameter Reference

Current relevant defaults in `Config::default()`:

```text
# Architecture
embed_dim                    = 192
num_layers                   = 12
num_heads                    = 6
conv_layers                  = 0

# Training / batching
physical_batch_size          = 16000
batch_size                   = None
shuffle_buffer_size          = 100000
disable_training_shuffle     = false
num_workers                  = 4
num_devices                  = 1

# Loss weights (initial values; GradNorm adapts them)
policy_loss_weight           = 0.30
value_loss_weight            = 0.10
time_usage_loss_weight       = 0.0
aux_loss_weight              = 0.06
calibration_loss_weight      = 0.10

# GradNorm priorities
gradnorm_policy_priority     = 5.0
gradnorm_value_priority      = 1.0
gradnorm_time_priority       = 1.0
gradnorm_aux_priority        = 1.5
gradnorm_calibration_priority = 2.0

# Policy loss config
policy_label_smoothing       = 0.005
focal_loss_gamma             = 0.0

# Value config
value_ply_ramp_start         = 10
value_ply_ramp_full          = 30
value_tower_layers           = 2    # config exists; active model path still uses one value block

# Optimizer / regularization
muon_base_lr                 = 0.0225
adamw_base_lr                = 3.375e-4
embedding_base_lr            = 0.1125
weight_decay                 = 0.003
gradient_clip                = 3.0
cautious_weight_decay        = true

# Scheduler
warmup_multiplier            = 2.0
lr_window_size               = 120
lr_improvement_threshold     = 0.02
lr_reduction_factor          = 0.5
lr_min                       = 1e-6

# Data sampling
enable_ply_sampling          = true
enable_elo_sampling          = true
elo_priority_boost           = 3.0
puzzle_sampling_ratio        = 0.05

# Logging / checkpoints
full_metrics_interval        = 50
checkpoint_interval          = 100

# Calibration
calibration_db_path          = <data-path>/calibration.db if present
```

Research-loop overrides mainly affect:

- batch size
- warmup
- metrics cadence
- checkpointing
- TUI

They do not currently override calibration-specific defaults.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/config.rs` | core defaults, CLI overrides, runtime config accessors |
| `src/model.rs` | main forward pass, all heads, all loss construction including CPL calibration |
| `src/factorized_policy.rs` | factorized policy head |
| `src/relative_position_transformer.rs` | transformer blocks, Smolgen attention, FiLM-RMSNorm, SwiGLU |
| `src/gradnorm.rs` | five-task GradNorm logic including calibration |
| `src/custom_training.rs` | training loop, worker orchestration, batching, logging, rendering |
| `src/dataset.rs` | `ChessExample`, `ChessBatch`, calibration label lookup, dense CPL batch tensors |
| `src/training_stream.rs` | shared human-training stream used by training and ordered calibration generation |
| `src/calibration.rs` | SQLite calibration DB, CPL bins, sparse move-loss blobs, labels |
| `src/pgn_processor.rs` | PGN streaming and deterministic file iteration |
| `src/stockfish.rs` | Stockfish integration for offline labeling |
| `src/main.rs` | CLI entrypoints including `train` and `create-calibration-db` |
| `research_loop.py` | automated research orchestration and calibration-aware scoring |
| `monitor.py` | live TUI/CLI monitor with calibration charting |
| `docs/calibrated-strength-spec.md` | design spec for the CPL calibration feature |

