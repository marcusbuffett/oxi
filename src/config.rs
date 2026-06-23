use clap::{Args, ValueEnum};
use once_cell::sync::Lazy;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Normal};
use std::sync::OnceLock;

pub const NUM_GLOBALS: usize = 11;
pub const LEGAL_MOVES: usize = 64 * 76;
// Per-square feature layout. Pruned aggressively after the 2026-06 channel
// ablation study (see docs/feature_ablation_2026_06.md): pins, pinned
// defender, pawn structure flags, weak squares, open file, passed pawn,
// dark square, and en passant were all measured near zero reliance on the
// trained model and removed.
// - Piece identity group (12): white/black one-hots for all roles
// - Tactical group (24): attackers by role + material + count for both
//   colors (8 each), hanging flag, square control, per-side SEE capture
//   outcomes (2), per-side x-ray attacker count + material (4)
// - Positional group (17): legal-move count, rank one-hot, file one-hot
// - Misc group (1): local castling right
// - Recency channels (4): white/black from/to of recent moves, decayed
// - History occupancy (7x12): piece one-hots of the past 7 positions,
//   most recent first (Maia-3-style board history)
pub const PIECE_IDENTITY_FEATURES: usize = 12;
pub const TACTICAL_FEATURES: usize = 24;
pub const POSITIONAL_FEATURES: usize = 17;
pub const MISC_FEATURES: usize = 1;
pub const RECENCY_FEATURES: usize = 4; // white_from, white_to, black_from, black_to
pub const PREVIOUS_POSITIONS: usize = 7; // history horizon (occupancy planes + recency decay)
pub const HISTORY_OCCUPANCY_FEATURES: usize = PREVIOUS_POSITIONS * PIECE_IDENTITY_FEATURES;

pub const BOARD_FEATURES_PER_TOKEN: usize =
    PIECE_IDENTITY_FEATURES + TACTICAL_FEATURES + POSITIONAL_FEATURES + MISC_FEATURES;
pub const FEATURES_PER_SQUARE_POSITION: usize = BOARD_FEATURES_PER_TOKEN;
pub const FEATURES_PER_TOKEN: usize =
    BOARD_FEATURES_PER_TOKEN + RECENCY_FEATURES + HISTORY_OCCUPANCY_FEATURES;
pub const HISTORY_DECAY: f32 = 0.8; // Exponential decay factor for historical positions

// Global config storage
static GLOBAL_CONFIG: OnceLock<Config> = OnceLock::new();

/// Minimum Elo rating for both players to include games
pub const MIN_ELO: i32 = 1000;
pub const MIN_PLY: usize = 0;
pub const MAX_ELO: i32 = 3000;
pub const MAX_ELO_DIFF: i32 = 200;
/// Games qualify only with base time above this (seconds). Excludes bullet
/// and short blitz like 2+1, whose time-scramble moves aren't worth modeling.
pub const MIN_TIME_CONTROL: u32 = 121;

/// Minimum clock time (in seconds) to include moves
pub const MIN_CLOCK_TIME: u32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelSize {
    #[default]
    Full,
    Mini,
}

/// Unified configuration for OXI chess engine training and inference.
/// Defaults are defined in the `Default` impl. Use `ConfigOverrides` for CLI parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Named model-size preset used to create this config. Saved into
    /// params.json so serving/deploy tooling can distinguish full vs mini.
    #[serde(default)]
    pub model_size: ModelSize,

    // === DATA AND RUNTIME ===
    /// Path to data (PGN directory, PGN file, or CSV file)
    pub data_path: Option<std::path::PathBuf>,

    /// Directory for training logs (train.log, metrics_logs/)
    pub log_dir: Option<std::path::PathBuf>,

    pub max_samples: Option<usize>,

    /// Number of initial samples to skip during PGN processing
    pub skip: Option<usize>,

    pub timeout: Option<u64>,

    /// Resume training from the last saved model checkpoint
    #[serde(default)]
    pub resume: Option<bool>,

    // We only do one pass through the data, so this is mostly unused
    pub train_ratio: f32,

    /// Batch size for training
    pub batch_size: Option<usize>,

    /// Physical batch size (for gradient accumulation).
    /// 0 = auto: derive from model parameter count (see auto_physical_batch_size).
    pub physical_batch_size: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Number of data loader workers
    pub num_workers: usize,

    /// Final learning rate at the end of the WSD decay phase
    pub lr_min: f64,

    /// WSD schedule: fraction of the run budget (--timeout or --max-samples)
    /// spent in the final linear decay phase. The rest of the run after warmup
    /// holds LR constant.
    #[serde(default = "default_wsd_decay_fraction")]
    pub wsd_decay_fraction: f64,

    /// Hold LR constant indefinitely (no decay phase). For open-ended runs
    /// without a budget; anneal later by resuming the checkpoint with a budget.
    #[serde(default)]
    pub lr_hold: Option<bool>,

    /// Multiplier applied to the learning rate calculated from batch size
    pub lr_multiplier: f64,

    /// Base learning rate for Muon optimizer (at d=256, before batch-size scaling)
    /// Scales as sqrt(256 / embed_dim) when embed_dim changes.
    #[serde(default = "default_muon_base_lr")]
    pub muon_base_lr: f64,

    /// Base learning rate for AdamW optimizer (at d=256, before batch-size scaling)
    /// Scales as 256 / embed_dim when embed_dim changes.
    #[serde(default = "default_adamw_base_lr")]
    pub adamw_base_lr: f64,

    /// Base learning rate for embedding parameters (width-independent per μP)
    /// Does NOT scale with embed_dim — set once and it transfers across widths.
    #[serde(default = "default_embedding_base_lr")]
    pub embedding_base_lr: f64,

    /// Fraction of the run budget spent warming up. Uses --max-samples and/or
    /// --timeout progress, so it is stable across hardware speed.
    #[serde(default = "default_warmup_fraction")]
    pub warmup_fraction: f64,

    /// Weight for policy loss
    pub policy_loss_weight: f32,

    /// Label smoothing applied to the policy targets (smoothed over legal moves only)
    pub policy_label_smoothing: f32,

    /// Weight for value loss
    pub value_loss_weight: f32,

    /// Entropy regularization weight for value predictions (0.0 recommended for decoupled value tower)
    pub value_entropy_weight: f32,

    pub time_usage_loss_weight: f32,

    /// Weight decay for optimizer
    pub weight_decay: f64,

    /// Enable cautious weight decay (only applies decay when gradient and update align)
    #[serde(default)]
    pub cautious_weight_decay: Option<bool>,

    /// Adam epsilon for numerical stability
    pub adam_epsilon: f32,

    /// Gradient clipping norm (0 to disable)
    pub gradient_clip: f64,

    /// Enable verbose gradient norm breakdown logging
    #[serde(default)]
    pub log_gradient_breakdown: Option<bool>,

    /// Number of attention heads to include when logging gradient breakdowns
    #[serde(default)]
    pub gradient_head_limit: usize,

    /// Number of layers/modules to include when logging gradient breakdowns
    #[serde(default)]
    pub gradient_layer_limit: usize,

    /// Embedding dimension for tokens
    pub embed_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Number of attention heads (head_dim = embed_dim / num_heads)
    pub num_heads: usize,

    /// Number of convolutional layers applied over the 8x8 board grid before token embedding
    #[serde(default)]
    pub conv_layers: usize,

    /// Only include positions with a single legal move
    pub single_legal_move_only: Option<bool>,

    /// Disable terminal UI
    pub disable_tui: Option<bool>,

    /// Only include positions that are checkmate
    pub checkmate_only: Option<bool>,

    /// Probability of logging individual items for debugging (0.0 to 1.0)
    pub item_log_probability: f32,

    /// Enable detailed tensor norm logging during forward passes
    #[serde(default)]
    pub log_tensor_norms: Option<bool>,

    /// Maximum number of tensor elements to print when previewing small tensors
    #[serde(default)]
    pub norm_preview_limit: usize,

    /// Focal loss gamma parameter for policy head (0.0 disables focal loss)
    pub focal_loss_gamma: f32,

    pub smolgen_hidden: usize,

    pub smolgen_global_dim: usize,

    pub smolgen_gen_size: usize,

    /// Enable forward pass timing instrumentation for profiling
    pub enable_forward_timing: Option<bool>,

    /// Sample interval for forward timing (time every Nth forward pass)
    pub forward_timing_interval: u64,

    pub num_devices: usize,

    /// Probability of dropping positions based on ply (80% at ply 0, 0% at ply 10+)
    pub enable_ply_sampling: Option<bool>,

    /// Maximum ply (inclusive) to include training samples for. Samples with
    /// `current_ply > max_ply` are skipped. `None` disables the upper bound.
    #[serde(default)]
    pub max_ply: Option<usize>,

    /// Probability of dropping games based on Elo (75% at 1000 Elo, 0% at 2000+ Elo)
    pub enable_elo_sampling: Option<bool>,

    /// Number of iterations between checkpoints
    pub checkpoint_interval: usize,

    /// Decay factor for the EMA copy of the weights kept as training
    /// instrumentation (and saved as model_ema.mpk at each checkpoint).
    #[serde(default = "default_ema_beta")]
    pub ema_beta: f64,

    /// Allie test set JSONL evaluated at every checkpoint with both the live
    /// and EMA weights (metrics: allie_top1_live / allie_top1_ema). The EMA
    /// curve shows real progress through the constant-LR noise floor.
    #[serde(default)]
    pub ema_eval_dataset: Option<std::path::PathBuf>,

    /// Number of games loaded from the eval dataset (~50 positions each)
    #[serde(default = "default_ema_eval_games")]
    pub ema_eval_games: usize,

    /// Number of TCEC (computer engine) samples to use for pretraining (0 to disable)
    pub pretrain_samples: usize,

    /// Size of shuffle buffer for streaming data loading (number of examples to buffer before sampling)
    pub shuffle_buffer_size: usize,

    /// Disable shuffle-buffer randomization and consume buffered examples in stream order.
    #[serde(default)]
    pub disable_training_shuffle: Option<bool>,

    /// Interval for computing expensive metrics (top-5 accuracy, debug predictions, gradient breakdown, L2 penalty, tensor norms). 0 = never, 1 = every iteration.
    pub full_metrics_interval: usize,

    /// Priority boost for advanced/expert ELO games (2000+). Value of 1.0 = 2x boost at 2500 ELO,
    /// 2.0 = 3x boost, 3.0 = 4x boost. Set to 0.0 to disable.
    pub elo_priority_boost: f64,

    /// Ratio of puzzle examples to mix into training (0.0 to 1.0). Default 0.05 = 5% puzzles.
    pub puzzle_sampling_ratio: f64,

    /// Path to puzzle CSV file (defaults to <data-path>/puzzles/lichess_db_puzzle.csv.zst)
    pub puzzle_path: Option<std::path::PathBuf>,

    /// Path to SQLite DB with precomputed centipawn-loss labels
    #[serde(default)]
    pub calibration_db_path: Option<std::path::PathBuf>,

    /// Weight for centipawn-loss calibration losses
    #[serde(default = "default_calibration_loss_weight")]
    pub calibration_loss_weight: f32,

    /// Weight for the per-move policy regret hinge:
    ///   L = (Σ_m policy(m) * max(0, cp_loss(m) - cp_loss(human))) averaged over
    ///   positions with calibration labels.
    /// Punishes probability mass placed on moves worse than the move the human
    /// actually played. The per-position `cp_loss(human)` target already encodes
    /// player skill, so no extra Elo scaling is applied.
    ///
    /// The hinge is measured in raw centipawns (~50-150 at steady state), so the
    /// weight is deliberately small to match the effective gradient magnitude of
    /// the existing calibration loss. Set to 0 to disable.
    #[serde(default = "default_policy_regret_loss_weight")]
    pub policy_regret_loss_weight: f32,

    /// Reference scale (centipawns) for the policy-regret hinge gate. Early in
    /// training the policy is near-uniform, so the hinge sits at hundreds of cp
    /// with gradients to match, crowding other tasks out of the gradient-clip
    /// budget. When the batch hinge exceeds this reference, the loss term is
    /// rescaled to the reference (gradients damped proportionally); at or below
    /// it the gate is identity, so steady-state behavior is unchanged.
    #[serde(default = "default_policy_regret_ref_cp")]
    pub policy_regret_ref_cp: f32,

    // === VALUE TOWER CONFIGURATION ===
    /// Number of transformer layers in the value tower (separate from trunk)
    #[serde(default = "default_value_tower_layers")]
    pub value_tower_layers: usize,

    /// Starting ply for value example weighting ramp (positions before this get 0 weight)
    #[serde(default = "default_value_ply_ramp_start")]
    pub value_ply_ramp_start: usize,

    /// Ending ply for value example weighting ramp (positions at or after this get full weight)
    #[serde(default = "default_value_ply_ramp_full")]
    pub value_ply_ramp_full: usize,

    /// Whether to train value head on puzzle positions
    #[serde(default)]
    pub value_train_on_puzzles: Option<bool>,

    /// Run LR range finder instead of training
    #[serde(default)]
    pub lr_range_finder: Option<bool>,

    /// Use Muon optimizer for 2D+ weight matrices (false = use AdamW for everything)
    #[serde(default = "default_use_muon")]
    pub use_muon: Option<bool>,

    /// Optimizer update for the Muon parameter group: "aurora" or "muon"
    #[serde(default = "default_muon_optimizer")]
    pub muon_optimizer: Option<String>,

    /// Aurora projection iterations for the Muon parameter group
    #[serde(default = "default_aurora_pp_iterations")]
    pub aurora_pp_iterations: usize,

    /// Aurora row-normalization damping exponent
    #[serde(default = "default_aurora_pp_beta")]
    pub aurora_pp_beta: f32,

    /// Muon LR adjustment function: "original" or "match_rms_adamw"
    #[serde(default)]
    pub muon_lr_adjust: Option<String>,

    /// Weight for auxiliary losses (mobility + material prediction)
    #[serde(default = "default_aux_loss_weight")]
    pub aux_loss_weight: f32,

    /// Enable bf16 mixed precision training (cast inputs to bf16, keep optimizer in f32)
    #[serde(default)]
    pub mixed_precision: Option<bool>,

    /// Compute whitening.json next to the trained model after training exits.
    #[serde(default = "default_whiten_after_training")]
    pub whiten_after_training: Option<bool>,

    /// Number of positions to use for the post-training whitening transform.
    #[serde(default = "default_whitening_positions")]
    pub whitening_positions: usize,

    /// Inference batch size for the post-training whitening pass.
    #[serde(default = "default_whitening_batch_size")]
    pub whitening_batch_size: usize,
}

// Serde default functions for backwards compatibility with older params.json files
fn default_value_tower_layers() -> usize {
    2
}
fn default_value_ply_ramp_start() -> usize {
    10
}
fn default_value_ply_ramp_full() -> usize {
    30
}
fn default_use_muon() -> Option<bool> {
    Some(true)
}
fn default_muon_optimizer() -> Option<String> {
    Some("aurora".to_string())
}
fn default_aurora_pp_iterations() -> usize {
    2
}
fn default_aurora_pp_beta() -> f32 {
    0.5
}
fn default_muon_base_lr() -> f64 {
    0.0225 // Was 0.015; increased 50% for faster convergence in limited-iteration regime
}
fn default_adamw_base_lr() -> f64 {
    3.375e-4 // Was 2.25e-4; increased 50% for faster convergence
}
fn default_embedding_base_lr() -> f64 {
    0.1125 // Was 0.075; increased 50% for faster convergence. Width-independent per μP.
}
fn default_warmup_fraction() -> f64 {
    0.02
}
fn default_whiten_after_training() -> Option<bool> {
    Some(true)
}
fn default_whitening_positions() -> usize {
    200_000
}
fn default_whitening_batch_size() -> usize {
    256
}
fn default_wsd_decay_fraction() -> f64 {
    0.15
}
fn default_ema_beta() -> f64 {
    0.999
}
fn default_ema_eval_games() -> usize {
    200
}
fn default_aux_loss_weight() -> f32 {
    0.06 // Was 0.04; increased to strengthen trunk-level auxiliary supervision signal
}
fn default_calibration_loss_weight() -> f32 {
    0.10
}
fn default_policy_regret_loss_weight() -> f32 {
    0.01
}
fn default_policy_regret_ref_cp() -> f32 {
    // Steady-state hinge is ~18-36cp, so 50 leaves converged training untouched
    // while damping the ~650cp near-uniform-policy phase by >10x.
    50.0
}
/// Command-line overrides for Config. All fields are optional.
/// Use `Config::with_overrides()` to merge with defaults.
#[derive(Debug, Clone, Args, Default)]
pub struct ConfigOverrides {
    /// Named model-size preset to start from before applying explicit
    /// overrides. `mini` is sized for low-latency embedding/policy inference.
    #[arg(long, value_enum)]
    pub model_size: Option<ModelSize>,

    /// Path to data (PGN directory, PGN file, or CSV file)
    #[arg(long)]
    pub data_path: Option<std::path::PathBuf>,

    /// Directory for training logs (train.log, metrics_logs/)
    #[arg(long)]
    pub log_dir: Option<std::path::PathBuf>,

    #[arg(long)]
    pub max_samples: Option<usize>,

    /// Number of initial samples to skip during PGN processing
    #[arg(long)]
    pub skip: Option<usize>,

    #[arg(long)]
    pub timeout: Option<u64>,

    /// Resume training from the last saved model checkpoint
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub resume: Option<bool>,

    /// Ratio of data used for training (vs validation)
    #[arg(long)]
    pub train_ratio: Option<f32>,

    /// Batch size for training
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Physical batch size (for gradient accumulation). 0 = auto-derive from
    /// model parameter count so larger models get a safe (smaller) batch.
    #[arg(long)]
    pub physical_batch_size: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of data loader workers
    #[arg(long)]
    pub num_workers: Option<usize>,

    /// Final learning rate at the end of the WSD decay phase
    #[arg(long)]
    pub lr_min: Option<f64>,

    /// WSD schedule: fraction of the run budget spent in the final decay phase
    #[arg(long)]
    pub wsd_decay_fraction: Option<f64>,

    /// Hold LR constant indefinitely (no decay phase; for open-ended runs)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub lr_hold: Option<bool>,

    /// Multiplier applied to the learning rate calculated from batch size
    #[arg(long)]
    pub lr_multiplier: Option<f64>,

    /// Base learning rate for Muon optimizer (at d=256, before batch-size scaling)
    #[arg(long)]
    pub muon_base_lr: Option<f64>,

    /// Base learning rate for AdamW optimizer (at d=256, before batch-size scaling)
    #[arg(long)]
    pub adamw_base_lr: Option<f64>,

    /// Base learning rate for embedding parameters (width-independent, no d-scaling)
    #[arg(long)]
    pub embedding_base_lr: Option<f64>,

    /// Run LR range finder test instead of training
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub lr_range_finder: Option<bool>,

    /// Fraction of run budget spent warming up. Uses --timeout/--max-samples progress.
    #[arg(long)]
    pub warmup_fraction: Option<f64>,

    /// Weight for policy loss
    #[arg(long)]
    pub policy_loss_weight: Option<f32>,

    /// Label smoothing applied to the policy targets
    #[arg(long)]
    pub policy_label_smoothing: Option<f32>,

    /// Weight for value loss
    #[arg(long)]
    pub value_loss_weight: Option<f32>,

    /// Entropy regularization weight for value predictions
    #[arg(long)]
    pub value_entropy_weight: Option<f32>,

    /// Weight for time usage loss
    #[arg(long)]
    pub time_usage_loss_weight: Option<f32>,

    /// Weight decay for optimizer
    #[arg(long)]
    pub weight_decay: Option<f64>,

    /// Enable cautious weight decay
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub cautious_weight_decay: Option<bool>,

    /// Gradient clipping norm (0 to disable)
    #[arg(long)]
    pub gradient_clip: Option<f64>,

    /// Enable verbose gradient norm breakdown logging
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub log_gradient_breakdown: Option<bool>,

    /// Number of attention heads to include when logging gradient breakdowns
    #[arg(long)]
    pub gradient_head_limit: Option<usize>,

    /// Number of layers/modules to include when logging gradient breakdowns
    #[arg(long)]
    pub gradient_layer_limit: Option<usize>,

    /// Embedding dimension for tokens
    #[arg(long)]
    pub embed_dim: Option<usize>,

    /// Number of transformer layers
    #[arg(long)]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[arg(long)]
    pub num_heads: Option<usize>,

    /// Number of convolutional layers applied over the 8x8 board grid
    #[arg(long)]
    pub conv_layers: Option<usize>,

    /// Only include positions with a single legal move
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub single_legal_move_only: Option<bool>,

    /// Disable terminal UI
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub disable_tui: Option<bool>,

    /// Only include positions that are checkmate
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub checkmate_only: Option<bool>,

    /// Probability of logging individual items for debugging
    #[arg(long)]
    pub item_log_probability: Option<f32>,

    /// Enable detailed tensor norm logging during forward passes
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub log_tensor_norms: Option<bool>,

    /// Maximum number of tensor elements to print when previewing small tensors
    #[arg(long)]
    pub norm_preview_limit: Option<usize>,

    /// Focal loss gamma parameter for policy head
    #[arg(long)]
    pub focal_loss_gamma: Option<f32>,

    /// Smolgen hidden dimension
    #[arg(long)]
    pub smolgen_hidden: Option<usize>,

    /// Smolgen global dimension
    #[arg(long)]
    pub smolgen_global_dim: Option<usize>,

    /// Smolgen generator size
    #[arg(long)]
    pub smolgen_gen_size: Option<usize>,

    /// Enable forward pass timing instrumentation
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_forward_timing: Option<bool>,

    /// Sample interval for forward timing
    #[arg(long)]
    pub forward_timing_interval: Option<u64>,

    /// Number of devices to use
    #[arg(long)]
    pub num_devices: Option<usize>,

    /// Enable ply-based position sampling
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_ply_sampling: Option<bool>,

    /// Maximum ply (inclusive) to include training samples for. Samples with
    /// `current_ply > max_ply` are skipped. Omit to disable the upper bound.
    #[arg(long)]
    pub max_ply: Option<usize>,

    /// Enable Elo-based game sampling
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_elo_sampling: Option<bool>,

    /// Number of iterations between checkpoints
    #[arg(long)]
    pub checkpoint_interval: Option<usize>,

    /// EMA decay factor for the instrumentation copy of the weights
    #[arg(long)]
    pub ema_beta: Option<f64>,

    /// Allie test set JSONL for checkpoint-cadence live/EMA eval
    #[arg(long)]
    pub ema_eval_dataset: Option<std::path::PathBuf>,

    /// Number of games loaded from the EMA eval dataset
    #[arg(long)]
    pub ema_eval_games: Option<usize>,

    /// Number of TCEC samples to use for pretraining
    #[arg(long)]
    pub pretrain_samples: Option<usize>,

    /// Size of shuffle buffer for streaming data loading
    #[arg(long)]
    pub shuffle_buffer_size: Option<usize>,

    /// Disable shuffle-buffer randomization and consume buffered examples in stream order
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub disable_training_shuffle: Option<bool>,

    /// Interval for computing expensive metrics
    #[arg(long)]
    pub full_metrics_interval: Option<usize>,

    /// Priority boost for advanced/expert ELO games
    #[arg(long)]
    pub elo_priority_boost: Option<f64>,

    /// Ratio of puzzle examples to mix into training (0.0 to 1.0)
    #[arg(long)]
    pub puzzle_sampling_ratio: Option<f64>,

    /// Path to puzzle CSV file
    #[arg(long)]
    pub puzzle_path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub calibration_db_path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub calibration_loss_weight: Option<f32>,

    #[arg(long)]
    pub policy_regret_loss_weight: Option<f32>,

    /// Reference scale (centipawns) for the policy-regret hinge gate
    #[arg(long)]
    pub policy_regret_ref_cp: Option<f32>,

    /// Number of transformer layers in the value tower
    #[arg(long)]
    pub value_tower_layers: Option<usize>,

    /// Starting ply for value example weighting ramp
    #[arg(long)]
    pub value_ply_ramp_start: Option<usize>,

    /// Ending ply for value example weighting ramp
    #[arg(long)]
    pub value_ply_ramp_full: Option<usize>,

    /// Whether to train value head on puzzle positions
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub value_train_on_puzzles: Option<bool>,

    /// Use Muon optimizer for 2D+ weight matrices (false = AdamW for everything)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub use_muon: Option<bool>,

    /// Optimizer update for the Muon parameter group: "aurora" or "muon"
    #[arg(long)]
    pub muon_optimizer: Option<String>,

    /// Aurora projection iterations for the Muon parameter group
    #[arg(long)]
    pub aurora_pp_iterations: Option<usize>,

    /// Aurora row-normalization damping exponent
    #[arg(long)]
    pub aurora_pp_beta: Option<f32>,

    /// Muon LR adjustment function: "original" or "match_rms_adamw"
    #[arg(long)]
    pub muon_lr_adjust: Option<String>,

    /// Weight for auxiliary losses (mobility + material prediction)
    #[arg(long)]
    pub aux_loss_weight: Option<f32>,

    /// Enable bf16 mixed precision training (cast inputs to bf16, keep optimizer in f32)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub mixed_precision: Option<bool>,

    /// Compute whitening.json next to the trained model after training exits.
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub whiten_after_training: Option<bool>,

    /// Number of positions to use for post-training whitening.
    #[arg(long)]
    pub whitening_positions: Option<usize>,

    /// Inference batch size for post-training whitening.
    #[arg(long)]
    pub whitening_batch_size: Option<usize>,
}

impl Config {
    /// Create a Config from defaults, applying any overrides that are Some
    pub fn with_overrides(overrides: ConfigOverrides) -> Self {
        let mut config = Config::default();
        if let Some(model_size) = overrides.model_size {
            config.apply_model_size_preset(model_size);
        }

        if let Some(v) = overrides.data_path {
            config.data_path = Some(v);
        }
        if let Some(v) = overrides.log_dir {
            config.log_dir = Some(v);
        }
        if let Some(v) = overrides.max_samples {
            config.max_samples = Some(v);
        }
        if let Some(v) = overrides.skip {
            config.skip = Some(v);
        }
        if let Some(v) = overrides.timeout {
            config.timeout = Some(v);
        }
        if let Some(v) = overrides.resume {
            config.resume = Some(v);
        }
        if let Some(v) = overrides.train_ratio {
            config.train_ratio = v;
        }
        if let Some(v) = overrides.batch_size {
            config.batch_size = Some(v);
        }
        if let Some(v) = overrides.physical_batch_size {
            config.physical_batch_size = v;
        }
        if let Some(v) = overrides.seed {
            config.seed = v;
        }
        if let Some(v) = overrides.num_workers {
            config.num_workers = v;
        }
        if let Some(v) = overrides.lr_min {
            config.lr_min = v;
        }
        if let Some(v) = overrides.wsd_decay_fraction {
            config.wsd_decay_fraction = v;
        }
        if let Some(v) = overrides.lr_hold {
            config.lr_hold = Some(v);
        }
        if let Some(v) = overrides.lr_multiplier {
            config.lr_multiplier = v;
        }
        if let Some(v) = overrides.muon_base_lr {
            config.muon_base_lr = v;
        }
        if let Some(v) = overrides.adamw_base_lr {
            config.adamw_base_lr = v;
        }
        if let Some(v) = overrides.embedding_base_lr {
            config.embedding_base_lr = v;
        }
        if let Some(v) = overrides.warmup_fraction {
            config.warmup_fraction = v;
        }
        if let Some(v) = overrides.policy_loss_weight {
            config.policy_loss_weight = v;
        }
        if let Some(v) = overrides.policy_label_smoothing {
            config.policy_label_smoothing = v;
        }
        if let Some(v) = overrides.value_loss_weight {
            config.value_loss_weight = v;
        }
        if let Some(v) = overrides.value_entropy_weight {
            config.value_entropy_weight = v;
        }
        if let Some(v) = overrides.time_usage_loss_weight {
            config.time_usage_loss_weight = v;
        }
        if let Some(v) = overrides.weight_decay {
            config.weight_decay = v;
        }
        if let Some(v) = overrides.gradient_clip {
            config.gradient_clip = v;
        }
        if let Some(v) = overrides.log_gradient_breakdown {
            config.log_gradient_breakdown = Some(v);
        }
        if let Some(v) = overrides.gradient_head_limit {
            config.gradient_head_limit = v;
        }
        if let Some(v) = overrides.gradient_layer_limit {
            config.gradient_layer_limit = v;
        }
        if let Some(v) = overrides.embed_dim {
            config.embed_dim = v;
        }
        if let Some(v) = overrides.num_layers {
            config.num_layers = v;
        }
        if let Some(v) = overrides.num_heads {
            config.num_heads = v;
        }

        if let Some(v) = overrides.conv_layers {
            config.conv_layers = v;
        }
        if let Some(v) = overrides.single_legal_move_only {
            config.single_legal_move_only = Some(v);
        }
        if let Some(v) = overrides.disable_tui {
            config.disable_tui = Some(v);
        }
        if let Some(v) = overrides.checkmate_only {
            config.checkmate_only = Some(v);
        }
        if let Some(v) = overrides.item_log_probability {
            config.item_log_probability = v;
        }
        if let Some(v) = overrides.log_tensor_norms {
            config.log_tensor_norms = Some(v);
        }
        if let Some(v) = overrides.norm_preview_limit {
            config.norm_preview_limit = v;
        }
        if let Some(v) = overrides.focal_loss_gamma {
            config.focal_loss_gamma = v;
        }
        if let Some(v) = overrides.smolgen_hidden {
            config.smolgen_hidden = v;
        }
        if let Some(v) = overrides.smolgen_global_dim {
            config.smolgen_global_dim = v;
        }
        if let Some(v) = overrides.smolgen_gen_size {
            config.smolgen_gen_size = v;
        }
        if let Some(v) = overrides.enable_forward_timing {
            config.enable_forward_timing = Some(v);
        }
        if let Some(v) = overrides.forward_timing_interval {
            config.forward_timing_interval = v;
        }
        if let Some(v) = overrides.num_devices {
            config.num_devices = v;
        }
        if let Some(v) = overrides.enable_ply_sampling {
            config.enable_ply_sampling = Some(v);
        }
        if let Some(v) = overrides.max_ply {
            config.max_ply = Some(v);
        }
        if let Some(v) = overrides.enable_elo_sampling {
            config.enable_elo_sampling = Some(v);
        }
        if let Some(v) = overrides.checkpoint_interval {
            config.checkpoint_interval = v;
        }
        if let Some(v) = overrides.ema_beta {
            config.ema_beta = v;
        }
        if let Some(v) = overrides.ema_eval_dataset.clone() {
            config.ema_eval_dataset = Some(v);
        }
        if let Some(v) = overrides.ema_eval_games {
            config.ema_eval_games = v;
        }
        if let Some(v) = overrides.pretrain_samples {
            config.pretrain_samples = v;
        }
        if let Some(v) = overrides.shuffle_buffer_size {
            config.shuffle_buffer_size = v;
        }
        if let Some(v) = overrides.disable_training_shuffle {
            config.disable_training_shuffle = Some(v);
        }
        if let Some(v) = overrides.full_metrics_interval {
            config.full_metrics_interval = v;
        }
        if let Some(v) = overrides.elo_priority_boost {
            config.elo_priority_boost = v;
        }
        if let Some(v) = overrides.puzzle_sampling_ratio {
            config.puzzle_sampling_ratio = v;
        }
        if let Some(v) = overrides.puzzle_path {
            config.puzzle_path = Some(v);
        }
        if let Some(v) = overrides.calibration_db_path {
            config.calibration_db_path = Some(v);
        }
        if let Some(v) = overrides.calibration_loss_weight {
            config.calibration_loss_weight = v;
        }
        if let Some(v) = overrides.policy_regret_loss_weight {
            config.policy_regret_loss_weight = v;
        }
        if let Some(v) = overrides.policy_regret_ref_cp {
            config.policy_regret_ref_cp = v;
        }
        if let Some(v) = overrides.value_tower_layers {
            config.value_tower_layers = v;
        }
        if let Some(v) = overrides.value_ply_ramp_start {
            config.value_ply_ramp_start = v;
        }
        if let Some(v) = overrides.value_ply_ramp_full {
            config.value_ply_ramp_full = v;
        }
        if let Some(v) = overrides.value_train_on_puzzles {
            config.value_train_on_puzzles = Some(v);
        }
        if let Some(v) = overrides.lr_range_finder {
            config.lr_range_finder = Some(v);
        }
        if let Some(v) = overrides.use_muon {
            config.use_muon = Some(v);
        }
        if let Some(v) = overrides.muon_optimizer {
            config.muon_optimizer = Some(v);
        }
        if let Some(v) = overrides.aurora_pp_iterations {
            config.aurora_pp_iterations = v;
        }
        if let Some(v) = overrides.aurora_pp_beta {
            config.aurora_pp_beta = v;
        }
        if let Some(v) = overrides.muon_lr_adjust {
            config.muon_lr_adjust = Some(v);
        }
        if let Some(v) = overrides.aux_loss_weight {
            config.aux_loss_weight = v;
        }
        if let Some(v) = overrides.mixed_precision {
            config.mixed_precision = Some(v);
        }
        if let Some(v) = overrides.whiten_after_training {
            config.whiten_after_training = Some(v);
        }
        if let Some(v) = overrides.whitening_positions {
            config.whitening_positions = v;
        }
        if let Some(v) = overrides.whitening_batch_size {
            config.whitening_batch_size = v;
        }
        config
    }
}

impl Config {
    pub fn mini() -> Self {
        let mut config = Self::default();
        config.apply_model_size_preset(ModelSize::Mini);
        config
    }

    pub fn apply_model_size_preset(&mut self, model_size: ModelSize) {
        self.model_size = model_size;
        match model_size {
            ModelSize::Full => {}
            ModelSize::Mini => {
                // Target roughly 1/10th of the full model's parameter count.
                // The measured policy-forward tradeoff is best with fewer
                // layers; use the aggressive 128d/3L preset unless quality
                // says we need to buy back width.
                self.embed_dim = 128;
                self.num_layers = 3;
                self.num_heads = 4;
                self.smolgen_hidden = 12;
                self.smolgen_global_dim = 80;
                self.smolgen_gen_size = 80;

                // Mini is intended for fast policy + embedding serving. Keep
                // auxiliary/value heads out of the initial local run unless
                // explicitly re-enabled by overrides.
                self.policy_loss_weight = 1.0;
                self.value_loss_weight = 0.0;
                self.value_entropy_weight = 0.0;
                self.time_usage_loss_weight = 0.0;
                self.aux_loss_weight = 0.0;
                self.calibration_loss_weight = 0.0;
                self.policy_regret_loss_weight = 0.0;

                self.physical_batch_size = 8192;
                self.full_metrics_interval = 100;
                self.checkpoint_interval = 200;
                self.whiten_after_training = Some(true);
                self.whitening_positions = 20000;
                self.whitening_batch_size = 512;
            }
        }
    }

    /// Create new config with explicit parameters for testing
    pub fn new(embed_dim: usize, num_layers: usize) -> Self {
        // Keep head_dim at 32 regardless of width so test/bench configs with
        // non-default embed_dim don't trip the divisibility assert in
        // `num_heads()` (the default num_heads only matches the default width).
        let num_heads = (embed_dim / 32).max(1);
        Self {
            embed_dim,
            num_layers,
            num_heads,
            ..Default::default()
        }
    }

    // === MODEL ARCHITECTURE ACCESSORS ===

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
    pub fn global_dim(&self) -> usize {
        NUM_GLOBALS
    }
    pub fn non_global_dim(&self) -> usize {
        self.embed_dim - self.global_dim()
    }
    pub fn num_heads(&self) -> usize {
        assert!(
            self.embed_dim % self.num_heads == 0,
            "embed_dim ({}) must be divisible by num_heads ({})",
            self.embed_dim,
            self.num_heads
        );
        self.num_heads
    }
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads()
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn conv_layers(&self) -> usize {
        self.conv_layers
    }
    pub fn seq_len(&self) -> usize {
        64
    }

    pub fn log_tensor_norms(&self) -> bool {
        self.log_tensor_norms.unwrap_or(false)
    }

    pub fn norm_preview_limit(&self) -> usize {
        self.norm_preview_limit.max(1)
    }

    pub fn log_gradient_breakdown(&self) -> bool {
        self.log_gradient_breakdown.unwrap_or(false)
    }

    pub fn gradient_head_limit(&self) -> usize {
        self.gradient_head_limit.max(1)
    }

    pub fn gradient_layer_limit(&self) -> usize {
        self.gradient_layer_limit.max(1)
    }

    pub fn forward_timing_enabled(&self) -> bool {
        self.enable_forward_timing.unwrap_or(false)
    }

    pub fn use_muon(&self) -> bool {
        self.use_muon.unwrap_or(true)
    }

    pub fn muon_optimizer(&self) -> &str {
        self.muon_optimizer.as_deref().unwrap_or("aurora")
    }

    pub fn muon_lr_adjust(&self) -> &str {
        self.muon_lr_adjust.as_deref().unwrap_or("original")
    }

    pub fn mixed_precision(&self) -> bool {
        self.mixed_precision.unwrap_or(false)
    }

    pub fn forward_timing_interval(&self) -> u64 {
        self.forward_timing_interval.max(1)
    }

    pub fn smolgen_hidden(&self) -> usize {
        self.smolgen_hidden
    }

    pub fn smolgen_global_dim(&self) -> usize {
        self.smolgen_global_dim
    }

    pub fn smolgen_gen_size(&self) -> usize {
        self.smolgen_gen_size
    }

    // === VALUE TOWER ACCESSORS ===

    pub fn value_tower_layers(&self) -> usize {
        self.value_tower_layers
    }

    pub fn value_ply_ramp_start(&self) -> usize {
        self.value_ply_ramp_start
    }

    pub fn value_ply_ramp_full(&self) -> usize {
        self.value_ply_ramp_full
    }

    pub fn value_train_on_puzzles(&self) -> bool {
        self.value_train_on_puzzles.unwrap_or(false)
    }

    pub fn calibration_db_path(&self) -> Option<std::path::PathBuf> {
        if let Some(path) = &self.calibration_db_path {
            return Some(path.clone());
        }
        self.data_path.as_ref().and_then(|data_path| {
            let candidate = data_path.join("calibration.db");
            candidate.exists().then_some(candidate)
        })
    }

    pub fn calibration_loss_weight(&self) -> f32 {
        self.calibration_loss_weight.max(0.0)
    }

    pub fn policy_regret_loss_weight(&self) -> f32 {
        self.policy_regret_loss_weight.max(0.0)
    }

    /// Calculate value example weight based on ply (0 before start, ramps to 1 at full)
    pub fn value_ply_weight(&self, ply: usize) -> f32 {
        if ply < self.value_ply_ramp_start {
            0.0
        } else if ply >= self.value_ply_ramp_full {
            1.0
        } else {
            let range = (self.value_ply_ramp_full - self.value_ply_ramp_start) as f32;
            let progress = (ply - self.value_ply_ramp_start) as f32;
            progress / range
        }
    }
}

/// Set the global config (should be called once at startup)
#[allow(clippy::result_large_err)]
pub fn set_global_config(config: Config) -> Result<(), Config> {
    GLOBAL_CONFIG.set(config)
}

/// Get the global config, falling back to default if not set
pub fn get_global_config() -> &'static Config {
    GLOBAL_CONFIG.get().unwrap()
}

pub fn global_config() -> Option<&'static Config> {
    GLOBAL_CONFIG.get()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_size: ModelSize::Full,
            data_path: Some(std::path::PathBuf::from("/lambda/nfs/chessbook")),
            log_dir: None,
            train_ratio: 1.0,
            batch_size: None,
            physical_batch_size: 16000,
            seed: 42,
            num_workers: 4,
            weight_decay: 0.005,
            cautious_weight_decay: Some(true),
            adam_epsilon: 1e-8,
            gradient_clip: 3.0,
            log_gradient_breakdown: Some(false),
            gradient_head_limit: 128,
            gradient_layer_limit: 128,
            lr_min: 0.000001,
            wsd_decay_fraction: default_wsd_decay_fraction(),
            lr_hold: Some(false),
            lr_multiplier: 1.0, // Was 1.5 (tuned on short research iters); destabilized the value head ~2.5k steps into an 8h run
            muon_base_lr: default_muon_base_lr(),
            adamw_base_lr: default_adamw_base_lr(),
            embedding_base_lr: default_embedding_base_lr(),
            warmup_fraction: default_warmup_fraction(),
            policy_loss_weight: 0.30,
            policy_label_smoothing: 0.0,
            value_loss_weight: 0.10,
            value_entropy_weight: 0.0,
            // ~30M params (was 192/8/6 at ~9.7M). num_heads keeps head_dim at 32;
            // base LRs are defined at LR_REFERENCE_DIM=256 and μP-scaled to this width.
            embed_dim: 320,
            num_layers: 16,
            num_heads: 10,

            conv_layers: 0,
            max_samples: None,
            skip: None,
            timeout: None,
            resume: Some(false),
            single_legal_move_only: Some(false),
            checkmate_only: Some(false),
            item_log_probability: 0.01,
            log_tensor_norms: Some(false),
            norm_preview_limit: 6,
            time_usage_loss_weight: 0.0,
            disable_tui: Some(false),
            smolgen_hidden: 24,
            smolgen_global_dim: 128,
            smolgen_gen_size: 128,
            enable_forward_timing: Some(false),
            forward_timing_interval: 100,
            num_devices: 1,
            focal_loss_gamma: 0.0,
            enable_ply_sampling: Some(true),
            max_ply: None,
            enable_elo_sampling: Some(true),
            checkpoint_interval: 100,
            ema_beta: default_ema_beta(),
            ema_eval_dataset: None,
            ema_eval_games: default_ema_eval_games(),
            pretrain_samples: 0,
            shuffle_buffer_size: 100000,
            disable_training_shuffle: Some(false),
            full_metrics_interval: 50,
            elo_priority_boost: 3.0,
            puzzle_sampling_ratio: 0.0,
            puzzle_path: None,
            calibration_db_path: None,
            calibration_loss_weight: default_calibration_loss_weight(),
            policy_regret_loss_weight: default_policy_regret_loss_weight(),
            policy_regret_ref_cp: default_policy_regret_ref_cp(),
            value_tower_layers: 2,
            value_ply_ramp_start: 10,
            value_ply_ramp_full: 30,
            value_train_on_puzzles: Some(false),
            lr_range_finder: Some(false),

            use_muon: Some(true),
            muon_optimizer: default_muon_optimizer(),
            aurora_pp_iterations: default_aurora_pp_iterations(),
            aurora_pp_beta: default_aurora_pp_beta(),
            muon_lr_adjust: None,
            aux_loss_weight: default_aux_loss_weight(),
            mixed_precision: Some(false),
            whiten_after_training: default_whiten_after_training(),
            whitening_positions: default_whitening_positions(),
            whitening_batch_size: default_whitening_batch_size(),
        }
    }
}

impl Config {
    pub fn full_metrics_interval(&self) -> Option<usize> {
        if self.full_metrics_interval == 0 {
            None
        } else {
            Some(self.full_metrics_interval)
        }
    }

    pub fn warmup_fraction_clamped(&self) -> f64 {
        self.warmup_fraction.clamp(0.0, 1.0)
    }

    pub fn lr_hold(&self) -> bool {
        self.lr_hold.unwrap_or(false)
    }

    pub fn whiten_after_training(&self) -> bool {
        self.whiten_after_training.unwrap_or(true)
    }

    /// The WSD schedule needs a run budget to know when to start decaying.
    /// Errors when neither --timeout nor --max-samples is set, unless
    /// --lr-hold explicitly opts into an open-ended constant-LR run.
    pub fn validate_lr_schedule(&self) -> Result<(), String> {
        if self.lr_hold() || self.timeout.is_some() || self.max_samples.is_some() {
            Ok(())
        } else {
            Err(
                "WSD LR schedule needs a run budget: pass --timeout <secs> or \
                 --max-samples <n> (decay covers the final wsd_decay_fraction \
                 of the budget), or pass --lr-hold for an open-ended \
                 constant-LR run that you anneal later"
                    .to_string(),
            )
        }
    }
}

/// Known-good anchor for auto physical batch sizing on this hardware class:
/// the 192-dim / 8-layer / 6-head model (AUTO_BATCH_REFERENCE_PARAMS params)
/// trains comfortably at physical batch 2048 on an M4 Max via LibTorch/MPS,
/// while 4096 collapses (~4x slower per sample). We keep the per-step working
/// set roughly constant by scaling batch inversely with parameter count.
pub const AUTO_BATCH_REFERENCE_PARAMS: usize = 9_731_513; // measured: 192-dim/8-layer/6-head
pub const AUTO_BATCH_REFERENCE_BATCH: usize = 2048;

/// Derive a safe physical batch size from the model's parameter count.
/// Returns the power of two nearest to
/// `AUTO_BATCH_REFERENCE_BATCH * AUTO_BATCH_REFERENCE_PARAMS / num_params`,
/// clamped to [64, AUTO_BATCH_REFERENCE_BATCH]. The cap means we never exceed
/// the empirically validated batch even for models smaller than the reference,
/// since the MPS cliff was observed at the reference size.
pub fn auto_physical_batch_size(num_params: usize) -> usize {
    let target = AUTO_BATCH_REFERENCE_BATCH as f64 * AUTO_BATCH_REFERENCE_PARAMS as f64
        / num_params.max(1) as f64;
    let exponent = target.log2().round().clamp(6.0, 11.0) as u32; // 64..=2048
    1usize << exponent
}

/// Calculate probability of keeping a position based on ply number
/// 80% drop at ply 0, grading down to 0% drop at ply 10+
pub fn ply_keep_probability(ply: usize) -> f64 {
    if ply >= 10 {
        1.0
    } else {
        // Linear interpolation from 0.2 (20% keep) at ply 0 to 1.0 (100% keep) at ply 10
        0.2 + (0.8 * ply as f64 / 10.0)
    }
}

const ELO_DISTRIBUTION_MEAN: f64 = 1672.0;
const ELO_DISTRIBUTION_STD: f64 = 404.0;
const ELO_FLATTENING_FACTOR: f64 = 0.05;

static ELO_DISTRIBUTION: Lazy<Normal> = Lazy::new(|| {
    Normal::new(ELO_DISTRIBUTION_MEAN, ELO_DISTRIBUTION_STD)
        .expect("Failed to create normal distribution")
});

const ADVANCED_ELO_THRESHOLD: f64 = 2000.0;
const ADVANCED_ELO_RANGE: f64 = 500.0;

pub fn elo_keep_probability(avg_elo: f64) -> f64 {
    elo_keep_probability_with_boost(avg_elo, 0.0)
}

pub fn elo_keep_probability_with_boost(avg_elo: f64, priority_boost: f64) -> f64 {
    if avg_elo < MIN_ELO as f64 || avg_elo > MAX_ELO as f64 {
        return 0.0;
    }

    let natural_frequency = ELO_DISTRIBUTION.pdf(avg_elo);
    if natural_frequency == 0.0 {
        return 0.0;
    }

    let peak_density = ELO_DISTRIBUTION.pdf(ELO_DISTRIBUTION_MEAN);
    let relative_frequency = natural_frequency / peak_density;
    let flatten_prob = (ELO_FLATTENING_FACTOR / relative_frequency).clamp(0.0, 1.0);

    let elo_clamped = avg_elo.clamp(MIN_ELO as f64, MAX_ELO as f64);
    let graduated_prob = (elo_clamped - MIN_ELO as f64) / (MAX_ELO - MIN_ELO) as f64;

    let base_prob = flatten_prob * graduated_prob;

    if priority_boost > 0.0 && avg_elo >= ADVANCED_ELO_THRESHOLD {
        let boost_progress = ((avg_elo - ADVANCED_ELO_THRESHOLD) / ADVANCED_ELO_RANGE).min(1.0);
        let boost_factor = 1.0 + priority_boost * boost_progress;
        (base_prob * boost_factor).min(1.0)
    } else {
        base_prob
    }
}

/// Randomly decide whether to log this item based on `item_log_probability`.
/// Use this to guard expensive host syncs (e.g., tensor .to_data()) in hot paths.
pub fn should_log_item() -> bool {
    let config = get_global_config();
    let mut rng = rand::rng();
    (rng.next_u32() as f32 / u32::MAX as f32) < config.item_log_probability
}

/// Execute the provided logger closure with probability `item_log_probability`.
/// Example:
///   shd_log(|| tracing::info!("my message: {}", value));
pub fn shd_log<F: FnOnce()>(f: F) {
    if should_log_item() {
        f();
    }
}

/// Check if a position should be kept based on ply and random sampling
pub fn should_keep_position_by_ply(ply: usize, rng_value: f64) -> bool {
    let config = get_global_config();
    // Apply the hard upper ply bound (if configured) regardless of whether the
    // probabilistic ply-sampling pipeline is enabled.
    if let Some(max_ply) = config.max_ply {
        if ply > max_ply {
            return false;
        }
    }
    if !config.enable_ply_sampling.unwrap_or(true) {
        return true;
    }
    rng_value < ply_keep_probability(ply)
}

pub fn should_keep_game_by_elo(white_elo: i32, black_elo: i32, rng_value: f64) -> bool {
    let config = get_global_config();
    if !config.enable_elo_sampling.unwrap_or(true) {
        return true;
    }
    let avg_elo = (white_elo + black_elo) as f64 / 2.0;
    let keep_prob = elo_keep_probability_with_boost(avg_elo, config.elo_priority_boost);
    rng_value < keep_prob
}

// Legacy type aliases for backward compatibility during transition
pub type ModelConfig = Config;
pub type TrainingConfig = Config;

#[cfg(test)]
mod auto_batch_tests {
    use super::*;

    #[test]
    fn reference_model_gets_reference_batch() {
        assert_eq!(
            auto_physical_batch_size(AUTO_BATCH_REFERENCE_PARAMS),
            AUTO_BATCH_REFERENCE_BATCH
        );
    }

    #[test]
    fn double_params_halves_batch() {
        assert_eq!(
            auto_physical_batch_size(AUTO_BATCH_REFERENCE_PARAMS * 2),
            AUTO_BATCH_REFERENCE_BATCH / 2
        );
    }

    #[test]
    fn rounds_to_nearest_power_of_two() {
        // 1.6x params -> target 1280 -> nearest power of two is 1024
        let params = (AUTO_BATCH_REFERENCE_PARAMS as f64 * 1.6) as usize;
        assert_eq!(auto_physical_batch_size(params), 1024);
        // 1.3x params -> target ~1575 -> nearest power of two is 2048
        let params = (AUTO_BATCH_REFERENCE_PARAMS as f64 * 1.3) as usize;
        assert_eq!(auto_physical_batch_size(params), 2048);
    }

    #[test]
    fn never_exceeds_reference_batch() {
        assert_eq!(
            auto_physical_batch_size(AUTO_BATCH_REFERENCE_PARAMS / 8),
            AUTO_BATCH_REFERENCE_BATCH
        );
    }

    #[test]
    fn huge_model_clamps_to_floor() {
        assert_eq!(auto_physical_batch_size(usize::MAX / 2), 64);
    }
}

#[cfg(test)]
mod warmup_tests {
    use super::*;

    #[test]
    fn default_warmup_fraction_is_two_percent() {
        let config = Config::default();
        assert_eq!(config.warmup_fraction_clamped(), 0.02);
    }

    #[test]
    fn explicit_fraction_override_is_clamped() {
        let overrides = ConfigOverrides {
            warmup_fraction: Some(1.5),
            ..Default::default()
        };
        let config = Config::with_overrides(overrides);
        assert_eq!(config.warmup_fraction_clamped(), 1.0);
    }
}

#[cfg(test)]
mod model_size_tests {
    use super::*;

    #[test]
    fn mini_preset_sets_latency_focused_architecture() {
        let config = Config::with_overrides(ConfigOverrides {
            model_size: Some(ModelSize::Mini),
            ..Default::default()
        });

        assert_eq!(config.model_size, ModelSize::Mini);
        assert_eq!(config.embed_dim, 128);
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim(), 32);
        assert_eq!(config.policy_loss_weight, 1.0);
        assert_eq!(config.value_loss_weight, 0.0);
        assert_eq!(config.aux_loss_weight, 0.0);
        assert!(config.whiten_after_training());
    }

    #[test]
    fn explicit_architecture_overrides_win_after_mini_preset() {
        let config = Config::with_overrides(ConfigOverrides {
            model_size: Some(ModelSize::Mini),
            embed_dim: Some(192),
            num_layers: Some(4),
            num_heads: Some(6),
            ..Default::default()
        });

        assert_eq!(config.model_size, ModelSize::Mini);
        assert_eq!(config.embed_dim, 192);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_heads, 6);
        assert_eq!(config.head_dim(), 32);
    }
}
