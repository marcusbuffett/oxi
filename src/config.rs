use clap::Args;
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Normal};
use std::sync::OnceLock;

pub const NUM_GLOBALS: usize = 10;
pub const LEGAL_MOVES: usize = 64 * 76;
// Per-square features (current position only):
// - 12 piece one-hots (white/black 6 each)
// - 1 en passant
// - 1 castling right at this square
// - 6 attackers (white, one-hots by role)
// - 1 total attacker material (white), normalized by 24
// - 1 number of attackers (white), normalized by 6
// - 6 attackers (black, one-hots by role)
// - 1 total attacker material (black), normalized by 24
// - 1 number of attackers (black), normalized by 6
// Feature grouping per square (current position only, excluding recency channels):
// - Piece identity group (12): white/black one-hots for all roles
// - Tactical group (22): attackers, pins, pin target, hanging flag, has_pinned_defender, square control
// - Positional group (25): legal moves, pawn structure, weak squares, open file, passed pawn, dark-square flag, rank/file one-hots
// - Misc group (2): en passant target, local castling right
pub const PIECE_IDENTITY_FEATURES: usize = 12;
pub const TACTICAL_FEATURES: usize = 22;
pub const POSITIONAL_FEATURES: usize = 25;
pub const MISC_FEATURES: usize = 2;
pub const RECENCY_FEATURES: usize = 4; // white_from, white_to, black_from, black_to

pub const BOARD_FEATURES_PER_TOKEN: usize =
    PIECE_IDENTITY_FEATURES + TACTICAL_FEATURES + POSITIONAL_FEATURES + MISC_FEATURES;
pub const FEATURES_PER_SQUARE_POSITION: usize = BOARD_FEATURES_PER_TOKEN;
pub const FEATURES_PER_TOKEN: usize = BOARD_FEATURES_PER_TOKEN + RECENCY_FEATURES;
pub const PREVIOUS_POSITIONS: usize = 5; // Used for decay horizon only
pub const HISTORY_DECAY: f32 = 0.8; // Exponential decay factor for historical positions

// Global config storage
static GLOBAL_CONFIG: OnceLock<Config> = OnceLock::new();

/// Minimum Elo rating for both players to include games
pub const MIN_ELO: i32 = 1000;
pub const MIN_PLY: usize = 0;
pub const MAX_ELO: i32 = 3000;
pub const MAX_ELO_DIFF: i32 = 200;
pub const MIN_TIME_CONTROL: u32 = 61;

/// Minimum clock time (in seconds) to include moves
pub const MIN_CLOCK_TIME: u32 = 30;

/// Unified configuration for OXI chess engine training and inference.
/// Defaults are defined in the `Default` impl. Use `ConfigOverrides` for CLI parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
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

    /// Physical batch size (for gradient accumulation)
    pub physical_batch_size: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Number of data loader workers
    pub num_workers: usize,

    /// Minimum learning rate (end of training)
    pub lr_min: f64,

    /// Window size for plateau detection (number of iterations to compare)
    pub lr_window_size: usize,

    /// Minimum relative improvement threshold for plateau detection (e.g., 0.0005 = 0.05% improvement required)
    pub lr_improvement_threshold: f64,

    /// Factor to reduce learning rate by when plateau is detected (e.g., 0.5 means halve the LR)
    pub lr_reduction_factor: f64,

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

    /// Warmup multiplier: warmup lasts for warmup_multiplier * effective_batch_size optimizer steps
    pub warmup_multiplier: f64,

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

    /// Enable adaptive GradNorm reweighting across heads
    #[serde(default)]
    pub enable_gradnorm: Option<bool>,

    /// Optimizer steps between GradNorm weight updates
    #[serde(default)]
    pub gradnorm_interval: usize,

    /// Alpha hyperparameter for GradNorm target scaling
    pub gradnorm_alpha: f32,

    /// Multiplicative learning rate used when adjusting GradNorm weights
    pub gradnorm_learning_rate: f32,

    /// Priority multiplier applied to policy GradNorm target (1.0 = neutral)
    pub gradnorm_policy_priority: f32,

    /// Priority multiplier applied to value GradNorm target (1.0 = neutral)
    pub gradnorm_value_priority: f32,

    /// Priority multiplier applied to time-usage GradNorm target (1.0 = neutral)
    pub gradnorm_time_priority: f32,

    /// Priority multiplier applied to auxiliary GradNorm target (1.0 = neutral)
    pub gradnorm_aux_priority: f32,

    /// Number of samples to materialize on the lead device when probing GradNorm weights
    #[serde(default)]
    pub gradnorm_probe_size: usize,

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

    /// Probability of dropping games based on Elo (75% at 1000 Elo, 0% at 2000+ Elo)
    pub enable_elo_sampling: Option<bool>,

    /// Number of iterations between checkpoints
    pub checkpoint_interval: usize,

    /// Number of TCEC (computer engine) samples to use for pretraining (0 to disable)
    pub pretrain_samples: usize,

    /// Size of shuffle buffer for streaming data loading (number of examples to buffer before sampling)
    pub shuffle_buffer_size: usize,

    /// Interval for computing expensive metrics (top-5 accuracy, debug predictions, gradient breakdown, L2 penalty, tensor norms). 0 = never, 1 = every iteration.
    pub full_metrics_interval: usize,

    /// Priority boost for advanced/expert ELO games (2000+). Value of 1.0 = 2x boost at 2500 ELO,
    /// 2.0 = 3x boost, 3.0 = 4x boost. Set to 0.0 to disable.
    pub elo_priority_boost: f64,

    /// Ratio of puzzle examples to mix into training (0.0 to 1.0). Default 0.05 = 5% puzzles.
    pub puzzle_sampling_ratio: f64,

    /// Path to puzzle CSV file (defaults to <data-path>/puzzles/lichess_db_puzzle.csv.zst)
    pub puzzle_path: Option<std::path::PathBuf>,

    // === VALUE TOWER CONFIGURATION ===
    /// Number of transformer layers in the value tower (separate from trunk)
    #[serde(default = "default_value_tower_layers")]
    pub value_tower_layers: usize,

    /// Whether to backpropagate value gradients to trunk (false = stop_grad)
    #[serde(default)]
    pub value_backprop_to_trunk: Option<bool>,

    /// Starting ply for value example weighting ramp (positions before this get 0 weight)
    #[serde(default = "default_value_ply_ramp_start")]
    pub value_ply_ramp_start: usize,

    /// Ending ply for value example weighting ramp (positions at or after this get full weight)
    #[serde(default = "default_value_ply_ramp_full")]
    pub value_ply_ramp_full: usize,

    /// Whether to train value head on puzzle positions
    #[serde(default)]
    pub value_train_on_puzzles: Option<bool>,

    /// Skip the policy training stage and go directly to value tower only training.
    /// This freezes all params except value tower, resets LR to initial, disables puzzles,
    /// and sets policy loss weight to zero.
    #[serde(default)]
    pub skip_policy_loss: Option<bool>,

    /// Run LR range finder instead of training
    #[serde(default)]
    pub lr_range_finder: Option<bool>,

    /// Use Muon optimizer for 2D+ weight matrices (false = use AdamW for everything)
    #[serde(default = "default_use_muon")]
    pub use_muon: Option<bool>,

    /// Muon LR adjustment function: "original" or "match_rms_adamw"
    #[serde(default)]
    pub muon_lr_adjust: Option<String>,

    /// Weight for auxiliary losses (mobility + material prediction)
    #[serde(default = "default_aux_loss_weight")]
    pub aux_loss_weight: f32,

    /// Enable bf16 mixed precision training (cast inputs to bf16, keep optimizer in f32)
    #[serde(default)]
    pub mixed_precision: Option<bool>,
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
fn default_muon_base_lr() -> f64 {
    0.0225 // Was 0.015; increased 50% for faster convergence in limited-iteration regime
}
fn default_adamw_base_lr() -> f64 {
    3.375e-4 // Was 2.25e-4; increased 50% for faster convergence
}
fn default_embedding_base_lr() -> f64 {
    0.1125 // Was 0.075; increased 50% for faster convergence. Width-independent per μP.
}
fn default_aux_loss_weight() -> f32 {
    0.06 // Was 0.04; increased to strengthen trunk-level auxiliary supervision signal
}
/// Command-line overrides for Config. All fields are optional.
/// Use `Config::with_overrides()` to merge with defaults.
#[derive(Debug, Clone, Args, Default)]
pub struct ConfigOverrides {
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

    /// Physical batch size (for gradient accumulation)
    #[arg(long)]
    pub physical_batch_size: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of data loader workers
    #[arg(long)]
    pub num_workers: Option<usize>,

    /// Minimum learning rate (end of training)
    #[arg(long)]
    pub lr_min: Option<f64>,

    /// Window size for plateau detection (number of iterations to compare)
    #[arg(long)]
    pub lr_window_size: Option<usize>,

    /// Minimum relative improvement threshold for plateau detection
    #[arg(long)]
    pub lr_improvement_threshold: Option<f64>,

    /// Factor to reduce learning rate by when plateau is detected
    #[arg(long)]
    pub lr_reduction_factor: Option<f64>,

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

    /// Warmup multiplier: warmup lasts for warmup_multiplier * effective_batch_size optimizer steps
    #[arg(long)]
    pub warmup_multiplier: Option<f64>,

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

    /// Enable adaptive GradNorm reweighting across heads
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_gradnorm: Option<bool>,

    /// Optimizer steps between GradNorm weight updates
    #[arg(long)]
    pub gradnorm_interval: Option<usize>,

    /// Alpha hyperparameter for GradNorm target scaling
    #[arg(long)]
    pub gradnorm_alpha: Option<f32>,

    /// Multiplicative learning rate used when adjusting GradNorm weights
    #[arg(long)]
    pub gradnorm_learning_rate: Option<f32>,

    /// Priority multiplier applied to policy GradNorm target
    #[arg(long)]
    pub gradnorm_policy_priority: Option<f32>,

    /// Priority multiplier applied to value GradNorm target
    #[arg(long)]
    pub gradnorm_value_priority: Option<f32>,

    /// Priority multiplier applied to time-usage GradNorm target
    #[arg(long)]
    pub gradnorm_time_priority: Option<f32>,

    /// Priority multiplier applied to auxiliary GradNorm target
    #[arg(long)]
    pub gradnorm_aux_priority: Option<f32>,

    /// Number of samples to materialize on the lead device when probing GradNorm weights
    #[arg(long)]
    pub gradnorm_probe_size: Option<usize>,

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

    /// Enable Elo-based game sampling
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_elo_sampling: Option<bool>,

    /// Number of iterations between checkpoints
    #[arg(long)]
    pub checkpoint_interval: Option<usize>,

    /// Number of TCEC samples to use for pretraining
    #[arg(long)]
    pub pretrain_samples: Option<usize>,

    /// Size of shuffle buffer for streaming data loading
    #[arg(long)]
    pub shuffle_buffer_size: Option<usize>,

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

    /// Number of transformer layers in the value tower
    #[arg(long)]
    pub value_tower_layers: Option<usize>,

    /// Whether to backpropagate value gradients to trunk
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub value_backprop_to_trunk: Option<bool>,

    /// Starting ply for value example weighting ramp
    #[arg(long)]
    pub value_ply_ramp_start: Option<usize>,

    /// Ending ply for value example weighting ramp
    #[arg(long)]
    pub value_ply_ramp_full: Option<usize>,

    /// Whether to train value head on puzzle positions
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub value_train_on_puzzles: Option<bool>,

    /// Skip the policy training stage and go directly to value tower only training
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub skip_policy_loss: Option<bool>,

    /// Use Muon optimizer for 2D+ weight matrices (false = AdamW for everything)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub use_muon: Option<bool>,

    /// Muon LR adjustment function: "original" or "match_rms_adamw"
    #[arg(long)]
    pub muon_lr_adjust: Option<String>,

    /// Weight for auxiliary losses (mobility + material prediction)
    #[arg(long)]
    pub aux_loss_weight: Option<f32>,

    /// Enable bf16 mixed precision training (cast inputs to bf16, keep optimizer in f32)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub mixed_precision: Option<bool>,
}

impl Config {
    /// Create a Config from defaults, applying any overrides that are Some
    pub fn with_overrides(overrides: ConfigOverrides) -> Self {
        let mut config = Config::default();

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
        if let Some(v) = overrides.lr_window_size {
            config.lr_window_size = v;
        }
        if let Some(v) = overrides.lr_improvement_threshold {
            config.lr_improvement_threshold = v;
        }
        if let Some(v) = overrides.lr_reduction_factor {
            config.lr_reduction_factor = v;
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
        if let Some(v) = overrides.warmup_multiplier {
            config.warmup_multiplier = v;
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
        if let Some(v) = overrides.enable_gradnorm {
            config.enable_gradnorm = Some(v);
        }
        if let Some(v) = overrides.gradnorm_interval {
            config.gradnorm_interval = v;
        }
        if let Some(v) = overrides.gradnorm_alpha {
            config.gradnorm_alpha = v;
        }
        if let Some(v) = overrides.gradnorm_learning_rate {
            config.gradnorm_learning_rate = v;
        }
        if let Some(v) = overrides.gradnorm_policy_priority {
            config.gradnorm_policy_priority = v;
        }
        if let Some(v) = overrides.gradnorm_value_priority {
            config.gradnorm_value_priority = v;
        }
        if let Some(v) = overrides.gradnorm_time_priority {
            config.gradnorm_time_priority = v;
        }
        if let Some(v) = overrides.gradnorm_aux_priority {
            config.gradnorm_aux_priority = v;
        }
        if let Some(v) = overrides.gradnorm_probe_size {
            config.gradnorm_probe_size = v;
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
        if let Some(v) = overrides.enable_elo_sampling {
            config.enable_elo_sampling = Some(v);
        }
        if let Some(v) = overrides.checkpoint_interval {
            config.checkpoint_interval = v;
        }
        if let Some(v) = overrides.pretrain_samples {
            config.pretrain_samples = v;
        }
        if let Some(v) = overrides.shuffle_buffer_size {
            config.shuffle_buffer_size = v;
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
        if let Some(v) = overrides.value_tower_layers {
            config.value_tower_layers = v;
        }
        if let Some(v) = overrides.value_backprop_to_trunk {
            config.value_backprop_to_trunk = Some(v);
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
        if let Some(v) = overrides.skip_policy_loss {
            config.skip_policy_loss = Some(v);
        }
        if let Some(v) = overrides.lr_range_finder {
            config.lr_range_finder = Some(v);
        }
        if let Some(v) = overrides.use_muon {
            config.use_muon = Some(v);
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
        config
    }
}

impl Config {
    /// Create new config with explicit parameters for testing
    pub fn new(embed_dim: usize, num_layers: usize) -> Self {
        Self {
            embed_dim,
            num_layers,
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

    pub fn gradnorm_enabled(&self) -> bool {
        self.enable_gradnorm.unwrap_or(true)
    }

    pub fn gradnorm_interval(&self) -> usize {
        self.gradnorm_interval.max(1)
    }

    pub fn gradnorm_alpha(&self) -> f32 {
        self.gradnorm_alpha
    }

    pub fn gradnorm_learning_rate(&self) -> f32 {
        self.gradnorm_learning_rate
    }

    pub fn gradnorm_policy_priority(&self) -> f32 {
        self.gradnorm_policy_priority.max(0.0)
    }

    pub fn gradnorm_value_priority(&self) -> f32 {
        self.gradnorm_value_priority.max(0.0)
    }

    pub fn gradnorm_time_priority(&self) -> f32 {
        self.gradnorm_time_priority.max(0.0)
    }

    pub fn gradnorm_aux_priority(&self) -> f32 {
        self.gradnorm_aux_priority.max(0.0)
    }

    pub fn gradnorm_probe_size(&self) -> usize {
        self.gradnorm_probe_size.max(1)
    }

    pub fn forward_timing_enabled(&self) -> bool {
        self.enable_forward_timing.unwrap_or(false)
    }

    pub fn use_muon(&self) -> bool {
        self.use_muon.unwrap_or(true)
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

    pub fn value_backprop_to_trunk(&self) -> bool {
        self.value_backprop_to_trunk.unwrap_or(false)
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

    pub fn skip_policy_loss(&self) -> bool {
        self.skip_policy_loss.unwrap_or(false)
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

impl Default for Config {
    fn default() -> Self {
        Self {
            data_path: Some(std::path::PathBuf::from("/lambda/nfs/chessbook")),
            log_dir: None,
            train_ratio: 1.0,
            batch_size: None,
            physical_batch_size: 16000,
            seed: 42,
            num_workers: 4,
            weight_decay: 0.003,
            cautious_weight_decay: Some(true),
            adam_epsilon: 1e-8,
            gradient_clip: 3.0,
            log_gradient_breakdown: Some(false),
            gradient_head_limit: 128,
            gradient_layer_limit: 128,
            enable_gradnorm: Some(true),
            gradnorm_interval: 20,
            gradnorm_alpha: 0.5,
            gradnorm_learning_rate: 0.1,
            gradnorm_policy_priority: 5.0,
            gradnorm_value_priority: 2.5,
            gradnorm_time_priority: 1.0,
            gradnorm_aux_priority: 1.5,
            gradnorm_probe_size: 256,
            lr_min: 0.000001,
            lr_window_size: 120,
            lr_improvement_threshold: 0.015,
            lr_reduction_factor: 0.5,
            lr_multiplier: 1.0,
            muon_base_lr: default_muon_base_lr(),
            adamw_base_lr: default_adamw_base_lr(),
            embedding_base_lr: default_embedding_base_lr(),
            warmup_multiplier: 2.0,
            policy_loss_weight: 0.30,
            policy_label_smoothing: 0.005,
            value_loss_weight: 0.0001,
            value_entropy_weight: 0.0,
            embed_dim: 384,
            num_layers: 24,
            num_heads: 6,

            conv_layers: 0,
            max_samples: None,
            skip: None,
            timeout: None,
            resume: Some(false),
            single_legal_move_only: Some(false),
            checkmate_only: Some(false),
            item_log_probability: 1.00,
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
            enable_elo_sampling: Some(true),
            checkpoint_interval: 100,
            pretrain_samples: 0,
            shuffle_buffer_size: 100000,
            full_metrics_interval: 50,
            elo_priority_boost: 3.0,
            puzzle_sampling_ratio: 0.05,
            puzzle_path: None,
            value_tower_layers: 2,
            value_backprop_to_trunk: Some(false),
            value_ply_ramp_start: 10,
            value_ply_ramp_full: 30,
            value_train_on_puzzles: Some(false),
            skip_policy_loss: Some(false),
            lr_range_finder: Some(false),

            use_muon: Some(true),
            muon_lr_adjust: None,
            aux_loss_weight: default_aux_loss_weight(),
            mixed_precision: Some(false),
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
    let mut rng = rand::thread_rng();
    rng.gen::<f32>() < config.item_log_probability
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
    if !config.enable_ply_sampling.unwrap_or(true) {
        return true;
    }
    rng_value < ply_keep_probability(ply)
}

pub fn should_keep_game_by_elo(white_elo: i32, black_elo: i32, rng_value: f64) -> bool {
    let config = get_global_config();
    if !config.enable_elo_sampling.unwrap_or(true) {
        eprintln!("DEBUG: ELO sampling is DISABLED!");
        return true;
    }
    let avg_elo = (white_elo + black_elo) as f64 / 2.0;
    let keep_prob = elo_keep_probability_with_boost(avg_elo, config.elo_priority_boost);
    rng_value < keep_prob
}

// Legacy type aliases for backward compatibility during transition
pub type ModelConfig = Config;
pub type TrainingConfig = Config;