use clap::Parser;
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Normal};
use std::sync::OnceLock;

pub const NUM_GLOBALS: usize = 7;
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
pub const MAX_ELO: i32 = 2500;
pub const MAX_ELO_DIFF: i32 = 200;
pub const MIN_TIME_CONTROL: u32 = 61;

/// Minimum clock time (in seconds) to include moves
pub const MIN_CLOCK_TIME: u32 = 30;

/// Unified configuration for OXI chess engine training and inference
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct Config {
    // === DATA AND RUNTIME ===
    /// Path to data (PGN directory, PGN file, or CSV file)
    #[arg(long, default_value = "/lambda/nfs/chessbook")]
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
    #[serde(default)]
    pub resume: Option<bool>,

    // We only do one pass through the data, so this is mostly unused
    #[arg(long, default_value = "1.0")]
    pub train_ratio: f32,

    /// Batch size for training
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Physical batch size (for gradient accumulation)
    #[arg(long, default_value = "16000")]
    pub physical_batch_size: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of data loader workers
    #[arg(long, default_value = "4")]
    pub num_workers: usize,

    /// Minimum learning rate (end of training)
    #[arg(long, default_value = "1e-6")]
    pub lr_min: f64,

    /// Window size for plateau detection (number of iterations to compare)
    #[arg(long, default_value = "1000")]
    pub lr_window_size: usize,

    /// Minimum relative improvement threshold for plateau detection (e.g., 0.0005 = 0.05% improvement required)
    #[arg(long, default_value = "0.0005")]
    pub lr_improvement_threshold: f64,

    /// Factor to reduce learning rate by when plateau is detected (e.g., 0.5 means halve the LR)
    #[arg(long, default_value = "0.5")]
    pub lr_reduction_factor: f64,

    /// Multiplier applied to the learning rate calculated from batch size
    #[arg(long, default_value = "1.0")]
    pub lr_multiplier: f64,

    /// Warmup multiplier: warmup lasts for warmup_multiplier * effective_batch_size samples
    #[arg(long, default_value = "2.0")]
    pub warmup_multiplier: f64,

    /// Weight for policy loss
    #[arg(long, default_value = "0.15")]
    pub policy_loss_weight: f32,

    /// Label smoothing applied to the policy targets (smoothed over legal moves only)
    #[arg(long, default_value = "0.00")]
    pub policy_label_smoothing: f32,

    /// Weight for value loss
    #[arg(long, default_value = "0.0001")]
    pub value_loss_weight: f32,

    /// Entropy regularization weight for value predictions
    #[arg(long, default_value = "0.05")]
    pub value_entropy_weight: f32,

    #[arg(long, default_value = "0.0")]
    pub time_usage_loss_weight: f32,

    /// Weight decay for optimizer
    #[arg(long, default_value = "0.00001")]
    pub weight_decay: f64,

    /// Gradient clipping norm (0 to disable)
    #[arg(long, default_value = "3.0")]
    pub gradient_clip: f64,

    /// Enable verbose gradient norm breakdown logging
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    #[serde(default)]
    pub log_gradient_breakdown: Option<bool>,

    /// Number of attention heads to include when logging gradient breakdowns
    #[arg(long, default_value = "128")]
    #[serde(default)]
    pub gradient_head_limit: usize,

    /// Number of layers/modules to include when logging gradient breakdowns
    #[arg(long, default_value = "128")]
    #[serde(default)]
    pub gradient_layer_limit: usize,

    /// Enable adaptive GradNorm reweighting across heads
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    #[serde(default)]
    pub enable_gradnorm: Option<bool>,

    /// Optimizer steps between GradNorm weight updates
    #[arg(long, default_value = "20")]
    #[serde(default)]
    pub gradnorm_interval: usize,

    /// Alpha hyperparameter for GradNorm target scaling
    #[arg(long, default_value = "0.5")]
    pub gradnorm_alpha: f32,

    /// Multiplicative learning rate used when adjusting GradNorm weights
    #[arg(long, default_value = "0.5")]
    pub gradnorm_learning_rate: f32,

    /// Priority multiplier applied to policy GradNorm target (1.0 = neutral)
    #[arg(long, default_value = "20.0")]
    pub gradnorm_policy_priority: f32,

    /// Priority multiplier applied to value GradNorm target (1.0 = neutral)
    #[arg(long, default_value = "1.0")]
    pub gradnorm_value_priority: f32,

    /// Priority multiplier applied to time-usage GradNorm target (1.0 = neutral)
    #[arg(long, default_value = "1.0")]
    pub gradnorm_time_priority: f32,

    /// Number of samples to materialize on the lead device when probing GradNorm weights
    #[arg(long, default_value = "256")]
    #[serde(default)]
    pub gradnorm_probe_size: usize,

    /// Embedding dimension for tokens
    #[arg(long, default_value = "512")]
    pub embed_dim: usize,

    /// Number of transformer layers
    #[arg(long, default_value = "14")]
    pub num_layers: usize,

    /// Number of attention heads (head_dim = embed_dim / num_heads)
    #[arg(long, default_value = "8")]
    pub num_heads: usize,

    /// MLP hidden dimension ratio
    #[arg(long, default_value = "4.0")]
    pub mlp_ratio: f32,

    /// Number of convolutional layers applied over the 8x8 board grid before token embedding
    #[arg(long, default_value = "0")]
    #[serde(default)]
    pub conv_layers: usize,

    /// Only include positions with a single legal move
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub single_legal_move_only: Option<bool>,

    /// Disable terminal UI
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub disable_tui: Option<bool>,

    /// Only include positions that are checkmate
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub checkmate_only: Option<bool>,

    /// Probability of logging individual items for debugging (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub item_log_probability: f32,

    /// Enable detailed tensor norm logging during forward passes
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    #[serde(default)]
    pub log_tensor_norms: Option<bool>,

    /// Maximum number of tensor elements to print when previewing small tensors
    #[arg(long, default_value = "6")]
    #[serde(default)]
    pub norm_preview_limit: usize,

    /// Focal loss gamma parameter for policy head (0.0 disables focal loss)
    #[arg(long, default_value = "2.0")]
    pub focal_loss_gamma: f32,

    #[arg(long, default_value = "24")]
    pub smolgen_hidden: usize,

    #[arg(long, default_value = "128")]
    pub smolgen_global_dim: usize,

    #[arg(long, default_value = "128")]
    pub smolgen_gen_size: usize,

    /// Enable forward pass timing instrumentation for profiling
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_forward_timing: Option<bool>,

    /// Sample interval for forward timing (time every Nth forward pass)
    #[arg(long, default_value = "100")]
    pub forward_timing_interval: u64,

    #[arg(long, default_value = "1")]
    pub num_devices: usize,

    /// Probability of dropping positions based on ply (80% at ply 0, 0% at ply 10+)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_ply_sampling: Option<bool>,

    /// Probability of dropping games based on Elo (75% at 1000 Elo, 0% at 2000+ Elo)
    #[arg(long, default_missing_value="true", num_args=0..=1)]
    pub enable_elo_sampling: Option<bool>,

    /// Number of iterations between checkpoints
    #[arg(long, default_value = "100")]
    pub checkpoint_interval: usize,

    /// Number of TCEC (computer engine) samples to use for pretraining (0 to disable)
    #[arg(long, default_value = "0")]
    pub pretrain_samples: usize,

    /// Size of shuffle buffer for streaming data loading (number of examples to buffer before sampling)
    #[arg(long, default_value = "100000")]
    pub shuffle_buffer_size: usize,

    /// Interval for computing expensive metrics (top-5 accuracy, debug predictions, gradient breakdown, L2 penalty, tensor norms). 0 = never, 1 = every iteration.
    #[arg(long, default_value = "50")]
    pub full_metrics_interval: usize,

    /// Priority boost for advanced/expert ELO games (2000+). Value of 1.0 = 2x boost at 2500 ELO,
    /// 2.0 = 3x boost, 3.0 = 4x boost. Set to 0.0 to disable.
    #[arg(long, default_value = "3.0")]
    pub elo_priority_boost: f64,
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
    pub fn mlp_ratio(&self) -> f32 {
        self.mlp_ratio
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

    pub fn gradnorm_probe_size(&self) -> usize {
        self.gradnorm_probe_size.max(1)
    }

    pub fn forward_timing_enabled(&self) -> bool {
        self.enable_forward_timing.unwrap_or(false)
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
            weight_decay: 0.00001,
            gradient_clip: 3.0,
            log_gradient_breakdown: Some(false),
            gradient_head_limit: 128,
            gradient_layer_limit: 128,
            enable_gradnorm: Some(true),
            gradnorm_interval: 20,
            gradnorm_alpha: 0.5,
            gradnorm_learning_rate: 0.5,
            gradnorm_policy_priority: 20.0,
            gradnorm_value_priority: 1.0,
            gradnorm_time_priority: 1.0,
            gradnorm_probe_size: 256,
            lr_min: 0.000001,
            lr_window_size: 1000,
            lr_improvement_threshold: 0.0005,
            lr_reduction_factor: 0.5,
            lr_multiplier: 1.0,
            warmup_multiplier: 2.0,
            policy_loss_weight: 0.15,
            policy_label_smoothing: 0.03,
            value_loss_weight: 0.0001,
            value_entropy_weight: 0.05,
            embed_dim: 512,
            num_layers: 14,
            num_heads: 8,
            mlp_ratio: 4.0,
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
            focal_loss_gamma: 1.0,
            enable_ply_sampling: Some(true),
            enable_elo_sampling: Some(true),
            checkpoint_interval: 100,
            pretrain_samples: 0,
            shuffle_buffer_size: 100000,
            full_metrics_interval: 50,
            elo_priority_boost: 3.0,
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
