use clap::Parser;
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Normal};
use std::sync::OnceLock;

pub const NUM_GLOBALS: usize = 6;
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
// - Tactical group (38): attackers, pins, pin target, hanging flag, square control, diagonal/cardinal ray features
// - Positional group (25): legal moves, pawn structure, weak squares, open file, passed pawn, dark-square flag, rank/file one-hots
// - Misc group (2): en passant target, local castling right
pub const PIECE_IDENTITY_FEATURES: usize = 12;
pub const TACTICAL_FEATURES: usize = 38;
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
    #[arg(long)]
    pub data_path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub max_samples: Option<usize>,

    /// Number of initial samples to skip during PGN processing
    #[arg(long)]
    pub skip: Option<usize>,

    #[arg(long)]
    pub timeout: Option<u64>,

    /// Resume training from the last saved model checkpoint
    #[arg(long)]
    #[serde(default)]
    pub resume: bool,

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

    /// Maximum learning rate (after warmup)
    #[arg(long, default_value = "0.0005")]
    pub lr_max: f64,

    /// Minimum learning rate (end of training)
    #[arg(long, default_value = "0.000001")]
    pub lr_min: f64,

    /// Learning rate scalar multiplier for scale parameters (scale_qk, scale_v)
    #[arg(long, default_value = "100.0")]
    pub lr_scalar: f64,

    /// Number of warmup steps (batches, not optimizer steps). If not specified, defaults to 10% of total batches
    #[arg(long)]
    pub warmup_steps: Option<usize>,

    /// Number of validation samples
    #[arg(long)]
    pub validation_samples: Option<usize>,

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
    #[arg(long, default_value = "false")]
    #[serde(default)]
    pub log_gradient_breakdown: bool,

    /// How often (in optimizer steps) to log gradient breakdown details
    #[arg(long, default_value = "16")]
    #[serde(default)]
    pub gradient_breakdown_interval: usize,

    /// Number of attention heads to include when logging gradient breakdowns
    #[arg(long, default_value = "128")]
    #[serde(default)]
    pub gradient_head_limit: usize,

    /// Number of layers/modules to include when logging gradient breakdowns
    #[arg(long, default_value = "128")]
    #[serde(default)]
    pub gradient_layer_limit: usize,

    /// How often (in optimizer steps) to log L2 penalty from weight decay
    #[arg(long, default_value = "100")]
    #[serde(default)]
    pub l2_penalty_log_interval: usize,

    /// Enable adaptive GradNorm reweighting across heads
    #[arg(long, default_value = "true")]
    #[serde(default)]
    pub enable_gradnorm: bool,

    /// Optimizer steps between GradNorm weight updates
    #[arg(long, default_value = "8")]
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
    #[arg(long, default_value = "768")]
    pub embed_dim: usize,

    /// Number of transformer layers
    #[arg(long, default_value = "14")]
    pub num_layers: usize,

    /// Number of key/value heads used for grouped-query attention (defaults to num_heads)
    #[arg(long)]
    pub num_kv_heads: Option<usize>,

    /// MLP hidden dimension ratio
    #[arg(long, default_value = "4.0")]
    pub mlp_ratio: f32,

    /// Number of convolutional layers applied over the 8x8 board grid before token embedding
    #[arg(long, default_value = "0")]
    #[serde(default)]
    pub conv_layers: usize,

    /// Only include positions with a single legal move
    #[arg(long)]
    pub single_legal_move_only: bool,

    /// Only include positions with a single legal move
    #[arg(long, default_value = "false")]
    pub disable_tui: bool,

    /// Only include positions that are checkmate
    #[arg(long)]
    pub checkmate_only: bool,

    /// Probability of logging individual items for debugging (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub item_log_probability: f32,

    /// Enable detailed tensor norm logging during forward passes
    #[arg(long, default_value = "false")]
    #[serde(default)]
    pub log_tensor_norms: bool,

    /// How often (in forward passes) to record tensor norm snapshots
    #[arg(long, default_value = "1")]
    #[serde(default)]
    pub norm_log_interval: usize,

    /// Maximum number of tensor elements to print when previewing small tensors
    #[arg(long, default_value = "6")]
    #[serde(default)]
    pub norm_preview_limit: usize,

    /// Focal loss gamma parameter for policy head (0.0 disables focal loss)
    #[arg(long, default_value = "2.0")]
    pub focal_loss_gamma: f32,

    /// Disable Shaw-style relative positional representations in attention
    #[arg(long, default_value = "false")]
    pub disable_shaw_pr: bool,

    #[arg(long, default_value = "1")]
    pub num_devices: usize,

    /// Probability of dropping positions based on ply (80% at ply 0, 0% at ply 10+)
    #[arg(long, default_value = "true")]
    pub enable_ply_sampling: bool,

    /// Probability of dropping games based on Elo (75% at 1000 Elo, 0% at 2000+ Elo)
    #[arg(long, default_value = "true")]
    pub enable_elo_sampling: bool,

    /// Number of iterations between checkpoints
    #[arg(long, default_value = "100")]
    pub checkpoint_interval: usize,

    #[arg(long, default_value = "1000000")]
    pub num_pretrain_steps: usize,

    /// Maximum number of easy positions to load for pretraining
    #[arg(long)]
    pub max_easy_positions: Option<usize>,
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
        12
    }
    pub fn non_global_dim(&self) -> usize {
        self.embed_dim - self.global_dim()
    }
    pub fn head_dim(&self) -> usize {
        // head_dim is min(64, embed_dim)
        64.min(self.embed_dim)
    }
    pub fn num_heads(&self) -> usize {
        // num_heads = embed_dim / head_dim (rounded down)
        self.embed_dim / self.head_dim()
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads())
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
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads() * self.head_dim()
    }
    pub fn gqa_group_size(&self) -> usize {
        return 1;
        let num_kv = self.num_kv_heads();
        let num_heads = self.num_heads();
        assert!(
            num_heads % num_kv == 0,
            "num_heads ({}) must be divisible by num_kv_heads ({})",
            num_heads,
            num_kv
        );
        num_heads / num_kv
    }

    pub fn mlp_ratio(&self) -> f32 {
        self.mlp_ratio
    }

    pub fn log_tensor_norms(&self) -> bool {
        self.log_tensor_norms
    }

    pub fn norm_log_interval(&self) -> usize {
        self.norm_log_interval.max(1)
    }

    pub fn norm_preview_limit(&self) -> usize {
        self.norm_preview_limit.max(1)
    }

    pub fn log_gradient_breakdown(&self) -> bool {
        self.log_gradient_breakdown
    }

    pub fn gradient_breakdown_interval(&self) -> usize {
        self.gradient_breakdown_interval.max(1)
    }

    pub fn gradient_head_limit(&self) -> usize {
        self.gradient_head_limit.max(1)
    }

    pub fn gradient_layer_limit(&self) -> usize {
        self.gradient_layer_limit.max(1)
    }

    pub fn l2_penalty_log_interval(&self) -> usize {
        self.l2_penalty_log_interval.max(1)
    }

    pub fn gradnorm_enabled(&self) -> bool {
        self.enable_gradnorm
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

    // etc... (other accessors)
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
            train_ratio: 1.0,
            batch_size: None,
            physical_batch_size: 16000,
            seed: 42,
            num_workers: 4,
            weight_decay: 0.00001,
            gradient_clip: 3.0,
            log_gradient_breakdown: false,
            gradient_breakdown_interval: 16,
            gradient_head_limit: 128,
            gradient_layer_limit: 128,
            l2_penalty_log_interval: 100,
            enable_gradnorm: true,
            gradnorm_interval: 8,
            gradnorm_alpha: 0.5,
            gradnorm_learning_rate: 0.5,
            gradnorm_policy_priority: 20.0,
            gradnorm_value_priority: 1.0,
            gradnorm_time_priority: 1.0,
            gradnorm_probe_size: 256,
            lr_max: 0.0005,
            lr_min: 0.000001,
            lr_scalar: 100.0,
            warmup_steps: Some(1000),
            validation_samples: None,
            policy_loss_weight: 0.15,
            policy_label_smoothing: 0.03,
            value_loss_weight: 0.0001,
            value_entropy_weight: 0.05,
            embed_dim: 768,
            num_layers: 14,
            num_kv_heads: None,
            mlp_ratio: 4.0,
            conv_layers: 0,
            max_samples: Some(240000000),
            skip: None,
            timeout: None,
            resume: false,
            single_legal_move_only: false,
            checkmate_only: false,
            item_log_probability: 1.00,
            log_tensor_norms: false,
            norm_log_interval: 100,
            norm_preview_limit: 6,
            time_usage_loss_weight: 0.0,
            disable_tui: false,
            disable_shaw_pr: false,
            num_devices: 1,
            focal_loss_gamma: 1.0,
            enable_ply_sampling: true,
            enable_elo_sampling: true,
            checkpoint_interval: 100,
            num_pretrain_steps: 1000000,
            max_easy_positions: Some(0),
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

static ELO_DISTRIBUTION: Lazy<Normal> = Lazy::new(|| {
    // Fix mean to 1500; adjust std based on data (try 350-400 for blitz/rapid)
    Normal::new(1500.0, 400.0).expect("Failed to create normal distribution")
});

pub fn elo_keep_probability(avg_elo: f64) -> f64 {
    let normalize_keep_prob = {
        let natural_frequency = ELO_DISTRIBUTION.pdf(avg_elo);
        if natural_frequency == 0.0 {
            return 0.0;
        } // Edge case for extreme ELO

        let peak_density = ELO_DISTRIBUTION.pdf(1500.0);
        let relative_frequency = natural_frequency / peak_density; // 0 to 1

        // Tunable: 0.1 = keep ~10% at mean, flat up to ~ELO 2570 (with std=400, z~2.15)
        // Increase to 0.3-0.5 for less aggressive (more 1500 games, but flat only to ~2000-2200)
        // Decrease to 0.01 for more aggressive (like your current, flat farther but fewer center games)
        let flattening_factor = 0.1;

        // Inverse: boost tails, downsample center
        (flattening_factor / relative_frequency).clamp(0.0, 1.0)
    };

    let graduated_keep_prob = {
        let min_elo = 1000;
        let max_elo = 2300;
        let elo_clamped = avg_elo.clamp(min_elo as f64, max_elo as f64);
        let elo_normalized = (elo_clamped - min_elo as f64) / (max_elo - min_elo) as f64;
        let keep_prob = 0.2 + (0.8 * elo_normalized);
        keep_prob
    };

    normalize_keep_prob * graduated_keep_prob
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
    if !config.enable_ply_sampling {
        return true;
    }
    rng_value < ply_keep_probability(ply)
}

/// Check if a game should be kept based on Elo and random sampling
pub fn should_keep_game_by_elo(white_elo: i32, black_elo: i32, rng_value: f64) -> bool {
    let config = get_global_config();
    if !config.enable_elo_sampling {
        eprintln!("DEBUG: ELO sampling is DISABLED!");
        return true;
    }
    let avg_elo = (white_elo + black_elo) as f64 / 2.0;
    let keep_prob = elo_keep_probability(avg_elo);
    let result = rng_value < keep_prob;

    result
}

// Legacy type aliases for backward compatibility during transition
pub type ModelConfig = Config;
pub type TrainingConfig = Config;
