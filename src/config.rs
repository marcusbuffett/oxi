use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub const NUM_GLOBALS: usize = 7;
pub const LEGAL_MOVES: usize = 64 * 76;
// Per-position square features:
// - 12 piece one-hots (white/black 6 each)
// - 1 en passant
// - 1 castling right at this square
// - 6 attackers (white)
// - 6 attackers (black)
// - 1 legal move count normalized (moves / 20)
// - 1 pawn isolated
// - 1 pawn backward
// - 1 pawn doubled
// - 1 square control (kept as last per-position feature)
pub const FEATURES_PER_SQUARE_POSITION: usize = 31;
pub const PREVIOUS_POSITIONS: usize = 3;
pub const FEATURES_PER_TOKEN: usize = FEATURES_PER_SQUARE_POSITION * (PREVIOUS_POSITIONS + 1) + 1;

// Global config storage
static GLOBAL_CONFIG: OnceLock<Config> = OnceLock::new();

/// Unified configuration for OXI chess engine training and inference
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct Config {
    // === DATA AND RUNTIME ===
    /// Path to data (PGN directory, PGN file, or CSV file)
    #[arg(long)]
    pub data_path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub max_samples: Option<usize>,

    #[arg(long)]
    pub timeout: Option<u64>,

    /// Resume training from a specific checkpoint index
    #[arg(long)]
    pub checkpoint: Option<usize>,

    /// Train/validation split ratio (e.g., 0.9 for 90% train, 10% validation)
    #[arg(long, default_value = "0.9")]
    pub train_ratio: f32,

    /// Batch size for training
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Physical batch size (for gradient accumulation)
    #[arg(long, default_value = "256")]
    pub physical_batch_size: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of data loader workers
    #[arg(long, default_value = "4")]
    pub num_workers: usize,

    /// Number of training epochs
    #[arg(long, default_value = "10")]
    pub num_epochs: usize,

    /// Using NoamLrScheduler
    #[arg(long, default_value = "1.0")]
    pub learning_rate: f64,

    #[arg(long, default_value = "4000")]
    pub warmup: usize,

    #[arg(long)]
    pub learning_rate_min: Option<f64>,

    /// Number of validation samples
    #[arg(long)]
    pub validation_samples: Option<usize>,

    /// Weight for policy loss
    #[arg(long, default_value = "2.0")]
    pub policy_loss_weight: f32,

    /// Label smoothing applied to the policy targets (smoothed over legal moves only)
    #[arg(long, default_value = "0.05")]
    pub policy_label_smoothing: f32,

    /// Weight for value loss
    #[arg(long, default_value = "0.2")]
    pub value_loss_weight: f32,

    /// Weight for side info loss
    #[arg(long, default_value = "0.5")]
    pub side_info_loss_weight: f32,

    #[arg(long, default_value = "2.0")]
    pub time_usage_loss_weight: f32,

    /// Weight decay for optimizer
    #[arg(long, default_value = "0.01")]
    pub weight_decay: f64,

    /// Gradient clipping norm (0 to disable)
    #[arg(long, default_value = "1.0")]
    pub gradient_clip: f64,

    /// Embedding dimension for tokens
    #[arg(long, default_value = "384")]
    pub embed_dim: usize,

    /// Number of attention heads
    #[arg(long, default_value = "8")]
    pub num_heads: usize,

    /// Number of K/V head groups for grouped-query attention (GQA). Must divide num_heads.
    /// Set equal to num_heads to disable grouping; set to 1 for multi-query attention (MQA).
    #[arg(long)]
    pub kv_groups: Option<usize>,

    /// Number of transformer layers
    #[arg(long, default_value = "10")]
    pub num_layers: usize,

    /// ELO bins for player skill levels
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800"
    )]
    pub elo_bins: Vec<i32>,

    /// Embedding dimension for ELO
    #[arg(long, default_value = "256")]
    pub elo_embed_dim: usize,

    /// MLP hidden dimension ratio
    #[arg(long, default_value = "4.0")]
    pub mlp_ratio: f32,

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
    #[arg(long, default_value = "0.01")]
    pub item_log_probability: f32,

    /// Disable Shaw-style relative positional representations in attention
    #[arg(long, default_value = "false")]
    pub disable_shaw_pr: bool,

    /// Disable Rotary Positional Embeddings (RoPE)
    #[arg(long, default_value = "false")]
    pub disable_rope: bool,

    #[arg(long, default_value = "1")]
    pub num_devices: usize,
}

impl Config {
    /// Create new config with explicit parameters for testing
    pub fn new(embed_dim: usize, num_heads: usize, num_layers: usize) -> Self {
        Self {
            embed_dim,
            num_heads,
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
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn kv_groups(&self) -> usize {
        self.kv_groups.unwrap_or(self.num_heads)
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn seq_len(&self) -> usize {
        64
    }
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn elo_bins(&self) -> &[i32] {
        &self.elo_bins
    }
    pub fn elo_embed_dim(&self) -> usize {
        self.elo_embed_dim
    }
    pub fn mlp_ratio(&self) -> f32 {
        self.mlp_ratio
    }

    pub fn checkpoint_interval(&self) -> usize {
        1000
    }

    pub fn disable_rope(&self) -> bool {
        self.disable_rope
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
            data_path: None,
            train_ratio: 0.9,
            batch_size: None,
            physical_batch_size: 8,
            seed: 42,
            num_workers: 4,
            num_epochs: 10,
            weight_decay: 0.0001,
            gradient_clip: 1.0,
            learning_rate: 0.001,
            validation_samples: None,
            policy_loss_weight: 1.0,
            policy_label_smoothing: 0.05,
            value_loss_weight: 1.0,
            side_info_loss_weight: 0.5,
            embed_dim: 256,
            num_heads: 4,
            kv_groups: None,
            num_layers: 6,
            elo_bins: (800i32..=2800i32).step_by(200).collect(),
            elo_embed_dim: 128,
            mlp_ratio: 4.0,
            max_samples: None,
            timeout: None,
            checkpoint: None,
            single_legal_move_only: false,
            checkmate_only: false,
            item_log_probability: 1.00,
            time_usage_loss_weight: 0.0,
            disable_tui: true,
            learning_rate_min: None,
            warmup: 4000,
            disable_shaw_pr: false,
            disable_rope: false,
            num_devices: 1,
        }
    }
}

// Legacy type aliases for backward compatibility during transition
pub type ModelConfig = Config;
pub type TrainingConfig = Config;
