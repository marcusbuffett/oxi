use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Configurable parameters
    pub num_blocks: usize,
    pub channels: usize,
}

impl ModelConfig {
    pub fn new(num_blocks: usize, channels: usize) -> Self {
        Self {
            num_blocks,
            channels,
        }
    }

    // Fixed architecture parameters derived from channels
    pub fn se_channels(&self) -> usize {
        self.channels / 8 // SE reduction ratio of 8
    }

    pub fn policy_channels(&self) -> usize {
        (self.channels as f32 * 0.316).round() as usize // ~81/256 ratio from paper
    }

    pub fn embed_dim(&self) -> usize {
        self.channels // Same as channels
    }

    pub fn num_heads(&self) -> usize {
        (self.channels / 32).max(4) // Scale with channels, min 4
    }

    pub fn num_layers(&self) -> usize {
        6 // Fixed transformer layers
    }

    pub fn mlp_ratio(&self) -> f32 {
        2.0
    }

    pub fn max_seq_len(&self) -> usize {
        64
    }

    pub fn dropout(&self) -> f64 {
        0.2
    }

    pub fn attention_dropout(&self) -> f64 {
        0.2
    }

    pub fn num_board_channels(&self) -> usize {
        16
    }

    pub fn num_moves(&self) -> usize {
        4096 // 64x64 from-to squares
    }

    pub fn num_piece_types(&self) -> usize {
        6
    }

    pub fn elo_bins(&self) -> Vec<i32> {
        vec![1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 3000]
    }

    pub fn elo_embed_dim(&self) -> usize {
        32
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_blocks: 15,
            channels: 256,
        }
    }
}

// Fixed training configuration
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 1e-3,
            num_epochs: 10,
        }
    }
}

impl TrainingConfig {
    // Fixed parameters as methods
    pub fn weight_decay(&self) -> f64 {
        1e-5
    }

    pub fn warmup_steps(&self) -> usize {
        1000
    }

    pub fn gradient_clip(&self) -> f32 {
        0.0
    }

    pub fn policy_loss_weight(&self) -> f32 {
        2.0
    }

    pub fn value_loss_weight(&self) -> f32 {
        1.0
    }

    pub fn side_info_loss_weight(&self) -> f32 {
        5.0
    }

    pub fn num_workers(&self) -> usize {
        4
    }

    pub fn seed(&self) -> u64 {
        42
    }

    pub fn checkpoint_interval(&self) -> usize {
        1000
    }
}

// Simplified config - no more YAML files needed
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

impl Config {
    pub fn new(num_blocks: usize, channels: usize) -> Self {
        Self {
            model: ModelConfig::new(num_blocks, channels),
            training: TrainingConfig::default(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}
