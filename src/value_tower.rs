//! Value Tower: A dedicated transformer stack for WDL prediction.
//!
//! The value tower takes stop-grad tokens from the main trunk and processes them
//! through additional transformer blocks before the value head. This decouples
//! value training from the policy-optimized trunk.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};

use crate::config::{get_global_config, NUM_GLOBALS};
use crate::relative_position_transformer::TransformerBlock;
use crate::smolgen::SmolgenWeightGen;

#[cfg(feature = "train")]
use crate::forward_timing::TimingScope;
#[cfg(feature = "train")]
use crate::norm_debug::log_tensor_stats;
#[cfg(not(feature = "train"))]
use crate::train_stubs::{log_tensor_stats, TimingScope};

/// Value Tower: 2-layer (configurable) transformer stack for value prediction.
/// Runs on stop_grad tokens from the main trunk.
#[derive(Module, Debug)]
pub struct ValueTower<B: Backend> {
    /// Transformer blocks dedicated to value processing
    blocks: Vec<TransformerBlock<B>>,
    /// Shared Smolgen weight generator (separate from trunk's)
    smolgen_weight_gen: SmolgenWeightGen<B>,
    /// Final layer norm after value tower blocks
    norm: RmsNorm<B>,
    /// Attention pooling: project tokens to attention scores
    pool_fc1: Linear<B>,
    pool_fc2: Linear<B>,
    /// Value head hidden layer: pooled + globals -> embed_dim
    head_hidden: Linear<B>,
    /// Value head output: embed_dim -> 3 (loss, draw, win)
    head_output: Linear<B>,
}

impl<B: Backend> ValueTower<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();
        let num_layers = config.value_tower_layers();

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        // Create transformer blocks for value tower
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(device));
        }

        // Shared Smolgen weight generator for value tower
        let smolgen_weight_gen = SmolgenWeightGen::new(device);

        // Final layer norm
        let norm = RmsNormConfig::new(embed_dim).init(device);

        // Attention pooling components
        let pool_fc1 = LinearConfig::new(embed_dim, embed_dim)
            .with_initializer(std_init.clone())
            .init(device);
        let pool_fc2 = LinearConfig::new(embed_dim, 1)
            .with_initializer(std_init.clone())
            .init(device);

        // Value head MLP
        let head_hidden = LinearConfig::new(embed_dim + NUM_GLOBALS, embed_dim)
            .with_initializer(std_init.clone())
            .init(device);
        let head_output = LinearConfig::new(embed_dim, 3)
            .with_initializer(std_init.clone())
            .init(device);

        Self {
            blocks,
            smolgen_weight_gen,
            norm,
            pool_fc1,
            pool_fc2,
            head_hidden,
            head_output,
        }
    }

    /// Forward pass through value tower.
    ///
    /// Args:
    ///   - tokens: [batch, 64, embed_dim] - **should be stop_grad** from trunk
    ///   - globals: [batch, NUM_GLOBALS] - global features for FiLM conditioning
    ///
    /// Returns:
    ///   - value_logits: [batch, 3] - logits for (loss, draw, win)
    pub fn forward(&self, tokens: Tensor<B, 3>, globals: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = tokens.device();
        let _timing = TimingScope::new_with_sync::<B>("value_tower", &device);

        log_tensor_stats("value_tower.input", &tokens);

        // Pass through value tower blocks with FiLM conditioning
        let mut x = tokens;
        for (i, block) in self.blocks.iter().enumerate() {
            let _block_timing = TimingScope::new_with_sync::<B>("value_tower_block", &device);
            log_tensor_stats(&format!("value_tower.block{i}.input"), &x);
            x = block.forward_with_film(x, &self.smolgen_weight_gen, globals.clone());
            log_tensor_stats(&format!("value_tower.block{i}.output"), &x);
        }

        // Final layer norm
        x = self.norm.forward(x);
        log_tensor_stats("value_tower.post_norm", &x);

        // Attention pooling: compute attention weights over 64 squares
        let attn_hidden = silu(self.pool_fc1.forward(x.clone())); // [batch, 64, embed_dim]
        let attn_logits = self.pool_fc2.forward(attn_hidden); // [batch, 64, 1]
        let attn_weights = softmax(attn_logits.squeeze::<2>(), 1); // [batch, 64]
        log_tensor_stats("value_tower.attn_weights", &attn_weights);

        // Weighted sum: [batch, 64] @ [batch, 64, embed_dim] -> [batch, embed_dim]
        let attn_weights_expanded = attn_weights.unsqueeze_dim(1); // [batch, 1, 64]
        let pooled = attn_weights_expanded.matmul(x).squeeze::<2>(); // [batch, embed_dim]
        log_tensor_stats("value_tower.pooled", &pooled);

        // Concatenate pooled tokens with globals
        let combined = Tensor::cat(vec![pooled, globals], 1); // [batch, embed_dim + NUM_GLOBALS]
        log_tensor_stats("value_tower.combined", &combined);

        // Value head MLP
        let hidden = silu(self.head_hidden.forward(combined));
        log_tensor_stats("value_tower.head_hidden", &hidden);

        let value_logits = self.head_output.forward(hidden);
        log_tensor_stats("value_tower.value_logits", &value_logits);

        value_logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{set_global_config, Config};
    use crate::test_backend::{test_device, TestBackend};

    fn ensure_config() {
        let _ = set_global_config(Config::new(96, 2));
    }

    #[test]
    fn value_tower_smoke_shapes() {
        ensure_config();
        let device = test_device();
        let config = get_global_config();
        let tower = ValueTower::<TestBackend>::new(&device);

        let batch_size = 2usize;
        let seq_len = 64usize;
        let embed_dim = config.embed_dim();

        let tokens = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let globals = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let value_logits = tower.forward(tokens, globals);
        assert_eq!(value_logits.dims(), [batch_size, 3]);
    }
}
