use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::silu;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, Tensor};

use crate::config::Config;
use crate::smolgen::{SmolgenAttention, SmolgenWeightGen};

#[cfg(feature = "train")]
use crate::forward_timing::TimingScope;
#[cfg(feature = "train")]
use crate::norm_debug::log_tensor_stats;
#[cfg(not(feature = "train"))]
use crate::train_stubs::{log_tensor_stats, TimingScope};

/// FiLM-conditioned RmsNorm: generates its own gamma/beta from global features
/// Each instance lives inside a transformer block and conditions on globals passed at forward time
#[derive(Module, Debug)]
pub struct FiLMRmsNorm<B: Backend> {
    rms_norm: RmsNorm<B>,
    gamma_proj: Linear<B>,
    beta_proj: Linear<B>,
}

impl<B: Backend> FiLMRmsNorm<B> {
    pub fn new(device: &Device<B>, embed_dim: usize, global_dim: usize) -> Self {
        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        Self {
            rms_norm: RmsNormConfig::new(embed_dim).init(device),
            gamma_proj: LinearConfig::new(global_dim, embed_dim)
                .with_initializer(std_init.clone())
                .init(device),
            beta_proj: LinearConfig::new(global_dim, embed_dim)
                .with_initializer(std_init.clone())
                .init(device),
        }
    }

    /// Forward with FiLM conditioning from globals
    /// x: [batch, seq, embed_dim]
    /// globals: [batch, global_dim]
    pub fn forward(&self, x: Tensor<B, 3>, globals: Tensor<B, 2>) -> Tensor<B, 3> {
        let normed = self.rms_norm.forward(x);
        let gamma = self.gamma_proj.forward(globals.clone()) + 1.0;
        let beta = self.beta_proj.forward(globals);
        let gamma = gamma.unsqueeze_dim(1);
        let beta = beta.unsqueeze_dim(1);
        normed * gamma + beta
    }

    /// Forward without FiLM modulation (for inference without globals)
    pub fn forward_plain(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.rms_norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: SmolgenAttention<B>,
    norm1: FiLMRmsNorm<B>,
    norm2: FiLMRmsNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &Config, device: &Device<B>) -> Self {
        let attention = SmolgenAttention::new(config, device);
        let norm1 = FiLMRmsNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let norm2 = FiLMRmsNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let mlp = MLP::new(config, device);
        Self {
            attention,
            norm1,
            norm2,
            mlp,
        }
    }

    /// Create a TransformerBlock for a single-block head (policy/value).
    /// Uses depth=1 residual scaling instead of depth=num_layers.
    pub fn new_for_head(config: &Config, device: &Device<B>) -> Self {
        let attention = SmolgenAttention::new_for_head(config, device);
        let norm1 = FiLMRmsNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let norm2 = FiLMRmsNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let mlp = MLP::new_for_head(config, device);
        Self {
            attention,
            norm1,
            norm2,
            mlp,
        }
    }

    /// Forward pass without FiLM conditioning (pre-norm)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        log_tensor_stats("block.input", &x);

        // Pre-norm: norm before attention, then residual add
        let normed1 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_attn", &device);
            self.norm1.forward_plain(x.clone())
        };
        log_tensor_stats("block.norm1", &normed1);

        let attn_out = {
            let _t = TimingScope::new_with_sync::<B>("block_attention", &device);
            self.attention.forward(normed1, shared_weight_gen)
        };
        log_tensor_stats("block.attn_out", &attn_out);

        let x = x + attn_out;
        log_tensor_stats("block.post_attn_residual", &x);

        // Pre-norm: norm before MLP, then residual add
        let normed2 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_mlp", &device);
            self.norm2.forward_plain(x.clone())
        };
        log_tensor_stats("block.norm2", &normed2);

        let mlp_out = {
            let _t = TimingScope::new_with_sync::<B>("block_mlp", &device);
            self.mlp.forward(normed2)
        };
        log_tensor_stats("block.mlp_out", &mlp_out);

        let output = x + mlp_out;
        log_tensor_stats("block.output", &output);

        output
    }

    /// Forward pass with FiLM conditioning from globals (pre-norm)
    /// globals: [batch, NUM_GLOBALS] - passed to each FiLMRmsNorm to generate gamma/beta
    pub fn forward_with_film(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
        globals: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        log_tensor_stats("block.input", &x);

        // Pre-norm: norm before attention, then residual add
        let normed1 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_attn", &device);
            self.norm1.forward(x.clone(), globals.clone())
        };
        log_tensor_stats("block.norm1", &normed1);

        let attn_out = {
            let _t = TimingScope::new_with_sync::<B>("block_attention", &device);
            self.attention.forward(normed1, shared_weight_gen)
        };
        log_tensor_stats("block.attn_out", &attn_out);

        let x = x + attn_out;
        log_tensor_stats("block.post_attn_residual", &x);

        // Pre-norm: norm before MLP, then residual add
        let normed2 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_mlp", &device);
            self.norm2.forward(x.clone(), globals)
        };
        log_tensor_stats("block.norm2", &normed2);

        let mlp_out = {
            let _t = TimingScope::new_with_sync::<B>("block_mlp", &device);
            self.mlp.forward(normed2)
        };
        log_tensor_stats("block.mlp_out", &mlp_out);

        let output = x + mlp_out;
        log_tensor_stats("block.output", &output);

        output
    }

    /// Inference-only FiLM-conditioned forward pass that also returns this block's
    /// post-softmax attention weights.
    ///
    /// Mirrors `forward_with_film` but routes through `SmolgenAttention::forward_with_attn`
    /// to capture the attention map. The training paths (`forward`, `forward_with_film`)
    /// are untouched.
    ///
    /// Returns `(block_output, attn_weights)` where `attn_weights` has shape
    /// `[batch, num_heads, 64, 64]`.
    pub fn forward_with_attn(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
        globals: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let device = x.device();
        log_tensor_stats("block.input", &x);

        // Pre-norm: norm before attention, then residual add
        let normed1 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_attn", &device);
            self.norm1.forward(x.clone(), globals.clone())
        };
        log_tensor_stats("block.norm1", &normed1);

        let (attn_out, attn_weights) = {
            let _t = TimingScope::new_with_sync::<B>("block_attention", &device);
            self.attention.forward_with_attn(normed1, shared_weight_gen)
        };
        log_tensor_stats("block.attn_out", &attn_out);

        let x = x + attn_out;
        log_tensor_stats("block.post_attn_residual", &x);

        // Pre-norm: norm before MLP, then residual add
        let normed2 = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_mlp", &device);
            self.norm2.forward(x.clone(), globals)
        };
        log_tensor_stats("block.norm2", &normed2);

        let mlp_out = {
            let _t = TimingScope::new_with_sync::<B>("block_mlp", &device);
            self.mlp.forward(normed2)
        };
        log_tensor_stats("block.mlp_out", &mlp_out);

        let output = x + mlp_out;
        log_tensor_stats("block.output", &output);

        (output, attn_weights)
    }
}

/// SwiGLU MLP with fused gate+up projection for memory efficiency.
/// Uses a single Linear(D→2H) and splits into gate/up, avoiding extra allocations.
/// Formula: SwiGLU(x) = (SiLU(gate) * up) @ W_down
///   where [gate, up] = x @ W_fused (split along last dim)
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// Fused projection: embed_dim -> 2 * hidden_dim (gate and up combined)
    fused_gate_up: Linear<B>,
    /// Down projection: hidden_dim -> embed_dim
    down_proj: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn new(config: &Config, device: &Device<B>) -> Self {
        let hidden_dim = (config.embed_dim() as f32 * 2.5) as usize;

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        // Residual scaling: 1/sqrt(2*num_layers) for residual projections
        let residual_std = 0.02 / (2.0 * config.num_layers() as f64).sqrt();
        let residual_init = Initializer::Normal {
            mean: 0.0,
            std: residual_std,
        };

        // Fused projection outputs 2*hidden_dim, which we split into gate and up
        let fused_gate_up = LinearConfig::new(config.embed_dim(), 2 * hidden_dim)
            .with_initializer(std_init)
            .init(device);
        let down_proj = LinearConfig::new(hidden_dim, config.embed_dim())
            .with_initializer(residual_init)
            .init(device);
        Self {
            fused_gate_up,
            down_proj,
        }
    }

    pub fn new_for_head(config: &Config, device: &Device<B>) -> Self {
        let hidden_dim = (config.embed_dim() as f32 * 2.5) as usize;

        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        // Residual scaling for a single block: 1/sqrt(2*1) = 1/sqrt(2)
        let residual_std = 0.02 / (2.0_f64).sqrt();
        let residual_init = Initializer::Normal {
            mean: 0.0,
            std: residual_std,
        };

        let fused_gate_up = LinearConfig::new(config.embed_dim(), 2 * hidden_dim)
            .with_initializer(std_init)
            .init(device);
        let down_proj = LinearConfig::new(hidden_dim, config.embed_dim())
            .with_initializer(residual_init)
            .init(device);
        Self {
            fused_gate_up,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        log_tensor_stats("mlp.input", &x);

        // Fused projection: [B, seq, D] -> [B, seq, 2H]
        let fused = self.fused_gate_up.forward(x);
        log_tensor_stats("mlp.fused_out", &fused);

        let [b, s, h2] = fused.dims();
        let hidden_dim = h2 / 2;
        let gate = fused.clone().slice([0..b, 0..s, 0..hidden_dim]);
        let up = fused.slice([0..b, 0..s, hidden_dim..h2]);

        // SwiGLU: SiLU(gate) * up
        let activated = silu(gate) * up;
        log_tensor_stats("mlp.swiglu", &activated);

        // Down projection
        let output = self.down_proj.forward(activated);
        log_tensor_stats("mlp.down_out", &output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::test_backend::{test_device, TestBackend};

    #[test]
    fn transformer_block_smoke_shapes() {
        let device = test_device();
        let config = Config::new(96, 2);
        let block = TransformerBlock::<TestBackend>::new(&config, &device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&config, &device);
        let batch_size = 2usize;
        let seq_len = 64usize;
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x, &weight_gen);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }

    #[test]
    #[should_panic]
    fn transformer_block_panics_on_wrong_seq_len() {
        let device = test_device();
        let config = Config::new(96, 2);
        let block = TransformerBlock::<TestBackend>::new(&config, &device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&config, &device);
        let batch_size = 1usize;
        let seq_len = 32usize;
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let _ = block.forward(x, &weight_gen);
    }

    #[test]
    #[ignore]
    fn transformer_block_large_batch_shapes() {
        let device = test_device();
        let config = Config::new(96, 2);
        let block = TransformerBlock::<TestBackend>::new(&config, &device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&config, &device);
        let batch_size = 1024usize;
        let seq_len = 64usize;
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x, &weight_gen);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }
}
