use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, Tensor};

use crate::config::get_global_config;
use crate::smolgen::{SmolgenAttention, SmolgenWeightGen};

#[cfg(feature = "train")]
use crate::forward_timing::TimingScope;
#[cfg(feature = "train")]
use crate::norm_debug::log_tensor_stats;
#[cfg(not(feature = "train"))]
use crate::train_stubs::{log_tensor_stats, TimingScope};

/// FiLM-conditioned LayerNorm: generates its own gamma/beta from global features
/// Each instance lives inside a transformer block and conditions on globals passed at forward time
#[derive(Module, Debug)]
pub struct FiLMLayerNorm<B: Backend> {
    layer_norm: LayerNorm<B>,
    gamma_proj: Linear<B>,
    beta_proj: Linear<B>,
}

impl<B: Backend> FiLMLayerNorm<B> {
    pub fn new(device: &Device<B>, embed_dim: usize, global_dim: usize) -> Self {
        Self {
            layer_norm: LayerNormConfig::new(embed_dim).init(device),
            gamma_proj: LinearConfig::new(global_dim, embed_dim).init(device),
            beta_proj: LinearConfig::new(global_dim, embed_dim).init(device),
        }
    }

    /// Forward with FiLM conditioning from globals
    /// x: [batch, seq, embed_dim]
    /// globals: [batch, global_dim]
    pub fn forward(&self, x: Tensor<B, 3>, globals: Tensor<B, 2>) -> Tensor<B, 3> {
        let normed = self.layer_norm.forward(x);
        let gamma = self.gamma_proj.forward(globals.clone()) + 1.0;
        let beta = self.beta_proj.forward(globals);
        let gamma = gamma.unsqueeze_dim(1);
        let beta = beta.unsqueeze_dim(1);
        normed * gamma + beta
    }

    /// Forward without FiLM modulation (for inference without globals)
    pub fn forward_plain(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layer_norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: SmolgenAttention<B>,
    norm_post_attn: FiLMLayerNorm<B>,
    norm_post_mlp: FiLMLayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let attention = SmolgenAttention::new(device);
        let norm_post_attn =
            FiLMLayerNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let norm_post_mlp =
            FiLMLayerNorm::new(device, config.embed_dim(), crate::config::NUM_GLOBALS);
        let mlp = MLP::new(device);
        Self {
            attention,
            norm_post_attn,
            norm_post_mlp,
            mlp,
        }
    }

    /// Forward pass without FiLM conditioning
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        log_tensor_stats("block.input", &x);

        let attn_out = {
            let _t = TimingScope::new_with_sync::<B>("block_attention", &device);
            self.attention.forward(x.clone(), shared_weight_gen)
        };
        log_tensor_stats("block.attn_out", &attn_out);

        let residual_attn = x + attn_out;
        log_tensor_stats("block.post_attn", &residual_attn);

        let post_attn = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_attn", &device);
            self.norm_post_attn.forward_plain(residual_attn)
        };
        log_tensor_stats("block.norm_post_attn", &post_attn);

        let mlp_out = {
            let _t = TimingScope::new_with_sync::<B>("block_mlp", &device);
            self.mlp.forward(post_attn.clone())
        };
        log_tensor_stats("block.mlp_out", &mlp_out);

        let residual_mlp = post_attn + mlp_out;

        let output = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_mlp", &device);
            self.norm_post_mlp.forward_plain(residual_mlp)
        };
        log_tensor_stats("block.output", &output);

        output
    }

    /// Forward pass with FiLM conditioning from globals
    /// globals: [batch, NUM_GLOBALS] - passed to each LayerNorm to generate gamma/beta
    pub fn forward_with_film(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
        globals: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        log_tensor_stats("block.input", &x);

        let attn_out = {
            let _t = TimingScope::new_with_sync::<B>("block_attention", &device);
            self.attention.forward(x.clone(), shared_weight_gen)
        };
        log_tensor_stats("block.attn_out", &attn_out);

        let residual_attn = x + attn_out;
        log_tensor_stats("block.post_attn", &residual_attn);

        let post_attn = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_attn", &device);
            self.norm_post_attn.forward(residual_attn, globals.clone())
        };
        log_tensor_stats("block.norm_post_attn", &post_attn);

        let mlp_out = {
            let _t = TimingScope::new_with_sync::<B>("block_mlp", &device);
            self.mlp.forward(post_attn.clone())
        };
        log_tensor_stats("block.mlp_out", &mlp_out);

        let residual_mlp = post_attn + mlp_out;

        let output = {
            let _t = TimingScope::new_with_sync::<B>("block_norm_mlp", &device);
            self.norm_post_mlp.forward(residual_mlp, globals)
        };
        log_tensor_stats("block.output", &output);

        output
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
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let hidden_dim = (config.embed_dim() as f32 * 2.5) as usize;
        // Fused projection outputs 2*hidden_dim, which we split into gate and up
        let fused_gate_up = LinearConfig::new(config.embed_dim(), 2 * hidden_dim).init(device);
        let down_proj = LinearConfig::new(hidden_dim, config.embed_dim()).init(device);
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
    use crate::config::{set_global_config, Config};
    use crate::test_backend::{test_device, TestBackend};

    fn ensure_config() {
        let _ = set_global_config(Config::new(128, 2));
    }

    #[test]
    fn transformer_block_smoke_shapes() {
        ensure_config();
        let device = test_device();
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&device);
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
        ensure_config();
        let device = test_device();
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&device);
        let batch_size = 1usize;
        let seq_len = 32usize;
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let _ = block.forward(x, &weight_gen);
    }

    #[test]
    #[ignore]
    fn transformer_block_large_batch_shapes() {
        ensure_config();
        let device = test_device();
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&device);
        let batch_size = 1024usize;
        let seq_len = 64usize;
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x, &weight_gen);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }
}
