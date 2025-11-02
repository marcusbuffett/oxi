use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, Tensor};

use crate::config::get_global_config;
use crate::norm_debug::log_tensor_stats;
use crate::shaw::ShawRelativePositionAttention;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: ShawRelativePositionAttention<B>,
    norm_post_attn: LayerNorm<B>, // New: Post-residual norm for attention
    norm_post_mlp: LayerNorm<B>,  // Repurposed from norm_final: Post-residual norm for MLP
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let attention = ShawRelativePositionAttention::new(device);
        let norm_post_attn = LayerNormConfig::new(config.embed_dim()).init(device);
        let norm_post_mlp = LayerNormConfig::new(config.embed_dim()).init(device);
        let mlp = MLP::new(device);
        Self {
            attention,
            norm_post_attn,
            norm_post_mlp,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        log_tensor_stats("block.input", &x);

        // Attention without pre-norm
        let attn_out = self.attention.forward(x.clone());
        log_tensor_stats("block.attn_out", &attn_out);

        // Attention residual
        let residual_attn = x + attn_out;
        log_tensor_stats("block.post_attn", &residual_attn);

        // Post-residual norm (Peri-LN key change)
        let post_attn = self.norm_post_attn.forward(residual_attn);
        log_tensor_stats("block.norm_post_attn", &post_attn);

        // MLP without pre-norm
        let mlp_out = self.mlp.forward(post_attn.clone());
        log_tensor_stats("block.mlp_out", &mlp_out);

        // MLP residual
        let residual_mlp = post_attn + mlp_out;

        // Final post-residual norm (Peri-LN for MLP)
        let output = self.norm_post_mlp.forward(residual_mlp.clone());
        log_tensor_stats("block.output", &output);

        output
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let hidden_dim = (config.embed_dim() as f32 * config.mlp_ratio()) as usize;
        let fc1 = LinearConfig::new(config.embed_dim(), hidden_dim).init(device);
        let fc2 = LinearConfig::new(hidden_dim, config.embed_dim()).init(device);
        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        log_tensor_stats("mlp.input", &x);
        let hidden = self.fc1.forward(x);
        log_tensor_stats("mlp.fc1_out", &hidden);
        let activated = gelu(hidden);
        log_tensor_stats("mlp.gelu", &activated);
        let output = self.fc2.forward(activated);
        log_tensor_stats("mlp.fc2_out", &output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{set_global_config, Config};

    #[cfg(target_os = "macos")]
    type TestBackend = burn::backend::Metal;
    #[cfg(not(target_os = "macos"))]
    type TestBackend = burn::backend::LibTorch<f32>;

    fn ensure_config() {
        // Try setting a small config; ignore error if already set
        let _ = set_global_config(Config::new(128, 2));
    }

    #[test]
    fn transformer_block_smoke_shapes() {
        ensure_config();
        #[cfg(target_os = "macos")]
        let device = burn::backend::metal::MetalDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let batch_size = 2usize;
        let seq_len = 64usize; // 8x8 board
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }

    #[test]
    #[should_panic]
    fn transformer_block_panics_on_wrong_seq_len() {
        ensure_config();
        #[cfg(target_os = "macos")]
        let device = burn::backend::metal::MetalDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let batch_size = 1usize;
        let seq_len = 32usize; // wrong: not 8x8
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let _ = block.forward(x); // should panic inside attention due to seq len mismatch
    }

    #[test]
    #[ignore]
    fn transformer_block_large_batch_shapes() {
        ensure_config();
        #[cfg(target_os = "macos")]
        let device = burn::backend::metal::MetalDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;
        let config = get_global_config();
        let block = TransformerBlock::<TestBackend>::new(&device);
        let batch_size = 1024usize;
        let seq_len = 64usize; // 8x8 board
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }
}
