use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::activation::gelu;
use burn::tensor::Device;
use burn::tensor::{backend::Backend, Tensor};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
};

use crate::config::get_global_config;
use crate::rope::RopeRelativePositionAttention;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: RopeRelativePositionAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let attention = RopeRelativePositionAttention::new(device);
        let norm1 = LayerNormConfig::new(config.embed_dim()).init(device);
        let norm2 = LayerNormConfig::new(config.embed_dim()).init(device);
        let mlp = MLP::new(device);
        Self {
            attention,
            norm1,
            norm2,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_out = self.attention.forward(self.norm1.forward(x.clone()));
        let x = x + attn_out;
        let mlp_out = self.mlp.forward(self.norm2.forward(x.clone()));
        x + mlp_out
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
        let x = self.fc1.forward(x);
        let x = gelu(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{set_global_config, Config};
    use burn_ndarray::NdArray;

    fn ensure_config() {
        // Try setting a small config; ignore error if already set
        let _ = set_global_config(Config::new(128, 8, 2));
    }

    #[test]
    fn transformer_block_smoke_shapes() {
        ensure_config();
        let device = Default::default();
        let config = get_global_config();
        let block = TransformerBlock::<NdArray>::new(&device);
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
        let device = Default::default();
        let config = get_global_config();
        let block = TransformerBlock::<NdArray>::new(&device);
        let batch_size = 1usize;
        let seq_len = 32usize; // wrong: not 8x8
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let _ = block.forward(x); // should panic inside attention due to seq len mismatch
    }

    #[test]
    fn transformer_block_large_batch_shapes() {
        ensure_config();
        let device = Default::default();
        let config = get_global_config();
        let block = TransformerBlock::<NdArray>::new(&device);
        let batch_size = 1024usize;
        let seq_len = 64usize; // 8x8 board
        let embed_dim = config.embed_dim();
        let x = Tensor::zeros([batch_size, seq_len, embed_dim], &device);
        let y = block.forward(x);
        assert_eq!(y.dims(), [batch_size, seq_len, embed_dim]);
    }
}
