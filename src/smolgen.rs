use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{backend::Backend, Device, Tensor};

use crate::config::get_global_config;

#[cfg(feature = "train")]
use crate::forward_timing::TimingScope;
#[cfg(feature = "train")]
use crate::norm_debug::log_tensor_stats;
#[cfg(not(feature = "train"))]
use crate::train_stubs::{log_tensor_stats, TimingScope};

/// Smolgen: Position-dependent attention bias generator
/// Generates per-head vectors from global board state that are then projected
/// by the shared SmolgenWeightGen into [batch, heads, 64, 64] attention biases.
///
/// This is a per-layer component (each transformer block has its own Smolgen).
#[derive(Module, Debug)]
pub struct Smolgen<B: Backend> {
    /// Per-token compression: embed_dim -> smolgen_hidden
    compress: Linear<B>,

    /// Global state extraction: (64 * smolgen_hidden) -> global_dim
    dense1: Linear<B>,
    ln1: LayerNorm<B>,

    /// Per-head projection: global_dim -> (num_heads * gen_size)
    dense2: Linear<B>,
    ln2: LayerNorm<B>,
}

/// Shared weight generator across all Smolgen instances in all transformer blocks.
/// This projects per-head vectors into 64x64 attention biases.
///
/// CRITICAL: This should be created ONCE and passed to all transformer blocks.
#[derive(Module, Debug)]
pub struct SmolgenWeightGen<B: Backend> {
    /// Final projection: gen_size -> 4096 (64x64)
    weight_gen: Linear<B>,
}

impl<B: Backend> Smolgen<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();
        let num_heads = config.num_heads();
        let smolgen_hidden = config.smolgen_hidden();
        let global_dim = config.smolgen_global_dim();
        let gen_size = config.smolgen_gen_size();

        let compress = LinearConfig::new(embed_dim, smolgen_hidden)
            .with_bias(true)
            .init(device);

        let dense1 = LinearConfig::new(64 * smolgen_hidden, global_dim)
            .with_bias(true)
            .init(device);
        let ln1 = LayerNormConfig::new(global_dim).init(device);

        let dense2 = LinearConfig::new(global_dim, num_heads * gen_size)
            .with_bias(true)
            .init(device);
        let ln2 = LayerNormConfig::new(gen_size).init(device);

        Self {
            compress,
            dense1,
            ln1,
            dense2,
            ln2,
        }
    }

    /// Generate per-head vectors from input tokens.
    /// Input: [batch, 64, embed_dim]
    /// Output: [batch, num_heads, gen_size]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let config = get_global_config();
        let device = x.device();
        let [batch_size, seq_len, _embed_dim] = x.dims();
        let num_heads = config.num_heads();
        let gen_size = config.smolgen_gen_size();
        let smolgen_hidden = config.smolgen_hidden();

        debug_assert_eq!(seq_len, 64, "Smolgen expects seq_len=64 (8x8 board)");

        let compressed = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_compress", &device);
            self.compress.forward(x)
        };
        log_tensor_stats("smolgen.compressed", &compressed);

        let flat = compressed.reshape([batch_size, 64 * smolgen_hidden]);
        let global = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_dense1", &device);
            self.dense1.forward(flat)
        };
        let global = global.clone() * sigmoid(global);
        let global = self.ln1.forward(global);
        log_tensor_stats("smolgen.global", &global);

        let per_head = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_dense2", &device);
            self.dense2.forward(global)
        };
        let per_head = per_head.reshape([batch_size, num_heads, gen_size]);
        let per_head = per_head.clone() * sigmoid(per_head);
        let per_head = self.ln2.forward(per_head);
        log_tensor_stats("smolgen.per_head", &per_head);

        per_head
    }
}

impl<B: Backend> SmolgenWeightGen<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let gen_size = config.smolgen_gen_size();

        let weight_gen = LinearConfig::new(gen_size, 64 * 64)
            .with_bias(false)
            .init(device);

        Self { weight_gen }
    }

    /// Generate attention bias from per-head vectors.
    /// Input: [batch, num_heads, gen_size]
    /// Output: [batch, num_heads, 64, 64]
    pub fn forward(&self, per_head: Tensor<B, 3>) -> Tensor<B, 4> {
        let device = per_head.device();
        let [batch_size, num_heads, gen_size] = per_head.dims();

        let flat = per_head.reshape([batch_size * num_heads, gen_size]);

        let logits = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_weight_gen", &device);
            self.weight_gen.forward(flat)
        };
        log_tensor_stats("smolgen.weight_gen_output", &logits);

        logits.reshape([batch_size, num_heads, 64, 64])
    }
}

/// Smolgen-enhanced attention mechanism with dynamic position-dependent biases.
#[derive(Module, Debug)]
pub struct SmolgenAttention<B: Backend> {
    qkv_proj: Linear<B>,
    o_proj: Linear<B>,
    smolgen: Smolgen<B>,
}

impl<B: Backend> SmolgenAttention<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();

        let qkv_proj = LinearConfig::new(embed_dim, 3 * embed_dim).init(device);
        let o_proj = LinearConfig::new(embed_dim, embed_dim).init(device);

        let smolgen = Smolgen::new(device);

        Self {
            qkv_proj,
            o_proj,
            smolgen,
        }
    }

    /// Forward pass with shared weight generator.
    /// The SmolgenWeightGen must be passed from the model level (shared across all layers).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        shared_weight_gen: &SmolgenWeightGen<B>,
    ) -> Tensor<B, 3> {
        let config = get_global_config();
        let device = x.device();
        let [batch_size, seq_len, embed_dim] = x.dims();

        debug_assert_eq!(
            seq_len, 64,
            "SmolgenAttention expects seq_len=64 (8x8 board)"
        );
        debug_assert_eq!(embed_dim, config.embed_dim());

        let num_heads = config.num_heads();
        let head_dim = config.head_dim();

        log_tensor_stats("smolgen_attn.input", &x);

        let qkv = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_qkv", &device);
            self.qkv_proj.forward(x.clone())
        };

        let (q, k, v) = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_reshape", &device);
            let q = qkv
                .clone()
                .narrow(2, 0, embed_dim)
                .reshape([batch_size, seq_len, num_heads, head_dim])
                .permute([0, 2, 1, 3]);
            let k = qkv
                .clone()
                .narrow(2, embed_dim, embed_dim)
                .reshape([batch_size, seq_len, num_heads, head_dim])
                .permute([0, 2, 1, 3]);
            let v = qkv
                .narrow(2, 2 * embed_dim, embed_dim)
                .reshape([batch_size, seq_len, num_heads, head_dim])
                .permute([0, 2, 1, 3]);
            (q, k, v)
        };

        let scale = (head_dim as f32).sqrt();
        let q_scaled = q.clone().div_scalar(scale);

        let attn_scores = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_qk_matmul", &device);
            q_scaled.matmul(k.clone().swap_dims(2, 3))
        };
        log_tensor_stats("smolgen_attn.base_scores", &attn_scores);

        let per_head = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_forward", &device);
            self.smolgen.forward(x)
        };
        let smolgen_bias = shared_weight_gen.forward(per_head);
        log_tensor_stats("smolgen_attn.smolgen_bias", &smolgen_bias);

        let attn_scores = attn_scores + smolgen_bias;
        log_tensor_stats("smolgen_attn.combined_scores", &attn_scores);

        let attn_weights = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_softmax", &device);
            softmax(attn_scores, 3)
        };
        log_tensor_stats("smolgen_attn.attn_weights", &attn_weights);

        let context = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_v_matmul", &device);
            attn_weights.matmul(v)
        };
        log_tensor_stats("smolgen_attn.context", &context);

        let context_merged = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_merge", &device);
            context
                .permute([0, 2, 1, 3])
                .reshape([batch_size, seq_len, embed_dim])
        };

        let output = {
            let _t = TimingScope::new_with_sync::<B>("smolgen_attn_o_proj", &device);
            self.o_proj.forward(context_merged)
        };
        log_tensor_stats("smolgen_attn.output", &output);

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{set_global_config, Config};

    #[cfg(target_os = "macos")]
    type TestBackend = burn::backend::Wgpu;
    #[cfg(not(target_os = "macos"))]
    type TestBackend = burn::backend::LibTorch<f32>;

    fn ensure_config() {
        let _ = set_global_config(Config::new(128, 2));
    }

    #[test]
    fn smolgen_shapes() {
        ensure_config();
        #[cfg(target_os = "macos")]
        let device = burn::backend::wgpu::WgpuDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;

        let config = get_global_config();
        let smolgen = Smolgen::<TestBackend>::new(&device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&device);

        let batch_size = 2usize;
        let x = Tensor::zeros([batch_size, 64, config.embed_dim()], &device);

        let per_head = smolgen.forward(x);
        assert_eq!(
            per_head.dims(),
            [batch_size, config.num_heads(), config.smolgen_gen_size()]
        );

        let bias = weight_gen.forward(per_head);
        assert_eq!(bias.dims(), [batch_size, config.num_heads(), 64, 64]);
    }

    #[test]
    fn smolgen_attention_shapes() {
        ensure_config();
        #[cfg(target_os = "macos")]
        let device = burn::backend::wgpu::WgpuDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;

        let config = get_global_config();
        let attention = SmolgenAttention::<TestBackend>::new(&device);
        let weight_gen = SmolgenWeightGen::<TestBackend>::new(&device);

        let batch_size = 2usize;
        let x = Tensor::zeros([batch_size, 64, config.embed_dim()], &device);

        let output = attention.forward(x, &weight_gen);
        assert_eq!(output.dims(), [batch_size, 64, config.embed_dim()]);
    }
}
