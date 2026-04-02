use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

#[cfg(feature = "train")]
use crate::norm_debug::log_tensor_stats;
#[cfg(not(feature = "train"))]
use crate::train_stubs::log_tensor_stats;

/// Spatial convolution: 3x3 neighbor gather + linear projection (channels-last, no permute).
/// For each square, gathers its 3x3 neighborhood with zero-padding at edges,
/// then projects the 9*channels neighborhood features back to `in_channels`.
#[derive(Module, Debug)]
pub struct SpatialConv<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> SpatialConv<B> {
    pub fn new(device: &Device<B>, in_channels: usize) -> Self {
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let proj = LinearConfig::new(9 * in_channels, in_channels)
            .with_initializer(std_init)
            .init(device);
        Self { proj }
    }

    /// Forward pass.
    ///
    /// Input:  `x` with shape `[batch, 64, channels]`
    /// Output: tensor with shape `[batch, 64, channels]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [batch_size, _seq, channels] = x.dims();
        let board = x.reshape([batch_size, 8, 8, channels]);
        let mut patches: Vec<Tensor<B, 4>> = Vec::with_capacity(9);
        for dr in [-1i32, 0, 1] {
            for dc in [-1i32, 0, 1] {
                let mut patch = Tensor::zeros([batch_size, 8, 8, channels], &device);
                let (sr0, sr1) = ((-dr).max(0) as usize, (8 - dr).min(8) as usize);
                let (sc0, sc1) = ((-dc).max(0) as usize, (8 - dc).min(8) as usize);
                let (dr0, dr1) = (dr.max(0) as usize, (8 + dr).min(8) as usize);
                let (dc0, dc1) = (dc.max(0) as usize, (8 + dc).min(8) as usize);
                let src = board
                    .clone()
                    .slice([0..batch_size, sr0..sr1, sc0..sc1, 0..channels]);
                patch = patch.slice_assign([0..batch_size, dr0..dr1, dc0..dc1, 0..channels], src);
                patches.push(patch);
            }
        }
        let neighborhood = Tensor::cat(patches, 3); // [batch, 8, 8, 9*channels]
        let neighborhood_flat = neighborhood.reshape([batch_size, 64, 9 * channels]);
        log_tensor_stats("embed.spatial_conv.input", &neighborhood_flat);
        let out = silu(self.proj.forward(neighborhood_flat));
        log_tensor_stats("embed.spatial_conv.output", &out);
        out
    }
}
