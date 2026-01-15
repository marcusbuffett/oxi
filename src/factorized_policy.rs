use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use crate::config::get_global_config;

const POLICY_RANK: usize = 64;
const NUM_PROMO_PIECES: usize = 4;

#[derive(Module, Debug)]
pub struct FactorizedPolicyHead<B: Backend> {
    source_proj: Linear<B>,
    target_proj: Linear<B>,
    promo_from_proj: Linear<B>,
    promo_to_proj: Linear<B>,
}

impl<B: Backend> FactorizedPolicyHead<B> {
    pub fn new(device: &B::Device) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        Self {
            source_proj: LinearConfig::new(embed_dim, POLICY_RANK)
                .with_initializer(std_init.clone())
                .init(device),
            target_proj: LinearConfig::new(embed_dim, POLICY_RANK)
                .with_initializer(std_init.clone())
                .init(device),
            promo_from_proj: LinearConfig::new(embed_dim, NUM_PROMO_PIECES)
                .with_initializer(std_init.clone())
                .init(device),
            promo_to_proj: LinearConfig::new(embed_dim, NUM_PROMO_PIECES)
                .with_initializer(std_init.clone())
                .init(device),
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _embed_dim] = tokens.dims();
        debug_assert_eq!(seq_len, 64, "Expected 64 squares");
        let device = tokens.device();

        let base_logits = self.compute_base_logits(tokens.clone());
        let promo_logits = self.compute_promotion_logits(tokens);

        self.combine_logits_vectorized(base_logits, promo_logits, batch_size, &device)
    }

    fn compute_base_logits(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let source = self.source_proj.forward(tokens.clone());
        let target = self.target_proj.forward(tokens);
        source.matmul(target.transpose())
    }

    fn compute_promotion_logits(&self, tokens: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, _, embed_dim] = tokens.dims();

        let white_from = tokens.clone().slice([0..batch_size, 48..56, 0..embed_dim]);
        let white_to = tokens.clone().slice([0..batch_size, 56..64, 0..embed_dim]);
        let black_from = tokens.clone().slice([0..batch_size, 8..16, 0..embed_dim]);
        let black_to = tokens.slice([0..batch_size, 0..8, 0..embed_dim]);

        let from_squares = Tensor::cat(vec![white_from, black_from], 1);
        let to_squares = Tensor::cat(vec![white_to, black_to], 1);

        let from_proj = self.promo_from_proj.forward(from_squares);
        let to_proj = self.promo_to_proj.forward(to_squares);

        from_proj.unsqueeze_dim(2) + to_proj.unsqueeze_dim(1)
    }

    fn combine_logits_vectorized(
        &self,
        base_logits: Tensor<B, 3>,
        promo_logits: Tensor<B, 4>,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut output = Tensor::zeros([batch_size, 64, 76], device);

        output = output.slice_assign([0..batch_size, 0..64, 0..64], base_logits.clone());

        for from_file in 0..8usize {
            let white_from_sq = 48 + from_file;

            for to_offset in 0..3usize {
                let to_file = from_file as i32 + (to_offset as i32 - 1);
                if to_file < 0 || to_file >= 8 {
                    continue;
                }
                let white_to_sq = 56 + to_file as usize;

                let base_val = base_logits
                    .clone()
                    .slice([
                        0..batch_size,
                        white_from_sq..white_from_sq + 1,
                        white_to_sq..white_to_sq + 1,
                    ])
                    .reshape([batch_size, 1]);

                let promo_4 = promo_logits
                    .clone()
                    .slice([
                        0..batch_size,
                        from_file..from_file + 1,
                        to_file as usize..(to_file as usize) + 1,
                        0..4,
                    ])
                    .reshape([batch_size, 4]);

                let combined = (base_val + promo_4).reshape([batch_size, 1, 4]);

                let dir_idx = to_offset;
                let start_idx = 64 + dir_idx * 4;

                output = output.slice_assign(
                    [
                        0..batch_size,
                        white_from_sq..white_from_sq + 1,
                        start_idx..start_idx + 4,
                    ],
                    combined,
                );
            }
        }

        for from_file in 0..8usize {
            let black_from_sq = 8 + from_file;
            let promo_from_idx = 8 + from_file;

            for to_offset in 0..3usize {
                let to_file = from_file as i32 + (to_offset as i32 - 1);
                if to_file < 0 || to_file >= 8 {
                    continue;
                }
                let black_to_sq = to_file as usize;
                let promo_to_idx = 8 + to_file as usize;

                let base_val = base_logits
                    .clone()
                    .slice([
                        0..batch_size,
                        black_from_sq..black_from_sq + 1,
                        black_to_sq..black_to_sq + 1,
                    ])
                    .reshape([batch_size, 1]);

                let promo_4 = promo_logits
                    .clone()
                    .slice([
                        0..batch_size,
                        promo_from_idx..promo_from_idx + 1,
                        promo_to_idx..promo_to_idx + 1,
                        0..4,
                    ])
                    .reshape([batch_size, 4]);

                let combined = (base_val + promo_4).reshape([batch_size, 1, 4]);

                let dir_idx = to_offset;
                let start_idx = 64 + dir_idx * 4;

                output = output.slice_assign(
                    [
                        0..batch_size,
                        black_from_sq..black_from_sq + 1,
                        start_idx..start_idx + 4,
                    ],
                    combined,
                );
            }
        }

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
    fn test_factorized_policy_output_shape() {
        ensure_config();
        let device = test_device();

        let head = FactorizedPolicyHead::<TestBackend>::new(&device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, 128], &device);

        let output = head.forward(tokens);
        assert_eq!(output.dims(), [batch_size, 64, 76]);
    }

    #[test]
    fn test_base_logits_shape() {
        ensure_config();
        let device = test_device();

        let head = FactorizedPolicyHead::<TestBackend>::new(&device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, 128], &device);

        let base = head.compute_base_logits(tokens);
        assert_eq!(base.dims(), [batch_size, 64, 64]);
    }

    #[test]
    fn test_promo_logits_shape() {
        ensure_config();
        let device = test_device();

        let head = FactorizedPolicyHead::<TestBackend>::new(&device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, 128], &device);

        let promo = head.compute_promotion_logits(tokens);
        assert_eq!(promo.dims(), [batch_size, 16, 16, 4]);
    }
}
