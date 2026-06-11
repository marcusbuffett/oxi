use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;
use once_cell::sync::Lazy;

use crate::config::{Config, NUM_GLOBALS};
use crate::move_encoding::policy_slot_gather_indices;

const NUM_PROMO_PIECES: usize = 4;

/// Precomputed routing from (from_square, policy_slot) to the from x to-square
/// logit that slot represents. See `policy_slot_gather_indices`.
static SLOT_GATHER_INDICES: Lazy<[i32; 64 * 76]> = Lazy::new(policy_slot_gather_indices);

#[derive(Module, Debug)]
pub struct FactorizedPolicyHead<B: Backend> {
    source_proj: Linear<B>,
    target_proj: Linear<B>,
    promo_from_proj: Linear<B>,
    promo_to_proj: Linear<B>,
    source_gamma_proj: Linear<B>,
    source_beta_proj: Linear<B>,
    target_gamma_proj: Linear<B>,
    target_beta_proj: Linear<B>,
}

impl<B: Backend> FactorizedPolicyHead<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        let embed_dim = config.embed_dim();

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        // Smaller init for FiLM projections so they start near-identity (gamma≈1, beta≈0)
        let film_init = Initializer::Normal {
            mean: 0.0,
            std: 0.01,
        };

        Self {
            source_proj: LinearConfig::new(embed_dim, embed_dim)
                .with_initializer(std_init.clone())
                .init(device),
            target_proj: LinearConfig::new(embed_dim, embed_dim)
                .with_initializer(std_init.clone())
                .init(device),
            promo_from_proj: LinearConfig::new(embed_dim, NUM_PROMO_PIECES)
                .with_initializer(std_init.clone())
                .init(device),
            promo_to_proj: LinearConfig::new(embed_dim, NUM_PROMO_PIECES)
                .with_initializer(std_init.clone())
                .init(device),
            source_gamma_proj: LinearConfig::new(NUM_GLOBALS, embed_dim)
                .with_initializer(film_init.clone())
                .init(device),
            source_beta_proj: LinearConfig::new(NUM_GLOBALS, embed_dim)
                .with_initializer(film_init.clone())
                .init(device),
            target_gamma_proj: LinearConfig::new(NUM_GLOBALS, embed_dim)
                .with_initializer(film_init.clone())
                .init(device),
            target_beta_proj: LinearConfig::new(NUM_GLOBALS, embed_dim)
                .with_initializer(film_init.clone())
                .init(device),
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 3>, globals: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _embed_dim] = tokens.dims();
        debug_assert_eq!(seq_len, 64, "Expected 64 squares");
        let device = tokens.device();

        let base_logits = self.compute_base_logits(tokens.clone(), globals);
        let promo_logits = self.compute_promotion_logits(tokens);

        self.combine_logits_vectorized(base_logits, promo_logits, batch_size, &device)
    }

    fn compute_base_logits(&self, tokens: Tensor<B, 3>, globals: Tensor<B, 2>) -> Tensor<B, 3> {
        let source = self.source_proj.forward(tokens.clone());
        let target = self.target_proj.forward(tokens);

        // FiLM conditioning: modulate source/target features based on global features (Elo, time, etc.)
        // gamma = gamma_proj(globals) + 1  (starts near identity due to small init)
        // beta = beta_proj(globals)        (starts near zero due to small init)
        let source_gamma = self.source_gamma_proj.forward(globals.clone()) + 1.0;
        let source_beta = self.source_beta_proj.forward(globals.clone());
        let target_gamma = self.target_gamma_proj.forward(globals.clone()) + 1.0;
        let target_beta = self.target_beta_proj.forward(globals);

        // Reshape for broadcasting: [batch, embed_dim] -> [batch, 1, embed_dim]
        let source_gamma = source_gamma.unsqueeze_dim(1);
        let source_beta = source_beta.unsqueeze_dim(1);
        let target_gamma = target_gamma.unsqueeze_dim(1);
        let target_beta = target_beta.unsqueeze_dim(1);

        let source = source * source_gamma + source_beta;
        let target = target * target_gamma + target_beta;

        source.matmul(target.transpose())
    }

    fn compute_promotion_logits(&self, tokens: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, _, embed_dim] = tokens.dims();

        // Positions are color-canonicalized so the side to move is always the
        // white-frame player; only rank-7 -> rank-8 promotions can occur (the
        // move encoding cannot even represent downward promotions).
        let from_squares = tokens.clone().slice([0..batch_size, 48..56, 0..embed_dim]);
        let to_squares = tokens.slice([0..batch_size, 56..64, 0..embed_dim]);

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
        // Route every slot to the from x destination-square logit it represents:
        // slot j of the 76-wide axis is a direction-distance code, not a square,
        // so the bilinear from x to logits must be gathered through the
        // (from, slot) -> destination mapping rather than copied column-wise.
        let gather_idx = Tensor::<B, 1, Int>::from_data(
            TensorData::from(SLOT_GATHER_INDICES.as_slice()),
            device,
        );
        let flat = base_logits.reshape([batch_size, 64 * 64]);
        let output = flat.select(1, gather_idx).reshape([batch_size, 64, 76]);

        // Add promotion-specific logits onto the promotion slots. Slot layout
        // must match move_encoding::build_tables: straight = 64..68, capture
        // toward the a-file = 68..72, capture toward the h-file = 72..76, each
        // block ordered knight, bishop, rook, queen.
        let mut promo_add = Tensor::zeros([batch_size, 64, 76], device);
        for from_file in 0..8usize {
            let from_sq = 48 + from_file;

            for to_file_delta in -1i32..=1 {
                let to_file = from_file as i32 + to_file_delta;
                if !(0..8).contains(&to_file) {
                    continue;
                }
                let dir_idx = match to_file_delta {
                    0 => 0usize,
                    -1 => 1,
                    _ => 2,
                };
                let start_idx = 64 + dir_idx * 4;

                let promo_4 = promo_logits
                    .clone()
                    .slice([
                        0..batch_size,
                        from_file..from_file + 1,
                        to_file as usize..(to_file as usize) + 1,
                        0..4,
                    ])
                    .reshape([batch_size, 1, 4]);

                promo_add = promo_add.slice_assign(
                    [
                        0..batch_size,
                        from_sq..from_sq + 1,
                        start_idx..start_idx + 4,
                    ],
                    promo_4,
                );
            }
        }

        output + promo_add
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend::{test_device, TestBackend};

    #[test]
    fn test_factorized_policy_output_shape() {
        let device = test_device();
        let config = Config::new(96, 2);

        let head = FactorizedPolicyHead::<TestBackend>::new(&config, &device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, config.embed_dim()], &device);
        let globals = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let output = head.forward(tokens, globals);
        assert_eq!(output.dims(), [batch_size, 64, 76]);
    }

    #[test]
    fn test_base_logits_shape() {
        let device = test_device();
        let config = Config::new(96, 2);

        let head = FactorizedPolicyHead::<TestBackend>::new(&config, &device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, config.embed_dim()], &device);
        let globals = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let base = head.compute_base_logits(tokens, globals);
        assert_eq!(base.dims(), [batch_size, 64, 64]);
    }

    #[test]
    fn test_promo_logits_shape() {
        let device = test_device();
        let config = Config::new(96, 2);

        let head = FactorizedPolicyHead::<TestBackend>::new(&config, &device);
        let batch_size = 2usize;
        let tokens = Tensor::zeros([batch_size, 64, config.embed_dim()], &device);

        let promo = head.compute_promotion_logits(tokens);
        assert_eq!(promo.dims(), [batch_size, 8, 8, 4]);
    }

    /// The head's output slots must agree with `encode_move`'s slot semantics:
    /// the logit at (from, slot) must be the bilinear logit for the move's
    /// actual destination square, plus the matching promotion logit for
    /// promotion slots. Guards against the direction-distance slot layout
    /// drifting out of sync with the from x to-square factorization.
    #[test]
    fn test_slot_semantics_match_move_encoding() {
        use crate::move_encoding::encode_move;
        use shakmaty::uci::UciMove;

        let device = test_device();
        let config = Config::new(96, 2);
        let head = FactorizedPolicyHead::<TestBackend>::new(&config, &device);

        // base[from][to] = from*64 + to, so the gathered value identifies the
        // exact square pair the slot consulted.
        let base: Vec<f32> = (0..64 * 64).map(|i| i as f32).collect();
        let base_logits =
            Tensor::<TestBackend, 1>::from_data(TensorData::from(base.as_slice()), &device)
                .reshape([1, 64, 64]);

        // promo[from_file][to_file][role] = 10_000 + flat index, offset so the
        // promotion contribution is distinguishable from any base value.
        let promo: Vec<f32> = (0..8 * 8 * 4).map(|i| 10_000.0 + i as f32).collect();
        let promo_logits =
            Tensor::<TestBackend, 1>::from_data(TensorData::from(promo.as_slice()), &device)
                .reshape([1, 8, 8, 4]);

        let output = head.combine_logits_vectorized(base_logits, promo_logits, 1, &device);
        let out = output.into_data().to_vec::<f32>().unwrap();

        let slot_value = |uci: &str| -> f32 {
            let m: UciMove = uci.parse().unwrap();
            let (from, slot) = encode_move(&m).unwrap();
            out[from as usize * 76 + slot as usize]
        };
        let base_val = |from: usize, to: usize| (from * 64 + to) as f32;
        let promo_val = |from_file: usize, to_file: usize, role: usize| {
            10_000.0 + ((from_file * 8 + to_file) * 4 + role) as f32
        };

        // Normal moves: pure base logit at the true destination square.
        assert_eq!(slot_value("e2e4"), base_val(12, 28));
        assert_eq!(slot_value("g1f3"), base_val(6, 21));
        assert_eq!(slot_value("a1a8"), base_val(0, 56));
        assert_eq!(slot_value("d1e1"), base_val(3, 4));

        // Promotions: base at the destination plus the matching promo logit.
        // Role order within each slot block is knight, bishop, rook, queen.
        assert_eq!(slot_value("e7e8q"), base_val(52, 60) + promo_val(4, 4, 3));
        assert_eq!(slot_value("e7d8n"), base_val(52, 59) + promo_val(4, 3, 0));
        assert_eq!(slot_value("e7f8b"), base_val(52, 61) + promo_val(4, 5, 1));
        assert_eq!(slot_value("a7b8r"), base_val(48, 57) + promo_val(0, 1, 2));
        assert_eq!(slot_value("h7h8q"), base_val(55, 63) + promo_val(7, 7, 3));
    }
}
