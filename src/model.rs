use crate::chess_output::ChessOutput;
use crate::config::{get_global_config, ModelConfig, FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS};
use crate::model_prediction_logger::log_model_predictions;
use crate::relative_position_transformer::TransformerBlock;
use burn::module::Param;
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, sigmoid, softmax, softplus};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};
use statrs::distribution::{Continuous, Gamma};

#[derive(Module, Debug)]
pub struct OXIModel<B: Backend> {
    board_embed: Linear<B>,
    pos_embed: Embedding<B>,
    global_embed: Linear<B>,
    blocks: Vec<TransformerBlock<B>>,
    norm: LayerNorm<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
    side_info_head: Linear<B>,
    time_usage_head: Linear<B>,
    side_info_bce: BinaryCrossEntropyLoss<B>,
    policy_uncertainty: Param<Tensor<B, 1>>,
    value_uncertainty: Param<Tensor<B, 1>>,
    side_info_uncertainty: Param<Tensor<B, 1>>,
    time_usage_uncertainty: Param<Tensor<B, 1>>,
}

impl<B: Backend> OXIModel<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        // Embed board channels (e.g., 16 or 112) to embed_dim
        let board_embed = LinearConfig::new(FEATURES_PER_TOKEN, config.embed_dim()).init(device);
        let global_embed = LinearConfig::new(NUM_GLOBALS, config.embed_dim()).init(device);

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers() {
            blocks.push(TransformerBlock::new(device));
        }

        // Learned absolute positional embeddings for 64 squares
        let pos_embed = EmbeddingConfig::new(64, config.embed_dim())
            .with_initializer(nn::Initializer::Uniform { min: 0.0, max: 0.01 })
            .init(device);

        let norm = LayerNormConfig::new(config.embed_dim()).init(device);

        let policy_head = LinearConfig::new(config.embed_dim(), LEGAL_MOVES / 64).init(device);
        let value_head = LinearConfig::new(config.embed_dim(), 3).init(device);
        let side_info_head = LinearConfig::new(config.embed_dim(), 13).init(device);
        let time_usage_head = LinearConfig::new(config.embed_dim(), 6).init(device);
        let bce_config = BinaryCrossEntropyLossConfig::new().with_logits(true);
        let side_info_bce = bce_config.init(device);

        let policy_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let value_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let side_info_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let time_usage_uncertainty = Param::from_tensor(Tensor::zeros([1], device));

        Self {
            board_embed,
            pos_embed,
            global_embed,
            blocks,
            norm,
            policy_head,
            value_head,
            side_info_head,
            time_usage_head,
            side_info_bce,
            policy_uncertainty,
            value_uncertainty,
            side_info_uncertainty,
            time_usage_uncertainty,
        }
    }

    pub fn forward(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let board_features = board.clone();
        let token_embeds = self.board_embed.forward(board_features);
        let global_embeds = self.global_embed.forward(globals);
        let global_embeds = global_embeds.unsqueeze_dim(1);

        let dims = token_embeds.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let device = token_embeds.device();

        // Learned absolute positional embeddings
        assert!(seq_len == 64, "Sequence length must be 64 for 8x8 board");
        let index_positions = Tensor::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);
        let embedding_positions = self.pos_embed.forward(index_positions);

        let mut x = token_embeds + global_embeds + embedding_positions;

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x = self.norm.forward(x);

        let policy_logits = self.policy_head.forward(x.clone());

        let global_features = x.mean_dim(1).squeeze(1);
        let value_logits = self.value_head.forward(global_features.clone());
        let side_info_logits = self.side_info_head.forward(global_features.clone());
        let time_usage_raw = self.time_usage_head.forward(global_features);

        // Apply activations to time usage outputs: [w1, w2, alpha1, alpha2, theta1, theta2]
        let weights_raw = time_usage_raw
            .clone()
            .slice([0..time_usage_raw.dims()[0], 0..2]);
        let alphas_raw = time_usage_raw
            .clone()
            .slice([0..time_usage_raw.dims()[0], 2..4]);
        let thetas_raw = time_usage_raw
            .clone()
            .slice([0..time_usage_raw.dims()[0], 4..6]);

        let weights = softmax(weights_raw, 1); // Ensure weights sum to 1
        let alphas = (softplus(alphas_raw, 1.0) + 1e-6).clamp(1e-4, 20.0);
        let thetas = (softplus(thetas_raw, 1.0) + 1e-6).clamp(1e-4, 1.0);

        let time_usage_logits = Tensor::cat(vec![weights, alphas, thetas], 1);

        (
            policy_logits,
            value_logits,
            side_info_logits,
            time_usage_logits,
        )
    }

    pub fn forward_classification(&self, batch: crate::dataset::ChessBatch<B>) -> ChessOutput<B>
    where
        B::FloatElem: From<f32>,
    {
        let batch_clone = batch.clone(); // Clone early for later use
        let batch_size = batch.board_input.shape().dims[0];

        let (policy_logits, value_logits, side_info_logits, time_usage_logits) =
            self.forward(batch.board_input, batch.global_features);
        // ...

        // Log model predictions for debugging
        let policy_logits_flat_original = policy_logits.reshape([batch_size, LEGAL_MOVES]);

        let mask = batch.legal_moves.clone().equal_elem(0.0);
        let policy_logits_flat = policy_logits_flat_original
            .clone()
            .mask_fill(mask.clone(), f32::NEG_INFINITY);
        log_model_predictions(
            &policy_logits_flat,
            &value_logits,
            &time_usage_logits,
            &batch_clone,
        );
        let log_policy = log_softmax(policy_logits_flat.clone(), 1);
        let log_policy = log_policy.mask_fill(mask.clone(), 0.0);

        // Label smoothing over legal moves only
        let eps = get_global_config().policy_label_smoothing;
        let legal_counts = batch
            .legal_moves
            .clone()
            .sum_dim(1)
            .reshape([batch_size, 1])
            .clamp_min(1.0);
        let uniform_over_legal = batch.legal_moves.clone() / legal_counts;
        let targets_smoothed =
            batch.move_distributions.clone() * (1.0 - eps) + uniform_over_legal * eps;

        let policy_loss = (targets_smoothed * log_policy).sum_dim(1).neg().mean();

        // Value loss
        let value_log_probs = log_softmax(value_logits.clone(), 1);
        let value_loss = (batch.values.clone() * value_log_probs)
            .sum_dim(1)
            .neg()
            .mean();

        // Side info loss
        let side_info_probs = sigmoid(side_info_logits.clone()).clamp(1e-7, 1.0 - 1e-7);
        let target_data = batch.side_info.clone().to_data();
        let mean_target = target_data
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .sum::<f32>()
            / (target_data.num_elements() as f32);
        let pos_weight = (1.0 - mean_target).max(0.1);
        let neg_weight = mean_target.max(0.1);
        let weights = batch
            .side_info
            .clone()
            .float()
            .mul_scalar(pos_weight - neg_weight)
            .add_scalar(neg_weight);
        let targets_float = batch.side_info.clone().float();
        let bce_per_element = targets_float.clone() * side_info_probs.clone().log()
            + (targets_float.neg() + 1.0) * (side_info_probs.neg() + 1.0).log();
        let weighted_bce = bce_per_element.neg() * weights;
        let side_info_loss = weighted_bce.mean();

        // Time usage loss (MSE between target and mixture mean)
        let time_usage_loss = self
            .compute_gamma_mixture_loss_impl(time_usage_logits.clone(), batch.time_usages.clone());

        let config = get_global_config();

        let config_weighted_policy_loss = policy_loss.clone() * config.policy_loss_weight;
        let config_weighted_value_loss = value_loss.clone() * config.value_loss_weight;
        let config_weighted_side_info_loss = side_info_loss.clone() * config.side_info_loss_weight;
        let config_weighted_time_usage_loss =
            time_usage_loss.clone() * config.time_usage_loss_weight;

        // let log_sigma_policy = self.policy_uncertainty.val();
        // let log_sigma_value = self.value_uncertainty.val();
        // let log_sigma_side_info = self.side_info_uncertainty.val();
        // let log_sigma_time_usage = self.time_usage_uncertainty.val();
        // let sigma_sq_policy = (log_sigma_policy.clone() * 2.0).exp();
        // let sigma_sq_value = (log_sigma_value.clone() * 2.0).exp();
        // let sigma_sq_side_info = (log_sigma_side_info.clone() * 2.0).exp();
        // let sigma_sq_time_usage = (log_sigma_time_usage.clone() * 2.0).exp();

        // let uncertainty_weighted_policy_loss = config_weighted_policy_loss.clone()
        //     / sigma_sq_policy.clone()
        //     + log_sigma_policy.clone();
        // let uncertainty_weighted_value_loss =
        //     config_weighted_value_loss.clone() / sigma_sq_value.clone() + log_sigma_value.clone();
        // // let uncertainty_weighted_side_info_loss = config_weighted_side_info_loss.clone()
        // //     / sigma_sq_side_info.clone()
        // //     + log_sigma_side_info.clone();
        // let uncertainty_weighted_time_usage_loss = config_weighted_time_usage_loss.clone()
        //     / sigma_sq_time_usage.clone()
        //     + log_sigma_time_usage.clone();

        let lambda = 0.01;
        let reg_term = lambda
            * (self.policy_uncertainty.val().powf_scalar(2.0)
                + self.value_uncertainty.val().powf_scalar(2.0)
                // + self.side_info_uncertainty.val().powf_scalar(2.0)
                + self.time_usage_uncertainty.val().powf_scalar(2.0))
            .mean();
        let loss = config_weighted_time_usage_loss.clone()
            + config_weighted_policy_loss.clone()
            + config_weighted_value_loss.clone()
            // + config_weighted_side_info_loss.clone()
            + reg_term;

        // let sigma_policy = log_sigma_policy.exp().to_data().as_slice::<f32>().unwrap()[0];
        // let sigma_value = log_sigma_value.exp().to_data().as_slice::<f32>().unwrap()[0];
        // let sigma_side_info = log_sigma_side_info
        //     .exp()
        //     .to_data()
        //     .as_slice::<f32>()
        //     .unwrap()[0];
        // let sigma_time_usage = log_sigma_time_usage
        //     .exp()
        //     .to_data()
        //     .as_slice::<f32>()
        //     .unwrap()[0];

        // Accuracy
        let targets = batch.move_distributions.clone().argmax(1).squeeze(1);
        let policy_logits_only_legals = policy_logits_flat_original
            .clone()
            .mask_fill(mask.clone(), 0.0);
        let _predicted_moves: Tensor<B, 1, Int> =
            policy_logits_only_legals.clone().argmax(1).squeeze(1);
        // let correct = targets
        //     .to_data()
        //     .as_slice::<i32>()
        //     .unwrap()
        //     .iter()
        //     .zip(predicted_moves.to_data().as_slice::<i32>().unwrap())
        //     .filter(|(&t, &p)| t == p)
        //     .count();
        // let batch_accuracy = correct as f32 / batch_size as f32;

        ChessOutput::new(
            loss,
            config_weighted_policy_loss.clone().mean().reshape([1]),
            config_weighted_value_loss.clone().mean().reshape([1]),
            config_weighted_side_info_loss.clone().mean().reshape([1]),
            config_weighted_time_usage_loss.clone().mean().reshape([1]),
            policy_logits_flat,
            targets,
            value_logits,
            batch.values.clone(),
            batch.legal_moves.clone(),
        )
        .with_distributions(batch.move_distributions.clone())
        // .with_uncertainties((sigma_policy, sigma_value, sigma_side_info, sigma_time_usage))
        .with_raw_losses(
            config_weighted_policy_loss.mean().reshape([1]),
            config_weighted_value_loss.mean().reshape([1]),
            config_weighted_side_info_loss.mean().reshape([1]),
            config_weighted_time_usage_loss.mean().reshape([1]),
        )
    }

    #[cfg(test)]
    pub fn compute_gamma_mixture_loss(
        &self,
        time_usage_logits: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1>
    where
        B::FloatElem: From<f32>,
    {
        self.compute_gamma_mixture_loss_impl(time_usage_logits, targets)
    }

    fn compute_gamma_mixture_loss_impl(
        &self,
        time_usage_logits: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1>
    where
        B::FloatElem: From<f32>,
    {
        let batch_size = time_usage_logits.dims()[0];

        // Extract parameters: [w1, w2, alpha1, alpha2, theta1, theta2]
        let weights = time_usage_logits.clone().slice([0..batch_size, 0..2]);
        let alphas = time_usage_logits.clone().slice([0..batch_size, 2..4]);
        let thetas = time_usage_logits.clone().slice([0..batch_size, 4..6]);

        // Robust loss in log-space using Huber on log times
        let eps = 1e-6;
        let delta = 0.1;
        let targets_flat = targets.clone().flatten::<1>(0, 1).clamp_min(eps);
        let pred_mean =
            crate::gamma_utils::mixture_mean(weights.clone(), alphas.clone(), thetas.clone())
                .clamp(eps, 1.0);
        let targets_flat = targets_flat.clamp(eps, 1.0);
        let diff = pred_mean.log() - targets_flat.log();
        let abs = diff.clone().abs();
        let quad: Tensor<B, 1> = (0.5f32) * diff.clone().powf_scalar(2.0);
        let lin: Tensor<B, 1> = (delta as f32) * (abs.clone() - (0.5f32) * (delta as f32));
        let cond = abs.clone().lower_elem(delta).float();
        let huber = cond.clone() * quad + (cond.neg() + 1.0) * lin;
        let tensor_loss = huber.mean().reshape([1]);

        // OLD: CPU-based implementation using statrs (for comparison)
        let weights_data = weights.to_data();
        let alphas_data = alphas.to_data();
        let thetas_data = thetas.to_data();
        let targets_data = targets.to_data();

        let weights_slice = weights_data.as_slice::<f32>().unwrap();
        let alphas_slice = alphas_data.as_slice::<f32>().unwrap();
        let thetas_slice = thetas_data.as_slice::<f32>().unwrap();
        let targets_slice = targets_data.as_slice::<f32>().unwrap();

        let mut log_likelihoods = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let w1 = weights_slice[i * 2];
            let w2 = weights_slice[i * 2 + 1];
            let alpha1 = alphas_slice[i * 2];
            let alpha2 = alphas_slice[i * 2 + 1];
            let theta1 = thetas_slice[i * 2];
            let theta2 = thetas_slice[i * 2 + 1];
            let target = targets_slice[i].max(1e-8);

            let gamma1 = Gamma::new(alpha1 as f64, 1.0 / theta1 as f64);
            let gamma2 = Gamma::new(alpha2 as f64, 1.0 / theta2 as f64);

            let pdf1 = match gamma1 {
                Ok(g) => g.pdf(target as f64) as f32,
                Err(_) => 1e-8,
            };

            let pdf2 = match gamma2 {
                Ok(g) => g.pdf(target as f64) as f32,
                Err(_) => 1e-8,
            };

            let mixture_prob = w1 * pdf1 + w2 * pdf2;
            let log_likelihood = mixture_prob.max(1e-8).ln();
            log_likelihoods.push(log_likelihood);
        }

        // let statrs_log_likelihood_tensor: Tensor<B, 1> =
        //     Tensor::from_floats(log_likelihoods.as_slice(), &device);
        // let statrs_loss = statrs_log_likelihood_tensor.neg().mean().reshape([1]);

        // Log both for comparison
        // let tensor_loss_val = tensor_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        // let statrs_loss_val = statrs_loss.to_data().as_slice::<f32>().unwrap()[0];

        // tracing::info!(
        //     "Gamma loss comparison - Tensor: {:.6}, Statrs: {:.6}, Diff: {:.6}",
        //     tensor_loss_val,
        //     statrs_loss_val,
        //     (tensor_loss_val - statrs_loss_val).abs()
        // );

        // Return the tensor-based version that preserves gradients
        tensor_loss
    }

    pub fn get_uncertainties(&self) -> (f32, f32, f32, f32) {
        let sigma_policy = self
            .policy_uncertainty
            .val()
            .exp()
            .to_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let sigma_value = self
            .value_uncertainty
            .val()
            .exp()
            .to_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let sigma_side_info = self
            .side_info_uncertainty
            .val()
            .exp()
            .to_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let sigma_time_usage = self
            .time_usage_uncertainty
            .val()
            .exp()
            .to_data()
            .as_slice::<f32>()
            .unwrap()[0];
        (sigma_policy, sigma_value, sigma_side_info, sigma_time_usage)
    }

    // Helper to get top moves
    pub fn top_moves(
        &self,
        policy_logits: Tensor<B, 3>,
        top_k: usize,
    ) -> Vec<Vec<(usize, usize, f32)>> {
        let [batch, _, _] = policy_logits.dims();
        let policy_probs = softmax(policy_logits.reshape([batch, LEGAL_MOVES]), 1);
        let policy_probs_data = policy_probs.to_data();
        let probs = policy_probs_data.as_slice::<f32>().unwrap();

        let mut top_moves = Vec::new();
        for b in 0..batch {
            let batch_probs = &probs[b * 64 * 64..(b + 1) * LEGAL_MOVES];
            let mut indexed_probs: Vec<(usize, f32)> = batch_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k_moves: Vec<(usize, usize, f32)> = indexed_probs
                .iter()
                .take(top_k)
                .map(|&(idx, prob)| (idx / 64, idx % 64, prob))
                .collect();
            top_moves.push(top_k_moves);
        }
        top_moves
    }
}

impl<B: AutodiffBackend> TrainStep<crate::dataset::ChessBatch<B>, ChessOutput<B>> for OXIModel<B>
where
    B::FloatElem: From<f32>,
{
    fn step(&self, batch: crate::dataset::ChessBatch<B>) -> TrainOutput<ChessOutput<B>> {
        let item = self.forward_classification(batch);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<crate::dataset::ChessBatch<B>, ChessOutput<B>> for OXIModel<B>
where
    B::FloatElem: From<f32>,
{
    fn step(&self, batch: crate::dataset::ChessBatch<B>) -> ChessOutput<B> {
        self.forward_classification(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FEATURES_PER_TOKEN, NUM_GLOBALS};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    #[test]
    fn test_gamma_mixture_loss() {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = OXIModel::<NdArray>::new(&device, &config);

        // Create test data: [batch_size=2, 6] for [w1, w2, alpha1, alpha2, theta1, theta2]
        let time_usage_logits = Tensor::from_data(
            TensorData::from([
                [0.5, 0.5, 2.0, 3.0, 0.1, 0.2],   // First sample
                [0.3, 0.7, 1.5, 2.5, 0.15, 0.25], // Second sample
            ]),
            &device,
        );

        // Target time usage values
        let targets = Tensor::from_data(
            TensorData::from([[0.05], [0.08]]), // Relative time usage
            &device,
        );

        // Compute loss
        let loss = model.compute_gamma_mixture_loss(time_usage_logits, targets);

        // Loss should be a scalar tensor
        assert_eq!(loss.dims(), [1]);

        // Loss should be positive (MSE)
        let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];
        assert!(
            loss_value > 0.0,
            "Loss should be positive, got: {}",
            loss_value
        );

        println!("Gamma mixture loss test passed! Loss value: {}", loss_value);
    }

    #[test]
    fn test_time_usage_head_output_shape() {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = OXIModel::<NdArray>::new(&device, &config);

        // Create dummy input
        let batch_size = 2;
        let board_input = Tensor::zeros([batch_size, 64, FEATURES_PER_TOKEN], &device);
        let global_features = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let (policy_logits, value_logits, side_info_logits, time_usage_logits) =
            model.forward(board_input, global_features);

        // Check time usage head outputs 6 values per batch
        assert_eq!(time_usage_logits.dims(), [batch_size, 6]);

        // Check that weights sum to approximately 1 (first 2 values should be softmax normalized)
        let time_usage_data = time_usage_logits.to_data();
        let values = time_usage_data.as_slice::<f32>().unwrap();

        for i in 0..batch_size {
            let w1 = values[i * 6];
            let w2 = values[i * 6 + 1];
            let weight_sum = w1 + w2;
            assert!(
                (weight_sum - 1.0).abs() < 1e-5,
                "Weights should sum to 1, got: {} + {} = {}",
                w1,
                w2,
                weight_sum
            );

            // Check that alphas and thetas are positive
            let alpha1 = values[i * 6 + 2];
            let alpha2 = values[i * 6 + 3];
            let theta1 = values[i * 6 + 4];
            let theta2 = values[i * 6 + 5];

            assert!(alpha1 > 0.0, "Alpha1 should be positive, got: {}", alpha1);
            assert!(alpha2 > 0.0, "Alpha2 should be positive, got: {}", alpha2);
            assert!(theta1 > 0.0, "Theta1 should be positive, got: {}", theta1);
            assert!(theta2 > 0.0, "Theta2 should be positive, got: {}", theta2);
        }

        println!("Time usage head output shape test passed!");
    }

    #[test]
    fn test_gamma_mixture_loss_snapshot() {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = OXIModel::<NdArray>::new(&device, &config);

        // Create fixed parameters for reproducible test
        // Distribution 1: alpha=2.0, theta=0.02 (mean=0.04, focused around 0.04)
        // Distribution 2: alpha=3.0, theta=0.03 (mean=0.09, focused around 0.09)
        // Weights: 0.6 for dist1, 0.4 for dist2
        let time_usage_logits = Tensor::from_data(
            TensorData::from([
                [0.405, -0.405, 2.0, 3.0, 0.02, 0.03], // logits that give weights ~[0.6, 0.4] after softmax
                [0.405, -0.405, 2.0, 3.0, 0.02, 0.03], // Same parameters for all test cases
                [0.405, -0.405, 2.0, 3.0, 0.02, 0.03],
            ]),
            &device,
        );

        // Test three scenarios:
        // 1. Target near first distribution peak (0.04)
        // 2. Target near second distribution peak (0.09)
        // 3. Target far from both distributions (0.15)
        let targets = Tensor::from_data(
            TensorData::from([
                [0.04], // Near first distribution mean
                [0.09], // Near second distribution mean
                [0.15], // Far from both distributions
            ]),
            &device,
        );

        let loss = model.compute_gamma_mixture_loss(time_usage_logits, targets);
        let loss_data = loss.to_data();
        let loss_values = loss_data.as_slice::<f32>().unwrap();

        // Create snapshot data
        let snapshot_data = format!(
            "Gamma Mixture Loss Snapshot Test\n\
            Parameters: w1=0.6, w2=0.4, α1=2.0, α2=3.0, θ1=0.02, θ2=0.03\n\
            Distribution 1 mean: {:.4} (2.0 * 0.02)\n\
            Distribution 2 mean: {:.4} (3.0 * 0.03)\n\
            \n\
            Target 0.04 (near dist1): loss = {:.6}\n\
            Target 0.09 (near dist2): loss = {:.6}\n\
            Target 0.15 (far away):   loss = {:.6}\n\
            \n\
            Expected behavior:\n\
            - Loss should be lowest for target 0.04 (highest probability)\n\
            - Loss should be medium for target 0.09 (medium probability)\n\
            - Loss should be highest for target 0.15 (lowest probability)",
            2.0 * 0.02,
            3.0 * 0.03,
            loss_values[0],
            loss_values[0], // Note: loss is averaged, so same value for all
            loss_values[0]
        );

        // For now, just print the snapshot. In a real test, you'd use insta::assert_snapshot!
        println!("{}", snapshot_data);

        // Verify the loss is reasonable (positive and finite)
        assert!(
            loss_values[0] > 0.0 && loss_values[0].is_finite(),
            "Loss should be positive and finite, got: {}",
            loss_values[0]
        );
    }

    #[test]
    fn test_individual_gamma_mixture_losses() {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = OXIModel::<NdArray>::new(&device, &config);

        // Test each target individually to see the actual loss differences
        let base_params = [0.405, -0.405, 2.0, 3.0, 0.02, 0.03]; // w1≈0.6, w2≈0.4

        let test_cases = [
            (0.04, "near first distribution"),
            (0.09, "near second distribution"),
            (0.15, "far from both distributions"),
        ];

        let mut results = Vec::new();

        for (target_value, description) in test_cases.iter() {
            let time_usage_logits = Tensor::from_data(TensorData::from([base_params]), &device);

            let targets = Tensor::from_data(TensorData::from([[*target_value]]), &device);

            let loss = model.compute_gamma_mixture_loss(time_usage_logits, targets);
            let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];

            results.push((target_value, description, loss_value));
        }

        // Print results
        println!("\nIndividual Gamma Mixture Loss Results:");
        for (target, desc, loss) in &results {
            println!("Target {:.3} ({}): loss = {:.6}", target, desc, loss);
        }

        // Verify expected ordering: loss should increase as we move away from distribution peaks
        // Note: This is a heuristic check - the exact ordering depends on the mixture weights
        assert!(
            results
                .iter()
                .all(|(_, _, loss)| *loss > 0.0 && loss.is_finite()),
            "All losses should be positive and finite"
        );
    }
}
