use crate::chess_output::ChessOutput;
use crate::config::{
    get_global_config, ModelConfig, BOARD_FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS,
    RECENCY_FEATURES,
};
use crate::distribution_utils::beta_log_pdf;
use crate::model_prediction_logger::log_model_predictions;
use crate::norm_debug::{log_tensor_stats, LayerScope, NormDebugScope, StreamScope};
use crate::relative_position_transformer::TransformerBlock;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::{gelu, log_softmax, sigmoid, softmax, softplus};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};

#[derive(Module, Debug)]
pub struct OXIModel<B: Backend> {
    token_embed: Linear<B>,
    conv_layers: Vec<Conv2d<B>>,
    global_embed: Linear<B>,
    // Per-stream normalization and gating to balance token/global embeddings
    token_norm: LayerNorm<B>,
    global_norm: LayerNorm<B>,
    gate_logits: Param<Tensor<B, 1>>, // [2] -> softmax to get gates for [token, global]
    blocks: Vec<TransformerBlock<B>>,
    norm: LayerNorm<B>,
    policy_head: Linear<B>,
    // Value head: attention pooling + 2-layer MLP (with globals concatenation)
    value_pool_fc1: Linear<B>,
    value_pool_fc2: Linear<B>,
    value_head_hidden: Linear<B>,
    value_head: Linear<B>,
    side_info_head: Linear<B>,
    // Time-usage head: attention pooling + 2-layer MLP (with globals concatenation)
    time_pool_fc1: Linear<B>,
    time_pool_fc2: Linear<B>,
    time_usage_head_hidden: Linear<B>,
    time_usage_head: Linear<B>,
    side_info_bce: BinaryCrossEntropyLoss<B>,
    policy_uncertainty: Param<Tensor<B, 1>>,
    value_uncertainty: Param<Tensor<B, 1>>,
    side_info_uncertainty: Param<Tensor<B, 1>>,
    time_usage_uncertainty: Param<Tensor<B, 1>>,
    recency_norm: LayerNorm<B>,
    policy_block: TransformerBlock<B>,
    value_block: TransformerBlock<B>,
    time_block: TransformerBlock<B>,
}

impl<B: Backend> OXIModel<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let base_embed_dim = config.embed_dim().saturating_sub(RECENCY_FEATURES);
        assert!(
            base_embed_dim >= 2,
            "embed_dim must exceed recency features to allocate token/global streams"
        );
        let token_embed = LinearConfig::new(BOARD_FEATURES_PER_TOKEN, base_embed_dim).init(device);

        let mut conv_layers = Vec::new();
        if config.conv_layers() > 0 {
            for _ in 0..config.conv_layers() {
                let conv =
                    Conv2dConfig::new([BOARD_FEATURES_PER_TOKEN, BOARD_FEATURES_PER_TOKEN], [3, 3])
                        .with_padding(PaddingConfig2d::Same)
                        .init(device);
                conv_layers.push(conv);
            }
        }

        let global_embed = LinearConfig::new(NUM_GLOBALS, base_embed_dim).init(device);

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers() {
            blocks.push(TransformerBlock::new(device));
        }

        // Per-stream layer norms
        let token_norm = LayerNormConfig::new(base_embed_dim).init(device);
        let global_norm = LayerNormConfig::new(base_embed_dim).init(device);
        let recency_norm = LayerNormConfig::new(RECENCY_FEATURES).init(device);

        // Gating logits initialized to values that give a softmax of ~ [0.55, 0.45],
        // providing a mild prior toward token features
        let gate_logits =
            Param::from_tensor(Tensor::from_data(TensorData::from([1.3, 1.0]), device));

        let norm = LayerNormConfig::new(config.embed_dim()).init(device);

        let policy_head = LinearConfig::new(config.embed_dim(), LEGAL_MOVES / 64).init(device);

        // Value head components
        let value_pool_fc1 = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let value_pool_fc2 = LinearConfig::new(config.embed_dim(), 1).init(device);
        let value_head_hidden =
            LinearConfig::new(config.embed_dim() + NUM_GLOBALS, config.embed_dim()).init(device);
        let value_head = LinearConfig::new(config.embed_dim(), 3).init(device);
        let side_info_head = LinearConfig::new(config.embed_dim(), 13).init(device);
        // Time-usage head components
        let time_pool_fc1 = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let time_pool_fc2 = LinearConfig::new(config.embed_dim(), 1).init(device);
        let time_usage_head_hidden =
            LinearConfig::new(config.embed_dim() + NUM_GLOBALS, config.embed_dim()).init(device);
        let time_usage_head = LinearConfig::new(config.embed_dim(), 2).init(device);
        let bce_config = BinaryCrossEntropyLossConfig::new().with_logits(true);
        let side_info_bce = bce_config.init(device);

        let policy_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let value_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let side_info_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let time_usage_uncertainty = Param::from_tensor(Tensor::zeros([1], device));

        let policy_block = TransformerBlock::new(device);
        let value_block = TransformerBlock::new(device);
        let time_block = TransformerBlock::new(device);

        Self {
            token_embed,
            conv_layers,
            global_embed,
            token_norm,
            global_norm,
            gate_logits,
            blocks,
            norm,
            policy_head,
            value_pool_fc1,
            value_pool_fc2,
            value_head_hidden,
            value_head,
            side_info_head,
            time_pool_fc1,
            time_pool_fc2,
            time_usage_head_hidden,
            time_usage_head,
            side_info_bce,
            policy_uncertainty,
            value_uncertainty,
            side_info_uncertainty,
            time_usage_uncertainty,
            recency_norm,
            policy_block,
            value_block,
            time_block,
        }
    }

    pub fn forward(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let _norm_scope = NormDebugScope::start("OXIModel::forward");
        let _root_stream = StreamScope::enter("root");

        log_tensor_stats("input.board_raw", &board);
        log_tensor_stats("input.globals_raw", &globals);

        let board_features = board.clone();
        log_tensor_stats("input.board_features", &board_features);

        let main_features = board_features
            .clone()
            .narrow(2, 0, BOARD_FEATURES_PER_TOKEN);
        log_tensor_stats("input.main_features", &main_features);

        let recency_features = board_features.narrow(2, BOARD_FEATURES_PER_TOKEN, RECENCY_FEATURES);
        log_tensor_stats("input.recency_features", &recency_features);

        let mut token_features = main_features.clone();
        log_tensor_stats("embed.token_features_initial", &token_features);

        if !self.conv_layers.is_empty() {
            let dims = token_features.dims();
            let batch_size = dims[0];
            let seq_len = dims[1];
            debug_assert_eq!(
                seq_len, 64,
                "Sequence length must be 64 for 8x8 board when applying convolution layers"
            );
            let channels = dims[2];
            debug_assert_eq!(
                channels, BOARD_FEATURES_PER_TOKEN,
                "Convolution stack expects BOARD_FEATURES_PER_TOKEN channels per square"
            );
            let mut conv_activations = token_features
                .reshape([batch_size, 8, 8, channels])
                .permute([0, 3, 1, 2]);
            for (layer_idx, conv) in self.conv_layers.iter().enumerate() {
                log_tensor_stats(&format!("embed.conv{layer_idx}.input"), &conv_activations);
                conv_activations = conv.forward(conv_activations);
                log_tensor_stats(&format!("embed.conv{layer_idx}.output"), &conv_activations);
                conv_activations = gelu(conv_activations);
                log_tensor_stats(&format!("embed.conv{layer_idx}.gelu"), &conv_activations);
            }
            token_features = conv_activations
                .permute([0, 2, 3, 1])
                .reshape([batch_size, seq_len, channels]);
            log_tensor_stats("embed.token_features_convolved", &token_features);
        }

        let token_embeds = self.token_embed.forward(token_features);
        log_tensor_stats("embed.token_embeds", &token_embeds);

        let global_embeds = self.global_embed.forward(globals.clone()).unsqueeze_dim(1);
        log_tensor_stats("embed.global_embeds", &global_embeds);
        debug_assert_eq!(
            token_embeds.dims()[1],
            64,
            "Sequence length must be 64 for 8x8 board"
        );

        // Normalize each stream
        let token_normed = self.token_norm.forward(token_embeds);
        let global_normed = self.global_norm.forward(global_embeds);
        let recency_normed = self.recency_norm.forward(recency_features);

        log_tensor_stats("embed.token_normed", &token_normed);
        log_tensor_stats("embed.global_normed", &global_normed);
        log_tensor_stats("embed.recency_normed", &recency_normed);

        // Softmax gates over the two streams
        let gates = softmax(self.gate_logits.val(), 0); // [2]
        log_tensor_stats("embed.gates_softmax", &gates);
        let gate_token = gates.clone().slice([0..1]).reshape([1, 1, 1]);
        let gate_global = gates.clone().slice([1..2]).reshape([1, 1, 1]);

        // Log gate values probabilistically to avoid host-device sync every step
        crate::config::shd_log(|| {
            let gate_vals = gates.clone().to_data();
            if let Ok(g) = gate_vals.as_slice::<f32>() {
                tracing::info!(
                    "Embedding gates (softmax): token={:.6} global={:.6}",
                    g[0],
                    g[1]
                );
            }
        });

        // Combine token/global streams with learned gates
        let combined = token_normed * gate_token + global_normed * gate_global;
        log_tensor_stats("embed.combined", &combined);
        let mut x = Tensor::cat(vec![combined, recency_normed], 2);
        log_tensor_stats("encoder.input_tokens", &x);

        {
            let _encoder_stream = StreamScope::enter("encoder");
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                let _layer_scope = LayerScope::enter(layer_idx);
                log_tensor_stats("encoder.pre_block", &x);
                x = block.forward(x);
                log_tensor_stats("encoder.post_block", &x);
            }
        }

        x = self.norm.forward(x);
        log_tensor_stats("encoder.post_norm", &x);

        let policy_logits = {
            let _stream = StreamScope::enter("policy");
            let tokens = self.policy_block.forward(x.clone());
            log_tensor_stats("policy.tokens", &tokens);
            let logits = self.policy_head.forward(tokens);
            log_tensor_stats("policy.logits", &logits);
            logits
        };

        let value_logits = {
            let _stream = StreamScope::enter("value");
            let value_tokens = self.value_block.forward(x.clone());
            log_tensor_stats("value.tokens", &value_tokens);
            let value_pool_hidden = self.value_pool_fc1.forward(value_tokens.clone());
            log_tensor_stats("value.pool_fc1", &value_pool_hidden);
            let value_pool_act = gelu(value_pool_hidden);
            log_tensor_stats("value.pool_fc1_gelu", &value_pool_act);
            let value_scores = self.value_pool_fc2.forward(value_pool_act);
            log_tensor_stats("value.attn_scores", &value_scores);
            let value_attn = softmax(value_scores.clone(), 1); // [batch, seq, 1]
            log_tensor_stats("value.attn_weights", &value_attn);
            let pooled_tokens = (value_tokens.clone() * value_attn.clone())
                .sum_dim(1)
                .squeeze_dim(1);
            log_tensor_stats("value.pooled_tokens", &pooled_tokens);
            let value_input = Tensor::cat(vec![pooled_tokens.clone(), globals.clone()], 1);
            log_tensor_stats("value.concat_input", &value_input);
            let value_hidden = gelu(self.value_head_hidden.forward(value_input));
            log_tensor_stats("value.hidden", &value_hidden);
            let logits = self.value_head.forward(value_hidden);
            log_tensor_stats("value.logits", &logits);
            logits
        };

        let side_info_logits = {
            let _stream = StreamScope::enter("side_info");
            let mean_features = x.clone().mean_dim(1).squeeze_dim(1);
            log_tensor_stats("side.mean_features", &mean_features);
            let logits = self.side_info_head.forward(mean_features);
            log_tensor_stats("side.logits", &logits);
            logits
        };

        let time_usage_logits = {
            let _stream = StreamScope::enter("time");
            let time_tokens = self.time_block.forward(x.clone());
            log_tensor_stats("time.tokens", &time_tokens);
            let time_pool_hidden = self.time_pool_fc1.forward(time_tokens.clone());
            log_tensor_stats("time.pool_fc1", &time_pool_hidden);
            let time_pool_act = gelu(time_pool_hidden);
            log_tensor_stats("time.pool_fc1_gelu", &time_pool_act);
            let time_scores = self.time_pool_fc2.forward(time_pool_act);
            log_tensor_stats("time.attn_scores", &time_scores);
            let time_attn = softmax(time_scores.clone(), 1); // [batch, seq, 1]
            log_tensor_stats("time.attn_weights", &time_attn);
            let time_pooled_tokens = (time_tokens.clone() * time_attn.clone())
                .sum_dim(1)
                .squeeze_dim(1);
            log_tensor_stats("time.pooled_tokens", &time_pooled_tokens);
            let time_input = Tensor::cat(vec![time_pooled_tokens.clone(), globals.clone()], 1);
            log_tensor_stats("time.concat_input", &time_input);
            let time_hidden = gelu(self.time_usage_head_hidden.forward(time_input));
            log_tensor_stats("time.hidden", &time_hidden);
            let time_usage_raw = self.time_usage_head.forward(time_hidden);
            log_tensor_stats("time.raw_logits", &time_usage_raw);

            // Reparameterize time usage outputs into Beta(alpha, beta)
            let mean_raw = time_usage_raw
                .clone()
                .slice([0..time_usage_raw.dims()[0], 0..1]);
            let concentration_raw = time_usage_raw
                .clone()
                .slice([0..time_usage_raw.dims()[0], 1..2]);
            log_tensor_stats("time.mean_raw", &mean_raw);
            log_tensor_stats("time.concentration_raw", &concentration_raw);

            let eps = 1e-4f32;
            let mean = sigmoid(mean_raw).clamp(eps, 1.0 - eps);
            let min_concentration = 2.0;
            let concentration = softplus(concentration_raw, 1.0).add_scalar(min_concentration);
            log_tensor_stats("time.mean", &mean);
            log_tensor_stats("time.concentration", &concentration);

            let alphas = mean.clone() * concentration.clone();
            let betas = (Tensor::ones_like(&mean) - mean) * concentration.clone();
            log_tensor_stats("time.alphas", &alphas);
            log_tensor_stats("time.betas", &betas);

            let logits = Tensor::cat(vec![alphas, betas], 1);
            log_tensor_stats("time.logits", &logits);
            logits
        };

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
        let batch_clone = batch.clone();
        let config = get_global_config();
        let batch_size = batch.board_input.shape().dims[0];

        let (policy_logits, value_logits, side_info_logits, time_usage_logits) =
            self.forward(batch.board_input, batch.global_features);

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
        let eps = config.policy_label_smoothing;
        let legal_counts = batch
            .legal_moves
            .clone()
            .sum_dim(1)
            .reshape([batch_size, 1])
            .clamp_min(1.0);
        let uniform_over_legal = batch.legal_moves.clone() / legal_counts;
        let targets_smoothed =
            batch.move_distributions.clone() * (1.0 - eps) + uniform_over_legal * eps;

        // Focal Loss: FL(p_t) = -(1 - p_t)^γ * log(p_t)
        let gamma = config.focal_loss_gamma;
        let policy_probs = softmax(policy_logits_flat.clone(), 1);
        let policy_probs = policy_probs.mask_fill(mask.clone(), 0.0);

        // Compute p_t for each target
        let p_t = (targets_smoothed.clone() * policy_probs).sum_dim(1); // Sum over classes for each sample
        let focal_weight = (Tensor::ones_like(&p_t) - p_t.clone()).powf_scalar(gamma);

        // Standard cross-entropy loss per sample
        let ce_loss_per_sample = (targets_smoothed * log_policy).sum_dim(1).neg();

        // Apply focal weight
        let policy_loss = (focal_weight * ce_loss_per_sample).mean();

        // Value loss
        let value_log_probs = log_softmax(value_logits.clone(), 1);
        let value_probs = value_log_probs.clone().exp();
        let ce_per_sample = (batch.values.clone() * value_log_probs.clone())
            .sum_dim(1)
            .neg();

        let batch_range = 0..batch_size;
        let loss_probs = value_probs.clone().slice([batch_range.clone(), 0..1]);
        let win_probs = value_probs.clone().slice([batch_range, 2..3]);
        let decisive_mass = loss_probs.clone() + win_probs.clone();
        let denom = decisive_mass.clone().clamp_min(1e-8);
        let loss_norm = loss_probs.clone() / denom.clone();
        let win_norm = win_probs.clone() / denom;
        let binary_entropy = (loss_norm.clone() * (loss_norm.clone().add_scalar(1e-8).log())
            + win_norm.clone() * (win_norm.clone().add_scalar(1e-8).log()))
        .neg();
        let entropy_bonus = (binary_entropy * decisive_mass).sum_dim(1);

        // Helper for creating zero tensors
        let zero_like = || Tensor::zeros([1], &policy_logits_flat_original.device());

        // Only compute value loss if weight is non-zero
        let (base_value_loss, value_term) = if config.value_loss_weight > 0.0 {
            let value_loss = (ce_per_sample - entropy_bonus * config.value_entropy_weight).mean();
            let weighted = value_loss.clone() * config.value_loss_weight;
            (value_loss, weighted)
        } else {
            let zero = zero_like();
            (zero.clone(), zero)
        };

        // Side info loss removed from training hot path (kept zero)

        // Only compute time usage loss if weight is non-zero
        let (base_time_usage_loss, time_usage_term) = if config.time_usage_loss_weight > 0.0 {
            let time_usage_loss = self
                .compute_time_usage_loss_impl(time_usage_logits.clone(), batch.time_usages.clone());
            let weighted = time_usage_loss.clone() * config.time_usage_loss_weight;
            (time_usage_loss, weighted)
        } else {
            let zero = zero_like();
            (zero.clone(), zero)
        };

        // Policy loss is always computed (always has non-zero weight in practice)
        let base_policy_loss = policy_loss.clone();
        let config_weighted_policy_loss = base_policy_loss.clone() * config.policy_loss_weight;

        let loss =
            config_weighted_policy_loss.clone() + value_term.clone() + time_usage_term.clone();

        // Accuracy
        let targets = batch.move_distributions.clone().argmax(1).squeeze_dim(1);
        let policy_logits_only_legals = policy_logits_flat_original
            .clone()
            .mask_fill(mask.clone(), 0.0);
        // Removed unused predicted move argmax to avoid extra compute
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
            config_weighted_policy_loss.clone(),
            value_term.clone(),
            time_usage_term.clone(),
            policy_logits_flat,
            targets,
            value_logits,
            batch.values.clone(),
            batch.legal_moves.clone(),
        )
        // .with_uncertainties((sigma_policy, sigma_value, sigma_time_usage))
        .with_raw_losses(
            base_policy_loss.clone(),
            base_value_loss.clone(),
            base_time_usage_loss.clone(),
        )
        .with_base_losses(base_policy_loss, base_value_loss, base_time_usage_loss)
    }

    #[cfg(test)]
    pub fn compute_time_usage_loss(
        &self,
        time_usage_logits: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1>
    where
        B::FloatElem: From<f32>,
    {
        self.compute_time_usage_loss_impl(time_usage_logits, targets)
    }

    fn compute_time_usage_loss_impl(
        &self,
        time_usage_logits: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1>
    where
        B::FloatElem: From<f32>,
    {
        let batch_size = time_usage_logits.dims()[0];

        // Extract parameters: [alpha, beta] for Beta distribution
        let alphas = time_usage_logits
            .clone()
            .slice([0..batch_size, 0..1])
            .flatten::<1>(0, 1);
        let betas = time_usage_logits
            .clone()
            .slice([0..batch_size, 1..2])
            .flatten::<1>(0, 1);

        let eps = 1e-3f32;
        let offset = (1.0f32 / eps).ln();
        let targets_flat = targets.flatten::<1>(0, 1).clamp(eps, 1.0 - eps);
        let log_pdf = beta_log_pdf(targets_flat, alphas, betas);
        let nll = log_pdf.neg();
        let offset_tensor = Tensor::ones_like(&nll) * offset;

        (nll + offset_tensor).mean().reshape([1])
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
            let batch_probs = &probs[b * LEGAL_MOVES..(b + 1) * LEGAL_MOVES];
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
                .map(|&(idx, prob)| (idx / 76, idx % 76, prob))
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

    #[cfg(target_os = "macos")]
    type TestBackend = burn::backend::Metal;
    #[cfg(not(target_os = "macos"))]
    type TestBackend = burn::backend::LibTorch<f32>;

    #[test]
    fn test_beta_loss_matches_manual_nll() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = ModelConfig::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<TestBackend>::new(&device, &config);

        let time_usage_params =
            Tensor::from_data(TensorData::from([[2.0, 3.0], [5.0, 2.5]]), &device);

        let targets = Tensor::from_data(TensorData::from([[0.25], [0.7]]), &device);

        let loss = model.compute_time_usage_loss(time_usage_params.clone(), targets.clone());
        assert_eq!(loss.dims(), [1]);

        let manual = {
            let alphas = time_usage_params
                .clone()
                .slice([0..2, 0..1])
                .flatten::<1>(0, 1);
            let betas = time_usage_params.slice([0..2, 1..2]).flatten::<1>(0, 1);
            let eps = 1e-3f32;
            let offset = (1.0f32 / eps).ln();
            let targets_flat = targets.flatten::<1>(0, 1).clamp(eps, 1.0 - eps);
            let log_pdf = beta_log_pdf(targets_flat, alphas, betas);
            let nll = log_pdf.neg();
            let offset_tensor = Tensor::ones_like(&nll) * offset;
            (nll + offset_tensor).mean().reshape([1])
        };

        let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];
        let manual_value = manual.to_data().as_slice::<f32>().unwrap()[0];

        assert!(
            (loss_value - manual_value).abs() < 1e-5,
            "Loss mismatch: got {loss_value}, expected {manual_value}"
        );
    }

    #[test]
    fn test_uncertainty_weighted_loss_non_negative_for_small_losses() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        // Very small raw losses (simulate < 0.02 edge case)
        let small_losses = [0.02f32, 0.005f32, 1e-6f32];

        for &l in &small_losses {
            let raw_loss = Tensor::<TestBackend, 1>::from_data([l], &device);
            let log_sigma = Tensor::<TestBackend, 1>::from_data([0.0f32], &device); // sigma=1

            let sigma_sq = (log_sigma.clone() * 2.0).exp();
            let penalty = (Tensor::ones_like(&sigma_sq) + sigma_sq.clone()).log() * 0.5f32;
            let weighted = (raw_loss / sigma_sq) * 0.5f32 + penalty;

            let v = weighted.to_data().as_slice::<f32>().unwrap()[0];
            assert!(
                v.is_finite() && v >= 0.0,
                "Weighted loss should be non-negative, got {} for raw {}",
                v,
                l
            );
        }
    }

    #[test]
    fn test_time_usage_head_output_shape() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = ModelConfig::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<TestBackend>::new(&device, &config);

        // Create dummy input
        let batch_size = 2;
        let board_input = Tensor::zeros([batch_size, 64, FEATURES_PER_TOKEN], &device);
        let global_features = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let (policy_logits, value_logits, side_info_logits, time_usage_logits) =
            model.forward(board_input, global_features);

        // Check time usage head outputs 2 values per batch (alpha, beta)
        assert_eq!(time_usage_logits.dims(), [batch_size, 2]);

        // Check that alpha and beta parameters are positive
        let time_usage_data = time_usage_logits.to_data();
        let values = time_usage_data.as_slice::<f32>().unwrap();

        for i in 0..batch_size {
            let alpha = values[i * 2];
            let beta = values[i * 2 + 1];

            assert!(alpha > 0.0, "Alpha should be positive, got: {}", alpha);
            assert!(beta > 0.0, "Beta should be positive, got: {}", beta);
        }

        println!("Time usage head output shape test passed!");
    }

    #[test]
    fn test_beta_loss_handles_extreme_targets() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = ModelConfig::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<TestBackend>::new(&device, &config);

        let time_usage_params =
            Tensor::from_data(TensorData::from([[2.0, 5.0], [8.0, 1.5]]), &device);
        let targets = Tensor::from_data(TensorData::from([[1e-5], [1.0 - 1e-5]]), &device);

        let loss = model.compute_time_usage_loss(time_usage_params, targets);
        let value = loss.to_data().as_slice::<f32>().unwrap()[0];

        assert!(value.is_finite() && value > 0.0);
    }
}
