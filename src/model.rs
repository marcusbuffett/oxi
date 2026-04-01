use crate::config::{
    ModelConfig, BOARD_FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS, RECENCY_FEATURES,
};
use crate::distribution_utils::beta_log_pdf;
use crate::factorized_policy::FactorizedPolicyHead;
use crate::relative_position_transformer::TransformerBlock;
use crate::smolgen::SmolgenWeightGen;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig2d, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, silu, softmax};

#[cfg(feature = "train")]
use crate::config::get_global_config;
#[cfg(feature = "train")]
use crate::forward_timing::{finish_and_log_forward_pass, start_forward_pass, TimingScope};
#[cfg(feature = "train")]
use crate::model_prediction_logger::log_model_predictions;
#[cfg(feature = "train")]
use crate::norm_debug::{log_tensor_stats, LayerScope, NormDebugScope, StreamScope};
#[cfg(feature = "train")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "train")]
use burn::train::{TrainOutput, TrainStep, ValidStep};

#[cfg(not(feature = "train"))]
use crate::train_stubs::*;

#[derive(Module, Debug)]
pub struct OXIModel<B: Backend> {
    token_embed: Linear<B>,
    conv_layers: Vec<Conv2d<B>>,
    token_norm: RmsNorm<B>,
    smolgen_weight_gen: SmolgenWeightGen<B>,
    blocks: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    policy_head: FactorizedPolicyHead<B>,
    value_pool_fc1: Linear<B>,
    value_pool_fc2: Linear<B>,
    value_head_hidden: Linear<B>,
    value_head: Linear<B>,
    side_info_head: Linear<B>,
    time_pool_fc1: Linear<B>,
    time_pool_fc2: Linear<B>,
    time_usage_head_hidden: Linear<B>,
    time_usage_head: Linear<B>,
    side_info_bce: BinaryCrossEntropyLoss<B>,
    policy_uncertainty: Param<Tensor<B, 1>>,
    value_uncertainty: Param<Tensor<B, 1>>,
    side_info_uncertainty: Param<Tensor<B, 1>>,
    time_usage_uncertainty: Param<Tensor<B, 1>>,
    recency_norm: RmsNorm<B>,
    policy_block: TransformerBlock<B>,
    value_block: TransformerBlock<B>,
    // Disabled: time_block was unused in forward pass, wastes parameters
    // time_block: TransformerBlock<B>,
    aux_mobility_head: Linear<B>,
    aux_material_head: Linear<B>,
    aux_from_square_head: Linear<B>,
    aux_to_square_head: Linear<B>,
    aux_from_square_hidden: Linear<B>,
    aux_to_square_hidden: Linear<B>,
    aux_trunk_from_square_head: Linear<B>,
    aux_trunk_from_square_hidden: Linear<B>,
    aux_trunk_to_square_head: Linear<B>,
    aux_trunk_to_square_hidden: Linear<B>,
}

impl<B: Backend> OXIModel<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let base_embed_dim = config.embed_dim().saturating_sub(RECENCY_FEATURES);
        assert!(
            base_embed_dim >= 2,
            "embed_dim must exceed recency features"
        );

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        let token_embed = LinearConfig::new(BOARD_FEATURES_PER_TOKEN, base_embed_dim)
            .with_initializer(std_init.clone())
            .init(device);

        let mut conv_layers = Vec::new();
        if config.conv_layers() > 0 {
            for _ in 0..config.conv_layers() {
                let conv =
                    Conv2dConfig::new([BOARD_FEATURES_PER_TOKEN, BOARD_FEATURES_PER_TOKEN], [3, 3])
                        .with_padding(PaddingConfig2d::Same)
                        .with_initializer(std_init.clone())
                        .init(device);
                conv_layers.push(conv);
            }
        }

        let smolgen_weight_gen = SmolgenWeightGen::new(device);

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers() {
            blocks.push(TransformerBlock::new(device));
        }

        let token_norm = RmsNormConfig::new(base_embed_dim).init(device);
        let recency_norm = RmsNormConfig::new(RECENCY_FEATURES).init(device);

        let norm = RmsNormConfig::new(config.embed_dim()).init(device);

        let policy_head = FactorizedPolicyHead::new(device);

        // Value head components - standard initialization
        let value_pool_fc1 = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let value_pool_fc2 = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let value_head_hidden =
            LinearConfig::new(config.embed_dim() + NUM_GLOBALS, config.embed_dim())
                .with_initializer(std_init.clone())
                .init(device);
        let value_head = LinearConfig::new(config.embed_dim(), 3)
            .with_initializer(std_init.clone())
            .init(device);
        let side_info_head = LinearConfig::new(config.embed_dim(), 13)
            .with_initializer(std_init.clone())
            .init(device);
        // Time-usage head components - standard initialization
        let time_pool_fc1 = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let time_pool_fc2 = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let time_usage_head_hidden =
            LinearConfig::new(config.embed_dim() + NUM_GLOBALS, config.embed_dim())
                .with_initializer(std_init.clone())
                .init(device);
        let time_usage_head = LinearConfig::new(config.embed_dim(), 2)
            .with_initializer(std_init.clone())
            .init(device);
        let bce_config = BinaryCrossEntropyLossConfig::new().with_logits(true);
        let side_info_bce = bce_config.init(device);

        let policy_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let value_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let side_info_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let time_usage_uncertainty = Param::from_tensor(Tensor::zeros([1], device));

        let policy_block = TransformerBlock::new(device);
        let value_block = TransformerBlock::new(device);
        // Disabled: time_block was unused in forward pass, wastes parameters
        // let time_block = TransformerBlock::new(device);

        // Auxiliary prediction heads
        let aux_mobility_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_material_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_from_square_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_to_square_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_from_square_hidden = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let aux_to_square_hidden = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let aux_trunk_from_square_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_trunk_from_square_hidden = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let aux_trunk_to_square_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_trunk_to_square_hidden = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);

        Self {
            token_embed,
            conv_layers,
            token_norm,
            smolgen_weight_gen,
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
            // time_block,
            aux_mobility_head,
            aux_material_head,
            aux_from_square_head,
            aux_to_square_head,
            aux_from_square_hidden,
            aux_to_square_hidden,
            aux_trunk_from_square_head,
            aux_trunk_from_square_hidden,
            aux_trunk_to_square_head,
            aux_trunk_to_square_hidden,
        }
    }

    pub fn forward(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (policy, value, side_info, time_usage, _trunk, _policy_tokens) =
            self.forward_with_trunk(board, globals);
        (policy, value, side_info, time_usage)
    }

    fn forward_with_trunk(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        start_forward_pass();
        let device = board.device();
        let total_timing = TimingScope::new_with_sync::<B>("forward_total", &device);
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
                conv_activations = silu(conv_activations);
                log_tensor_stats(&format!("embed.conv{layer_idx}.silu"), &conv_activations);
            }
            token_features = conv_activations
                .permute([0, 2, 3, 1])
                .reshape([batch_size, seq_len, channels]);
            log_tensor_stats("embed.token_features_convolved", &token_features);
        }

        let token_embeds = {
            let _t = TimingScope::new_with_sync::<B>("token_embed", &device);
            self.token_embed.forward(token_features)
        };
        log_tensor_stats("embed.token_embeds", &token_embeds);
        debug_assert_eq!(
            token_embeds.dims()[1],
            64,
            "Sequence length must be 64 for 8x8 board"
        );

        let token_normed = self.token_norm.forward(token_embeds);
        let recency_normed = self.recency_norm.forward(recency_features);

        log_tensor_stats("embed.token_normed", &token_normed);
        log_tensor_stats("embed.recency_normed", &recency_normed);

        let mut x = Tensor::cat(vec![token_normed, recency_normed], 2);
        log_tensor_stats("encoder.input_tokens", &x);

        {
            let _encoder_stream = StreamScope::enter("encoder");
            let _encoder_timing = TimingScope::new_with_sync::<B>("encoder_blocks", &device);
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                let _layer_scope = LayerScope::enter(layer_idx);
                let _block_timing = TimingScope::new_with_sync::<B>("encoder_block", &device);
                log_tensor_stats("encoder.pre_block", &x);

                x = block.forward_with_film(x, &self.smolgen_weight_gen, globals.clone());
                log_tensor_stats("encoder.post_block", &x);
            }
        }

        x = self.norm.forward(x);
        log_tensor_stats("encoder.post_norm", &x);

        let (policy_logits, policy_tokens) = {
            let _stream = StreamScope::enter("policy");
            let _timing = TimingScope::new_with_sync::<B>("policy_head", &device);
            let tokens = {
                let _t = TimingScope::new_with_sync::<B>("policy_block", &device);
                self.policy_block
                    .forward_with_film(x.clone(), &self.smolgen_weight_gen, globals.clone())
            };
            log_tensor_stats("policy.tokens", &tokens);
            let logits = {
                let _t = TimingScope::new_with_sync::<B>("factorized_policy", &device);
                self.policy_head.forward(tokens.clone())
            };
            log_tensor_stats("policy.logits", &logits);
            (logits, tokens)
        };

        let aux_batch_size = board.dims()[0];
        let embed_dim = x.dims()[2];

        // Value head: value_block → attention pooling → hidden → WDL logits
        let value_logits = {
            let _stream = StreamScope::enter("value");
            let _timing = TimingScope::new_with_sync::<B>("value_head", &device);
            let value_tokens = self.value_block.forward(x.clone(), &self.smolgen_weight_gen);
            log_tensor_stats("value.tokens", &value_tokens);

            // Attention pooling: fc1 → silu → fc2 → softmax → weighted sum
            let pool_hidden = silu(self.value_pool_fc1.forward(value_tokens.clone()));
            let pool_weights = softmax(self.value_pool_fc2.forward(pool_hidden).reshape([aux_batch_size, 64]), 1)
                .reshape([aux_batch_size, 64, 1]);
            let pooled = (value_tokens * pool_weights).sum_dim(1).reshape([aux_batch_size, embed_dim]);
            log_tensor_stats("value.pooled", &pooled);

            // Concat with globals, hidden layer, output
            let with_globals = Tensor::cat(vec![pooled, globals.clone()], 1);
            let hidden = silu(self.value_head_hidden.forward(with_globals));
            self.value_head.forward(hidden)
        };
        log_tensor_stats("value.logits", &value_logits);

        let trunk_pooled = x.clone().mean_dim(1).reshape([aux_batch_size, embed_dim]);
        let side_info_logits = self.side_info_head.forward(trunk_pooled);
        let time_usage_logits = Tensor::zeros([aux_batch_size, 2], &device);

        drop(total_timing);
        finish_and_log_forward_pass();
        (
            policy_logits,
            value_logits,
            side_info_logits,
            time_usage_logits,
            x,
            policy_tokens,
        )
    }

    #[cfg(feature = "train")]
    pub fn forward_classification(
        &self,
        batch: crate::dataset::ChessBatch<B>,
    ) -> crate::chess_output::ChessOutput<B>
    where
        B::FloatElem: From<f32>,
    {
        let batch_clone = batch.clone();
        let config = get_global_config();
        let batch_size = batch.board_input.shape().dims[0];

        let (policy_logits, value_logits, _side_info_logits, time_usage_logits, trunk_output, policy_tokens) =
            self.forward_with_trunk(batch.board_input, batch.global_features);

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

        // Standard cross-entropy loss per sample
        let ce_loss_per_sample = (targets_smoothed.clone() * log_policy.clone()).sum_dim(1).neg();

        // Focal Loss: FL(p_t) = -(1 - p_t)^γ * log(p_t)
        let gamma = config.focal_loss_gamma;
        let policy_loss = if gamma > 0.0 {
            // Derive softmax from log_softmax: softmax(x) = exp(log_softmax(x))
            let policy_probs = log_policy.clone().exp();
            let policy_probs = policy_probs.mask_fill(mask.clone(), 0.0);

            // Compute p_t for each target
            let p_t = (targets_smoothed * policy_probs).sum_dim(1);
            let focal_weight = (Tensor::ones_like(&p_t) - p_t.clone()).powf_scalar(gamma);

            // Apply focal weight
            (focal_weight * ce_loss_per_sample).mean()
        } else {
            // gamma=0: standard cross-entropy (focal weight is 1.0)
            ce_loss_per_sample.mean()
        };

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

        // Policy loss is always computed
        let base_policy_loss = policy_loss.clone();
        let config_weighted_policy_loss = base_policy_loss.clone() * config.policy_loss_weight;

        // Auxiliary losses (mobility + material prediction)
        let (base_aux_loss, aux_mobility_loss_f32, aux_material_loss_f32, aux_mobility_mae_f32, aux_material_mae_f32) = if config.aux_loss_weight > 0.0 {
            // Mobility: predict legal move count per square from trunk output
            let legal_per_square = batch
                .legal_moves
                .clone()
                .reshape([batch_size, 64, 76])
                .sum_dim(2)
                .reshape([batch_size, 64]);
            let legal_per_square_norm = legal_per_square / 27.0f32;

            let mobility_pred = self
                .aux_mobility_head
                .forward(trunk_output.clone())
                .reshape([batch_size, 64]);
            let mobility_diff = mobility_pred - legal_per_square_norm;
            let mobility_mse = mobility_diff.clone().powf_scalar(2.0).mean();
            let mobility_mae = mobility_diff.abs().mean();

            // Material: predict material imbalance from mean-pooled trunk
            let embed_dim = trunk_output.dims()[2];
            let trunk_pooled = trunk_output.clone().mean_dim(1).reshape([batch_size, embed_dim]);
            let material_pred = self
                .aux_material_head
                .forward(trunk_pooled)
                .reshape([batch_size]);
            let material_target = batch.material_imbalance.clone();
            let material_diff = material_pred - material_target;
            let material_mse = material_diff.clone().powf_scalar(2.0).mean();
            let material_mae = material_diff.abs().mean();

            // Extract f32 values for metrics (detached, no grad)
            let mob_loss_f32 = mobility_mse.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);
            let mat_loss_f32 = material_mse.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);
            let mob_mae_f32 = mobility_mae.into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);
            let mat_mae_f32 = material_mae.into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            (mobility_mse + material_mse, mob_loss_f32, mat_loss_f32, mob_mae_f32, mat_mae_f32)
        } else {
            (zero_like(), 0.0, 0.0, 0.0, 0.0)
        };

        // Maia 2-style auxiliary losses: side info, from-square, to-square
        let (maia_loss, aux_side_info_loss_f32, aux_from_sq_loss_f32, aux_to_sq_loss_f32, aux_from_sq_acc_f32, aux_to_sq_acc_f32) = if config.aux_loss_weight > 0.0 {
            let embed_dim = trunk_output.dims()[2];

            // Side info: piece moved/captured/check (first 13 values)
            let side_info_target_int = batch.side_info.clone()
                .slice([0..batch_size, 0..13]);
            let trunk_pooled_si = trunk_output.clone().mean_dim(1).reshape([batch_size, embed_dim]);
            let side_info_logits = self.side_info_head.forward(trunk_pooled_si);
            let side_info_bce = self.side_info_bce.forward(side_info_logits, side_info_target_int);

            let si_loss_f32 = side_info_bce.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            // From-square: per-token prediction using policy tokens → [batch, 64]
            let from_sq_hidden = silu(self.aux_from_square_hidden
                .forward(policy_tokens.clone()));
            let from_sq_logits = self.aux_from_square_head
                .forward(from_sq_hidden)
                .reshape([batch_size, 64]);
            let from_sq_target_int = batch.side_info.clone()
                .slice([0..batch_size, 13..77]);

            // Use cross-entropy instead of BCE: from-square is a 64-class classification
            let from_sq_log_probs = log_softmax(from_sq_logits.clone(), 1);
            let from_sq_target_float = from_sq_target_int.clone().float();
            let from_sq_ce = (from_sq_target_float * from_sq_log_probs).sum_dim(1).neg().mean();

            let from_loss_f32 = from_sq_ce.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            // From-square accuracy: argmax match
            let from_pred = from_sq_logits.argmax(1).squeeze_dim::<1>(1);
            let from_true = from_sq_target_int.argmax(1).squeeze_dim::<1>(1);
            let from_correct = from_pred.equal(from_true).float().mean();
            let from_acc_f32 = from_correct.into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            // To-square: per-token prediction using policy tokens → [batch, 64]
            let to_sq_hidden = silu(self.aux_to_square_hidden
                .forward(policy_tokens.clone()));
            let to_sq_logits = self.aux_to_square_head
                .forward(to_sq_hidden)
                .reshape([batch_size, 64]);
            let to_sq_target_int = batch.side_info.clone()
                .slice([0..batch_size, 77..141]);

            // Use cross-entropy instead of BCE: to-square is a 64-class classification
            let to_sq_log_probs = log_softmax(to_sq_logits.clone(), 1);
            let to_sq_target_float = to_sq_target_int.clone().float();
            let to_sq_ce = (to_sq_target_float * to_sq_log_probs).sum_dim(1).neg().mean();

            let to_loss_f32 = to_sq_ce.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            // To-square accuracy: argmax match
            let to_pred = to_sq_logits.argmax(1).squeeze_dim::<1>(1);
            let to_true = to_sq_target_int.argmax(1).squeeze_dim::<1>(1);
            let to_correct = to_pred.equal(to_true).float().mean();
            let to_acc_f32 = to_correct.into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);

            // Trunk-level from/to square prediction: direct aux supervision on trunk tokens
            // This gives the trunk direct gradient signal about which squares matter,
            // rather than only getting it indirectly through the policy_block.
            let trunk_from_sq_target = batch.side_info.clone()
                .slice([0..batch_size, 13..77]);
            let trunk_from_sq_hidden = silu(self.aux_trunk_from_square_hidden
                .forward(trunk_output.clone()));
            let trunk_from_sq_logits = self.aux_trunk_from_square_head
                .forward(trunk_from_sq_hidden)
                .reshape([batch_size, 64]);
            let trunk_from_sq_log_probs = log_softmax(trunk_from_sq_logits, 1);
            let trunk_from_sq_target_float = trunk_from_sq_target.float();
            let trunk_from_sq_ce = (trunk_from_sq_target_float * trunk_from_sq_log_probs).sum_dim(1).neg().mean();

            let trunk_to_sq_target = batch.side_info.clone()
                .slice([0..batch_size, 77..141]);
            let trunk_to_sq_hidden = silu(self.aux_trunk_to_square_hidden
                .forward(trunk_output.clone()));
            let trunk_to_sq_logits = self.aux_trunk_to_square_head
                .forward(trunk_to_sq_hidden)
                .reshape([batch_size, 64]);
            let trunk_to_sq_log_probs = log_softmax(trunk_to_sq_logits, 1);
            let trunk_to_sq_target_float = trunk_to_sq_target.float();
            let trunk_to_sq_ce = (trunk_to_sq_target_float * trunk_to_sq_log_probs).sum_dim(1).neg().mean();

            (side_info_bce + from_sq_ce + to_sq_ce + trunk_from_sq_ce + trunk_to_sq_ce, si_loss_f32, from_loss_f32, to_loss_f32, from_acc_f32, to_acc_f32)
        } else {
            (zero_like(), 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let base_aux_loss = base_aux_loss + maia_loss;
        let aux_term = base_aux_loss.clone() * config.aux_loss_weight;

        let loss = config_weighted_policy_loss.clone()
            + value_term.clone()
            + time_usage_term.clone()
            + aux_term.clone();

        // Accuracy
        let targets = batch.move_distributions.clone().argmax(1).squeeze_dim(1);
        let policy_logits_only_legals = policy_logits_flat_original
            .clone()
            .mask_fill(mask.clone(), 0.0);

        crate::chess_output::ChessOutput::new(
            loss,
            config_weighted_policy_loss.clone(),
            value_term.clone(),
            time_usage_term.clone(),
            aux_term,
            policy_logits_flat,
            targets,
            value_logits,
            batch.values.clone(),
            batch.legal_moves.clone(),
        )
        .with_raw_losses(
            base_policy_loss.clone(),
            base_value_loss.clone(),
            base_time_usage_loss.clone(),
        )
        .with_base_losses(
            base_policy_loss,
            base_value_loss,
            base_time_usage_loss,
            base_aux_loss,
        )
        .with_aux_metrics(
            aux_mobility_loss_f32,
            aux_material_loss_f32,
            aux_mobility_mae_f32,
            aux_material_mae_f32,
        )
        .with_maia_metrics(
            aux_side_info_loss_f32,
            aux_from_sq_loss_f32,
            aux_to_sq_loss_f32,
            aux_from_sq_acc_f32,
            aux_to_sq_acc_f32,
        )
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

#[cfg(feature = "train")]
impl<B: AutodiffBackend>
    TrainStep<crate::dataset::ChessBatch<B>, crate::chess_output::ChessOutput<B>> for OXIModel<B>
where
    B::FloatElem: From<f32>,
{
    fn step(
        &self,
        batch: crate::dataset::ChessBatch<B>,
    ) -> TrainOutput<crate::chess_output::ChessOutput<B>> {
        let item = self.forward_classification(batch);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

#[cfg(feature = "train")]
impl<B: Backend> ValidStep<crate::dataset::ChessBatch<B>, crate::chess_output::ChessOutput<B>>
    for OXIModel<B>
where
    B::FloatElem: From<f32>,
{
    fn step(&self, batch: crate::dataset::ChessBatch<B>) -> crate::chess_output::ChessOutput<B> {
        self.forward_classification(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FEATURES_PER_TOKEN, NUM_GLOBALS};
    use crate::test_backend::{test_device, TestBackend};
    use burn::tensor::TensorData;

    #[test]
    fn test_beta_loss_matches_manual_nll() {
        let device = test_device();
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
        let device = test_device();

        let small_losses = [0.02f32, 0.005f32, 1e-6f32];

        for &l in &small_losses {
            let raw_loss = Tensor::<TestBackend, 1>::from_data([l], &device);
            let log_sigma = Tensor::<TestBackend, 1>::from_data([0.0f32], &device);

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
        let device = test_device();
        let config = ModelConfig::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<TestBackend>::new(&device, &config);

        let batch_size = 2;
        let board_input = Tensor::zeros([batch_size, 64, FEATURES_PER_TOKEN], &device);
        let global_features = Tensor::zeros([batch_size, NUM_GLOBALS], &device);

        let (policy_logits, _value_logits, _side_info_logits, time_usage_logits) =
            model.forward(board_input, global_features);

        assert_eq!(
            policy_logits.dims(),
            [batch_size, 64, 76],
            "policy_logits shape"
        );
        assert_eq!(
            time_usage_logits.dims(),
            [batch_size, 2],
            "time_usage_logits shape"
        );
    }

    #[test]
    fn test_beta_loss_handles_extreme_targets() {
        let device = test_device();
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