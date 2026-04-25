use crate::calibration::calibration_skill_band_label;
use crate::calibration::RegretBin;
use crate::config::{
    ModelConfig, FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS, RETRIEVAL_OBJECTIVE_POLICY_OVERLAP,
};
use crate::distribution_utils::beta_log_pdf;
use crate::factorized_policy::FactorizedPolicyHead;
use crate::relative_position_transformer::TransformerBlock;
use crate::smolgen::SmolgenWeightGen;
use crate::spatial_conv::SpatialConv;
use burn::module::Param;
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
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
    embed_proj1: Linear<B>,
    embed_proj2: Linear<B>,
    square_embed: Param<Tensor<B, 2>>,
    spatial_conv: Option<SpatialConv<B>>,
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
    policy_block: TransformerBlock<B>,
    value_block: TransformerBlock<B>,
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
    cp_loss_head_hidden: Linear<B>,
    cp_loss_head: Linear<B>,
    retrieval_head: Linear<B>,
}

impl<B: Backend> OXIModel<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let embed_dim = config.embed_dim();
        let embed_hidden_dim = 256; // intermediate dim for two-layer embedding

        // Standard initialization: Normal(0, 0.02)
        let std_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        // Two-layer nonlinear embedding: all features (board + recency) → hidden → embed_dim
        let embed_proj1 = LinearConfig::new(FEATURES_PER_TOKEN, embed_hidden_dim)
            .with_initializer(std_init.clone())
            .init(device);
        let embed_proj2 = LinearConfig::new(embed_hidden_dim, embed_dim)
            .with_initializer(std_init.clone())
            .init(device);

        // Learnable per-square positional embedding (full embed_dim)
        let square_embed = Param::from_tensor(Tensor::random(
            [64, embed_dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        ));

        // Spatial conv: 3x3 neighbor gather + linear projection (channels-last, no permute)
        let spatial_conv = if config.conv_layers() > 0 {
            Some(SpatialConv::new(device, FEATURES_PER_TOKEN))
        } else {
            None
        };

        let smolgen_weight_gen = SmolgenWeightGen::new(config, device);

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers() {
            blocks.push(TransformerBlock::new(config, device));
        }

        let token_norm = RmsNormConfig::new(embed_dim).init(device);

        let norm = RmsNormConfig::new(config.embed_dim()).init(device);

        let policy_head = FactorizedPolicyHead::new(config, device);

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

        let policy_block = TransformerBlock::new_for_head(config, device);
        let value_block = TransformerBlock::new_for_head(config, device);
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
        let aux_trunk_from_square_hidden =
            LinearConfig::new(config.embed_dim(), config.embed_dim())
                .with_initializer(std_init.clone())
                .init(device);
        let aux_trunk_to_square_head = LinearConfig::new(config.embed_dim(), 1)
            .with_initializer(std_init.clone())
            .init(device);
        let aux_trunk_to_square_hidden = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);
        let cp_loss_head_hidden =
            LinearConfig::new(config.embed_dim() + NUM_GLOBALS, config.embed_dim())
                .with_initializer(std_init.clone())
                .init(device);
        let cp_loss_head = LinearConfig::new(config.embed_dim(), RegretBin::COUNT)
            .with_initializer(std_init.clone())
            .init(device);
        let retrieval_head = LinearConfig::new(config.embed_dim(), config.embed_dim())
            .with_initializer(std_init.clone())
            .init(device);

        Self {
            embed_proj1,
            embed_proj2,
            square_embed,
            spatial_conv,
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
            policy_block,
            value_block,
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
            cp_loss_head_hidden,
            cp_loss_head,
            retrieval_head,
        }
    }

    fn retrieval_embedding_from_pooled(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        let projected = self.retrieval_head.forward(pooled);
        let norm = projected
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-6);
        projected / norm
    }

    fn normalized_mean_pooled_trunk(&self, trunk: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = trunk.dims()[0];
        let embed_dim = trunk.dims()[2];
        let pooled = trunk.mean_dim(1).reshape([batch_size, embed_dim]);
        let norm = pooled
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-6);
        pooled / norm
    }

    fn retrieval_embedding_from_trunk(&self, trunk: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = trunk.dims()[0];
        let embed_dim = trunk.dims()[2];
        let pooled = trunk.mean_dim(1).reshape([batch_size, embed_dim]);
        self.retrieval_embedding_from_pooled(pooled)
    }

    #[cfg(feature = "train")]
    fn compute_same_move_retrieval_loss_from_embeddings(
        &self,
        embeddings: Tensor<B, 2>,
        items: &[crate::dataset::ChessItem],
    ) -> (Tensor<B, 1>, f32, f32, f32, f32, f32)
    where
        B::FloatElem: From<f32>,
    {
        let batch_size = embeddings.dims()[0];
        let device = embeddings.device();
        let zero = || Tensor::<B, 1>::zeros([1], &device);
        if batch_size < 2 || items.len() != batch_size {
            return (zero(), 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let target_indices = items
            .iter()
            .map(|item| item.move_distribution.iter().position(|p| *p > 0.0))
            .collect::<Vec<_>>();

        let mut positive_mask_data = vec![0.0f32; batch_size * batch_size];
        let mut negative_mask_data = vec![0.0f32; batch_size * batch_size];
        let mut positive_count = 0.0f32;
        let mut negative_count = 0.0f32;

        for i in 0..batch_size {
            let Some(query_move) = target_indices[i] else {
                continue;
            };
            for j in 0..batch_size {
                if i == j {
                    continue;
                }
                let Some(neighbor_move) = target_indices[j] else {
                    continue;
                };
                let query_legal_at_neighbor =
                    items[j].legal_moves.get(query_move).copied().unwrap_or(0.0) > 0.0;
                let neighbor_legal_at_query = items[i]
                    .legal_moves
                    .get(neighbor_move)
                    .copied()
                    .unwrap_or(0.0)
                    > 0.0;
                if !query_legal_at_neighbor || !neighbor_legal_at_query {
                    continue;
                }

                let offset = i * batch_size + j;
                if query_move == neighbor_move {
                    positive_mask_data[offset] = 1.0;
                    positive_count += 1.0;
                } else {
                    negative_mask_data[offset] = 1.0;
                    negative_count += 1.0;
                }
            }
        }

        let pair_count = positive_count + negative_count;
        if pair_count <= 0.0 {
            return (zero(), 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let positive_mask =
            Tensor::<B, 1>::from_data(TensorData::from(positive_mask_data.as_slice()), &device)
                .reshape([batch_size, batch_size]);
        let negative_mask =
            Tensor::<B, 1>::from_data(TensorData::from(negative_mask_data.as_slice()), &device)
                .reshape([batch_size, batch_size]);

        let sim = embeddings.clone().matmul(embeddings.transpose());
        let config = get_global_config();
        let logits = (sim.clone() - config.retrieval_margin()) * config.retrieval_logit_scale();

        // BCEWithLogits for labels in {0,1}. Logits are bounded by cosine scale,
        // so the direct softplus form is stable enough and lets us mask pairs.
        let bce = (logits.clone().exp() + 1.0).log() - logits * positive_mask.clone();

        let positive_loss = (bce.clone() * positive_mask.clone()).sum() / positive_count.max(1.0);
        let negative_loss = (bce * negative_mask.clone()).sum() / negative_count.max(1.0);
        let positive_active = if positive_count > 0.0 { 1.0 } else { 0.0 };
        let negative_active = if negative_count > 0.0 { 1.0 } else { 0.0 };
        let active_terms: f32 = positive_active + negative_active;
        let loss = (positive_loss * positive_active + negative_loss * negative_active)
            / active_terms.max(1.0);

        let positive_sim = if positive_count > 0.0 {
            ((sim.clone() * positive_mask).sum() / positive_count)
                .into_scalar()
                .elem::<f32>()
        } else {
            0.0
        };
        let negative_sim = if negative_count > 0.0 {
            ((sim * negative_mask).sum() / negative_count)
                .into_scalar()
                .elem::<f32>()
        } else {
            0.0
        };
        let loss_f32 = loss.clone().into_scalar().elem::<f32>();

        (
            loss,
            loss_f32,
            pair_count,
            positive_count,
            positive_sim,
            negative_sim,
        )
    }

    #[cfg(feature = "train")]
    fn off_diagonal_mask(&self, batch_size: usize, device: &B::Device) -> (Tensor<B, 2>, f32) {
        let mut mask_data = vec![1.0f32; batch_size * batch_size];
        for i in 0..batch_size {
            mask_data[i * batch_size + i] = 0.0;
        }
        let mask = Tensor::<B, 1>::from_data(TensorData::from(mask_data.as_slice()), device)
            .reshape([batch_size, batch_size]);
        (mask, (batch_size * batch_size - batch_size) as f32)
    }

    #[cfg(feature = "train")]
    fn compute_policy_overlap_retrieval_loss_from_embeddings(
        &self,
        embeddings: Tensor<B, 2>,
        policy_probs: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, f32, f32, f32, f32, f32)
    where
        B::FloatElem: From<f32>,
    {
        let batch_size = embeddings.dims()[0];
        let device = embeddings.device();
        let zero = || Tensor::<B, 1>::zeros([1], &device);
        if batch_size < 2 || policy_probs.dims()[0] != batch_size {
            return (zero(), 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let (pair_mask, pair_count) = self.off_diagonal_mask(batch_size, &device);
        let probs_i = policy_probs.clone().detach().unsqueeze_dim::<3>(1);
        let probs_j = policy_probs.detach().unsqueeze_dim::<3>(0);
        let target = probs_i
            .min_pair(probs_j)
            .sum_dim(2)
            .reshape([batch_size, batch_size])
            .detach();

        let sim = embeddings.clone().matmul(embeddings.transpose());
        let diff = sim.clone() - target.clone();
        let loss = (diff.powf_scalar(2.0) * pair_mask.clone()).sum() / pair_count;

        let config = get_global_config();
        let positive_mask = target
            .greater_elem(config.retrieval_policy_positive_threshold())
            .float()
            * pair_mask.clone();
        let positive_count = positive_mask.clone().sum().into_scalar().elem::<f32>();
        let negative_mask = pair_mask - positive_mask.clone();
        let negative_count = (pair_count - positive_count).max(0.0);

        let positive_sim = if positive_count > 0.0 {
            ((sim.clone() * positive_mask).sum() / positive_count)
                .into_scalar()
                .elem::<f32>()
        } else {
            0.0
        };
        let negative_sim = if negative_count > 0.0 {
            ((sim * negative_mask).sum() / negative_count)
                .into_scalar()
                .elem::<f32>()
        } else {
            0.0
        };
        let loss_f32 = loss.clone().into_scalar().elem::<f32>();

        (
            loss,
            loss_f32,
            pair_count,
            positive_count,
            positive_sim,
            negative_sim,
        )
    }

    #[cfg(feature = "train")]
    fn compute_retrieval_loss_from_embeddings(
        &self,
        embeddings: Tensor<B, 2>,
        items: &[crate::dataset::ChessItem],
        policy_probs: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 1>, f32, f32, f32, f32, f32)
    where
        B::FloatElem: From<f32>,
    {
        let config = get_global_config();
        if config.retrieval_objective() == RETRIEVAL_OBJECTIVE_POLICY_OVERLAP {
            if let Some(policy_probs) = policy_probs {
                return self.compute_policy_overlap_retrieval_loss_from_embeddings(
                    embeddings,
                    policy_probs,
                );
            }
        }

        self.compute_same_move_retrieval_loss_from_embeddings(embeddings, items)
    }

    #[cfg(feature = "train")]
    fn compute_retrieval_loss(
        &self,
        trunk_output: Tensor<B, 3>,
        items: &[crate::dataset::ChessItem],
        policy_probs: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 1>, f32, f32, f32, f32, f32)
    where
        B::FloatElem: From<f32>,
    {
        let retrieval_z = self.retrieval_embedding_from_trunk(trunk_output);
        self.compute_retrieval_loss_from_embeddings(retrieval_z, items, policy_probs)
    }

    #[cfg(feature = "train")]
    fn compute_trunk_retrieval_metrics(
        &self,
        trunk_output: Tensor<B, 3>,
        items: &[crate::dataset::ChessItem],
        policy_probs: Option<Tensor<B, 2>>,
    ) -> (f32, f32, f32, f32, f32)
    where
        B::FloatElem: From<f32>,
    {
        let trunk_z = self.normalized_mean_pooled_trunk(trunk_output.detach());
        let (_, loss_f32, pair_count, positive_count, positive_sim, negative_sim) =
            self.compute_retrieval_loss_from_embeddings(trunk_z, items, policy_probs);
        (
            loss_f32,
            pair_count,
            positive_count,
            positive_sim,
            negative_sim,
        )
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

    /// Inference-only forward pass that returns the same four outputs as `forward`,
    /// AND a per-layer vector of post-softmax attention weights, AND the post-
    /// norm trunk tensor (the per-square embeddings fed into the policy/value
    /// heads). Shape of the trunk is `[batch, 64, embed_dim]`.
    ///
    /// Used by analysis/bot paths that need attention plus the retrieval
    /// position embedding.
    pub fn forward_with_attention_and_trunk(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Vec<Tensor<B, 4>>,
        Tensor<B, 3>,
        Tensor<B, 2>,
    ) {
        let (policy, value, side_info, time_usage, attn, trunk, embedding) =
            self.forward_with_attention_inner(board, globals, true);
        (
            policy,
            value,
            side_info,
            time_usage,
            attn,
            trunk.expect("trunk must be present when capture_trunk=true"),
            embedding.expect("embedding must be present when capture_trunk=true"),
        )
    }

    /// Inference-only forward pass that returns the same four outputs as `forward`
    /// AND a per-layer vector of post-softmax attention weights for the main
    /// encoder blocks.
    ///
    /// The returned `Vec<Tensor<B, 4>>` has length equal to `self.blocks.len()`
    /// (num_layers), ordered from layer 0 to layer N-1. Each tensor has shape
    /// `[batch, num_heads, 64, 64]`.
    ///
    /// This duplicates `forward_with_trunk`'s path (minus the returned trunk /
    /// policy_tokens) and substitutes `TransformerBlock::forward_with_attn` for
    /// `forward_with_film` in the encoder loop. The training path
    /// (`forward`/`forward_with_trunk`) is untouched.
    pub fn forward_with_attention(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Vec<Tensor<B, 4>>,
    ) {
        let (policy, value, side_info, time_usage, attn, _trunk, _embedding) =
            self.forward_with_attention_inner(board, globals, false);
        (policy, value, side_info, time_usage, attn)
    }

    fn forward_with_attention_inner(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
        capture_trunk: bool,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Vec<Tensor<B, 4>>,
        Option<Tensor<B, 3>>,
        Option<Tensor<B, 2>>,
    ) {
        start_forward_pass();
        let device = board.device();
        let total_timing = TimingScope::new_with_sync::<B>("forward_total", &device);
        let _norm_scope = NormDebugScope::start("OXIModel::forward_with_attention");
        let _root_stream = StreamScope::enter("root");

        log_tensor_stats("input.board_raw", &board);
        log_tensor_stats("input.globals_raw", &globals);

        let board_features = board.clone();
        log_tensor_stats("input.board_features", &board_features);

        // Two-layer nonlinear embedding: all 65 features → SiLU → embed_dim
        let token_embeds = {
            let _t = TimingScope::new_with_sync::<B>("token_embed", &device);
            let hidden = silu(self.embed_proj1.forward(board_features));
            let embeds = self.embed_proj2.forward(hidden);
            // Add learnable per-square positional embeddings
            embeds + self.square_embed.val().unsqueeze_dim::<3>(0)
        };
        log_tensor_stats("embed.token_embeds", &token_embeds);
        debug_assert_eq!(
            token_embeds.dims()[1],
            64,
            "Sequence length must be 64 for 8x8 board"
        );

        let mut x = self.token_norm.forward(token_embeds);
        log_tensor_stats("encoder.input_tokens", &x);

        // Multi-scale trunk: average outputs from the last half of layers
        let num_layers = self.blocks.len();
        let avg_start = num_layers / 2;
        let mut layer_sum: Option<Tensor<B, 3>> = None;
        let mut avg_count = 0;

        // Collect per-layer attention weights in order.
        let mut attention_maps: Vec<Tensor<B, 4>> = Vec::with_capacity(num_layers);

        {
            let _encoder_stream = StreamScope::enter("encoder");
            let _encoder_timing = TimingScope::new_with_sync::<B>("encoder_blocks", &device);
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                let _layer_scope = LayerScope::enter(layer_idx);
                let _block_timing = TimingScope::new_with_sync::<B>("encoder_block", &device);
                log_tensor_stats("encoder.pre_block", &x);

                let (new_x, attn) =
                    block.forward_with_attn(x, &self.smolgen_weight_gen, globals.clone());
                x = new_x;
                attention_maps.push(attn);
                log_tensor_stats("encoder.post_block", &x);

                if layer_idx >= avg_start {
                    layer_sum = Some(match layer_sum {
                        Some(sum) => sum + x.clone(),
                        None => x.clone(),
                    });
                    avg_count += 1;
                }
            }
        }

        if let Some(sum) = layer_sum {
            x = sum / (avg_count as f32);
        }

        x = self.norm.forward(x);
        log_tensor_stats("encoder.post_norm", &x);

        let embedding_out = if capture_trunk {
            let batch_size = x.dims()[0];
            let embed_dim = x.dims()[2];
            let pooled = x.clone().mean_dim(1).reshape([batch_size, embed_dim]);
            Some(self.retrieval_embedding_from_pooled(pooled))
        } else {
            None
        };

        let policy_logits = {
            let _stream = StreamScope::enter("policy");
            let _timing = TimingScope::new_with_sync::<B>("policy_head", &device);
            let tokens = {
                let _t = TimingScope::new_with_sync::<B>("policy_block", &device);
                self.policy_block.forward_with_film(
                    x.clone(),
                    &self.smolgen_weight_gen,
                    globals.clone(),
                )
            };
            log_tensor_stats("policy.tokens", &tokens);
            let logits = {
                let _t = TimingScope::new_with_sync::<B>("factorized_policy", &device);
                self.policy_head.forward(tokens.clone(), globals.clone())
            };
            log_tensor_stats("policy.logits", &logits);
            logits
        };

        let aux_batch_size = board.dims()[0];
        let embed_dim = x.dims()[2];

        // Value head: value_block → attention pooling → hidden → WDL logits
        let value_logits = {
            let _stream = StreamScope::enter("value");
            let _timing = TimingScope::new_with_sync::<B>("value_head", &device);
            let value_tokens = self.value_block.forward_with_film(
                x.clone(),
                &self.smolgen_weight_gen,
                globals.clone(),
            );
            log_tensor_stats("value.tokens", &value_tokens);

            let pool_hidden = silu(self.value_pool_fc1.forward(value_tokens.clone()));
            let pool_logits = self
                .value_pool_fc2
                .forward(pool_hidden)
                .reshape([aux_batch_size, 64]);
            let scale = (embed_dim as f64).sqrt();
            let pool_logits = pool_logits / scale;
            let pool_weights = softmax(pool_logits, 1).reshape([aux_batch_size, 64, 1]);
            let pooled = (value_tokens * pool_weights)
                .sum_dim(1)
                .reshape([aux_batch_size, embed_dim]);
            log_tensor_stats("value.pooled", &pooled);

            let with_globals = Tensor::cat(vec![pooled, globals.clone()], 1);
            let hidden = silu(self.value_head_hidden.forward(with_globals));
            self.value_head.forward(hidden)
        };
        log_tensor_stats("value.logits", &value_logits);

        let trunk_pooled = x.clone().mean_dim(1).reshape([aux_batch_size, embed_dim]);
        let side_info_logits = self.side_info_head.forward(trunk_pooled);
        let time_usage_logits = Tensor::zeros([aux_batch_size, 2], &device);

        let trunk_out = if capture_trunk { Some(x.clone()) } else { None };

        drop(total_timing);
        finish_and_log_forward_pass();
        (
            policy_logits,
            value_logits,
            side_info_logits,
            time_usage_logits,
            attention_maps,
            trunk_out,
            embedding_out,
        )
    }

    fn forward_with_trunk(
        &self,
        board: Tensor<B, 3>,
        globals: Tensor<B, 2, Float>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 3>,
        Tensor<B, 3>,
    ) {
        start_forward_pass();
        let device = board.device();
        let total_timing = TimingScope::new_with_sync::<B>("forward_total", &device);
        let _norm_scope = NormDebugScope::start("OXIModel::forward");
        let _root_stream = StreamScope::enter("root");

        log_tensor_stats("input.board_raw", &board);
        log_tensor_stats("input.globals_raw", &globals);

        let board_features = board.clone();
        log_tensor_stats("input.board_features", &board_features);

        // Two-layer nonlinear embedding: all 65 features → SiLU → embed_dim
        let token_embeds = {
            let _t = TimingScope::new_with_sync::<B>("token_embed", &device);
            let hidden = silu(self.embed_proj1.forward(board_features));
            let embeds = self.embed_proj2.forward(hidden);
            // Add learnable per-square positional embeddings
            embeds + self.square_embed.val().unsqueeze_dim::<3>(0)
        };
        log_tensor_stats("embed.token_embeds", &token_embeds);
        debug_assert_eq!(
            token_embeds.dims()[1],
            64,
            "Sequence length must be 64 for 8x8 board"
        );

        let mut x = self.token_norm.forward(token_embeds);
        log_tensor_stats("encoder.input_tokens", &x);

        // Multi-scale trunk: average outputs from the last half of layers
        // to provide richer multi-level features and gradient shortcuts.
        let num_layers = self.blocks.len();
        let avg_start = num_layers / 2; // Average the second half of layers
        let mut layer_sum: Option<Tensor<B, 3>> = None;
        let mut avg_count = 0;

        {
            let _encoder_stream = StreamScope::enter("encoder");
            let _encoder_timing = TimingScope::new_with_sync::<B>("encoder_blocks", &device);
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                let _layer_scope = LayerScope::enter(layer_idx);
                let _block_timing = TimingScope::new_with_sync::<B>("encoder_block", &device);
                log_tensor_stats("encoder.pre_block", &x);

                x = block.forward_with_film(x, &self.smolgen_weight_gen, globals.clone());
                log_tensor_stats("encoder.post_block", &x);

                // Accumulate outputs from the second half of layers
                if layer_idx >= avg_start {
                    layer_sum = Some(match layer_sum {
                        Some(sum) => sum + x.clone(),
                        None => x.clone(),
                    });
                    avg_count += 1;
                }
            }
        }

        // Use averaged multi-scale representation instead of just final layer
        if let Some(sum) = layer_sum {
            x = sum / (avg_count as f32);
        }

        x = self.norm.forward(x);
        log_tensor_stats("encoder.post_norm", &x);

        let (policy_logits, policy_tokens) = {
            let _stream = StreamScope::enter("policy");
            let _timing = TimingScope::new_with_sync::<B>("policy_head", &device);
            let tokens = {
                let _t = TimingScope::new_with_sync::<B>("policy_block", &device);
                self.policy_block.forward_with_film(
                    x.clone(),
                    &self.smolgen_weight_gen,
                    globals.clone(),
                )
            };
            log_tensor_stats("policy.tokens", &tokens);
            let logits = {
                let _t = TimingScope::new_with_sync::<B>("factorized_policy", &device);
                self.policy_head.forward(tokens.clone(), globals.clone())
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
            let value_tokens = self.value_block.forward_with_film(
                x.clone(),
                &self.smolgen_weight_gen,
                globals.clone(),
            );
            log_tensor_stats("value.tokens", &value_tokens);

            // Attention pooling: fc1 → silu → fc2 → softmax → weighted sum
            let pool_hidden = silu(self.value_pool_fc1.forward(value_tokens.clone()));
            let pool_logits = self
                .value_pool_fc2
                .forward(pool_hidden)
                .reshape([aux_batch_size, 64]);
            // Scale logits by 1/sqrt(embed_dim) to prevent softmax saturation
            // Without scaling, the fc2 dot product over embed_dim dimensions produces
            // logits whose variance grows with embed_dim, causing the softmax to
            // concentrate on a single position and killing gradients through fc1/fc2.
            let scale = (embed_dim as f64).sqrt();
            let pool_logits = pool_logits / scale;
            let pool_weights = softmax(pool_logits, 1).reshape([aux_batch_size, 64, 1]);
            let pooled = (value_tokens * pool_weights)
                .sum_dim(1)
                .reshape([aux_batch_size, embed_dim]);
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

        let (
            policy_logits,
            value_logits,
            _side_info_logits,
            time_usage_logits,
            trunk_output,
            policy_tokens,
        ) = self.forward_with_trunk(batch.board_input, batch.global_features.clone());

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

        // Label smoothing over legal moves only. Disabled by default (eps == 0):
        // uniform-over-legal smoothing puts floor probability on every legal move,
        // including tactical blunders, which directly works against the policy-regret
        // hinge. Keep the knob so this can be re-enabled experimentally.
        let eps = config.policy_label_smoothing;
        let targets_smoothed = if eps > 0.0 {
            let legal_counts = batch
                .legal_moves
                .clone()
                .sum_dim(1)
                .reshape([batch_size, 1])
                .clamp_min(1.0);
            let uniform_over_legal = batch.legal_moves.clone() / legal_counts;
            batch.move_distributions.clone() * (1.0 - eps) + uniform_over_legal * eps
        } else {
            batch.move_distributions.clone()
        };

        // Standard cross-entropy loss per sample
        let ce_loss_per_sample = (targets_smoothed.clone() * log_policy.clone())
            .sum_dim(1)
            .neg();

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
        let value_weights = batch.value_weights.clone().reshape([batch_size, 1]);
        let value_weight_sum = value_weights.clone().sum().clamp_min(1e-8);

        // Helper for creating zero tensors
        let zero_like = || Tensor::zeros([1], &policy_logits_flat_original.device());

        // Only compute value loss if weight is non-zero
        let (base_value_loss, value_term) = if config.value_loss_weight > 0.0 {
            let value_loss_per_sample = ce_per_sample - entropy_bonus * config.value_entropy_weight;
            let value_loss =
                (value_loss_per_sample * value_weights.clone()).sum() / value_weight_sum.clone();
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
        let (
            base_aux_loss,
            aux_mobility_loss_f32,
            aux_material_loss_f32,
            aux_mobility_mae_f32,
            aux_material_mae_f32,
        ) = if config.aux_loss_weight > 0.0 {
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
            let trunk_pooled = trunk_output
                .clone()
                .mean_dim(1)
                .reshape([batch_size, embed_dim]);
            let material_pred = self
                .aux_material_head
                .forward(trunk_pooled)
                .reshape([batch_size]);
            let material_target = batch.material_imbalance.clone();
            let material_diff = material_pred - material_target;
            let material_mse = material_diff.clone().powf_scalar(2.0).mean();
            let material_mae = material_diff.abs().mean();

            // Extract f32 values for metrics (detached, no grad)
            let mob_loss_f32 = mobility_mse
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);
            let mat_loss_f32 = material_mse
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);
            let mob_mae_f32 = mobility_mae
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);
            let mat_mae_f32 = material_mae
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            (
                mobility_mse + material_mse,
                mob_loss_f32,
                mat_loss_f32,
                mob_mae_f32,
                mat_mae_f32,
            )
        } else {
            (zero_like(), 0.0, 0.0, 0.0, 0.0)
        };

        // Maia 2-style auxiliary losses: side info, from-square, to-square
        let (
            maia_loss,
            aux_side_info_loss_f32,
            aux_from_sq_loss_f32,
            aux_to_sq_loss_f32,
            aux_from_sq_acc_f32,
            aux_to_sq_acc_f32,
        ) = if config.aux_loss_weight > 0.0 {
            let embed_dim = trunk_output.dims()[2];

            // Side info: piece moved/captured/check (first 13 values)
            let side_info_target_int = batch.side_info.clone().slice([0..batch_size, 0..13]);
            let trunk_pooled_si = trunk_output
                .clone()
                .mean_dim(1)
                .reshape([batch_size, embed_dim]);
            let side_info_logits = self.side_info_head.forward(trunk_pooled_si);
            let side_info_bce = self
                .side_info_bce
                .forward(side_info_logits, side_info_target_int);

            let si_loss_f32 = side_info_bce
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            // From-square: per-token prediction using policy tokens → [batch, 64]
            let from_sq_hidden = silu(self.aux_from_square_hidden.forward(policy_tokens.clone()));
            let from_sq_logits = self
                .aux_from_square_head
                .forward(from_sq_hidden)
                .reshape([batch_size, 64]);
            let from_sq_target_int = batch.side_info.clone().slice([0..batch_size, 13..77]);

            // Use cross-entropy instead of BCE: from-square is a 64-class classification
            let from_sq_log_probs = log_softmax(from_sq_logits.clone(), 1);
            let from_sq_target_float = from_sq_target_int.clone().float();
            let from_sq_ce = (from_sq_target_float * from_sq_log_probs)
                .sum_dim(1)
                .neg()
                .mean();

            let from_loss_f32 = from_sq_ce
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            // From-square accuracy: argmax match
            let from_pred = from_sq_logits.argmax(1).squeeze_dim::<1>(1);
            let from_true = from_sq_target_int.argmax(1).squeeze_dim::<1>(1);
            let from_correct = from_pred.equal(from_true).float().mean();
            let from_acc_f32 = from_correct
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            // To-square: per-token prediction using policy tokens → [batch, 64]
            let to_sq_hidden = silu(self.aux_to_square_hidden.forward(policy_tokens.clone()));
            let to_sq_logits = self
                .aux_to_square_head
                .forward(to_sq_hidden)
                .reshape([batch_size, 64]);
            let to_sq_target_int = batch.side_info.clone().slice([0..batch_size, 77..141]);

            // Use cross-entropy instead of BCE: to-square is a 64-class classification
            let to_sq_log_probs = log_softmax(to_sq_logits.clone(), 1);
            let to_sq_target_float = to_sq_target_int.clone().float();
            let to_sq_ce = (to_sq_target_float * to_sq_log_probs)
                .sum_dim(1)
                .neg()
                .mean();

            let to_loss_f32 = to_sq_ce
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            // To-square accuracy: argmax match
            let to_pred = to_sq_logits.argmax(1).squeeze_dim::<1>(1);
            let to_true = to_sq_target_int.argmax(1).squeeze_dim::<1>(1);
            let to_correct = to_pred.equal(to_true).float().mean();
            let to_acc_f32 = to_correct
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default()
                .first()
                .copied()
                .unwrap_or(0.0);

            // Trunk-level from/to square prediction: direct aux supervision on trunk tokens
            // This gives the trunk direct gradient signal about which squares matter,
            // rather than only getting it indirectly through the policy_block.
            let trunk_from_sq_target = batch.side_info.clone().slice([0..batch_size, 13..77]);
            let trunk_from_sq_hidden = silu(
                self.aux_trunk_from_square_hidden
                    .forward(trunk_output.clone()),
            );
            let trunk_from_sq_logits = self
                .aux_trunk_from_square_head
                .forward(trunk_from_sq_hidden)
                .reshape([batch_size, 64]);
            let trunk_from_sq_log_probs = log_softmax(trunk_from_sq_logits, 1);
            let trunk_from_sq_target_float = trunk_from_sq_target.float();
            let trunk_from_sq_ce = (trunk_from_sq_target_float * trunk_from_sq_log_probs)
                .sum_dim(1)
                .neg()
                .mean();

            let trunk_to_sq_target = batch.side_info.clone().slice([0..batch_size, 77..141]);
            let trunk_to_sq_hidden = silu(
                self.aux_trunk_to_square_hidden
                    .forward(trunk_output.clone()),
            );
            let trunk_to_sq_logits = self
                .aux_trunk_to_square_head
                .forward(trunk_to_sq_hidden)
                .reshape([batch_size, 64]);
            let trunk_to_sq_log_probs = log_softmax(trunk_to_sq_logits, 1);
            let trunk_to_sq_target_float = trunk_to_sq_target.float();
            let trunk_to_sq_ce = (trunk_to_sq_target_float * trunk_to_sq_log_probs)
                .sum_dim(1)
                .neg()
                .mean();

            (
                side_info_bce + from_sq_ce + to_sq_ce + trunk_from_sq_ce + trunk_to_sq_ce,
                si_loss_f32,
                from_loss_f32,
                to_loss_f32,
                from_acc_f32,
                to_acc_f32,
            )
        } else {
            (zero_like(), 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let base_aux_loss = base_aux_loss + maia_loss;
        let aux_term = base_aux_loss.clone() * config.aux_loss_weight;

        let (
            base_retrieval_loss,
            retrieval_loss_f32,
            retrieval_pair_count_f32,
            retrieval_positive_count_f32,
            retrieval_positive_sim_f32,
            retrieval_negative_sim_f32,
            trunk_retrieval_loss_f32,
            trunk_retrieval_pair_count_f32,
            trunk_retrieval_positive_count_f32,
            trunk_retrieval_positive_sim_f32,
            trunk_retrieval_negative_sim_f32,
        ) = if config.retrieval_loss_weight() > 0.0 {
            let retrieval_policy_probs = if config.retrieval_uses_policy_overlap() {
                let temperature = config.retrieval_policy_temperature();
                let logits = policy_logits_flat.clone() / temperature;
                let probs = log_softmax(logits, 1)
                    .mask_fill(mask.clone(), 0.0)
                    .exp()
                    .mask_fill(mask.clone(), 0.0)
                    .detach();
                Some(probs)
            } else {
                None
            };

            let (
                base_retrieval_loss,
                retrieval_loss_f32,
                retrieval_pair_count_f32,
                retrieval_positive_count_f32,
                retrieval_positive_sim_f32,
                retrieval_negative_sim_f32,
            ) = self.compute_retrieval_loss(
                trunk_output.clone(),
                &batch.items,
                retrieval_policy_probs.clone(),
            );
            let (
                trunk_retrieval_loss_f32,
                trunk_retrieval_pair_count_f32,
                trunk_retrieval_positive_count_f32,
                trunk_retrieval_positive_sim_f32,
                trunk_retrieval_negative_sim_f32,
            ) = self.compute_trunk_retrieval_metrics(
                trunk_output.clone(),
                &batch.items,
                retrieval_policy_probs,
            );

            (
                base_retrieval_loss,
                retrieval_loss_f32,
                retrieval_pair_count_f32,
                retrieval_positive_count_f32,
                retrieval_positive_sim_f32,
                retrieval_negative_sim_f32,
                trunk_retrieval_loss_f32,
                trunk_retrieval_pair_count_f32,
                trunk_retrieval_positive_count_f32,
                trunk_retrieval_positive_sim_f32,
                trunk_retrieval_negative_sim_f32,
            )
        } else {
            (
                zero_like(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        };

        let (
            base_calibration_loss,
            calibration_head_loss_f32,
            calibration_policy_mae_f32,
            calibration_head_mae_f32,
            calibration_labeled_fraction_f32,
            calibration_overall_score_f32,
            calibration_policy_signed_error_by_elo,
            base_policy_regret_loss,
            policy_regret_loss_f32,
            argmax_cp_loss_by_elo,
        ) = if config.calibration_loss_weight() > 0.0 || config.policy_regret_loss_weight() > 0.0 {
            let calibration_mask = batch.calibration_mask.clone();
            let labeled_count = calibration_mask.clone().sum().into_scalar().elem::<f32>();
            if labeled_count > 0.0 {
                let embed_dim = trunk_output.dims()[2];
                let trunk_pooled = trunk_output
                    .clone()
                    .mean_dim(1)
                    .reshape([batch_size, embed_dim]);
                let cp_loss_hidden = silu(self.cp_loss_head_hidden.forward(Tensor::cat(
                    vec![trunk_pooled, batch.global_features.clone()],
                    1,
                )));
                let cp_loss_logits = self.cp_loss_head.forward(cp_loss_hidden);
                let cp_loss_log_probs = log_softmax(cp_loss_logits.clone(), 1);
                let cp_loss_probs = cp_loss_log_probs.clone().exp();

                let head_ce_per_sample = (batch.calibration_target_bins.clone()
                    * cp_loss_log_probs)
                    .sum_dim(1)
                    .neg()
                    .reshape([batch_size]);

                let centers: [f32; RegretBin::COUNT] = [
                    RegretBin::ExactZero.representative_cp(),
                    RegretBin::Cp1To10.representative_cp(),
                    RegretBin::Cp11To25.representative_cp(),
                    RegretBin::Cp26To50.representative_cp(),
                    RegretBin::Cp51To100.representative_cp(),
                    RegretBin::Cp101To200.representative_cp(),
                    RegretBin::Cp201To400.representative_cp(),
                    RegretBin::Cp400Plus.representative_cp(),
                ];
                let center_tensor = Tensor::<B, 1>::from_data(
                    TensorData::from(centers.as_slice()),
                    &policy_logits_flat_original.device(),
                )
                .reshape([1, RegretBin::COUNT]);
                let head_expected_cp = (cp_loss_probs * center_tensor)
                    .sum_dim(1)
                    .reshape([batch_size]);

                let policy_probs = log_policy.clone().exp().mask_fill(mask.clone(), 0.0);
                let policy_expected_cp = (policy_probs.clone()
                    * batch.calibration_move_cp_losses.clone())
                .sum_dim(1)
                .reshape([batch_size]);

                let target_cp = batch.calibration_target_cp_loss.clone();
                let mask_sum = calibration_mask.clone().sum().clamp_min(1.0);
                let head_ce =
                    (head_ce_per_sample * calibration_mask.clone()).sum() / mask_sum.clone();
                let policy_mae = ((policy_expected_cp.clone() - target_cp.clone()).abs()
                    * calibration_mask.clone())
                .sum()
                    / mask_sum.clone();
                // Metric-aligned policy calibration loss: directly optimizes the scoring metric exp(-|error|/15).
                // Unlike MAE (constant gradient), this gives stronger gradient for small errors where the
                // metric is steep, and weaker gradient for large errors where the metric is flat.
                // This prevents the model from wasting gradient budget on irredeemable miscalibrations
                // and focuses optimization where the metric is most sensitive.
                let policy_cp_error = (policy_expected_cp.clone() - target_cp.clone()).abs();
                let policy_cal_metric = (policy_cp_error
                    .neg()
                    .div_scalar(15.0f32)
                    .exp()
                    .neg()
                    .add_scalar(1.0f32)
                    * calibration_mask.clone())
                .sum()
                    / mask_sum.clone();
                let head_mae = ((head_expected_cp - target_cp).abs() * calibration_mask.clone())
                    .sum()
                    / mask_sum.clone();

                let base_loss =
                    head_ce.clone() + policy_cal_metric * 2.0 + head_mae.clone() * 0.005;

                let mut signed_error_sums =
                    std::collections::BTreeMap::<String, (f32, usize)>::new();
                let policy_expected_cp_values = policy_expected_cp
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                let target_cp_values = batch
                    .calibration_target_cp_loss
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                let calibration_mask_values = batch
                    .calibration_mask
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                for (idx, item) in batch.items.iter().enumerate() {
                    if calibration_mask_values.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                        continue;
                    }
                    let signed_error = policy_expected_cp_values.get(idx).copied().unwrap_or(0.0)
                        - target_cp_values.get(idx).copied().unwrap_or(0.0);
                    let bucket = calibration_skill_band_label(item.elo_self).to_string();
                    let entry = signed_error_sums.entry(bucket).or_insert((0.0, 0));
                    entry.0 += signed_error;
                    entry.1 += 1;
                }
                let calibration_policy_signed_error_by_elo = signed_error_sums
                    .into_iter()
                    .filter_map(|(bucket, (sum, count))| {
                        (count > 0).then_some((bucket, sum / count as f32))
                    })
                    .collect::<Vec<_>>();

                // --- Per-move policy-regret hinge ---
                // For each position, penalize probability mass the policy places on moves
                // that are WORSE than the move the human actually played at this position.
                // Per-move form: sum_m policy(m) * max(0, cp_loss(m) - cp_loss(human)).
                // The per-position target cp_loss(human) already encodes player skill
                // (high-Elo humans have low target_cp → tight threshold → more moves fire
                // the hinge; low-Elo humans have loose threshold), so no extra Elo scale.
                let target_cp_broadcast = batch
                    .calibration_target_cp_loss
                    .clone()
                    .reshape([batch_size, 1]);
                let per_move_excess =
                    (batch.calibration_move_cp_losses.clone() - target_cp_broadcast).clamp_min(0.0);
                let regret_per_sample = (policy_probs.clone() * per_move_excess)
                    .sum_dim(1)
                    .reshape([batch_size]);
                let base_policy_regret_loss_tensor = (regret_per_sample * calibration_mask.clone())
                    .sum()
                    .reshape([1])
                    / mask_sum.clone();
                let policy_regret_loss_f32 = base_policy_regret_loss_tensor
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .unwrap_or(0.0);

                // --- Argmax predicted move cp loss, bucketed by Elo band ---
                // For each position, look up the Stockfish cp loss of the model's
                // argmax predicted move. Average by Elo skill band.
                let argmax_indices = policy_logits_flat
                    .clone()
                    .argmax(1)
                    .squeeze_dim::<1>(1)
                    .into_data()
                    .convert::<i32>()
                    .to_vec::<i32>()
                    .unwrap_or_default();
                let move_cp_losses_values = batch
                    .calibration_move_cp_losses
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                let mut argmax_cp_sums = std::collections::BTreeMap::<String, (f32, usize)>::new();
                for (idx, item) in batch.items.iter().enumerate() {
                    if calibration_mask_values.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                        continue;
                    }
                    let move_idx = argmax_indices.get(idx).copied().unwrap_or(0) as usize;
                    if move_idx >= LEGAL_MOVES {
                        continue;
                    }
                    let cp_loss = move_cp_losses_values
                        .get(idx * LEGAL_MOVES + move_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let bucket = calibration_skill_band_label(item.elo_self).to_string();
                    let entry = argmax_cp_sums.entry(bucket).or_insert((0.0, 0));
                    entry.0 += cp_loss;
                    entry.1 += 1;
                }
                let argmax_cp_loss_by_elo = argmax_cp_sums
                    .into_iter()
                    .filter_map(|(bucket, (sum, count))| {
                        (count > 0).then_some((bucket, sum / count as f32))
                    })
                    .collect::<Vec<_>>();

                let calibration_overall_score = {
                    let mut total_score = 0.0f32;
                    let mut total_count = 0usize;
                    for idx in 0..policy_expected_cp_values.len() {
                        if calibration_mask_values.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                            continue;
                        }
                        let abs_error = (policy_expected_cp_values[idx]
                            - target_cp_values.get(idx).copied().unwrap_or(0.0))
                        .abs();
                        let bounded = (-abs_error / 15.0f32).exp();
                        total_score += bounded;
                        total_count += 1;
                    }
                    if total_count > 0 {
                        total_score / total_count as f32
                    } else {
                        0.0
                    }
                };

                (
                    base_loss,
                    head_ce
                        .clone()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0),
                    policy_mae
                        .clone()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0),
                    head_mae
                        .clone()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0),
                    labeled_count / batch_size as f32,
                    calibration_overall_score,
                    calibration_policy_signed_error_by_elo,
                    base_policy_regret_loss_tensor,
                    policy_regret_loss_f32,
                    argmax_cp_loss_by_elo,
                )
            } else {
                (
                    zero_like(),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Vec::new(),
                    zero_like(),
                    0.0,
                    Vec::new(),
                )
            }
        } else {
            (
                zero_like(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Vec::new(),
                zero_like(),
                0.0,
                Vec::new(),
            )
        };
        let calibration_term = base_calibration_loss.clone() * config.calibration_loss_weight();
        let policy_regret_term =
            base_policy_regret_loss.clone() * config.policy_regret_loss_weight();
        let retrieval_term = base_retrieval_loss.clone() * config.retrieval_loss_weight();

        let loss = config_weighted_policy_loss.clone()
            + value_term.clone()
            + time_usage_term.clone()
            + aux_term.clone()
            + calibration_term.clone()
            + policy_regret_term.clone()
            + retrieval_term.clone();

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
            calibration_term.clone(),
            retrieval_term.clone(),
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
        .with_base_calibration_loss(base_calibration_loss)
        .with_calibration_metrics(
            calibration_head_loss_f32,
            calibration_policy_mae_f32,
            calibration_head_mae_f32,
            calibration_labeled_fraction_f32,
            calibration_overall_score_f32,
            calibration_policy_signed_error_by_elo,
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
        .with_policy_regret(
            base_policy_regret_loss,
            policy_regret_term,
            policy_regret_loss_f32,
            argmax_cp_loss_by_elo,
        )
        .with_retrieval_metrics(
            base_retrieval_loss,
            retrieval_term,
            retrieval_loss_f32,
            retrieval_pair_count_f32,
            retrieval_positive_count_f32,
            retrieval_positive_sim_f32,
            retrieval_negative_sim_f32,
        )
        .with_trunk_retrieval_metrics(
            trunk_retrieval_loss_f32,
            trunk_retrieval_pair_count_f32,
            trunk_retrieval_positive_count_f32,
            trunk_retrieval_positive_sim_f32,
            trunk_retrieval_negative_sim_f32,
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
