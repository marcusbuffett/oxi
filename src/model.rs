use burn::module::Param;
use burn::nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig};
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::activation::log_softmax;
use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};

use crate::chess_output::ChessOutput;
use crate::config::ModelConfig;

/// Basic residual block for the ChessResNet encoder
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    se: Option<SqueezeExcitation<B>>,
}

impl<B: Backend> BasicBlock<B> {
    pub fn new(device: &Device<B>, channels: usize, se_channels: Option<usize>) -> Self {
        let conv1 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);

        let conv2 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(channels).init(device);

        let se = se_channels.map(|se_ch| SqueezeExcitation::new(device, channels, se_ch));

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            se,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);

        let x = if let Some(se) = &self.se {
            se.forward(x)
        } else {
            x
        };

        relu(x + residual)
    }
}

/// Squeeze-and-Excitation module
#[derive(Module, Debug)]
pub struct SqueezeExcitation<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> SqueezeExcitation<B> {
    pub fn new(device: &Device<B>, channels: usize, se_channels: usize) -> Self {
        let fc1 = LinearConfig::new(channels, se_channels).init(device);
        let fc2 = LinearConfig::new(se_channels, channels).init(device);

        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();

        // Global average pooling - flatten spatial dimensions and take mean
        let scale = x
            .clone()
            .reshape([batch, channels, height * width])
            .mean_dim(2)
            .squeeze::<2>(2);

        // FC layers with activation
        let scale = self.fc1.forward(scale);
        let scale = relu(scale);
        let scale = self.fc2.forward(scale);
        let scale = sigmoid(scale);

        // Reshape and multiply
        let scale = scale.reshape([batch, channels, 1, 1]);
        x * scale
    }
}

/// ChessResNet encoder - CNN backbone for position encoding
#[derive(Module, Debug)]
pub struct ChessResNet<B: Backend> {
    input_conv: Conv2d<B>,
    input_bn: BatchNorm<B, 2>,
    blocks: Vec<BasicBlock<B>>,
    policy_conv: Conv2d<B>,
    policy_bn: BatchNorm<B, 2>,
}

impl<B: Backend> ChessResNet<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        // Initial convolution
        let input_conv = Conv2dConfig::new([config.num_board_channels(), config.channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let input_bn = BatchNormConfig::new(config.channels).init(device);

        // Residual blocks
        let blocks = (0..config.num_blocks)
            .map(|_| BasicBlock::new(device, config.channels, Some(config.se_channels())))
            .collect();

        // Policy head convolution
        let policy_conv =
            Conv2dConfig::new([config.channels, config.policy_channels()], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(config.policy_channels()).init(device);

        Self {
            input_conv,
            input_bn,
            blocks,
            policy_conv,
            policy_bn,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Initial convolution
        let x = self.input_conv.forward(x);
        let x = self.input_bn.forward(x);
        let x = relu(x);

        // Residual blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Policy features
        let policy_features = self.policy_conv.forward(x.clone());
        let policy_features = self.policy_bn.forward(policy_features);
        let policy_features = relu(policy_features);

        (x, policy_features)
    }
}

/// Elo-aware attention mechanism
#[derive(Module, Debug)]
pub struct EloAwareAttention<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    elo_embeddings: Embedding<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    elo_proj: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> EloAwareAttention<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let head_dim = config.embed_dim() / config.num_heads();

        let elo_embeddings = EmbeddingConfig::new(
            config.elo_bins().len() * 2 + 2, // bins for each player + unknown
            config.elo_embed_dim(),
        )
        .init(device);

        let q_proj = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let k_proj = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let v_proj = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let elo_proj =
            LinearConfig::new(config.elo_embed_dim() * 2, config.embed_dim()).init(device);
        let out_proj = LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);

        let dropout = DropoutConfig::new(config.attention_dropout()).init();

        Self {
            num_heads: config.num_heads(),
            head_dim,
            elo_embeddings,
            q_proj,
            k_proj,
            v_proj,
            elo_proj,
            out_proj,
            dropout,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        white_elo_idx: Tensor<B, 2, Int>,
        black_elo_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, embed_dim] = x.dims();

        // Get Elo embeddings
        let white_elo_emb = self.elo_embeddings.forward(white_elo_idx);
        let black_elo_emb = self.elo_embeddings.forward(black_elo_idx);
        let elo_emb = Tensor::cat(vec![white_elo_emb, black_elo_emb], 2);
        let elo_emb: Tensor<B, 2> = elo_emb.squeeze(1);
        let elo_emb = self.elo_proj.forward(elo_emb);

        // Expand Elo embeddings to sequence length
        let elo_emb = elo_emb
            .reshape([batch, 1, embed_dim])
            .repeat_dim(1, seq_len);

        // Project queries, keys, values
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone() + elo_emb.clone());
        let v = self.v_proj.forward(x);

        // Reshape for multi-head attention
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let k_dims = k.dims();
        let scores = q.matmul(k.swap_dims(k_dims.len() - 2, k_dims.len() - 1)) / scale;
        let attn_weights = softmax(scores.clone(), scores.dims().len() - 1);
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply attention to values
        let attn_output = attn_weights.matmul(v);

        // Reshape back
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, embed_dim]);

        // Output projection
        self.out_proj.forward(attn_output)
    }
}

/// Transformer block with Elo-aware attention
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: EloAwareAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    mlp: MLP<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let attention = EloAwareAttention::new(device, config);
        let norm1 = LayerNormConfig::new(config.embed_dim()).init(device);
        let norm2 = LayerNormConfig::new(config.embed_dim()).init(device);
        let mlp = MLP::new(device, config);
        let dropout = DropoutConfig::new(config.dropout()).init();

        Self {
            attention,
            norm1,
            norm2,
            mlp,
            dropout,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        white_elo_idx: Tensor<B, 2, Int>,
        black_elo_idx: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        // Self-attention with residual
        let attn_out =
            self.attention
                .forward(self.norm1.forward(x.clone()), white_elo_idx, black_elo_idx);
        let x = x.clone() + self.dropout.forward(attn_out);

        // MLP with residual
        let mlp_out = self.mlp.forward(self.norm2.forward(x.clone()));
        x + self.dropout.forward(mlp_out)
    }
}

/// MLP module for transformer blocks
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MLP<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let hidden_dim = (config.embed_dim() as f32 * config.mlp_ratio()) as usize;

        let fc1 = LinearConfig::new(config.embed_dim(), hidden_dim).init(device);
        let fc2 = LinearConfig::new(hidden_dim, config.embed_dim()).init(device);
        let dropout = DropoutConfig::new(config.dropout()).init();

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = gelu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

/// Complete OXI model
#[derive(Module, Debug)]
pub struct OXIModel<B: Backend> {
    encoder: ChessResNet<B>,
    patch_embed: Linear<B>,
    pos_embed: Param<Tensor<B, 3>>,
    transformer_blocks: Vec<TransformerBlock<B>>,
    norm: LayerNorm<B>,

    // Output heads
    policy_head: Linear<B>,
    side_info_head: Linear<B>,
    side_info_bce: BinaryCrossEntropyLoss<B>,

    // Head-specific projections
    policy_projection: Linear<B>,
    value_projection: Linear<B>,
    side_info_projection: Linear<B>,

    value_hidden: Linear<B>,
    value_head: Linear<B>,
    value_dropout: Dropout,

    // Uncertainty parameters for loss weighting
    policy_uncertainty: Param<Tensor<B, 1>>,
    value_uncertainty: Param<Tensor<B, 1>>,
    side_info_uncertainty: Param<Tensor<B, 1>>,
}

impl<B: Backend> OXIModel<B> {
    pub fn new(device: &Device<B>, config: &ModelConfig) -> Self {
        let encoder = ChessResNet::new(device, config);

        // Patch embedding
        let patch_embed = LinearConfig::new(config.channels, config.embed_dim()).init(device);

        // Position embeddings
        let pos_embed = Param::from_tensor(Tensor::zeros(
            [1, config.max_seq_len(), config.embed_dim()],
            device,
        ));

        // Transformer blocks
        let transformer_blocks = (0..config.num_layers())
            .map(|_| TransformerBlock::new(device, config))
            .collect();

        let norm = LayerNormConfig::new(config.embed_dim()).init(device);

        // Output heads
        let policy_head = LinearConfig::new(
            config.policy_channels() * 64 + config.embed_dim(),
            config.num_moves(),
        )
        .init(device);

        let value_hidden = LinearConfig::new(config.embed_dim(), 128).init(device);
        let value_head = LinearConfig::new(128, 3).init(device);
        let value_dropout = DropoutConfig::new(config.dropout()).init();

        let side_info_head = LinearConfig::new(config.embed_dim(), 13).init(device);
        // Create config with logits=true
        let bce_config = BinaryCrossEntropyLossConfig::new().with_logits(true);

        // Init the loss module
        let side_info_bce: BinaryCrossEntropyLoss<B> = bce_config.init(device);

        // Head-specific projections to decouple representations
        let policy_projection =
            LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let value_projection =
            LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);
        let side_info_projection =
            LinearConfig::new(config.embed_dim(), config.embed_dim()).init(device);

        // Initialize uncertainty parameters (log(sigma) initialized to 0, so sigma = 1)
        let policy_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let value_uncertainty = Param::from_tensor(Tensor::zeros([1], device));
        let side_info_uncertainty = Param::from_tensor(Tensor::zeros([1], device));

        Self {
            encoder,
            patch_embed,
            pos_embed,
            transformer_blocks,
            norm,
            policy_projection,
            value_projection,
            side_info_projection,
            policy_head,
            value_head,
            value_hidden,
            value_dropout,
            side_info_head,
            side_info_bce,
            policy_uncertainty,
            value_uncertainty,
            side_info_uncertainty,
        }
    }

    pub fn forward(
        &self,
        board: Tensor<B, 4>,
        white_elo_idx: Tensor<B, 2, Int>,
        black_elo_idx: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _, height, width] = board.dims();

        // Encode board with CNN
        let (features, policy_features) = self.encoder.forward(board);

        // Flatten spatial dimensions for transformer
        let features = features.swap_dims(1, 2).swap_dims(2, 3);
        let feature_dim = features.dims()[3];
        let features = features.reshape([batch, height * width, feature_dim]);

        // Project to embedding dimension
        let x = self.patch_embed.forward(features);

        // Add position embeddings
        let seq_len = x.dims()[1];
        let embed_dim = x.dims()[2];
        let x = x + self.pos_embed.val().slice([0..1, 0..seq_len, 0..embed_dim]);

        // Apply transformer blocks
        let mut x = x;
        for block in &self.transformer_blocks {
            x = block.forward(x, white_elo_idx.clone(), black_elo_idx.clone());
        }

        x = self.norm.forward(x);

        // Global average pooling for value and side info heads
        let global_features = x.mean_dim(1).squeeze(1);

        // Apply head-specific projections to decouple representations
        let value_features = self.value_projection.forward(global_features.clone());
        let side_info_features = self.side_info_projection.forward(global_features.clone());
        let policy_global_features = self.policy_projection.forward(global_features);

        // Value prediction
        let value_hidden = self.value_hidden.forward(value_features);
        let value_hidden = activation::relu(value_hidden);
        let value_hidden = self.value_dropout.forward(value_hidden);
        let value = self.value_head.forward(value_hidden);

        // Side info prediction (castling rights, etc.)
        let side_info = self.side_info_head.forward(side_info_features);

        // Policy prediction
        let policy_channels = policy_features.dims()[1];
        let policy_features = policy_features.reshape([batch, policy_channels * 64]);
        let policy_input = Tensor::cat(vec![policy_features, policy_global_features], 1);
        let policy = self.policy_head.forward(policy_input);

        (policy, value, side_info)
    }

    /// Training forward pass with grouped data and KL divergence loss
    pub fn forward_classification(&self, batch: crate::dataset::ChessBatch<B>) -> ChessOutput<B>
    where
        B::FloatElem: From<f32>,
    {
        let batch_size = batch.board_input.shape().dims[0];
        tracing::info!("forward_classification: batch_size={}", batch_size);

        let (policy_logits, value_logits, side_info_logits) =
            self.forward(batch.board_input, batch.elo_self, batch.elo_oppo);

        // Log shapes
        tracing::info!(
            "forward_classification: policy_logits shape={:?}, value_logits shape={:?}, side_info_logits shape={:?}",
            policy_logits.shape().dims,
            value_logits.shape().dims,
            side_info_logits.shape().dims
        );

        // Log mask statistics
        let legal_moves_data = batch.legal_moves.clone().to_data();
        let legal_moves_slice = legal_moves_data.as_slice::<f32>().unwrap();
        let total_legal_moves: f32 = legal_moves_slice.iter().sum();
        let avg_legal_moves = total_legal_moves / batch_size as f32;
        tracing::info!(
            "forward_classification: total_legal_moves={}, avg_legal_moves_per_position={:.2}",
            total_legal_moves,
            avg_legal_moves
        );

        // Compute log softmax for numerical stability
        let log_policy = log_softmax(policy_logits.clone(), 1);

        // KL divergence loss: sum(p_target * log(p_target / p_predicted))
        // = sum(p_target * (log(p_target) - log(p_predicted)))
        // Since we have log(p_predicted), we compute: -sum(p_target * log(p_predicted))
        // We ignore the p_target * log(p_target) term as it's constant w.r.t. model parameters

        // Add small epsilon to avoid log(0)
        let epsilon = 1e-10;
        let target_dist = batch.move_distributions.clone().add_scalar(epsilon);

        // Log target distribution statistics
        let target_dist_data = batch.move_distributions.clone().to_data();
        let target_slice = target_dist_data.as_slice::<f32>().unwrap();
        let non_zero_targets = target_slice.iter().filter(|&&x| x > 0.0).count();
        let max_target = target_slice.iter().fold(0.0f32, |a, &b| a.max(b));
        tracing::info!(
            "forward_classification: non_zero_targets={}, max_target_prob={:.4}, target_dist_shape={:?}",
            non_zero_targets,
            max_target,
            batch.move_distributions.shape().dims
        );

        // Compute KL divergence (without the constant term)
        let policy_loss = (target_dist * log_policy).sum_dim(1).neg().mean();

        // Log policy loss
        let policy_loss_scalar = policy_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        tracing::info!(
            "forward_classification: policy_loss={:.6}",
            policy_loss_scalar
        );

        // Value loss (cross-entropy for win/draw/loss classification)
        let value_log_probs = log_softmax(value_logits.clone(), 1); // Log probabilities for loss

        // Log value target statistics
        let value_targets_data = batch.values.clone().to_data();
        let value_targets_slice = value_targets_data.as_slice::<f32>().unwrap();
        let wins = value_targets_slice
            .iter()
            .step_by(3)
            .filter(|&&x| x > 0.5)
            .count();
        let draws = value_targets_slice
            .iter()
            .skip(1)
            .step_by(3)
            .filter(|&&x| x > 0.5)
            .count();
        let losses = value_targets_slice
            .iter()
            .skip(2)
            .step_by(3)
            .filter(|&&x| x > 0.5)
            .count();
        tracing::info!(
            "forward_classification: value_targets - wins={}, draws={}, losses={}, total={}",
            wins,
            draws,
            losses,
            wins + draws + losses
        );

        // Cross-entropy loss: -sum(target * log(pred))
        let value_loss = (batch.values.clone() * value_log_probs)
            .sum_dim(1)
            .neg()
            .mean();

        // Log value loss details
        let value_loss_scalar = value_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        tracing::info!(
            "forward_classification: value_loss={:.6}",
            value_loss_scalar
        );

        // Side info loss (MSE after sigmoid)
        let side_info_probs = activation::sigmoid(side_info_logits.clone());
        let side_info_probs_clamped = side_info_probs.clamp(1e-7, 1.0 - 1e-7);

        // Log side info shape and statistics
        tracing::info!(
            "forward_classification: side_info_shape={:?}",
            batch.side_info.shape().dims
        );

        // Compute class weights
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

        // Manual BCE
        let targets_float = batch.side_info.clone().float();
        let bce_per_element = targets_float.clone() * side_info_probs_clamped.clone().log()
            + (targets_float.neg() + 1.0) * (side_info_probs_clamped.neg() + 1.0).log();
        let weighted_bce = bce_per_element.neg() * weights;
        let side_info_loss = weighted_bce.mean();

        // Log side info loss
        let side_info_loss_scalar = side_info_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        tracing::info!("forward_classification: side_info_loss={:.6}, mean_target={:.4}, pos_weight={:.4}, neg_weight={:.4}", 
            side_info_loss_scalar, mean_target, pos_weight, neg_weight);

        // Apply config weights first
        let config = crate::config::TrainingConfig::default();
        let config_weighted_policy_loss = policy_loss.clone() * config.policy_loss_weight();
        let config_weighted_value_loss = value_loss.clone() * config.value_loss_weight();
        let config_weighted_side_info_loss =
            side_info_loss.clone() * config.side_info_loss_weight();

        // Uncertainty-based loss weighting on top of config weights
        // We store log(sigma) as the parameter, so sigma = exp(log_sigma)
        // Loss_i = (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)

        // Get the log(sigma) values and compute sigma^2
        let log_sigma_policy = self.policy_uncertainty.val();
        let log_sigma_value = self.value_uncertainty.val();
        let log_sigma_side_info = self.side_info_uncertainty.val();

        // Compute sigma^2 = exp(2 * log_sigma)
        let sigma_sq_policy = (log_sigma_policy.clone() * 2.0).exp();
        let sigma_sq_value = (log_sigma_value.clone() * 2.0).exp();
        let sigma_sq_side_info = (log_sigma_side_info.clone() * 2.0).exp();

        // Apply uncertainty-based weighting to already config-weighted losses
        let uncertainty_weighted_policy_loss = config_weighted_policy_loss.clone()
            / (sigma_sq_policy.clone())
            + log_sigma_policy.clone();
        let uncertainty_weighted_value_loss =
            config_weighted_value_loss.clone() / (sigma_sq_value.clone()) + log_sigma_value.clone();
        let uncertainty_weighted_side_info_loss = config_weighted_side_info_loss.clone()
            / (sigma_sq_side_info.clone())
            + log_sigma_side_info.clone();

        // Total loss is sum of uncertainty-weighted losses
        let lambda = 0.01; // Small hyperparam, tune via experiments (e.g., 0.001 to 0.1)
        let reg_term = lambda
            * (self.policy_uncertainty.val().powf_scalar(2.0)
                + self.value_uncertainty.val().powf_scalar(2.0)
                + self.side_info_uncertainty.val().powf_scalar(2.0))
            .mean();
        let loss: Tensor<B, 1> = uncertainty_weighted_policy_loss.clone()
            + uncertainty_weighted_value_loss.clone()
            + uncertainty_weighted_side_info_loss.clone()
            + reg_term;

        // Log uncertainty values for monitoring
        let sigma_policy = log_sigma_policy.exp().to_data().as_slice::<f32>().unwrap()[0];
        let sigma_value = log_sigma_value.exp().to_data().as_slice::<f32>().unwrap()[0];
        let sigma_side_info = log_sigma_side_info
            .exp()
            .to_data()
            .as_slice::<f32>()
            .unwrap()[0];

        tracing::info!(
            "forward_classification: uncertainties - policy_sigma={:.3}, value_sigma={:.3}, side_info_sigma={:.3}",
            sigma_policy, sigma_value, sigma_side_info
        );

        let final_loss_scalar = loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        tracing::info!(
            "forward_classification: final_loss={:.6} (policy={:.3}, value={:.3}, side_info={:.3})",
            final_loss_scalar,
            policy_loss_scalar,
            value_loss_scalar,
            side_info_loss_scalar
        );

        // For compatibility with ClassificationOutput, we need to provide single targets
        // We'll use the argmax of the distribution
        let targets = batch.move_distributions.clone().argmax(1).squeeze(1);

        // Log targets info for accuracy calculation
        let targets_data = targets.clone().to_data();
        let targets_slice = targets_data.as_slice::<i32>().unwrap();
        let unique_targets = targets_slice
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        tracing::info!(
            "forward_classification: targets for accuracy - shape={:?}, unique_targets={}, sample_targets={:?}",
            targets.shape().dims,
            unique_targets,
            &targets_slice[..5.min(targets_slice.len())]
        );

        // Log output (masked_policy) info
        let output_argmax = policy_logits.clone().argmax(1);
        let output_argmax_data = output_argmax.to_data();
        let output_argmax_slice = output_argmax_data.as_slice::<i32>().unwrap();

        // Calculate and log accuracy for this batch
        let correct = targets_slice
            .iter()
            .zip(output_argmax_slice.iter())
            .filter(|(&t, &p)| t == p)
            .count();
        let batch_accuracy = correct as f32 / batch_size as f32;
        tracing::info!(
            "forward_classification: batch_accuracy={:.4} ({}/{}) - NOTE: This is move prediction accuracy, not value accuracy!",
            batch_accuracy,
            correct,
            batch_size
        );

        ChessOutput::new(
            loss,
            uncertainty_weighted_policy_loss.mean().reshape([1]),
            uncertainty_weighted_value_loss.mean().reshape([1]),
            uncertainty_weighted_side_info_loss.mean().reshape([1]),
            policy_logits,
            targets,
            value_logits,
            batch.values.clone(),
            batch.legal_moves.clone(),
        )
        .with_distributions(batch.move_distributions.clone())
        .with_uncertainties((sigma_policy, sigma_value, sigma_side_info))
        .with_raw_losses(
            config_weighted_policy_loss.mean().reshape([1]),
            config_weighted_value_loss.mean().reshape([1]),
            config_weighted_side_info_loss.mean().reshape([1]),
        )
    }

    /// Get the current uncertainty values (sigma, not log(sigma))
    pub fn get_uncertainties(&self) -> (f32, f32, f32) {
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
        (sigma_policy, sigma_value, sigma_side_info)
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
