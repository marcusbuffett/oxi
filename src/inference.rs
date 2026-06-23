use burn::prelude::*;
use burn::record::{DefaultRecorder, Recorder};
use burn::tensor::activation::log_softmax;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

#[cfg(all(target_os = "linux", feature = "backend-cuda"))]
use burn_cuda;

#[cfg(feature = "backend-candle")]
use burn_candle;

use crate::config::{Config, FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS, PREVIOUS_POSITIONS};
use crate::encoding::encode_position;
use crate::model::OXIModel;
use crate::move_encoding::{decode_move, encode_move};
use crate::moves::{mirror_fen, mirror_move};
use shakmaty::EnPassantMode;
use shakmaty::{fen::Fen, san::SanPlus, Chess, Color, Position, Square};

#[cfg(feature = "backend-ndarray")]
pub type InferenceBackend = burn_ndarray::NdArray<f32>;

#[cfg(all(not(feature = "backend-ndarray"), feature = "backend-tch"))]
pub type InferenceBackend = burn::backend::LibTorch<f32>;

#[cfg(all(not(feature = "backend-ndarray"), feature = "backend-cuda"))]
pub type InferenceBackend = burn_cuda::Cuda;

#[cfg(all(not(feature = "backend-ndarray"), feature = "backend-candle"))]
pub type InferenceBackend = burn_candle::Candle<f32, i64>;

/// Compute signed material imbalance (white - black) using standard piece values
pub fn compute_material_imbalance(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut white_score = 0i32;
    let mut black_score = 0i32;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let val = match piece.role {
                shakmaty::Role::Pawn => 1,
                shakmaty::Role::Knight => 3,
                shakmaty::Role::Bishop => 3,
                shakmaty::Role::Rook => 5,
                shakmaty::Role::Queen => 9,
                shakmaty::Role::King => 0,
            };
            match piece.color {
                shakmaty::Color::White => white_score += val,
                shakmaty::Color::Black => black_score += val,
            }
        }
    }
    white_score - black_score
}

/// Compute momentum features from material imbalance history
/// Returns (normalized_momentum, normalized_volatility)
pub fn compute_momentum_features(
    imbalance_history: &Vec<i32>,
    window: usize,
    alpha: f32,
) -> (f32, f32) {
    let n = imbalance_history.len();
    if n < 2 {
        return (0.5, 0.0); // Neutral momentum, zero volatility
    }

    // Limit to window
    let hist = if n > window {
        &imbalance_history[(n - window)..]
    } else {
        imbalance_history
    };

    // 1. Momentum: EMA of deltas
    let mut ema = 0.0_f32;
    let mut prev = hist[0] as f32;
    for &imb in hist.iter().skip(1) {
        let delta = (imb as f32 - prev) / 10.0; // Scale to [-1,1] approx
        ema = alpha * delta + (1.0 - alpha) * ema;
        prev = imb as f32;
    }
    let norm_momentum = (ema.clamp(-1.0, 1.0) * 0.5) + 0.5; // To [0,1]

    // 2. Volatility: Std dev of imbalances
    let mean = hist.iter().map(|&x| x as f32).sum::<f32>() / hist.len() as f32;
    let variance = hist
        .iter()
        .map(|&x| {
            let diff = x as f32 - mean;
            diff * diff
        })
        .sum::<f32>()
        / hist.len() as f32;
    let std_dev = variance.sqrt() / 15.0; // Scale by max imbalance ~15
    let norm_volatility = std_dev.clamp(0.0, 1.0); // [0,1], higher = more volatile

    (norm_momentum, norm_volatility)
}

/// Post-hoc whitening transform applied to trunk-mean position embeddings.
///
/// Computed offline (`oxi compute-whitening`) from the L2-normalized
/// trunk-mean embeddings of a corpus sample: `y = normalize((x - mean) * transform)`.
/// Stored as `whitening.json` next to `model.mpk`. The file is optional:
/// when it is absent the engine serves plain L2-normalized trunk-mean
/// embeddings, so checkpoints without the file always load.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhiteningTransform {
    /// Embedding dimension d; must match the model's embed_dim.
    pub dim: usize,
    /// Number of corpus samples the statistics were estimated from.
    pub samples: usize,
    /// Corpus mean of normalized trunk-mean embeddings, length d.
    pub mean: Vec<f32>,
    /// Row-major d×d ZCA matrix: output_j = Σ_i (x_i - mean_i) * transform[i*d + j].
    pub transform: Vec<f32>,
}

impl WhiteningTransform {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let parsed: Self = serde_json::from_str(&content)?;
        anyhow::ensure!(
            parsed.mean.len() == parsed.dim && parsed.transform.len() == parsed.dim * parsed.dim,
            "whitening.json dims inconsistent: dim={} mean={} transform={}",
            parsed.dim,
            parsed.mean.len(),
            parsed.transform.len()
        );
        Ok(parsed)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    /// Whiten an (already L2-normalized) trunk-mean embedding and re-normalize.
    pub fn apply(&self, embedding: &[f32]) -> Vec<f32> {
        let d = self.dim;
        debug_assert_eq!(embedding.len(), d);
        let centered: Vec<f32> = embedding
            .iter()
            .zip(&self.mean)
            .map(|(x, m)| x - m)
            .collect();
        let mut out = vec![0.0f32; d];
        for (i, &c) in centered.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let row = &self.transform[i * d..(i + 1) * d];
            for (o, w) in out.iter_mut().zip(row) {
                *o += c * w;
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in &mut out {
            *v /= norm;
        }
        out
    }
}

/// Inference engine for Oxi
pub struct InferenceEngine<B: Backend> {
    model: OXIModel<B>,
    /// Kept for checkpoint loading and public constructor compatibility.
    #[allow(dead_code)]
    config: Config,
    device: Device<B>,
    whitening: Option<WhiteningTransform>,
}

/// Simple move prediction with just move and probability
#[derive(Debug, Clone)]
pub struct MovePrediction {
    pub uci_move: String,
    pub probability: f32,
}

/// Win/Draw/Loss prediction
#[derive(Debug, Clone)]
pub struct WdlPrediction {
    pub win_prob: f32,
    pub draw_prob: f32,
    pub loss_prob: f32,
}

/// Time usage prediction modelled as a Beta distribution over the fraction of
/// the remaining clock time to invest in the move
#[derive(Debug, Clone)]
pub struct TimeUsagePrediction {
    /// Alpha parameter of the Beta distribution
    pub alpha: f32,
    /// Beta parameter of the Beta distribution
    pub beta: f32,
    /// Expected fraction of the clock to spend (alpha / (alpha + beta))
    pub expected_fraction: f32,
    /// Expected number of seconds to spend (fraction × time_remaining_self)
    pub expected_seconds: f32,
}

/// Complete prediction result from Oxi model
#[derive(Debug, Clone)]
pub struct OxiPrediction {
    pub moves: Vec<MovePrediction>,
    pub wdl: WdlPrediction,
    pub time_usage: TimeUsagePrediction,
}

/// Position + per-square trunk embeddings, pulled to CPU.
///
/// Captured from the encoder trunk path via `forward_with_attention_and_trunk`.
/// `embedding` is the L2-normalized mean-pooled trunk (optionally whitened)
/// used for cosine KNN. `per_square` remains the raw post-final-RmsNorm trunk
/// output for attention/debug views.
///
/// `per_square` is row-major with shape `[64, embed_dim]`, flat-indexed as
/// `sq * embed_dim + d`. Square indices follow the ABSOLUTE board frame
/// (a1 = 0 .. h8 = 63). For black-to-move positions the model internally
/// mirrors the board vertically; the batched inference path un-flips the
/// per-square embeddings via `model_sq ^ 56` before handing them back so
/// callers don't need to know about the mirror.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// Position embedding, shape `[embed_dim]`.
    pub embedding: Vec<f32>,
    /// Per-square embeddings in absolute frame, shape `[64, embed_dim]` flattened
    /// row-major. `per_square[sq * embed_dim + d]`.
    pub per_square: Vec<f32>,
    pub embed_dim: usize,
}

/// Per-layer post-softmax attention weights pulled to CPU.
///
/// Captured from one of the main encoder `TransformerBlock`s via the
/// `forward_with_attention` inference path. This is exclusively used for
/// introspection/visualization; the training forward path never produces one.
///
/// `data` is row-major with shape `[num_heads, 64, 64]`, flat-indexed as
/// `head * 64 * 64 + query_sq * 64 + key_sq`. Square indices follow the
/// model's internal (possibly mirrored-for-black) encoding used during the
/// forward pass.
///
/// The batched `predict_with_attention_batch` path produces one
/// `AttentionLayer` per input item by slicing each layer's
/// `[N, num_heads, 64, 64]` output row-by-row on CPU.
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    pub num_heads: usize,
    /// Row-major `[num_heads, 64, 64]`. Index: `head * 64 * 64 + query_sq * 64 + key_sq`.
    pub data: Vec<f32>,
}

/// Reconstructed move and position history from SAN moves
#[derive(Debug, Clone)]
pub struct PositionHistory {
    pub positions: Vec<Chess>,
    pub uci_moves: Vec<String>,
}

/// A single item inside a batched prediction call
/// (`predict_with_attention_batch` or `predict_batch`).
///
/// `positions` has the current position first, then previous positions (for
/// temporal context). At minimum one element (the current position).
/// `previous_moves` are the UCI moves leading to the current position
/// (ordered most-recent first). `globals`, `temperature`, `top_k` are all
/// per-item — the batched calls honor each item's own values.
///
/// Note: the batched prediction paths deliberately do NOT enforce a
/// max batch size. Callers should chunk based on available GPU memory.
#[derive(Debug, Clone)]
pub struct BatchItem {
    pub positions: Vec<Chess>,
    pub previous_moves: Vec<String>,
    pub globals: GlobalFeatures,
    pub temperature: f32,
    pub top_k: usize,
}

/// Global features for chess inference (public API)
#[derive(Debug, Clone)]
pub struct GlobalFeatures {
    /// Time remaining for self (in seconds)
    pub time_remaining_self: u32,
    /// Time remaining for opponent (in seconds)
    pub time_remaining_oppo: u32,
    /// Time control base time (in seconds)
    pub base_time: u32,
    /// Time control increment (in seconds)
    pub increment: u32,
    /// Move count in the game
    pub move_count: usize,
    /// Self Elo rating
    pub elo_self: i32,
    /// Opponent Elo rating
    pub elo_oppo: i32,
    /// Whether this is a puzzle position
    pub is_puzzle: bool,
    /// Whether the side to move is in check
    pub is_in_check: bool,
    /// Total number of pieces on the board (game phase proxy)
    pub total_pieces: u32,
}

/// Internal global features with material imbalance (used for prediction)
#[derive(Debug, Clone)]
pub struct GlobalFeaturesInternal {
    /// Time remaining for self (in seconds)
    pub time_remaining_self: u32,
    /// Time remaining for opponent (in seconds)
    pub time_remaining_oppo: u32,
    /// Time control base time (in seconds)
    pub base_time: u32,
    /// Time control increment (in seconds)
    pub increment: u32,
    /// Material imbalance (white - black)
    pub material_imbalance: i32,
    /// Count difference of major pieces (rooks + queens): white - black
    pub major_piece_imbalance: i32,
    /// Count difference of minor pieces (knights + bishops): white - black
    pub minor_piece_imbalance: i32,
    /// Count difference of pawns: white - black
    pub pawn_imbalance: i32,
    /// Move count in the game
    pub move_count: usize,
    /// Self Elo rating
    pub elo_self: i32,
    /// Opponent Elo rating
    pub elo_oppo: i32,
    /// Material imbalance history for momentum calculation
    pub material_imbalance_history: Vec<i32>,
    /// Whether this is a puzzle position
    pub is_puzzle: bool,
    /// Whether the side to move is in check
    pub is_in_check: bool,
    /// Total number of pieces on the board
    pub total_pieces: u32,
}

/// Normalized global features ready for model input
#[derive(Debug, Clone)]
pub struct GlobalFeaturesNormalized {
    pub time_self_normalized: f32,
    pub time_self_ratio: f32,
    pub time_oppo_normalized: f32,
    pub time_oppo_ratio: f32,
    pub increment_ratio: f32,
    pub move_count_normalized: f32,
    pub elo_self_normalized: f32,
    pub elo_oppo_normalized: f32,
    pub material_imbalance_normalized: f32,
    pub momentum: f32,
    pub volatility: f32,
}

impl GlobalFeatures {
    /// Convert to internal representation with computed material imbalance and history
    pub fn to_internal(
        &self,
        material_imbalance: i32,
        material_imbalance_history: Vec<i32>,
    ) -> GlobalFeaturesInternal {
        GlobalFeaturesInternal {
            time_remaining_self: self.time_remaining_self,
            time_remaining_oppo: self.time_remaining_oppo,
            base_time: self.base_time,
            increment: self.increment,
            material_imbalance,
            major_piece_imbalance: 0,
            minor_piece_imbalance: 0,
            pawn_imbalance: 0,
            move_count: self.move_count,
            elo_self: self.elo_self,
            elo_oppo: self.elo_oppo,
            material_imbalance_history,
            is_puzzle: self.is_puzzle,
            is_in_check: self.is_in_check,
            total_pieces: self.total_pieces,
        }
    }
}

impl GlobalFeaturesInternal {
    /// Compute normalized global features
    pub fn to_normalized(&self) -> GlobalFeaturesNormalized {
        let base_time = self.base_time.max(1);
        // Normalize Elo over 800..2800, clamping out-of-range ratings to the
        // nearest end of the scale.
        let elo_self_normalized = ((self.elo_self - 800) as f32 / 2000.0).clamp(0.0, 1.0);
        let elo_oppo_normalized = ((self.elo_oppo - 800) as f32 / 2000.0).clamp(0.0, 1.0);

        // Material imbalance: difference in total material (white - black) normalized to [0,1]
        // Max absolute imbalance ~15
        let material_imbalance_norm =
            ((self.material_imbalance as f32) / 15.0).clamp(-1.0, 1.0) * 0.5 + 0.5;

        // Compute momentum features (window=10, alpha=0.1 for EMA)
        let (momentum, volatility) =
            compute_momentum_features(&self.material_imbalance_history, 10, 0.1);

        GlobalFeaturesNormalized {
            time_self_normalized: (self.time_remaining_self as f32 / 1500.0).clamp(0.0, 1.0),
            time_self_ratio: (self.time_remaining_self as f32 / base_time as f32).clamp(0.0, 1.0),
            time_oppo_normalized: (self.time_remaining_oppo as f32 / 1500.0).clamp(0.0, 1.0),
            time_oppo_ratio: (self.time_remaining_oppo as f32 / base_time as f32).clamp(0.0, 1.0),
            increment_ratio: (self.increment as f32 / base_time as f32).clamp(0.0, 1.0),
            move_count_normalized: (self.move_count as f32 / 300.0).clamp(0.0, 1.0),
            elo_self_normalized,
            elo_oppo_normalized,
            material_imbalance_normalized: material_imbalance_norm,
            momentum,
            volatility,
        }
    }

    /// Compute global feature vector from the fields (following compute_global_features logic)
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let normalized = self.to_normalized();
        let globals = vec![
            normalized.time_self_normalized,
            normalized.time_self_ratio,
            normalized.time_oppo_normalized,
            normalized.time_oppo_ratio,
            normalized.increment_ratio,
            normalized.move_count_normalized,
            normalized.elo_self_normalized,
            normalized.elo_oppo_normalized,
            if self.is_puzzle { 1.0 } else { 0.0 },
            normalized.material_imbalance_normalized,
            (self.total_pieces as f32 / 32.0).clamp(0.0, 1.0),
        ];
        assert_eq!(globals.len(), NUM_GLOBALS);
        globals
    }
}

impl Default for GlobalFeatures {
    fn default() -> Self {
        Self {
            time_remaining_self: 1500,
            time_remaining_oppo: 1500,
            base_time: 1800,
            increment: 0,
            move_count: 20,
            elo_self: 1500,
            elo_oppo: 1500,
            is_puzzle: false,
            is_in_check: false,
            total_pieces: 32,
        }
    }
}

/// Convert a line of SAN moves to a sequence of chess positions
pub fn san_line_to_positions<T: AsRef<str>>(san_moves: &[T]) -> anyhow::Result<Vec<Chess>> {
    Ok(san_line_to_history(san_moves)?.positions)
}

/// Convert a SAN history to both positions and UCI moves
pub fn san_line_to_history<T: AsRef<str>>(san_moves: &[T]) -> anyhow::Result<PositionHistory> {
    let mut positions = vec![Chess::default()];
    let mut current_position = Chess::default();
    let mut uci_moves = Vec::with_capacity(san_moves.len());

    for san in san_moves.iter() {
        let san_plus: SanPlus = san
            .as_ref()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid SAN move: {}", san.as_ref()))?;

        let chess_move = san_plus
            .san
            .to_move(&current_position)
            .map_err(|_| anyhow::anyhow!("Cannot convert SAN to move: {}", san.as_ref()))?;

        let uci = chess_move
            .to_uci(shakmaty::CastlingMode::Standard)
            .to_string();
        uci_moves.push(uci);

        current_position = current_position
            .play(chess_move)
            .map_err(|_| anyhow::anyhow!("Illegal move: {}", san.as_ref()))?;

        positions.push(current_position.clone());
    }

    Ok(PositionHistory {
        positions,
        uci_moves,
    })
}

fn flip_chess_position(position: &Chess) -> anyhow::Result<Chess> {
    let fen_string = Fen::from_position(position, EnPassantMode::Legal).to_string();
    let mirrored_fen = mirror_fen(&fen_string);
    let flipped_position: Chess = mirrored_fen
        .parse::<Fen>()?
        .into_position(shakmaty::CastlingMode::Standard)?;
    Ok(flipped_position)
}

fn build_input_parts(
    positions: &[Chess],
    previous_moves: &[String],
    global_features: &GlobalFeatures,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Chess, Vec<i32>)> {
    if positions.is_empty() {
        return Err(anyhow::anyhow!("No positions provided"));
    }

    let max_positions = (PREVIOUS_POSITIONS + 1).min(positions.len());
    let relevant_positions = &positions[..max_positions];
    let moves_needed = max_positions.saturating_sub(1);
    let moves_available = previous_moves.len().min(moves_needed);
    let relevant_moves = &previous_moves[..moves_available];

    let current_position = &relevant_positions[0];
    let is_black_to_move = current_position.turn() == Color::Black;

    let (flipped_current, mut flipped_previous, mut flipped_previous_moves) = if is_black_to_move {
        let flipped_current = flip_chess_position(current_position)?;
        let flipped_previous: Result<Vec<Chess>, anyhow::Error> = relevant_positions[1..]
            .iter()
            .map(flip_chess_position)
            .collect();
        let flipped_prev_moves = relevant_moves
            .iter()
            .map(|mv| mirror_move(mv))
            .collect::<Vec<_>>();
        (flipped_current, flipped_previous?, flipped_prev_moves)
    } else {
        let previous_positions: Vec<Chess> = relevant_positions[1..].to_vec();
        let previous_moves_vec = relevant_moves.to_vec();
        (
            current_position.clone(),
            previous_positions,
            previous_moves_vec,
        )
    };

    if moves_available < flipped_previous.len() {
        flipped_previous.truncate(moves_available);
    }

    if moves_available < flipped_previous_moves.len() {
        flipped_previous_moves.truncate(moves_available);
    }

    let board_encoded =
        encode_position(&flipped_current, &flipped_previous, &flipped_previous_moves);

    let material_imbalance = compute_material_imbalance(&flipped_current);
    let material_imbalance_history: Vec<i32> = relevant_positions
        .iter()
        .skip(1)
        .map(compute_material_imbalance)
        .collect();

    let global_features_internal =
        global_features.to_internal(material_imbalance, material_imbalance_history.clone());
    let global_features_normalized = global_features_internal.to_normalized();
    tracing::debug!(
        "Global features (normalized): {:?}",
        global_features_normalized
    );

    Ok((
        board_encoded,
        global_features_internal.to_feature_vector(),
        flipped_current,
        material_imbalance_history,
    ))
}

pub fn load_model<B: Backend>(
    path: &Path,
    config: &Config,
    device: &Device<B>,
) -> anyhow::Result<OXIModel<B>> {
    let record = DefaultRecorder::new().load(path.to_path_buf(), device)?;

    let model = OXIModel::new(device, config).load_record(record);

    Ok(model)
}

impl<B: Backend> InferenceEngine<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    /// Create new inference engine from checkpoint
    pub fn from_checkpoint(
        checkpoint_path: &Path,
        config: Config,
        device: Device<B>,
    ) -> anyhow::Result<Self> {
        let model = load_model(checkpoint_path, &config, &device)?;
        let whitening = Self::load_whitening_for_checkpoint(checkpoint_path, &config);
        Ok(Self {
            model,
            config,
            device,
            whitening,
        })
    }

    /// Load `whitening.json` from the checkpoint's directory when the file
    /// exists. Missing or invalid files are non-fatal: the engine falls back
    /// to plain L2-normalized trunk-mean embeddings so checkpoints without
    /// the file always load.
    fn load_whitening_for_checkpoint(
        checkpoint_path: &Path,
        config: &Config,
    ) -> Option<WhiteningTransform> {
        let path = checkpoint_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("whitening.json");
        if !path.exists() {
            tracing::info!(
                "No whitening.json next to checkpoint ({:?}); serving plain trunk-mean embeddings",
                path
            );
            return None;
        }
        match WhiteningTransform::load(&path) {
            Ok(whitening) if whitening.dim == config.embed_dim => {
                tracing::info!(
                    "Loaded whitening transform from {:?} ({} samples)",
                    path,
                    whitening.samples
                );
                Some(whitening)
            }
            Ok(whitening) => {
                tracing::warn!(
                    "whitening.json dim {} != model embed_dim {}; falling back to trunk_mean",
                    whitening.dim,
                    config.embed_dim
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load whitening transform from {:?} ({e}); \
                     falling back to trunk_mean embeddings",
                    path
                );
                None
            }
        }
    }

    /// Create inference engine with existing model
    pub fn new(model: OXIModel<B>, config: Config, device: Device<B>) -> Self {
        Self {
            model,
            config,
            device,
            whitening: None,
        }
    }

    /// Replace the whitening transform (used by offline tooling).
    pub fn set_whitening(&mut self, whitening: Option<WhiteningTransform>) {
        self.whitening = whitening;
    }

    /// Create a simple inference engine with default device
    #[cfg(any(
        feature = "backend-ndarray",
        feature = "backend-tch",
        feature = "backend-cuda",
        feature = "backend-candle"
    ))]
    pub fn create_simple(
        model: OXIModel<InferenceBackend>,
        config: Config,
    ) -> InferenceEngine<InferenceBackend> {
        let device = Device::<InferenceBackend>::default();
        InferenceEngine::new(model, config, device)
    }

    /// Create input tensors for model forward pass from chess positions and global features
    /// This is the core tensor creation logic extracted from predict() for testability
    pub fn create_input_tensors(
        &self,
        positions: &[Chess],
        previous_moves: &[String],
        global_features: &GlobalFeatures,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 2>, Chess, Vec<i32>)>
    where
        B::FloatElem: From<f32>,
    {
        let (board_encoded, global_feature_vector, flipped_current, material_imbalance_history) =
            build_input_parts(positions, previous_moves, global_features)?;

        // Convert to tensor [batch=1, seq=64, features]
        let board_tensor = Tensor::<B, 1>::from_floats(board_encoded.as_slice(), &self.device)
            .reshape([1, 64, board_encoded.len() / 64]);

        let global_features_tensor =
            Tensor::<B, 1>::from_floats(global_feature_vector.as_slice(), &self.device)
                .reshape([1, NUM_GLOBALS]);

        Ok((
            board_tensor,
            global_features_tensor,
            flipped_current,
            material_imbalance_history,
        ))
    }

    /// Batched prediction that does NOT capture attention weights.
    ///
    /// Stacks the per-item `[1, 64, F]` board tensors along the batch axis into
    /// a single `[N, 64, F]` tensor (and likewise for globals), runs ONE
    /// forward pass with `forward`, then splits the `[N, ...]` outputs back
    /// per item. Result ordering matches `items` order.
    ///
    /// This avoids materializing the per-layer attention tensors, so it is
    /// strictly cheaper than `predict_with_attention_batch` when you don't
    /// need saliency.
    pub fn predict_batch(&self, items: &[BatchItem]) -> anyhow::Result<Vec<OxiPrediction>>
    where
        B::FloatElem: From<f32>,
    {
        self.predict_batch_with_channel_mask(items, None)
    }

    /// `predict_batch` with an optional per-channel input mask, used for
    /// feature-ablation studies: each of the `FEATURES_PER_TOKEN` board
    /// channels is multiplied by the corresponding mask entry after encoding
    /// (0.0 = channel zeroed, 1.0 = untouched). Globals are unaffected.
    pub fn predict_batch_with_channel_mask(
        &self,
        items: &[BatchItem],
        channel_mask: Option<&[f32]>,
    ) -> anyhow::Result<Vec<OxiPrediction>>
    where
        B::FloatElem: From<f32>,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let total_start = Instant::now();
        let build_start = Instant::now();
        let (batched_board, batched_globals, flipped_currents, is_black_to_move_flags, n) =
            self.build_batched_inputs(items)?;
        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

        let batched_board = if let Some(mask) = channel_mask {
            anyhow::ensure!(
                mask.len() == crate::config::FEATURES_PER_TOKEN,
                "channel mask has {} entries, expected {}",
                mask.len(),
                crate::config::FEATURES_PER_TOKEN
            );
            let mask_tensor =
                Tensor::<B, 1>::from_floats(mask, &self.device).reshape([1, 1, mask.len()]);
            batched_board * mask_tensor
        } else {
            batched_board
        };

        let forward_start = Instant::now();
        let policy_logits = self.model.forward_policy(batched_board, batched_globals);
        let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

        let finalize_start = Instant::now();
        let result = self.finalize_batched_predictions(
            items,
            n,
            &flipped_currents,
            &is_black_to_move_flags,
            policy_logits,
            None,
            None,
        );
        let finalize_ms = finalize_start.elapsed().as_secs_f64() * 1000.0;

        if std::env::var("OXI_INFERENCE_TIMING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
        {
            tracing::info!(
                batch = n,
                build_ms = build_ms,
                forward_call_ms = forward_ms,
                finalize_ms = finalize_ms,
                total_ms = total_start.elapsed().as_secs_f64() * 1000.0,
                "predict_batch_timing"
            );
        }

        result
    }

    /// Batched prediction that also captures per-layer post-softmax attention
    /// weights from the main encoder `TransformerBlock`s.
    ///
    /// Stacks the per-item `[1, 64, F]` board tensors along the batch axis into
    /// a single `[N, 64, F]` tensor (and likewise for globals), runs ONE forward
    /// pass with `forward_with_attention`, then splits the `[N, ...]` outputs
    /// back per item. Result ordering matches `items` order.
    ///
    /// Each item's inputs go through the same per-item preprocessing:
    /// `create_input_tensors` (which handles the black-to-move mirror so each
    /// stacked row is in model-frame for its own side) and its own
    /// `temperature`/`top_k` are applied in the post-processing per row. The
    /// caller is responsible for un-flipping the attention saliency back to
    /// absolute board coordinates when the corresponding item was black-to-move
    /// (the returned attention tensors are in model frame).
    ///
    /// NOTE: No max batch size is enforced here. The caller should chunk
    /// according to available GPU memory.
    pub fn predict_with_attention_batch(
        &self,
        items: &[BatchItem],
    ) -> anyhow::Result<Vec<(OxiPrediction, Vec<AttentionLayer>)>>
    where
        B::FloatElem: From<f32>,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let (batched_board, batched_globals, flipped_currents, is_black_to_move_flags, n) =
            self.build_batched_inputs(items)?;

        let (policy_logits, value_logits, _side_info_logits, time_usage_logits, attn_maps) = self
            .model
            .forward_with_attention(batched_board, batched_globals);

        // Pre-split attention tensors to CPU. Each attn layer is [N, num_heads, 64, 64];
        // flatten to a single Vec<f32> of length N * num_heads * 64 * 64 so we can
        // slice per-item without re-entering the backend.
        let num_layers = attn_maps.len();
        let mut attn_layer_buffers: Vec<(usize, Vec<f32>)> = Vec::with_capacity(num_layers);
        for attn in attn_maps.into_iter() {
            let dims = attn.dims();
            debug_assert_eq!(dims[0], n, "attention batch dim must equal N");
            debug_assert_eq!(dims[2], 64);
            debug_assert_eq!(dims[3], 64);
            let num_heads = dims[1];
            let data = attn.to_data().to_vec::<f32>().map_err(|e| {
                anyhow::anyhow!("Failed to convert attention tensor to f32: {:?}", e)
            })?;
            debug_assert_eq!(data.len(), n * num_heads * 64 * 64);
            attn_layer_buffers.push((num_heads, data));
        }

        let predictions = self.finalize_batched_predictions(
            items,
            n,
            &flipped_currents,
            &is_black_to_move_flags,
            policy_logits,
            Some(value_logits),
            Some(time_usage_logits),
        )?;

        // Combine predictions with per-item attention slices.
        let mut out: Vec<(OxiPrediction, Vec<AttentionLayer>)> = Vec::with_capacity(n);
        for (i, prediction) in predictions.into_iter().enumerate() {
            // Attention: slice row i out of each layer's flat buffer. Each row
            // is num_heads * 64 * 64 floats.
            let mut attention_layers: Vec<AttentionLayer> = Vec::with_capacity(num_layers);
            for (num_heads, buf) in attn_layer_buffers.iter() {
                let row_len = num_heads * 64 * 64;
                let start = i * row_len;
                let end = start + row_len;
                let data = buf[start..end].to_vec();
                attention_layers.push(AttentionLayer {
                    num_heads: *num_heads,
                    data,
                });
            }
            out.push((prediction, attention_layers));
        }

        Ok(out)
    }

    /// Batched prediction that returns policy output plus trunk-mean
    /// embeddings without materializing attention maps, value, or time heads.
    pub fn predict_with_embedding_batch(
        &self,
        items: &[BatchItem],
    ) -> anyhow::Result<Vec<(OxiPrediction, EmbeddingOutput)>>
    where
        B::FloatElem: From<f32>,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let (batched_board, batched_globals, flipped_currents, is_black_to_move_flags, n) =
            self.build_batched_inputs(items)?;

        let (policy_logits, trunk, _policy_tokens) =
            self.model
                .forward_policy_with_trunk(batched_board, batched_globals);

        let trunk_dims = trunk.dims();
        debug_assert_eq!(trunk_dims[0], n);
        debug_assert_eq!(trunk_dims[1], 64);
        let embed_dim = trunk_dims[2];
        let trunk_flat = trunk
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert trunk tensor to f32: {:?}", e))?;
        debug_assert_eq!(trunk_flat.len(), n * 64 * embed_dim);

        let predictions = self.finalize_batched_predictions(
            items,
            n,
            &flipped_currents,
            &is_black_to_move_flags,
            policy_logits,
            None,
            None,
        )?;

        let mut out = Vec::with_capacity(n);
        for (i, prediction) in predictions.into_iter().enumerate() {
            let row_start = i * 64 * embed_dim;
            let model_frame = &trunk_flat[row_start..row_start + 64 * embed_dim];
            let is_black = is_black_to_move_flags[i];
            let mut per_square = vec![0.0f32; 64 * embed_dim];
            if is_black {
                for model_sq in 0..64usize {
                    let absolute_sq = model_sq ^ 56;
                    let src = model_sq * embed_dim;
                    let dst = absolute_sq * embed_dim;
                    per_square[dst..dst + embed_dim]
                        .copy_from_slice(&model_frame[src..src + embed_dim]);
                }
            } else {
                per_square.copy_from_slice(model_frame);
            }

            let embedding = {
                let mut pooled = vec![0.0f32; embed_dim];
                for sq in 0..64usize {
                    let src = sq * embed_dim;
                    for dim in 0..embed_dim {
                        pooled[dim] += model_frame[src + dim];
                    }
                }
                for value in &mut pooled {
                    *value /= 64.0;
                }
                let norm = pooled
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                for value in &mut pooled {
                    *value /= norm;
                }
                match &self.whitening {
                    Some(whitening) => whitening.apply(&pooled),
                    None => pooled,
                }
            };

            out.push((
                prediction,
                EmbeddingOutput {
                    embedding,
                    per_square,
                    embed_dim,
                },
            ));
        }

        Ok(out)
    }

    /// Raw L2-normalized trunk-mean embeddings for a batch, with NO whitening
    /// applied regardless of the configured embedding source. Used by offline
    /// tooling (`compute-whitening`) that estimates whitening statistics.
    /// Mean pooling over all 64 squares is invariant to the black-to-move
    /// square flip, so no frame adjustment is needed.
    pub fn raw_trunk_mean_embeddings_batch(
        &self,
        items: &[BatchItem],
    ) -> anyhow::Result<Vec<Vec<f32>>>
    where
        B::FloatElem: From<f32>,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let (batched_board, batched_globals, _flipped_currents, _is_black_to_move_flags, n) =
            self.build_batched_inputs(items)?;
        let trunk = self.model.forward_trunk(batched_board, batched_globals);

        let trunk_dims = trunk.dims();
        debug_assert_eq!(trunk_dims[0], n);
        debug_assert_eq!(trunk_dims[1], 64);
        let embed_dim = trunk_dims[2];
        let trunk_flat = trunk
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert trunk tensor to f32: {:?}", e))?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &trunk_flat[i * 64 * embed_dim..(i + 1) * 64 * embed_dim];
            let mut pooled = vec![0.0f32; embed_dim];
            for sq in 0..64usize {
                let src = sq * embed_dim;
                for dim in 0..embed_dim {
                    pooled[dim] += row[src + dim];
                }
            }
            let norm = pooled
                .iter()
                .map(|v| (v / 64.0) * (v / 64.0))
                .sum::<f32>()
                .sqrt()
                .max(1e-6);
            for value in &mut pooled {
                *value = *value / 64.0 / norm;
            }
            out.push(pooled);
        }
        Ok(out)
    }

    /// Batched prediction that also captures per-layer post-softmax attention
    /// weights AND the trunk-level per-square embeddings.
    ///
    /// Used by viz/analysis and the bot `/predict` path when embedding output is
    /// requested. This keeps the attention response shape while adding the
    /// position embedding output.
    ///
    /// Returns one `(OxiPrediction, Vec<AttentionLayer>, EmbeddingOutput)`
    /// per input item, matching `items` order. The embedding is returned in
    /// the ABSOLUTE board frame: for black-to-move items, the per-square
    /// dimension is un-flipped via `sq ^ 56` so cross-position comparisons
    /// are frame-consistent. The position embedding is mean-pooled over all
    /// squares and needs no square-frame adjustment.
    pub fn predict_with_attention_and_embedding_batch(
        &self,
        items: &[BatchItem],
    ) -> anyhow::Result<Vec<(OxiPrediction, Vec<AttentionLayer>, EmbeddingOutput)>>
    where
        B::FloatElem: From<f32>,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let (batched_board, batched_globals, flipped_currents, is_black_to_move_flags, n) =
            self.build_batched_inputs(items)?;

        let (policy_logits, value_logits, _side_info_logits, time_usage_logits, attn_maps, trunk) =
            self.model
                .forward_with_attention_and_trunk(batched_board, batched_globals);

        // Pre-split attention tensors to CPU (mirrors `predict_with_attention_batch`).
        let num_layers = attn_maps.len();
        let mut attn_layer_buffers: Vec<(usize, Vec<f32>)> = Vec::with_capacity(num_layers);
        for attn in attn_maps.into_iter() {
            let dims = attn.dims();
            debug_assert_eq!(dims[0], n, "attention batch dim must equal N");
            debug_assert_eq!(dims[2], 64);
            debug_assert_eq!(dims[3], 64);
            let num_heads = dims[1];
            let data = attn.to_data().to_vec::<f32>().map_err(|e| {
                anyhow::anyhow!("Failed to convert attention tensor to f32: {:?}", e)
            })?;
            debug_assert_eq!(data.len(), n * num_heads * 64 * 64);
            attn_layer_buffers.push((num_heads, data));
        }

        // Pull trunk tensor to CPU once. Shape: [N, 64, embed_dim].
        let trunk_dims = trunk.dims();
        debug_assert_eq!(trunk_dims[0], n);
        debug_assert_eq!(trunk_dims[1], 64);
        let embed_dim = trunk_dims[2];
        let trunk_flat = trunk
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert trunk tensor to f32: {:?}", e))?;
        debug_assert_eq!(trunk_flat.len(), n * 64 * embed_dim);

        let predictions = self.finalize_batched_predictions(
            items,
            n,
            &flipped_currents,
            &is_black_to_move_flags,
            policy_logits,
            Some(value_logits),
            Some(time_usage_logits),
        )?;

        let mut out: Vec<(OxiPrediction, Vec<AttentionLayer>, EmbeddingOutput)> =
            Vec::with_capacity(n);
        for (i, prediction) in predictions.into_iter().enumerate() {
            // Attention: slice row i out of each layer's flat buffer.
            let mut attention_layers: Vec<AttentionLayer> = Vec::with_capacity(num_layers);
            for (num_heads, buf) in attn_layer_buffers.iter() {
                let row_len = num_heads * 64 * 64;
                let start = i * row_len;
                let end = start + row_len;
                let data = buf[start..end].to_vec();
                attention_layers.push(AttentionLayer {
                    num_heads: *num_heads,
                    data,
                });
            }

            // Trunk: slice row i (model frame), un-flip per-square for black-to-move.
            let row_start = i * 64 * embed_dim;
            let model_frame = &trunk_flat[row_start..row_start + 64 * embed_dim];
            let is_black = is_black_to_move_flags[i];
            let mut per_square = vec![0.0f32; 64 * embed_dim];
            if is_black {
                for model_sq in 0..64usize {
                    let absolute_sq = model_sq ^ 56;
                    let src = model_sq * embed_dim;
                    let dst = absolute_sq * embed_dim;
                    per_square[dst..dst + embed_dim]
                        .copy_from_slice(&model_frame[src..src + embed_dim]);
                }
            } else {
                per_square.copy_from_slice(model_frame);
            }

            let embedding = {
                let mut pooled = vec![0.0f32; embed_dim];
                for sq in 0..64usize {
                    let src = sq * embed_dim;
                    for dim in 0..embed_dim {
                        pooled[dim] += model_frame[src + dim];
                    }
                }
                for value in &mut pooled {
                    *value /= 64.0;
                }
                let norm = pooled
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                for value in &mut pooled {
                    *value /= norm;
                }
                match &self.whitening {
                    Some(whitening) => whitening.apply(&pooled),
                    None => pooled,
                }
            };

            out.push((
                prediction,
                attention_layers,
                EmbeddingOutput {
                    embedding,
                    per_square,
                    embed_dim,
                },
            ));
        }

        Ok(out)
    }

    /// Shared batched-input construction for `predict_batch` and
    /// `predict_with_attention_batch`. Each per-item board/globals tensor is
    /// built via the same `create_input_tensors` used historically, then
    /// concatenated along the batch axis.
    fn build_batched_inputs(
        &self,
        items: &[BatchItem],
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 2>, Vec<Chess>, Vec<bool>, usize)>
    where
        B::FloatElem: From<f32>,
    {
        let n = items.len();
        let built = items
            .par_iter()
            .map(|item| {
                if item.positions.is_empty() {
                    return Err(anyhow::anyhow!("BatchItem has an empty positions vec"));
                }
                let is_black_to_move = item.positions[0].turn() == Color::Black;
                let (board, globals, flipped_current, _imbalance_history) =
                    build_input_parts(&item.positions, &item.previous_moves, &item.globals)?;
                Ok((board, globals, flipped_current, is_black_to_move))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut board_data: Vec<f32> = Vec::with_capacity(n * 64 * FEATURES_PER_TOKEN);
        let mut global_data: Vec<f32> = Vec::with_capacity(n * NUM_GLOBALS);
        let mut flipped_currents: Vec<Chess> = Vec::with_capacity(n);
        let mut is_black_to_move_flags: Vec<bool> = Vec::with_capacity(n);

        for (board, globals, flipped_current, is_black_to_move) in built {
            debug_assert_eq!(board.len(), 64 * FEATURES_PER_TOKEN);
            debug_assert_eq!(globals.len(), NUM_GLOBALS);
            board_data.extend_from_slice(&board);
            global_data.extend_from_slice(&globals);
            flipped_currents.push(flipped_current);
            is_black_to_move_flags.push(is_black_to_move);
        }

        let batched_board = Tensor::<B, 1>::from_floats(board_data.as_slice(), &self.device)
            .reshape([n, 64, FEATURES_PER_TOKEN]);
        let batched_globals = Tensor::<B, 1>::from_floats(global_data.as_slice(), &self.device)
            .reshape([n, NUM_GLOBALS]);

        Ok((
            batched_board,
            batched_globals,
            flipped_currents,
            is_black_to_move_flags,
            n,
        ))
    }

    /// Shared per-row post-processing for both batched paths. Converts model
    /// outputs (policy / value / time-usage logits) into per-item
    /// `OxiPrediction`s, applying legal-move masking, per-row temperature, and
    /// un-mirroring the output moves when the row's item was black-to-move.
    fn finalize_batched_predictions(
        &self,
        items: &[BatchItem],
        n: usize,
        flipped_currents: &[Chess],
        is_black_to_move_flags: &[bool],
        policy_logits: Tensor<B, 3>,
        value_logits: Option<Tensor<B, 2>>,
        time_usage_logits: Option<Tensor<B, 2>>,
    ) -> anyhow::Result<Vec<OxiPrediction>>
    where
        B::FloatElem: From<f32>,
    {
        // Reshape policy logits to [N, LEGAL_MOVES].
        let policy_logits_flat = policy_logits.reshape([n, LEGAL_MOVES]);

        // Build the per-row legal-moves mask as a single [N, LEGAL_MOVES] tensor.
        let mut legal_mask_flat: Vec<f32> = vec![0.0f32; n * LEGAL_MOVES];
        for (i, flipped_current) in flipped_currents.iter().enumerate() {
            let row_off = i * LEGAL_MOVES;
            for legal_move in flipped_current.legal_moves() {
                if let Some((from_idx, promo_idx)) =
                    encode_move(&legal_move.to_uci(shakmaty::CastlingMode::Standard))
                {
                    let move_idx = from_idx as usize * 76 + promo_idx as usize;
                    legal_mask_flat[row_off + move_idx] = 1.0;
                }
            }
        }
        let legal_moves_tensor =
            Tensor::<B, 1>::from_floats(legal_mask_flat.as_slice(), &self.device)
                .reshape([n, LEGAL_MOVES]);

        let mask = legal_moves_tensor.equal_elem(0.0);
        let masked_logits = policy_logits_flat.mask_fill(mask, f32::NEG_INFINITY);

        // Apply per-row temperature by dividing each row by its own temperature
        // via a [N, 1] tensor. If all temperatures are 1.0 we skip the div.
        let all_temps_one = items.iter().all(|it| it.temperature == 1.0);
        let temped_logits = if all_temps_one {
            masked_logits
        } else {
            let temp_vec: Vec<f32> = items.iter().map(|it| it.temperature).collect();
            let temp_tensor =
                Tensor::<B, 1>::from_floats(temp_vec.as_slice(), &self.device).reshape([n, 1]);
            masked_logits.div(temp_tensor)
        };

        let log_probs = log_softmax(temped_logits, 1);
        let probs = log_probs.exp();

        // Pull everything to CPU as flat f32 for slicing. Each tensor is
        // [N, D] row-major.
        let probs_data = probs.to_data();
        let probs_slice = probs_data.as_slice::<f32>().unwrap();

        let value_data = value_logits.map(|logits| log_softmax(logits, 1).exp().to_data());
        let value_slice = value_data
            .as_ref()
            .map(|data| data.as_slice::<f32>().unwrap());

        let time_params_data = time_usage_logits.map(|logits| logits.to_data());
        let time_params_slice = time_params_data
            .as_ref()
            .map(|data| data.as_slice::<f32>().unwrap());

        let mut out: Vec<OxiPrediction> = Vec::with_capacity(n);
        for i in 0..n {
            let item = &items[i];
            let is_black_to_move = is_black_to_move_flags[i];

            // Top-k moves for row i.
            let row_off = i * LEGAL_MOVES;
            let mut move_probs: Vec<(usize, f32)> = probs_slice[row_off..row_off + LEGAL_MOVES]
                .iter()
                .enumerate()
                .filter(|(_, &prob)| prob > 0.0 && !prob.is_infinite())
                .map(|(j, &p)| (j, p))
                .collect();

            move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let move_predictions: Vec<MovePrediction> = move_probs
                .into_iter()
                .take(item.top_k)
                .filter_map(|(move_idx, prob)| {
                    let from_idx = (move_idx / 76) as u8;
                    let to_idx = (move_idx % 76) as u8;
                    decode_move(from_idx, to_idx).map(|uci_move| {
                        let final_uci = if is_black_to_move {
                            mirror_move(&uci_move.to_string())
                        } else {
                            uci_move.to_string()
                        };
                        MovePrediction {
                            uci_move: final_uci,
                            probability: prob,
                        }
                    })
                })
                .collect();

            // WDL for row i. Ordering: [loss, draw, win].
            let wdl = if let Some(value_slice) = value_slice {
                let v_off = i * 3;
                WdlPrediction {
                    loss_prob: value_slice[v_off],
                    draw_prob: value_slice[v_off + 1],
                    win_prob: value_slice[v_off + 2],
                }
            } else {
                WdlPrediction {
                    loss_prob: 1.0 / 3.0,
                    draw_prob: 1.0 / 3.0,
                    win_prob: 1.0 / 3.0,
                }
            };

            // Time usage params for row i (alpha, beta).
            let (alpha, beta) = if let Some(time_params_slice) = time_params_slice {
                let t_off = i * 2;
                (time_params_slice[t_off], time_params_slice[t_off + 1])
            } else {
                (0.0, 0.0)
            };
            let denom = alpha + beta;
            let expected_fraction = if denom > 0.0 { alpha / denom } else { 0.0 };
            let expected_seconds = expected_fraction * item.globals.time_remaining_self as f32;

            let time_usage = TimeUsagePrediction {
                alpha,
                beta,
                expected_fraction,
                expected_seconds,
            };

            out.push(OxiPrediction {
                moves: move_predictions,
                wdl,
                time_usage,
            });
        }

        Ok(out)
    }

    /// Predict the full policy distribution over ALL legal moves.
    ///
    /// Returns a HashMap mapping UCI move strings to probabilities, covering
    /// every legal move in the position. Unlike `predict()` which returns top-k,
    /// this returns the complete distribution needed for ECL computation.
    ///
    /// Moves are returned in the original coordinate system (not mirrored).
    pub fn predict_full_policy(
        &self,
        positions: &[Chess],
        global_features: &GlobalFeatures,
        temperature: f32,
    ) -> anyhow::Result<std::collections::HashMap<String, f32>>
    where
        B::FloatElem: From<f32>,
    {
        let (board_tensor, global_features_tensor, flipped_current, _material_imbalance_history) =
            self.create_input_tensors(positions, &[], global_features)?;

        let current_position = &positions[0];
        let is_black_to_move = current_position.turn() == Color::Black;

        let policy_logits = self
            .model
            .forward_policy(board_tensor, global_features_tensor);

        // Get legal moves mask
        let mut legal_moves_mask = vec![0.0f32; LEGAL_MOVES];
        for legal_move in flipped_current.legal_moves() {
            if let Some((from_idx, promo_idx)) =
                encode_move(&legal_move.to_uci(shakmaty::CastlingMode::Standard))
            {
                let move_idx = from_idx as usize * 76 + promo_idx as usize;
                legal_moves_mask[move_idx] = 1.0;
            }
        }

        let legal_moves_tensor =
            Tensor::<B, 1>::from_floats(legal_moves_mask.as_slice(), &self.device)
                .reshape([1, LEGAL_MOVES]);

        // Reshape policy logits to [batch, LEGAL_MOVES]
        let policy_logits_flat = policy_logits.reshape([1, LEGAL_MOVES]);

        // Apply legal move masking
        let mask = legal_moves_tensor.clone().equal_elem(0.0);
        let masked_logits = policy_logits_flat.mask_fill(mask, f32::NEG_INFINITY);

        // Apply temperature and softmax
        let log_probs = if temperature != 1.0 {
            log_softmax(masked_logits.div_scalar(temperature), 1)
        } else {
            log_softmax(masked_logits, 1)
        };
        let probs = log_probs.exp();

        // Extract all legal moves with probabilities
        let probs_data = probs.to_data();
        let probs_slice = probs_data.as_slice::<f32>().unwrap();

        let mut result = std::collections::HashMap::new();
        for (move_idx, &prob) in probs_slice[0..LEGAL_MOVES].iter().enumerate() {
            if prob > 0.0 && !prob.is_infinite() && !prob.is_nan() {
                let from_idx = (move_idx / 76) as u8;
                let to_idx = (move_idx % 76) as u8;
                if let Some(uci_move) = decode_move(from_idx, to_idx) {
                    let final_uci = if is_black_to_move {
                        mirror_move(&uci_move.to_string())
                    } else {
                        uci_move.to_string()
                    };
                    result.insert(final_uci, prob);
                }
            }
        }

        Ok(result)
    }

    /// Analyze a position from FEN and return detailed information
    pub fn analyze(
        &self,
        fen: &str,
        global_features: &GlobalFeatures,
        temperature: f32,
        top_k: usize,
    ) -> anyhow::Result<PositionAnalysis>
    where
        B::FloatElem: From<f32>,
    {
        let parsed_fen: Fen = fen.parse()?;
        let pos: Chess = parsed_fen.into_position(shakmaty::CastlingMode::Standard)?;

        let item = BatchItem {
            positions: vec![pos],
            previous_moves: Vec::new(),
            globals: global_features.clone(),
            temperature,
            top_k,
        };
        let mut predictions = self.predict_batch(&[item])?;
        let prediction = predictions.pop().ok_or_else(|| {
            anyhow::anyhow!("predict_batch returned no predictions for analyze()")
        })?;

        Ok(PositionAnalysis {
            fen: fen.to_string(),
            prediction,
        })
    }
}

/// Detailed position analysis
#[derive(Debug, Clone)]
pub struct PositionAnalysis {
    pub fen: String,
    pub prediction: OxiPrediction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine_creation() {
        let device = Device::<InferenceBackend>::default();
        let config = Config::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<InferenceBackend>::new(&device, &config);

        let engine = InferenceEngine::new(model, config, device);

        // Test prediction on starting position
        let starting_pos = Chess::default();
        let globals = GlobalFeatures::default();
        let items = vec![BatchItem {
            positions: vec![starting_pos],
            previous_moves: Vec::new(),
            globals,
            temperature: 1.0,
            top_k: 5,
        }];
        let result = engine.predict_batch(&items);

        assert!(result.is_ok());
        let mut predictions = result.unwrap();
        assert_eq!(predictions.len(), 1);
        let prediction = predictions.pop().unwrap();
        assert!(!prediction.moves.is_empty());
        assert!(prediction.moves.len() <= 5);

        // Check WDL probabilities sum to ~1
        let wdl_sum = prediction.wdl.win_prob + prediction.wdl.draw_prob + prediction.wdl.loss_prob;
        assert!((wdl_sum - 1.0).abs() < 0.01);

        // Time params are raw linear outputs — sign is meaningless on a
        // randomly initialized model (asserting positivity made this test
        // sensitive to struct-field RNG draw order), so only check finiteness.
        assert!(prediction.time_usage.alpha.is_finite());
        assert!(prediction.time_usage.beta.is_finite());
        assert!(prediction.time_usage.expected_fraction.is_finite());
    }

    #[test]
    fn test_analyze_position() {
        let device = Device::<InferenceBackend>::default();
        let config = Config::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<InferenceBackend>::new(&device, &config);
        let engine = InferenceEngine::new(model, config, device);

        let globals = GlobalFeatures::default();
        let result = engine.analyze(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &globals,
            1.0,
            3,
        );

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(
            analysis.fen,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        assert!(!analysis.prediction.moves.is_empty());
    }

    #[test]
    fn test_predict_with_attention_batch_matches_sequential() {
        let device = Device::<InferenceBackend>::default();
        let config = Config::default();
        let _ = crate::config::set_global_config(config.clone());
        let model = OXIModel::<InferenceBackend>::new(&device, &config);
        let engine = InferenceEngine::new(model, config, device);

        // Three different positions.
        let p0 = Chess::default();
        let p1 = san_line_to_positions(&["e4"])
            .unwrap()
            .last()
            .cloned()
            .unwrap();
        let p2 = san_line_to_positions(&["d4"])
            .unwrap()
            .last()
            .cloned()
            .unwrap();

        let globals = GlobalFeatures::default();
        let items: Vec<BatchItem> = vec![p0.clone(), p1.clone(), p2.clone()]
            .into_iter()
            .map(|p| BatchItem {
                positions: vec![p],
                previous_moves: Vec::new(),
                globals: globals.clone(),
                temperature: 1.0,
                top_k: 5,
            })
            .collect();

        // Batched attention path: single forward pass with attention, 3 outputs.
        let batched = engine
            .predict_with_attention_batch(&items)
            .expect("batched attention path should succeed");
        assert_eq!(batched.len(), 3);

        // Batched no-attention path: single forward pass without attention.
        let sequential: Vec<OxiPrediction> = engine
            .predict_batch(&items)
            .expect("batched no-attention path should succeed");
        assert_eq!(sequential.len(), 3);

        // Compare top move + attention shape on the batched-attention side.
        for i in 0..3 {
            let (b_pred, b_attn) = &batched[i];
            let s_pred = &sequential[i];

            // Top move should be identical across the two batched paths.
            assert!(!b_pred.moves.is_empty(), "batched[{}] has no moves", i);
            assert!(!s_pred.moves.is_empty(), "sequential[{}] has no moves", i);
            assert_eq!(
                b_pred.moves[0].uci_move, s_pred.moves[0].uci_move,
                "top move mismatch at position {}",
                i
            );

            // Attention layer count and shape sanity.
            assert_eq!(
                b_attn.len(),
                crate::config::get_global_config().num_layers(),
                "attention layers should match encoder depth"
            );
            for (layer_idx, bl) in b_attn.iter().enumerate() {
                assert_eq!(
                    bl.data.len(),
                    bl.num_heads * 64 * 64,
                    "attention buffer size != num_heads*64*64 at position {} layer {}",
                    i,
                    layer_idx
                );
            }

            // WDL values should match between the two batched paths (same
            // forward pass at the model level, just with/without attention
            // capture).
            for (field, bv, sv) in [
                ("win", b_pred.wdl.win_prob, s_pred.wdl.win_prob),
                ("draw", b_pred.wdl.draw_prob, s_pred.wdl.draw_prob),
                ("loss", b_pred.wdl.loss_prob, s_pred.wdl.loss_prob),
            ] {
                assert!(
                    (bv - sv).abs() < 1e-4,
                    "wdl.{} mismatch at position {}: batched={}, sequential={}",
                    field,
                    i,
                    bv,
                    sv,
                );
            }
        }
    }

    #[test]
    fn test_momentum_features_snapshot() {
        let device = Device::<InferenceBackend>::default();
        let config = Config::default();
        let _ = crate::config::set_global_config(config.clone());

        // Test line 1: e4 c5 d4 cxd4 c3 dxc3 (material changes with pawn captures)
        let san_moves_1 = ["e4", "c5", "d4", "cxd4", "c3", "dxc3"];
        let positions_1 = san_line_to_positions(&san_moves_1).unwrap();

        // Compute material imbalance history manually for line 1
        let material_history_1: Vec<i32> = positions_1
            .iter()
            .map(|pos| compute_material_imbalance(pos))
            .collect();

        let momentum_1 = compute_momentum_features(&material_history_1, 10, 0.1);

        // Test line 2: e4 c5 d4 cxd4 c3 dxc3 Nxc3 (same line but with knight recapture)
        let san_moves_2 = ["e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3"];
        let positions_2 = san_line_to_positions(&san_moves_2).unwrap();

        // Compute material imbalance history manually for line 2
        let material_history_2: Vec<i32> = positions_2
            .iter()
            .map(|pos| compute_material_imbalance(pos))
            .collect();

        let momentum_2 = compute_momentum_features(&material_history_2, 10, 0.1);

        // Create snapshot data
        let snapshot_data = format!(
            "Line 1 (e4 c5 d4 cxd4 c3 dxc3):\n\
             Positions: {}\n\
             Material History: {:?}\n\
             Momentum: {:.6}, Volatility: {:.6}\n\
             \n\
             Line 2 (e4 c5 d4 cxd4 c3 dxc3 Nxc3):\n\
             Positions: {}\n\
             Material History: {:?}\n\
             Momentum: {:.6}, Volatility: {:.6}",
            positions_1.len(),
            material_history_1,
            momentum_1.0,
            momentum_1.1,
            positions_2.len(),
            material_history_2,
            momentum_2.0,
            momentum_2.1
        );

        // Print for manual verification and potential snapshot assertion
        println!("\n=== Momentum Features Snapshot Test ===");
        println!("{}", snapshot_data);

        // Basic sanity checks
        assert!(!material_history_1.is_empty());
        assert!(!material_history_2.is_empty());
        assert!(
            material_history_2.len() > material_history_1.len(),
            "Line 2 should have more positions than line 1"
        );

        // Momentum should be in [0,1] range
        assert!(momentum_1.0 >= 0.0 && momentum_1.0 <= 1.0);
        assert!(momentum_1.1 >= 0.0 && momentum_1.1 <= 1.0);
        assert!(momentum_2.0 >= 0.0 && momentum_2.0 <= 1.0);
        assert!(momentum_2.1 >= 0.0 && momentum_2.1 <= 1.0);

        // Check specific expectations about the material changes
        // Line 1 ends with white down 2 pawns after dxc3 (d4 pawn was captured, c3 pawn was captured)
        assert_eq!(
            material_history_1.last(),
            Some(&-2),
            "Line 1 should end with -2 material (2 pawns down)"
        );
        // Line 2 should end with white down 1 after Nxc3 recaptures (knight for pawn)
        assert_eq!(
            material_history_2.last(),
            Some(&-1),
            "Line 2 should end with -1 material after Nxc3"
        );

        // The momentum should be different between the lines due to the recapture
        assert_ne!(
            momentum_1, momentum_2,
            "Momentum should differ between the two lines"
        );

        // Detailed analysis of the material changes
        println!("\nDetailed Analysis:");
        println!("Line 1 material changes: {:?}", material_history_1);
        println!("Line 2 material changes: {:?}", material_history_2);
        println!(
            "Line 1 deltas: {:?}",
            material_history_1
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect::<Vec<_>>()
        );
        println!(
            "Line 2 deltas: {:?}",
            material_history_2
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect::<Vec<_>>()
        );

        // Snapshot test for reproducibility
        let snapshot = format!(
            "Line 1 (e4 c5 d4 cxd4 c3 dxc3): {:?} -> Momentum: {:.6}, Volatility: {:.6}\n\
             Line 2 (e4 c5 d4 cxd4 c3 dxc3 Nxc3): {:?} -> Momentum: {:.6}, Volatility: {:.6}",
            material_history_1,
            momentum_1.0,
            momentum_1.1,
            material_history_2,
            momentum_2.0,
            momentum_2.1
        );

        // This serves as a regression test - if the momentum calculation changes,
        // this assertion will fail and alert us to verify the changes are intentional
        let expected_snapshot = "Line 1 (e4 c5 d4 cxd4 c3 dxc3): [0, 0, 0, 0, -1, -1, -2] -> Momentum: 0.490950, Volatility: 0.048562\n\
                                Line 2 (e4 c5 d4 cxd4 c3 dxc3 Nxc3): [0, 0, 0, 0, -1, -1, -2, -1] -> Momentum: 0.496855, Volatility: 0.046398";

        assert_eq!(
            snapshot, expected_snapshot,
            "Momentum calculation snapshot changed - verify this is intentional"
        );
    }
}
