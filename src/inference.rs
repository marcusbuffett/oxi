use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::log_softmax;
use std::path::Path;

use crate::config::{Config, LEGAL_MOVES, NUM_GLOBALS, PREVIOUS_POSITIONS};
use crate::encoding::encode_position;
use crate::model::OXIModel;
use crate::move_encoding::{decode_move, encode_move};
use crate::moves::{mirror_fen, mirror_move};
use shakmaty::EnPassantMode;
use shakmaty::{fen::Fen, san::SanPlus, Chess, Color, Position, Square};

/// Inference backend: Metal on macOS, LibTorch otherwise
#[cfg(target_os = "macos")]
pub type InferenceBackend = burn::backend::Metal;

#[cfg(not(target_os = "macos"))]
pub type InferenceBackend = burn::backend::LibTorch<f32>;

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

/// Inference engine for Oxi
pub struct InferenceEngine<B: Backend> {
    model: OXIModel<B>,
    config: Config,
    device: Device<B>,
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

/// Reconstructed move and position history from SAN moves
#[derive(Debug, Clone)]
pub struct PositionHistory {
    pub positions: Vec<Chess>,
    pub uci_moves: Vec<String>,
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
    /// Move count in the game
    pub move_count: usize,
    /// Self Elo rating
    pub elo_self: i32,
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
    /// Material imbalance history for momentum calculation
    pub material_imbalance_history: Vec<i32>,
}

/// Normalized global features ready for model input
#[derive(Debug, Clone)]
pub struct GlobalFeaturesNormalized {
    pub time_self_normalized: f32,
    pub time_self_ratio: f32,
    pub time_oppo_normalized: f32,
    pub time_oppo_ratio: f32,
    pub move_count_normalized: f32,
    pub elo_normalized: f32,
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
            material_imbalance,
            major_piece_imbalance: 0,
            minor_piece_imbalance: 0,
            pawn_imbalance: 0,
            move_count: self.move_count,
            elo_self: self.elo_self,
            material_imbalance_history,
        }
    }
}

impl GlobalFeaturesInternal {
    /// Compute normalized global features
    pub fn to_normalized(&self) -> GlobalFeaturesNormalized {
        // Normalize Elo to [0, 1] range (similar to dataset processing)
        let elo_self_normalized = if self.elo_self >= 800 && self.elo_self <= 2800 {
            (self.elo_self - 800) as f32 / (2800 - 800) as f32
        } else {
            0.5 // Default for out of range
        };

        // Material imbalance: difference in total material (white - black) normalized to [0,1]
        // Max absolute imbalance ~15
        let material_imbalance_norm =
            ((self.material_imbalance as f32) / 15.0).clamp(-1.0, 1.0) * 0.5 + 0.5;

        // Compute momentum features (window=10, alpha=0.1 for EMA)
        let (momentum, volatility) =
            compute_momentum_features(&self.material_imbalance_history, 10, 0.1);

        GlobalFeaturesNormalized {
            time_self_normalized: (self.time_remaining_self as f32 / 1500.0).clamp(0.0, 1.0),
            time_self_ratio: (self.time_remaining_self as f32 / self.base_time as f32)
                .clamp(0.0, 1.0),
            time_oppo_normalized: (self.time_remaining_oppo as f32 / 1500.0).clamp(0.0, 1.0),
            time_oppo_ratio: (self.time_remaining_oppo as f32 / self.base_time as f32)
                .clamp(0.0, 1.0),
            move_count_normalized: (self.move_count as f32 / 300.0).clamp(0.0, 1.0),
            elo_normalized: elo_self_normalized,
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
            normalized.move_count_normalized,
            normalized.elo_normalized,
            // normalized.material_imbalance_normalized,
            // normalized.momentum,   // normalized momentum [0,1]
            // normalized.volatility, // normalized volatility [0,1]
        ];
        assert_eq!(globals.len(), NUM_GLOBALS);
        globals
    }
}

impl Default for GlobalFeatures {
    fn default() -> Self {
        Self {
            time_remaining_self: 1500, // 25 minutes
            time_remaining_oppo: 1500, // 25 minutes
            base_time: 1800,           // 30 minutes
            move_count: 20,            // Mid-game
            elo_self: 1500,            // Average rating
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

pub fn load_model<B: Backend>(
    path: &Path,
    config: &Config,
    device: &Device<B>,
) -> anyhow::Result<OXIModel<B>> {
    let record = CompactRecorder::new().load(path.to_path_buf(), device)?;

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
        Ok(Self {
            model,
            config,
            device,
        })
    }

    /// Create inference engine with existing model
    pub fn new(model: OXIModel<B>, config: Config, device: Device<B>) -> Self {
        Self {
            model,
            config,
            device,
        }
    }

    /// Create a simple inference engine with default device
    pub fn create_simple(
        model: OXIModel<InferenceBackend>,
        config: Config,
    ) -> InferenceEngine<InferenceBackend> {
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
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
        if positions.is_empty() {
            return Err(anyhow::anyhow!("No positions provided"));
        }

        // Take up to PREVIOUS_POSITIONS + 1 positions (most recent first)
        let max_positions = (PREVIOUS_POSITIONS + 1).min(positions.len());
        let relevant_positions = &positions[..max_positions];
        let moves_needed = max_positions.saturating_sub(1);
        let moves_available = previous_moves.len().min(moves_needed);
        let relevant_moves = &previous_moves[..moves_available];

        // Current position is the first (most recent)
        let current_position = &relevant_positions[0];

        // Check if we need to flip the board (when it's Black to move)
        let is_black_to_move = current_position.turn() == Color::Black;

        // Apply board flipping if needed
        let (flipped_current, mut flipped_previous, mut flipped_previous_moves) =
            if is_black_to_move {
                let flipped_current = self.flip_position(current_position)?;
                let flipped_previous: Result<Vec<Chess>, anyhow::Error> = relevant_positions[1..]
                    .iter()
                    .map(|pos| self.flip_position(pos))
                    .collect();
                let flipped_prev_moves = relevant_moves
                    .iter()
                    .map(|mv| mirror_move(mv))
                    .collect::<Vec<_>>();
                (flipped_current, flipped_previous?, flipped_prev_moves)
            } else {
                let previous_positions: Vec<Chess> =
                    relevant_positions[1..].iter().cloned().collect();
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

        // Encode position with previous positions and moves for history features
        let board_encoded =
            encode_position(&flipped_current, &flipped_previous, &flipped_previous_moves);

        // Convert to tensor [batch=1, seq=64, features]
        let board_tensor = Tensor::<B, 1>::from_floats(board_encoded.as_slice(), &self.device)
            .reshape([1, 64, board_encoded.len() / 64]);

        // Compute material imbalance from the current position and convert to internal representation
        let material_imbalance = compute_material_imbalance(&flipped_current);

        // Compute material imbalance history from previous positions
        let material_imbalance_history: Vec<i32> = relevant_positions
            .iter()
            .skip(1)
            .map(|pos| compute_material_imbalance(pos))
            .collect();

        let global_features_internal =
            global_features.to_internal(material_imbalance, material_imbalance_history.clone());

        // Compute global features from the internal representation
        let global_features_normalized = global_features_internal.to_normalized();
        tracing::info!(
            "Global features (normalized): {:?}",
            global_features_normalized
        );

        let global_feature_vector = global_features_internal.to_feature_vector();
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

    /// Predict best moves from positions in descending chronological order (most recent first)
    /// Takes up to PREVIOUS_POSITIONS + 1 positions for encoding with history
    /// Automatically handles board flipping for Black's perspective
    pub fn predict(
        &self,
        positions: &[Chess],
        global_features: &GlobalFeatures,
        temperature: f32,
        top_k: usize,
    ) -> anyhow::Result<OxiPrediction>
    where
        B::FloatElem: From<f32>,
    {
        self.predict_with_history(positions, &[], global_features, temperature, top_k)
    }

    pub fn predict_with_history(
        &self,
        positions: &[Chess],
        previous_moves: &[String],
        global_features: &GlobalFeatures,
        temperature: f32,
        top_k: usize,
    ) -> anyhow::Result<OxiPrediction>
    where
        B::FloatElem: From<f32>,
    {
        let (board_tensor, global_features_tensor, flipped_current, _material_imbalance_history) =
            self.create_input_tensors(positions, previous_moves, global_features)?;

        let current_position = &positions[0];
        let is_black_to_move = current_position.turn() == Color::Black;

        // Forward pass
        let (policy_logits, value_logits, side_info_logits, time_usage_logits) =
            self.model.forward(board_tensor, global_features_tensor);

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

        // Apply temperature and log_softmax (consistent with training)
        let log_probs = if temperature != 1.0 {
            log_softmax(masked_logits.div_scalar(temperature), 1)
        } else {
            log_softmax(masked_logits, 1)
        };

        // Convert to probabilities for move selection
        let probs = log_probs.exp();

        // Extract top k moves
        let probs_data = probs.to_data();
        let probs_slice = probs_data.as_slice::<f32>().unwrap();
        let mut move_probs: Vec<(usize, f32)> = probs_slice[0..LEGAL_MOVES]
            .iter()
            .enumerate()
            .filter(|(_, &prob)| prob > 0.0 && !prob.is_infinite())
            .map(|(i, &p)| (i, p))
            .collect();

        move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let move_predictions: Vec<MovePrediction> = move_probs
            .into_iter()
            .take(top_k)
            .filter_map(|(move_idx, prob)| {
                let from_idx = (move_idx / 76) as u8;
                let to_idx = (move_idx % 76) as u8;
                decode_move(from_idx, to_idx).map(|uci_move| {
                    let final_uci = if is_black_to_move {
                        // Flip the move back to original coordinates
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

        // Get WDL predictions (consistent with training using log_softmax)
        let value_log_probs = log_softmax(value_logits.clone(), 1);
        let value_probs = value_log_probs.exp();
        let value_data = value_probs.to_data();
        let value_slice = value_data.as_slice::<f32>().unwrap();
        let wdl = WdlPrediction {
            loss_prob: value_slice[0], // Index 0 = loss
            draw_prob: value_slice[1], // Index 1 = draw
            win_prob: value_slice[2],  // Index 2 = win
        };

        // Get time usage predictions
        // NOTE: The model already maps raw logits into [min,max] via a sigmoid in model.rs, so outputs are parameters.
        // Do NOT re-apply activations here; just read alpha/beta directly.
        let params_data = time_usage_logits.to_data();
        let params_slice = params_data.as_slice::<f32>().unwrap();
        let alpha = params_slice[0];
        let beta = params_slice[1];
        let denom = alpha + beta;
        let expected_fraction = if denom > 0.0 { alpha / denom } else { 0.0 };
        let expected_seconds = expected_fraction * global_features.time_remaining_self as f32;

        let time_usage = TimeUsagePrediction {
            alpha,
            beta,
            expected_fraction,
            expected_seconds,
        };

        Ok(OxiPrediction {
            moves: move_predictions,
            wdl,
            time_usage,
        })
    }

    /// Helper function to flip a chess position for Black's perspective
    fn flip_position(&self, position: &Chess) -> anyhow::Result<Chess> {
        let fen_string = Fen::from_position(position, EnPassantMode::Legal).to_string();
        let mirrored_fen = mirror_fen(&fen_string);
        let flipped_position: Chess = mirrored_fen
            .parse::<Fen>()?
            .into_position(shakmaty::CastlingMode::Standard)?;
        Ok(flipped_position)
    }

    // Predict best move for a position (legacy method, kept for compatibility)
    // pub fn predict(
    //     &self,
    //     fen: &str,
    //     elo_self: i32,
    //     elo_oppo: i32,
    //     temperature: f32,
    //     top_k: usize,
    // ) -> anyhow::Result<Vec<MovePrediction>> {
    //     let parsed_fen: Fen = fen.parse()?;
    //     let pos: Chess = parsed_fen.into_position(shakmaty::CastlingMode::Standard)?;
    //
    //     // Encode position (no previous moves available in inference)
    //     let board_encoded = encode_position_with_previous_moves(&pos, &[]);
    //     // Convert to tensor
    //     let data = TensorData::from(board_encoded.as_slice());
    //     let tensor: Tensor<B, 1> = Tensor::from_data(data.convert::<B::FloatElem>(), &self.device);
    //     let board_encoded = tensor.reshape([20, 8, 8]).unsqueeze();
    //
    //     // Get Elo bins
    //     let elo_self_idx = get_elo_bin(elo_self, &self.config.elo_bins());
    //     let elo_oppo_idx = get_elo_bin(elo_oppo, &self.config.elo_bins());
    //
    //     // Create tensors
    //     let elo_self_data = TensorData::from([(elo_self_idx as i64).elem::<B::IntElem>()]);
    //     let elo_self_tensor_1d: Tensor<B, 1, Int> = Tensor::from_data(elo_self_data, &self.device);
    //     let elo_self_tensor = elo_self_tensor_1d.reshape([1, 1]);
    //     let elo_oppo_data = TensorData::from([(elo_oppo_idx as i64).elem::<B::IntElem>()]);
    //     let elo_oppo_tensor_1d: Tensor<B, 1, Int> = Tensor::from_data(elo_oppo_data, &self.device);
    //     let elo_oppo_tensor = elo_oppo_tensor_1d.reshape([1, 1]);
    //
    //     // Forward pass
    //     let (policy_logits, value_logits, _side_info) =
    //         self.model
    //             .forward(board_encoded, elo_self_tensor, elo_oppo_tensor);
    //
    //     // Get legal moves mask for 64x64 representation
    //     let mut legal_moves_mask = vec![0f32; 4096;
    //     for legal_move in pos.legal_moves() {
    //         let uci = legal_move
    //             .to_uci(shakmaty::CastlingMode::Standard)
    //             .to_string();
    //         if let Some((from_idx, to_idx)) = encode_move_az(&uci) {
    //             let flat_idx = from_idx * 73 + to_idx;
    //             legal_moves_mask[flat_idx] = 1.0;
    //         }
    //     }
    //
    //     let legal_moves_data = TensorData::from(legal_moves_mask.as_slice());
    //     let legal_moves_tensor_1d: Tensor<B, 1> =
    //         Tensor::from_data(legal_moves_data.convert::<B::FloatElem>(), &self.device);
    //     let legal_moves_tensor = legal_moves_tensor_1d.reshape([1, 4096]);
    //
    //     // Apply legal move masking
    //     let masked_logits = policy_logits + (legal_moves_tensor - 1.0) * 1e9;
    //
    //     // Apply temperature and softmax
    //     let probs = if temperature != 1.0 {
    //         activation::softmax(masked_logits / temperature, 1)
    //     } else {
    //         activation::softmax(masked_logits, 1)
    //     };
    //
    //     // Get value predictions (win/draw/loss probabilities)
    //     let value_probs = activation::softmax(value_logits, 1);
    //     let value_data = value_probs.squeeze::<1>(0).into_data();
    //     let value_slice = value_data.as_slice::<f32>().unwrap();
    //     let win_prob = value_slice[0];
    //     let draw_prob = value_slice[1];
    //     let loss_prob = value_slice[2];
    //
    //     // Extract probabilities and get top k
    //     let probs_tensor_data = probs.squeeze::<1>(0).into_data();
    //     let probs_data = probs_tensor_data.as_slice::<f32>().unwrap().to_vec();
    //     let mut move_probs: Vec<(usize, f32)> = probs_data
    //         .into_iter()
    //         .enumerate()
    //         .filter(|(idx, prob)| *prob > 0.0)
    //         .collect();
    //
    //     // Sort by probability
    //     move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    //
    //     // Take top k
    //     let predictions: Vec<MovePrediction> = move_probs
    //         .into_iter()
    //         .take(top_k)
    //         .map(|(idx, prob)| {
    //             // Convert flat index back to from-to indices
    //             let from_idx = idx / 64;
    //             let to_idx = idx % 64;
    //             let uci_move = decode_move_spatial(from_idx, to_idx);
    //             MovePrediction {
    //                 uci_move,
    //                 probability: prob,
    //                 win_prob,
    //                 draw_prob,
    //                 loss_prob,
    //             }
    //         })
    //         .collect();
    //
    //     Ok(predictions)
    // }

    // Batch prediction for multiple positions
    // pub fn predict_batch(
    //     &self,
    //     positions: &[(String, i32, i32)], // (fen, elo_self, elo_oppo)
    //     temperature: f32,
    //     top_k: usize,
    // ) -> anyhow::Result<Vec<Vec<MovePrediction>>> {
    //     let mut all_boards = Vec::new();
    //     let mut all_elo_self = Vec::new();
    //     let mut all_elo_oppo = Vec::new();
    //     let mut all_legal_masks = Vec::new();
    //
    //     // Process all positions
    //     for (fen, elo_self, elo_oppo) in positions {
    //         let parsed_fen: Fen = fen.parse()?;
    //         let pos: Chess = parsed_fen.into_position(shakmaty::CastlingMode::Standard)?;
    //
    //         // Encode board (no previous moves available in batch inference)
    //         let board_encoded = encode_position_with_previous_moves(&pos, &[]);
    //         // Convert to tensor
    //         let data = TensorData::from(board_encoded.as_slice());
    //         let tensor: Tensor<B, 1> =
    //             Tensor::from_data(data.convert::<B::FloatElem>(), &self.device);
    //         let board = tensor.reshape([20, 8, 8]).unsqueeze();
    //         all_boards.push(board);
    //
    //         // Elo bins
    //         let elo_self_idx = get_elo_bin(*elo_self, &self.config.elo_bins());
    //         let elo_oppo_idx = get_elo_bin(*elo_oppo, &self.config.elo_bins());
    //
    //         all_elo_self.push(elo_self_idx as i32);
    //         all_elo_oppo.push(elo_oppo_idx as i32);
    //
    //         // Legal moves mask
    //         let mut legal_mask = vec![0f32; 4096;
    //         for legal_move in pos.legal_moves() {
    //             let uci = legal_move
    //                 .to_uci(shakmaty::CastlingMode::Standard)
    //                 .to_string();
    //             if let Some((from_idx, to_idx)) = encode_move_az(&uci) {
    //                 let flat_idx = from_idx * 73 + to_idx;
    //                 legal_mask[flat_idx] = 1.0;
    //             }
    //         }
    //         all_legal_masks.push(legal_mask);
    //     }
    //
    //     let batch_size = positions.len();
    //
    //     // Stack tensors
    //     let boards = Tensor::cat(all_boards, 0);
    //     let elo_self_data: Vec<_> = all_elo_self
    //         .into_iter()
    //         .map(|x| (x as i64).elem::<B::IntElem>())
    //         .collect();
    //     let elo_oppo_data: Vec<_> = all_elo_oppo
    //         .into_iter()
    //         .map(|x| (x as i64).elem::<B::IntElem>())
    //         .collect();
    //
    //     let elo_self_tensor_data = TensorData::from(elo_self_data.as_slice());
    //     let elo_self_tensor_1d: Tensor<B, 1, Int> =
    //         Tensor::from_data(elo_self_tensor_data, &self.device);
    //     let elo_self_tensor = elo_self_tensor_1d.reshape([batch_size, 1]);
    //     let elo_oppo_tensor_data = TensorData::from(elo_oppo_data.as_slice());
    //     let elo_oppo_tensor_1d: Tensor<B, 1, Int> =
    //         Tensor::from_data(elo_oppo_tensor_data, &self.device);
    //     let elo_oppo_tensor = elo_oppo_tensor_1d.reshape([batch_size, 1]);
    //
    //     // Forward pass
    //     let (policy_logits, value_logits, _) =
    //         self.model.forward(boards, elo_self_tensor, elo_oppo_tensor);
    //
    //     // Process each position's results
    //     let mut results = Vec::new();
    //     // Get value predictions for all positions in the batch
    //     let value_probs = activation::softmax(value_logits, 1);
    //     let values_data = value_probs.into_data();
    //     let values_slice = values_data.as_slice::<f32>().unwrap();
    //
    //     // Extract win/draw/loss probabilities for each position
    //     let mut all_value_probs = Vec::new();
    //     for i in 0..batch_size {
    //         let start_idx = i * 3;
    //         all_value_probs.push((
    //             values_slice[start_idx],     // win
    //             values_slice[start_idx + 1], // draw
    //             values_slice[start_idx + 2], // loss
    //         ));
    //     }
    //
    //     for (i, legal_mask) in all_legal_masks.into_iter().enumerate() {
    //         // Get this position's logits
    //         let pos_logits = policy_logits.clone().slice([i..i + 1, 0..4096]);
    //
    //         // Apply legal mask
    //         let mask_data = TensorData::from(legal_mask.as_slice());
    //         let mask_tensor_1d: Tensor<B, 1> =
    //             Tensor::from_data(mask_data.convert::<B::FloatElem>(), &self.device);
    //         let mask_tensor = mask_tensor_1d.reshape([1, 4096]);
    //
    //         let masked = pos_logits + (mask_tensor - 1.0) * 1e9;
    //
    //         // Apply temperature and softmax
    //         let probs = if temperature != 1.0 {
    //             activation::softmax(masked / temperature, 1)
    //         } else {
    //             activation::softmax(masked, 1)
    //         };
    //
    //         // Extract top k moves
    //         let probs_tensor_data = probs.squeeze::<1>(0).into_data();
    //         let probs_data = probs_tensor_data.as_slice::<f32>().unwrap().to_vec();
    //         let mut move_probs: Vec<(usize, f32)> = probs_data
    //             .into_iter()
    //             .enumerate()
    //             .filter(|(idx, prob)| *prob > 0.0)
    //             .collect();
    //
    //         move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    //
    //         let predictions: Vec<MovePrediction> = move_probs
    //             .into_iter()
    //             .take(top_k)
    //             .map(|(idx, prob)| {
    //                 // Convert flat index back to from-to indices
    //                 let from_idx = idx / 64;
    //                 let to_idx = idx % 64;
    //                 let uci_move = decode_move_spatial(from_idx, to_idx);
    //                 let (win_prob, draw_prob, loss_prob) = all_value_probs[i];
    //                 MovePrediction {
    //                     uci_move,
    //                     probability: prob,
    //                     win_prob,
    //                     draw_prob,
    //                     loss_prob,
    //                 }
    //             })
    //             .collect();
    //
    //         results.push(predictions);
    //     }
    //
    //     Ok(results)
    // }

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

        let prediction = self.predict(&[pos], global_features, temperature, top_k)?;

        Ok(PositionAnalysis {
            fen: fen.to_string(),
            prediction,
        })
    }

    // Analyze a position and return detailed information (legacy method)
    // pub fn analyze_legacy(
    //     &self,
    //     fen: &str,
    //     elo_self: i32,
    //     elo_oppo: i32,
    // ) -> anyhow::Result<PositionAnalysis> {
    //     let parsed_fen: Fen = fen.parse()?;
    //     let pos: Chess = parsed_fen.into_position(shakmaty::CastlingMode::Standard)?;
    //
    //     // Get predictions
    //     let prediction = self.predict(fen, 1.0, 10)?;
    //
    //     // Get side info for top move (implement proper side info extraction later)
    //     let side_info = vec![0; 13];
    //
    //     Ok(PositionAnalysis {
    //         fen: fen.to_string(),
    //         prediction,
    //         side_info,
    //     })
    // }
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
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
        let config = Config::default();
        crate::config::set_global_config(config.clone()).unwrap();
        let model = OXIModel::<InferenceBackend>::new(&device, &config);

        let engine = InferenceEngine::new(model, config, device);

        // Test prediction on starting position
        let starting_pos = Chess::default();
        let globals = GlobalFeatures::default();
        let result = engine.predict(&[starting_pos], &globals, 1.0, 5);

        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert!(!prediction.moves.is_empty());
        assert!(prediction.moves.len() <= 5);

        // Check WDL probabilities sum to ~1
        let wdl_sum = prediction.wdl.win_prob + prediction.wdl.draw_prob + prediction.wdl.loss_prob;
        assert!((wdl_sum - 1.0).abs() < 0.01);

        // Check time usage parameters are positive
        assert!(prediction.time_usage.alpha > 0.0);
        assert!(prediction.time_usage.beta > 0.0);
        assert!(prediction.time_usage.expected_fraction >= 0.0);
    }

    #[test]
    fn test_analyze_position() {
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
        let config = Config::default();
        crate::config::set_global_config(config.clone()).unwrap();
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
    fn test_momentum_features_snapshot() {
        let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
        let config = Config::default();
        crate::config::set_global_config(config.clone()).unwrap();

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
