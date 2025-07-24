use burn::prelude::*;
use std::path::Path;

use crate::config::ModelConfig;
use crate::model::OXIModel;
use crate::training::load_model;

// Inference engine for Oxi
pub struct InferenceEngine<B: Backend> {
    _model: OXIModel<B>,
    _config: ModelConfig,
    _device: Device<B>,
}

/// Prediction result with move probabilities
#[derive(Debug, Clone)]
pub struct MovePrediction {
    pub uci_move: String,
    pub probability: f32,
    pub win_prob: f32,
    pub draw_prob: f32,
    pub loss_prob: f32,
}

impl<B: Backend> InferenceEngine<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    /// Create new inference engine from checkpoint
    pub fn from_checkpoint(
        checkpoint_path: &Path,
        config: ModelConfig,
        device: Device<B>,
    ) -> anyhow::Result<Self> {
        let model = load_model(checkpoint_path, &config, &device)?;
        Ok(Self {
            _model: model,
            _config: config,
            _device: device,
        })
    }

    /// Create inference engine with existing model
    pub fn new(model: OXIModel<B>, config: ModelConfig, device: Device<B>) -> Self {
        Self {
            _model: model,
            _config: config,
            _device: device,
        }
    }

    // Predict best move for a position
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

    // Analyze a position and return detailed information
    // pub fn analyze(
    //     &self,
    //     fen: &str,
    //     elo_self: i32,
    //     elo_oppo: i32,
    // ) -> anyhow::Result<PositionAnalysis> {
    //     let parsed_fen: Fen = fen.parse()?;
    //     let pos: Chess = parsed_fen.into_position(shakmaty::CastlingMode::Standard)?;
    //
    //     // Get predictions
    //     let predictions = self.predict(fen, elo_self, elo_oppo, 1.0, 10)?;
    //
    //     // Get side info for top move
    //     let top_move_uci = predictions
    //         .first()
    //         .ok_or_else(|| anyhow::anyhow!("No legal moves"))?
    //         .uci_move
    //         .clone();
    //     let top_move = predictions.first().unwrap();
    //     let win_prob = top_move.win_prob;
    //     let draw_prob = top_move.draw_prob;
    //     let loss_prob = top_move.loss_prob;
    //
    //     let (_, side_info_vec) = get_side_info(&pos, &top_move_uci);
    //
    //     Ok(PositionAnalysis {
    //         fen: fen.to_string(),
    //         top_moves: predictions,
    //         win_prob,
    //         draw_prob,
    //         loss_prob,
    //         side_info: side_info_vec,
    //     })
    // }
}

// Detailed position analysis
#[derive(Debug, Clone)]
pub struct PositionAnalysis {
    pub fen: String,
    pub top_moves: Vec<MovePrediction>,
    pub win_prob: f32,
    pub draw_prob: f32,
    pub loss_prob: f32,
    pub side_info: Vec<i32>,
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use burn_ndarray::NdArray;
//
//     #[test]
//     fn test_inference_engine_creation() {
//         let device = Default::default();
//         let config = ModelConfig::default();
//         let model = OXIModel::<NdArray>::new(&device, &config);
//
//         let engine = InferenceEngine::new(model, config, device);
//
//         // Test prediction on starting position
//         let result = engine.predict(
//             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
//             1500,
//             1500,
//             1.0,
//             5,
//         );
//
//         assert!(result.is_ok());
//         let predictions = result.unwrap();
//         assert!(!predictions.is_empty());
//         assert!(predictions.len() <= 5);
//     }
// }
