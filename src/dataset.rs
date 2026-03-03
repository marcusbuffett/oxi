use bincode::{Decode, Encode};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, Chess, Move, Position, Square};
use std::path::Path;

use crate::config::{get_global_config, ModelConfig, FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS};
use crate::encoding::encode_position;
use crate::inference::{compute_material_imbalance, GlobalFeatures};
use crate::move_encoding::encode_move;
use crate::moves::get_side_info;
use crate::pgn_processor::{process_pgn_directory_with_limit, process_pgn_file_with_limit};
// use shakmaty::Board;

/// Serializable representation of a chess move
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SerializableMove {
    pub from: Option<u8>, // Square index (0-63)
    pub to: u8,           // Square index (0-63)
}

impl SerializableMove {
    pub fn from_move(chess_move: &Move) -> Self {
        Self {
            from: chess_move.from().map(|sq| sq as u8),
            to: chess_move.to() as u8,
        }
    }

    pub fn to_move(&self) -> Option<Move> {
        let to_square = Square::new(self.to as u32);
        match self.from {
            Some(from_idx) => {
                let from_square = Square::new(from_idx as u32);
                Some(Move::Normal {
                    from: from_square,
                    to: to_square,
                    role: shakmaty::Role::Pawn, // Default role
                    capture: None,
                    promotion: None,
                })
            }
            None => Some(Move::Put {
                role: shakmaty::Role::Pawn, // Default, won't be used in our case
                to: to_square,
            }),
        }
    }
}

/// Training example with move probability distribution
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ChessExample {
    pub fen: String,
    pub move_uci: String,
    pub elo_self: i32,
    pub elo_oppo: i32,
    pub outcome: f32,
    pub previous_fens: Vec<String>,
    pub previous_moves: Vec<String>, // UCI moves that led to previous_fens positions
    // Time control data
    pub time_remaining_self: u32,          // seconds
    pub time_remaining_oppo: u32,          // seconds
    pub time_used_for_move: u32,           // seconds
    pub original_time_control: (u32, u32), // (base_time, increment)
    pub move_count: usize,
    /// Material imbalance history for momentum calculation
    pub material_imbalance_history: Vec<i32>,
    /// Whether this example comes from a puzzle (tactical training position)
    #[serde(default)]
    pub is_puzzle: bool,
}

/// Grouped dataset item - contains raw data with probability distributions
#[derive(Debug, Clone)]
pub struct ChessItem {
    pub board_encoded: Vec<f32>,     // 64 * FEATURES_PER_TOKEN
    pub move_distribution: Vec<f32>, // [4096] - probability distribution over moves
    pub legal_moves: Vec<f32>,       // [4096] - 64x64 from-to matrix
    pub side_info: Vec<i32>,         // [13] - grouped side info (needs thought)
    pub outcome: f32,
    pub material_imbalance: i32, // signed material difference (white - black)
    pub fen: String,             // The FEN position
    pub move_uci: String,
    pub elo_self: i32,
    pub elo_oppo: i32,
    pub time_used_for_move: u32,
    pub material_imbalance_history: Vec<i32>, // Material imbalance history for momentum
    // Global features (raw, unnormalized)
    pub global_features: GlobalFeatures,
    pub is_puzzle: bool,
}

/// Grouped dataset for Oxi training
pub struct OXIDataset {
    pub examples: Vec<ChessExample>,
    pub config: ModelConfig,
}

impl OXIDataset {
    pub fn new(examples: Vec<ChessExample>, config: ModelConfig) -> Self {
        Self { examples, config }
    }

    /// Load dataset from PGN file
    pub fn from_pgn_with_limit<P: AsRef<Path>>(
        path: P,
        config: ModelConfig,
        max_samples: Option<usize>,
    ) -> anyhow::Result<Self> {
        tracing::info!(
            "Loading PGN file: {:?} with limit {:?}",
            path.as_ref(),
            max_samples
        );
        let examples = process_pgn_file_with_limit(path.as_ref(), max_samples)?;
        tracing::info!("Loaded {} examples from PGN", examples.len());
        Ok(Self::new(examples, config))
    }

    /// Load dataset from directory of PGN files
    pub fn from_pgn_dir_with_limit<P: AsRef<Path>>(
        path: P,
        config: ModelConfig,
        max_samples: Option<usize>,
    ) -> anyhow::Result<Self> {
        tracing::info!(
            "Loading PGN directory: {:?} with limit {:?}",
            path.as_ref(),
            max_samples
        );
        let examples = process_pgn_directory_with_limit(path.as_ref(), max_samples)?;
        tracing::info!("Loaded {} total examples from directory", examples.len());
        Ok(Self::new(examples, config))
    }

    /// Process a single grouped example into raw data
    pub fn process_example(&self, example: &ChessExample) -> anyhow::Result<ChessItem> {
        let fen: Fen = example.fen.parse()?;
        let pos: Chess = fen
            .clone()
            .into_position(shakmaty::CastlingMode::Standard)?;
        let previous_positions = example
            .previous_fens
            .iter()
            .map(|f| f.parse().unwrap())
            .map(|f: Fen| f.into_position(shakmaty::CastlingMode::Standard).unwrap())
            .collect::<Vec<_>>();

        // Encode board position with previous moves
        let previous_moves_uci: Vec<String> = example.previous_moves.clone();
        let board_encoded = encode_position(&pos, &previous_positions, &previous_moves_uci);

        let mut move_distribution = vec![0.0; LEGAL_MOVES];

        let move_uci: UciMove = example.move_uci.parse().expect("move_uci invalid");
        let (from_idx, to_idx) = encode_move(&move_uci).unwrap();
        let move_idx = from_idx as usize * 76 + to_idx as usize;
        move_distribution[move_idx] = 1.0;

        // Normalize distribution (in case of rounding errors)
        let sum: f32 = move_distribution.iter().sum();
        if sum > 0.0 {
            for p in &mut move_distribution {
                *p /= sum;
            }
        }

        let side_info = get_side_info(&pos, &example.move_uci);
        let mut legal_moves = [0f32; LEGAL_MOVES];
        pos.legal_moves().iter().for_each(|m| {
            let (from, to) = encode_move(&m.to_uci(shakmaty::CastlingMode::Standard)).unwrap();
            let move_idx = from as usize * 76 + to as usize;
            legal_moves[move_idx] = 1.0;
        });

        let legal_move_count: f32 = legal_moves.iter().sum();
        assert!(
            legal_move_count > 0.0,
            "Position has no legal moves after encoding (fen={}, move={})",
            example.fen,
            example.move_uci
        );

        for i in 0..LEGAL_MOVES {
            if move_distribution[i] > 0.0 && legal_moves[i] == 0.0 {
                panic!("Illegal move in move_distribution");
            }
        }

        assert!(example.time_remaining_self > 0);
        assert!(example.time_remaining_oppo > 0);
        assert!(example.original_time_control.0 > 0);
        let material_imbalance = compute_material_imbalance(&pos);

        let global_features = GlobalFeatures {
            time_remaining_self: example.time_remaining_self,
            time_remaining_oppo: example.time_remaining_oppo,
            base_time: example.original_time_control.0,
            increment: example.original_time_control.1,
            move_count: example.move_count,
            elo_self: example.elo_self,
            is_puzzle: example.is_puzzle,
        };

        Ok(ChessItem {
            board_encoded,
            move_distribution,
            legal_moves: legal_moves.to_vec(),
            side_info,
            outcome: example.outcome,
            material_imbalance,
            fen: example.fen.clone(),
            move_uci: example.move_uci.clone(),
            elo_self: example.elo_self,
            elo_oppo: example.elo_oppo,
            time_used_for_move: example.time_used_for_move,
            material_imbalance_history: example.material_imbalance_history.clone(),
            global_features,
            is_puzzle: example.is_puzzle,
        })
    }

    /// Split dataset into train and validation sets
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let total_examples = self.examples.len();
        let train_size = (total_examples as f32 * train_ratio) as usize;

        let mut examples = self.examples;
        let validation_examples = examples.split_off(train_size);

        let train_dataset = Self::new(examples, self.config.clone());
        let valid_dataset = Self::new(validation_examples, self.config.clone());

        (train_dataset, valid_dataset)
    }
}

impl Dataset<ChessItem> for OXIDataset {
    fn get(&self, index: usize) -> Option<ChessItem> {
        let example = self.examples.get(index)?;
        self.process_example(example).ok()
    }

    fn len(&self) -> usize {
        self.examples.len()
    }
}

/// Batched grouped chess data
#[derive(Debug, Clone)]
pub struct ChessBatch<B: Backend> {
    pub board_input: Tensor<B, 3>, // [batch, 64, FEATURES_PER_TOKEN] (16 base + 4 for previous moves)
    pub move_distributions: Tensor<B, 2>, // [batch, 4096] - probability distributions
    pub time_usages: Tensor<B, 2>, // [batch, 1]
    pub legal_moves: Tensor<B, 2>, // [batch, 4096] - 64x64 from-to matrix
    pub side_info: Tensor<B, 2, Int>, // [batch, 141]
    pub values: Tensor<B, 2>,      // [batch, 3] - win, draw, loss probabilities
    pub fens: Vec<String>,         // FENs for each position in the batch
    pub global_features: Tensor<B, 2, Float>,
    pub value_weights: Tensor<B, 1>, // [batch] - per-sample weights for value loss (ply ramp + puzzle mask)
    // TODO: Remove these for less memory usage
    pub items: Vec<ChessItem>, // Original ChessItems for debugging/logging
}

impl<B: Backend> ChessBatch<B> {
    pub fn to_device(self, device: &Device<B>) -> Self {
        Self {
            board_input: self.board_input.to_device(device),
            move_distributions: self.move_distributions.to_device(device),
            time_usages: self.time_usages.to_device(device),
            legal_moves: self.legal_moves.to_device(device),
            side_info: self.side_info.to_device(device),
            values: self.values.to_device(device),
            fens: self.fens,
            global_features: self.global_features.to_device(device),
            value_weights: self.value_weights.to_device(device),
            items: self.items,
        }
    }
}

/// Batcher for creating batches from grouped chess items
#[derive(Debug, Clone)]
pub struct ChessBatcher<B: Backend> {
    _device: Device<B>,
}

impl<B: Backend> ChessBatcher<B> {
    pub fn new(device: Device<B>) -> Self {
        Self { _device: device }
    }
}

impl<B: Backend> Batcher<B, ChessItem, ChessBatch<B>> for ChessBatcher<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    fn batch(&self, items: Vec<ChessItem>, device: &Device<B>) -> ChessBatch<B> {
        let batch_size = items.len();
        let board_elem_count = 64 * FEATURES_PER_TOKEN;
        let side_info_len = if items.is_empty() {
            0
        } else {
            items[0].side_info.len()
        };

        let mut board_data = vec![0.0f32; batch_size * board_elem_count];
        let mut move_dist_data = vec![0.0f32; batch_size * LEGAL_MOVES];
        let mut legal_moves_data = vec![0.0f32; batch_size * LEGAL_MOVES];
        let mut side_info_data = vec![0i32; batch_size * side_info_len];
        let mut values_data = vec![0.0f32; batch_size * 3];
        let mut time_usages_data = vec![0.0f32; batch_size];
        let mut global_features_data = vec![0.0f32; batch_size * NUM_GLOBALS];
        let mut value_weights_data = vec![0.0f32; batch_size];
        let mut fens = Vec::with_capacity(batch_size);
        let config = get_global_config();

        for (i, item) in items.iter().enumerate() {
            let board_offset = i * board_elem_count;
            board_data[board_offset..board_offset + item.board_encoded.len()]
                .copy_from_slice(&item.board_encoded);

            let move_offset = i * LEGAL_MOVES;
            move_dist_data[move_offset..move_offset + LEGAL_MOVES]
                .copy_from_slice(&item.move_distribution);

            legal_moves_data[move_offset..move_offset + LEGAL_MOVES]
                .copy_from_slice(&item.legal_moves);

            let side_offset = i * side_info_len;
            side_info_data[side_offset..side_offset + side_info_len]
                .copy_from_slice(&item.side_info);

            let value_offset = i * 3;
            values_data[value_offset] = if item.outcome == 0.0 { 1.0 } else { 0.0 };
            values_data[value_offset + 1] = if item.outcome == 0.5 { 1.0 } else { 0.0 };
            values_data[value_offset + 2] = if item.outcome == 1.0 { 1.0 } else { 0.0 };

            time_usages_data[i] =
                item.time_used_for_move as f32 / item.global_features.time_remaining_self as f32;

            let global_offset = i * NUM_GLOBALS;
            let global_features_internal = item.global_features.to_internal(
                item.material_imbalance,
                item.material_imbalance_history.clone(),
            );
            let features = global_features_internal.to_feature_vector();
            global_features_data[global_offset..global_offset + NUM_GLOBALS]
                .copy_from_slice(&features);

            // Compute value weight: ply ramp * puzzle mask
            let ply = item.global_features.move_count;
            let ply_weight = config.value_ply_weight(ply);
            let puzzle_mask = if item.is_puzzle && !config.value_train_on_puzzles() {
                0.0
            } else {
                1.0
            };
            value_weights_data[i] = ply_weight * puzzle_mask;

            fens.push(item.fen.clone());
        }
        let board_input =
            Tensor::<B, 1>::from_data(TensorData::from(board_data.as_slice()), device).reshape([
                batch_size,
                64,
                FEATURES_PER_TOKEN,
            ]);

        let move_distributions =
            Tensor::<B, 1>::from_data(TensorData::from(move_dist_data.as_slice()), device)
                .reshape([batch_size, LEGAL_MOVES]);

        let legal_moves =
            Tensor::<B, 1>::from_data(TensorData::from(legal_moves_data.as_slice()), device)
                .reshape([batch_size, LEGAL_MOVES]);

        let side_info =
            Tensor::<B, 1, Int>::from_data(TensorData::from(side_info_data.as_slice()), device)
                .reshape([batch_size, side_info_len]);

        let values = Tensor::<B, 1>::from_data(TensorData::from(values_data.as_slice()), device)
            .reshape([batch_size, 3]);

        let time_usages =
            Tensor::<B, 1>::from_data(TensorData::from(time_usages_data.as_slice()), device)
                .reshape([batch_size, 1]);

        let global_features =
            Tensor::<B, 1>::from_data(TensorData::from(global_features_data.as_slice()), device)
                .reshape([batch_size, NUM_GLOBALS]);

        let value_weights =
            Tensor::<B, 1>::from_data(TensorData::from(value_weights_data.as_slice()), device);

        ChessBatch {
            board_input,
            move_distributions,
            time_usages,
            legal_moves,
            side_info,
            values,
            fens,
            items,
            global_features,
            value_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
}
