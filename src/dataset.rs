use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, Chess, Move, Position, Square};
use std::path::Path;

use crate::config::{get_global_config, ModelConfig, FEATURES_PER_TOKEN, LEGAL_MOVES, NUM_GLOBALS};
use crate::encoding::encode_position;
use crate::move_encoding::encode_move;
use crate::moves::get_side_info;
use crate::pgn_processor::{process_pgn_directory_with_limit, process_pgn_file_with_limit};
// use shakmaty::Board;

/// Compute global features from time control data
fn compute_global_features(item: &ChessItem) -> Vec<f32> {
    let (base_time, _increment) = item.original_time_control;

    // Material imbalance: difference in total material (white - black) normalized to [0,1]
    // Max absolute imbalance ~15
    let material_imbalance_norm = ((item.material_imbalance as f32) / 15.0).clamp(-1.0, 1.0) * 0.5
        + 0.5;

    let globals = vec![
        (item.time_remaining_self as f32 / 1500.0).clamp(0.0, 1.0),
        (item.time_remaining_self as f32 / base_time as f32).clamp(0.0, 1.0),
        (item.time_remaining_oppo as f32 / 1500.0).clamp(0.0, 1.0),
        (item.time_remaining_oppo as f32 / base_time as f32).clamp(0.0, 1.0),
        (item.move_count as f32 / 300.0).clamp(0.0, 1.0),
        item.elo_self_normalized, // already [0,1]
        material_imbalance_norm,
    ];
    assert_eq!(globals.len(), NUM_GLOBALS);
    globals
}

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessExample {
    pub fen: String,
    pub move_uci: String,
    pub elo_self: i32,
    pub elo_oppo: i32,
    pub outcome: f32,
    pub previous_fens: Vec<String>,
    // Time control data
    pub time_remaining_self: u32,          // seconds
    pub time_remaining_oppo: u32,          // seconds
    pub time_used_for_move: u32,           // seconds
    pub original_time_control: (u32, u32), // (base_time, increment)
    pub move_count: usize,
}

/// Grouped dataset item - contains raw data with probability distributions
#[derive(Debug, Clone)]
pub struct ChessItem {
    pub board_encoded: Vec<f32>,     // 64 * FEATURES_PER_TOKEN
    pub move_distribution: Vec<f32>, // [4096] - probability distribution over moves
    pub elo_self_normalized: f32,    // ELO bin index
    // pub elo_oppo_idx: usize,         // ELO bin index
    pub legal_moves: Vec<f32>, // [4096] - 64x64 from-to matrix
    pub side_info: Vec<i32>,   // [13] - grouped side info (needs thought)
    pub outcome: f32,
    pub material_imbalance: i32, // signed material difference (white - black)
    pub fen: String, // The FEN position
    pub move_uci: String,
    // Time control data (human-readable)
    pub time_remaining_self: u32,
    pub time_remaining_oppo: u32,
    pub time_used_for_move: u32,
    pub original_time_control: (u32, u32),
    pub move_count: usize,
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
        let board_encoded = encode_position(&pos, &previous_positions);

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

        let config = get_global_config();
        // Get ELO bin indices
        let elo_self_normalized =
            (example.elo_self as f32 / *config.elo_bins().last().unwrap() as f32).clamp(0.0, 1.0);
        // let elo_oppo_idx = get_elo_bin(example.elo_oppo, &self.config.elo_bins());

        let side_info = get_side_info(&pos, &example.move_uci);
        let mut legal_moves = [0f32; LEGAL_MOVES];
        pos.legal_moves().iter().for_each(|m| {
            let (from, to) = encode_move(&m.to_uci(shakmaty::CastlingMode::Standard)).unwrap();
            let move_idx = from as usize * 76 + to as usize;
            legal_moves[move_idx] = 1.0;
        });

        for i in 0..LEGAL_MOVES {
            if move_distribution[i] > 0.0 && legal_moves[i] == 0.0 {
                panic!("Illegal move in move_distribution");
            }
        }

        assert!(example.time_remaining_self > 0);
        assert!(example.time_remaining_oppo > 0);
        assert!(example.original_time_control.0 > 0);
        let material_imbalance = compute_material_imbalance(&pos);
        Ok(ChessItem {
            board_encoded,
            move_distribution,
            elo_self_normalized,
            // elo_oppo_idx,
            legal_moves: legal_moves.to_vec(),
            side_info,
            outcome: example.outcome,
            material_imbalance,
            fen: example.fen.clone(),
            move_uci: example.move_uci.clone(),
            time_remaining_self: example.time_remaining_self,
            time_remaining_oppo: example.time_remaining_oppo,
            time_used_for_move: example.time_used_for_move,
            original_time_control: example.original_time_control,
            move_count: example.move_count,
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
    // TODO: Remove these for less memory usage
    pub items: Vec<ChessItem>, // Original ChessItems for debugging/logging
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
        // Convert board inputs
        let board_inputs: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.board_encoded.as_slice()), device)
                    .reshape([1, 64, FEATURES_PER_TOKEN])
            })
            .collect();

        // Convert move distributions
        let move_distributions = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(
                    TensorData::from(item.move_distribution.as_slice()),
                    device,
                )
                .reshape([1, LEGAL_MOVES])
            })
            .collect::<Vec<_>>();

        // Convert legal moves
        let legal_moves = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.legal_moves.as_slice()), device)
                    .reshape([1, LEGAL_MOVES])
            })
            .collect::<Vec<_>>();

        // let illegal_moves = legal_moves
        //     .iter()
        //     .map(|m| {
        //         Tensor::<B, 1>::from_data(TensorData::from([1.0]), device).reshape([1, 4096]) - m
        //     })
        //     .collect::<Vec<_>>();

        // Convert side info
        let side_infos = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(TensorData::from(item.side_info.as_slice()), device)
                    .reshape([1, item.side_info.len()])
            })
            .collect::<Vec<_>>();

        // Convert values (win/draw/loss probabilities)
        let values = items
            .iter()
            .map(|item| {
                Tensor::<B, 2>::from_data(
                    TensorData::from([[
                        item.outcome == 0.0,
                        item.outcome == 0.5,
                        item.outcome == 1.0,
                    ]]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        let time_usages = items
            .iter()
            .map(|item| {
                Tensor::<B, 2>::from_data(
                    TensorData::from([[
                        item.time_used_for_move as f32 / item.time_remaining_self as f32
                    ]]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        // Collect FENs
        let fens = items.iter().map(|item| item.fen.clone()).collect();

        // Compute global features for each item
        let global_features = items
            .iter()
            .map(|item| {
                let features = compute_global_features(item);
                Tensor::<B, 1, Float>::from_data(TensorData::from(features.as_slice()), device)
                    .reshape([1, NUM_GLOBALS])
            })
            .collect::<Vec<_>>();

        ChessBatch {
            board_input: Tensor::cat(board_inputs, 0),
            move_distributions: Tensor::cat(move_distributions, 0),
            time_usages: Tensor::cat(time_usages, 0),
            legal_moves: Tensor::cat(legal_moves, 0),
            side_info: Tensor::cat(side_infos, 0),
            values: Tensor::cat(values, 0),
            fens,
            items,
            global_features: Tensor::cat(global_features, 0),
        }
    }
}

/// Compute signed material imbalance (white - black) using standard piece values
fn compute_material_imbalance(pos: &Chess) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
}
