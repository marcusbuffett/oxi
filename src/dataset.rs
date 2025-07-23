use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use shakmaty::{fen::Fen, Chess, Color, Position};
use std::collections::HashMap;
use std::path::Path;

use crate::config::ModelConfig;
use crate::encoding::{encode_position, get_elo_bin};
use crate::moves::{encode_move_spatial, get_side_info, mirror_move};
use crate::pgn_processor::{process_pgn_directory_with_limit, process_pgn_file_with_limit};

/// Raw training example from PGN files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawChessExample {
    /// FEN string of the position
    pub fen: String,
    /// UCI move that was played
    pub move_uci: String,
    /// ELO rating of the player making the move
    pub elo_self: i32,
    /// ELO rating of the opponent
    pub elo_oppo: i32,
    /// Whether the active player won the game (legacy field for backward compatibility)
    pub active_won: i32,
    /// Win probability for the active player (optional, computed from active_won if missing)
    pub win_prob: Option<f32>,
    /// Draw probability for the position (optional, computed from active_won if missing)
    pub draw_prob: Option<f32>,
    /// Loss probability for the active player (optional, computed from active_won if missing)
    pub loss_prob: Option<f32>,
}

/// Training example with move probability distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessExample {
    /// FEN string of the position
    pub fen: String,
    /// Move probabilities as a map from UCI move to probability
    pub move_probabilities: HashMap<String, f32>,
    /// Weighted average ELO rating of players making moves
    pub avg_elo_self: f32,
    /// Weighted average ELO rating of opponents
    pub avg_elo_oppo: f32,
    /// Weighted average outcome (proportion of games won) - legacy field
    pub avg_outcome: f32,
    /// Win probability for the active player
    pub win_prob: f32,
    /// Draw probability
    pub draw_prob: f32,
    /// Loss probability for the active player
    pub loss_prob: f32,
    /// Total count of examples grouped
    pub count: usize,
}

/// Grouped dataset item - contains raw data with probability distributions
#[derive(Debug, Clone)]
pub struct ChessItem {
    pub board_encoded: Vec<f32>,     // Flattened [16, 8, 8]
    pub move_distribution: Vec<f32>, // [4096] - probability distribution over moves
    pub elo_self_idx: usize,         // ELO bin index
    pub elo_oppo_idx: usize,         // ELO bin index
    pub legal_moves: Vec<f32>,       // [4096] - 64x64 from-to matrix
    pub side_info: Vec<i32>,         // [13] - grouped side info (needs thought)
    pub win_prob: f32,               // Win probability for active player
    pub draw_prob: f32,              // Draw probability
    pub loss_prob: f32,              // Loss probability for active player
    pub fen: String,                 // The FEN position
}

/// Grouped dataset for Oxi training
pub struct OXIDataset {
    examples: Vec<ChessExample>,
    config: ModelConfig,
}

impl OXIDataset {
    pub fn new(examples: Vec<ChessExample>, config: ModelConfig) -> Self {
        Self { examples, config }
    }

    /// Create dataset from regular examples
    pub fn from_examples(examples: Vec<RawChessExample>, config: ModelConfig) -> Self {
        let total_examples = examples.len();

        // Group examples by position (FEN)
        let mut position_map: HashMap<String, Vec<RawChessExample>> = HashMap::new();

        for example in examples {
            position_map
                .entry(example.fen.clone())
                .or_insert_with(Vec::new)
                .push(example);
        }

        // Convert to grouped examples
        let grouped_examples: Vec<ChessExample> = position_map
            .into_iter()
            .map(|(fen, examples)| {
                let count = examples.len() as f32;

                // Count moves and calculate probabilities
                let mut move_counts: HashMap<String, usize> = HashMap::new();
                let mut total_elo_self = 0.0;
                let mut total_elo_oppo = 0.0;
                let mut total_outcome = 0.0;
                let mut total_wins = 0.0;
                let mut total_draws = 0.0;
                let mut total_losses = 0.0;

                for example in &examples {
                    *move_counts.entry(example.move_uci.clone()).or_insert(0) += 1;
                    total_elo_self += example.elo_self as f32;
                    total_elo_oppo += example.elo_oppo as f32;
                    total_outcome += example.active_won as f32;

                    // If the new fields are available, use them
                    if example.win_prob.is_some()
                        && example.draw_prob.is_some()
                        && example.loss_prob.is_some()
                    {
                        total_wins += example.win_prob.unwrap();
                        total_draws += example.draw_prob.unwrap();
                        total_losses += example.loss_prob.unwrap();
                    } else {
                        // Legacy mode: use active_won to estimate win/loss (no draws)
                        match example.active_won {
                            1 => total_wins += 1.0,
                            0 => total_losses += 1.0,
                            _ => total_draws += 1.0,
                        }
                    }
                }

                // Convert counts to probabilities
                let move_probabilities: HashMap<String, f32> = move_counts
                    .into_iter()
                    .map(|(move_uci, move_count)| (move_uci, move_count as f32 / count))
                    .collect();

                ChessExample {
                    fen,
                    move_probabilities,
                    avg_elo_self: total_elo_self / count,
                    avg_elo_oppo: total_elo_oppo / count,
                    avg_outcome: total_outcome / count,
                    win_prob: total_wins / count,
                    draw_prob: total_draws / count,
                    loss_prob: total_losses / count,
                    count: examples.len(),
                }
            })
            .collect();

        tracing::info!(
            "Grouped {} examples into {} unique positions",
            total_examples,
            grouped_examples.len()
        );

        Self::new(grouped_examples, config)
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
        Ok(Self::from_examples(examples, config))
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
        Ok(Self::from_examples(examples, config))
    }

    /// Process a single grouped example into raw data
    pub fn process_example(&self, example: &ChessExample) -> anyhow::Result<ChessItem> {
        let fen: Fen = example.fen.parse()?;
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard)?;

        // Mirror position if black to play
        let (pos, mirrored) = if pos.turn() == Color::Black {
            let mirrored_fen: Fen = self.mirror_fen(&example.fen).parse()?;
            let mirrored_pos: Chess =
                mirrored_fen.into_position(shakmaty::CastlingMode::Standard)?;
            (mirrored_pos, true)
        } else {
            (pos, false)
        };

        // Encode board position
        let board_encoded = encode_position(&pos);

        // Create move probability distribution
        let mut move_distribution = vec![0.0; 4096];

        for (move_uci, prob) in &example.move_probabilities {
            let move_uci = if mirrored {
                mirror_move(move_uci)
            } else {
                move_uci.clone()
            };

            if let Some((from_idx, to_idx)) = encode_move_spatial(&move_uci) {
                let move_idx = from_idx * 64 + to_idx;
                move_distribution[move_idx] = *prob;
            }
        }

        // Normalize distribution (in case of rounding errors)
        let sum: f32 = move_distribution.iter().sum();
        if sum > 0.0 {
            for p in &mut move_distribution {
                *p /= sum;
            }
        }

        // Get ELO bin indices
        let elo_self_idx =
            get_elo_bin(example.avg_elo_self.round() as i32, &self.config.elo_bins());
        let elo_oppo_idx =
            get_elo_bin(example.avg_elo_oppo.round() as i32, &self.config.elo_bins());

        // Get legal moves mask
        // For side info, we'll use the most probable move for now
        let most_probable_move = example
            .move_probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(m, _)| if mirrored { mirror_move(m) } else { m.clone() })
            .unwrap_or_default();

        let (legal_moves, side_info) = get_side_info(&pos, &most_probable_move);

        Ok(ChessItem {
            board_encoded,
            move_distribution,
            elo_self_idx,
            elo_oppo_idx,
            legal_moves,
            side_info,
            win_prob: example.win_prob,
            draw_prob: example.draw_prob,
            loss_prob: example.loss_prob,
            fen: example.fen.clone(),
        })
    }

    /// Mirror FEN for black's perspective
    fn mirror_fen(&self, fen: &str) -> String {
        // Copy implementation from original dataset
        let parts: Vec<&str> = fen.split(' ').collect();
        if parts.is_empty() {
            return fen.to_string();
        }

        // Mirror the board position
        let board_parts: Vec<&str> = parts[0].split('/').collect();
        let mirrored_board: Vec<String> = board_parts
            .into_iter()
            .rev()
            .map(|rank| {
                rank.chars()
                    .map(|c| match c {
                        'P' => 'p',
                        'N' => 'n',
                        'B' => 'b',
                        'R' => 'r',
                        'Q' => 'q',
                        'K' => 'k',
                        'p' => 'P',
                        'n' => 'N',
                        'b' => 'B',
                        'r' => 'R',
                        'q' => 'Q',
                        'k' => 'K',
                        _ => c,
                    })
                    .collect()
            })
            .collect();

        let mut result = mirrored_board.join("/");

        // Flip turn
        if parts.len() > 1 {
            result.push(' ');
            result.push(if parts[1] == "w" { 'b' } else { 'w' });
        }

        // Mirror castling rights
        if parts.len() > 2 {
            result.push(' ');
            let castling = parts[2]
                .chars()
                .map(|c| match c {
                    'K' => 'k',
                    'Q' => 'q',
                    'k' => 'K',
                    'q' => 'Q',
                    _ => c,
                })
                .collect::<String>();
            result.push_str(&castling);
        }

        // En passant
        if parts.len() > 3 {
            result.push(' ');
            if parts[3] != "-" {
                let file = parts[3].chars().nth(0).unwrap();
                let rank = parts[3].chars().nth(1).unwrap();
                let new_rank = ((b'1' + b'8') - rank as u8) as char;
                result.push(file);
                result.push(new_rank);
            } else {
                result.push('-');
            }
        }

        // Halfmove and fullmove clocks
        for part in parts.iter().skip(4) {
            result.push(' ');
            result.push_str(part);
        }

        result
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
    pub board_input: Tensor<B, 4>,        // [batch, 16, 8, 8]
    pub move_distributions: Tensor<B, 2>, // [batch, 4096] - probability distributions
    pub elo_self: Tensor<B, 2, Int>,      // [batch, 1]
    pub elo_oppo: Tensor<B, 2, Int>,      // [batch, 1]
    pub legal_moves: Tensor<B, 2>,        // [batch, 4096] - 64x64 from-to matrix
    pub side_info: Tensor<B, 2, Int>,     // [batch, 141]
    pub values: Tensor<B, 2>,             // [batch, 3] - win, draw, loss probabilities
    pub fens: Vec<String>,                // FENs for each position in the batch
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
        let board_inputs: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.board_encoded.as_slice()), device)
                    .reshape([1, 16, 8, 8])
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
                .reshape([1, 4096])
            })
            .collect::<Vec<_>>();

        // Convert ELO indices
        let elo_selfs = items
            .iter()
            .map(|item| {
                Tensor::<B, 2, Int>::from_data(
                    TensorData::from([[item.elo_self_idx as i64]]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        let elo_oppos = items
            .iter()
            .map(|item| {
                Tensor::<B, 2, Int>::from_data(
                    TensorData::from([[item.elo_oppo_idx as i64]]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        // Convert legal moves
        let legal_moves = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.legal_moves.as_slice()), device)
                    .reshape([1, 4096])
            })
            .collect::<Vec<_>>();

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
                    TensorData::from([[item.win_prob, item.draw_prob, item.loss_prob]]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        // Collect FENs
        let fens = items.iter().map(|item| item.fen.clone()).collect();

        ChessBatch {
            board_input: Tensor::cat(board_inputs, 0),
            move_distributions: Tensor::cat(move_distributions, 0),
            elo_self: Tensor::cat(elo_selfs, 0),
            elo_oppo: Tensor::cat(elo_oppos, 0),
            legal_moves: Tensor::cat(legal_moves, 0),
            side_info: Tensor::cat(side_infos, 0),
            values: Tensor::cat(values, 0),
            fens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::moves::encode_move_spatial;

    #[test]
    fn test_move_aggregation() {
        // Load the test PGN file
        let pgn_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pgn/aggregation_test.pgn");

        let config = ModelConfig::default();
        let dataset =
            OXIDataset::from_pgn_with_limit(&pgn_path, config, None).expect("Failed to load PGN");

        // Find the starting position in the dataset
        let starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let starting_example = dataset
            .examples
            .iter()
            .find(|ex| ex.fen == starting_fen)
            .expect("Starting position not found");

        // Check that we have the correct move probabilities
        assert_eq!(starting_example.count, 3, "Should have grouped 3 games");
        assert_eq!(
            starting_example.move_probabilities.len(),
            2,
            "Should have 2 unique moves"
        );

        // e2e4 appears twice, d2d4 appears once
        let e4_prob = starting_example
            .move_probabilities
            .get("e2e4")
            .expect("e2e4 not found");
        let d4_prob = starting_example
            .move_probabilities
            .get("d2d4")
            .expect("d2d4 not found");

        assert!(
            (e4_prob - 0.6667).abs() < 0.001,
            "e2e4 probability should be ~2/3, got {}",
            e4_prob
        );
        assert!(
            (d4_prob - 0.3333).abs() < 0.001,
            "d2d4 probability should be ~1/3, got {}",
            d4_prob
        );

        // Check weighted average Elo
        // Game 1: 1500, Game 2: 1600, Game 3: 1550
        let expected_avg_elo = (1500.0 + 1600.0 + 1550.0) / 3.0;
        assert!(
            (starting_example.avg_elo_self - expected_avg_elo).abs() < 0.001,
            "Average Elo should be {}, got {}",
            expected_avg_elo,
            starting_example.avg_elo_self
        );

        // Check weighted average outcome
        // Game 1: 1 (white won), Game 2: 0 (white lost), Game 3: 1 (white won)
        let expected_avg_outcome = 2.0 / 3.0;
        assert!(
            (starting_example.avg_outcome - expected_avg_outcome).abs() < 0.001,
            "Average outcome should be {}, got {}",
            expected_avg_outcome,
            starting_example.avg_outcome
        );

        // Check win/draw/loss probabilities (using legacy conversion)
        // 2 wins, 1 loss, 0 draws
        let expected_win_prob = 2.0 / 3.0;
        let expected_draw_prob = 0.0;
        let expected_loss_prob = 1.0 / 3.0;

        assert!(
            (starting_example.win_prob - expected_win_prob).abs() < 0.001,
            "Win probability should be {}, got {}",
            expected_win_prob,
            starting_example.win_prob
        );
        assert!(
            (starting_example.draw_prob - expected_draw_prob).abs() < 0.001,
            "Draw probability should be {}, got {}",
            expected_draw_prob,
            starting_example.draw_prob
        );
        assert!(
            (starting_example.loss_prob - expected_loss_prob).abs() < 0.001,
            "Loss probability should be {}, got {}",
            expected_loss_prob,
            starting_example.loss_prob
        );
    }

    #[test]
    fn test_move_distribution_encoding() {
        let config = ModelConfig::default();
        let pgn_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pgn/aggregation_test.pgn");

        let dataset = OXIDataset::from_pgn_with_limit(&pgn_path, config.clone(), None)
            .expect("Failed to load PGN");

        // Get the first item (starting position)
        let item = dataset.get(0).expect("Failed to get first item");

        // Check that the move distribution is correctly encoded
        let e4_indices = encode_move_spatial("e2e4").expect("Failed to encode e2e4");
        let d4_indices = encode_move_spatial("d2d4").expect("Failed to encode d2d4");

        let e4_idx = e4_indices.0 * 64 + e4_indices.1;
        let d4_idx = d4_indices.0 * 64 + d4_indices.1;

        // Check probabilities at the correct indices
        assert!(
            item.move_distribution[e4_idx] > 0.6 && item.move_distribution[e4_idx] < 0.7,
            "e2e4 probability at index {} should be ~0.667, got {}",
            e4_idx,
            item.move_distribution[e4_idx]
        );
        assert!(
            item.move_distribution[d4_idx] > 0.3 && item.move_distribution[d4_idx] < 0.4,
            "d2d4 probability at index {} should be ~0.333, got {}",
            d4_idx,
            item.move_distribution[d4_idx]
        );

        // Check that probabilities sum to 1
        let total_prob: f32 = item.move_distribution.iter().sum();
        assert!(
            (total_prob - 1.0).abs() < 0.001,
            "Move probabilities should sum to 1.0, got {}",
            total_prob
        );

        // Check that all other indices are 0
        let non_zero_count = item.move_distribution.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(
            non_zero_count, 2,
            "Should have exactly 2 non-zero probabilities"
        );
    }

    #[test]
    fn test_legacy_active_won_conversion() {
        // Test that legacy active_won field is correctly converted to win/draw/loss
        let examples = vec![
            RawChessExample {
                fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                active_won: 1, // Win
                win_prob: None,
                draw_prob: None,
                loss_prob: None,
            },
            RawChessExample {
                fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                active_won: 0, // Loss (legacy behavior)
                win_prob: None,
                draw_prob: None,
                loss_prob: None,
            },
            RawChessExample {
                fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                active_won: -1, // Draw (if -1 is used for draws)
                win_prob: None,
                draw_prob: None,
                loss_prob: None,
            },
        ];

        let config = ModelConfig::default();
        let dataset = OXIDataset::from_examples(examples, config);

        let example = &dataset.examples[0];

        // 1 win, 1 loss, 1 draw out of 3 games
        assert!(
            (example.win_prob - 1.0 / 3.0).abs() < 0.001,
            "Win prob should be 1/3, got {}",
            example.win_prob
        );
        assert!(
            (example.draw_prob - 1.0 / 3.0).abs() < 0.001,
            "Draw prob should be 1/3, got {}",
            example.draw_prob
        );
        assert!(
            (example.loss_prob - 1.0 / 3.0).abs() < 0.001,
            "Loss prob should be 1/3, got {}",
            example.loss_prob
        );
    }

    #[test]
    fn test_new_probability_fields() {
        // Test that new probability fields are preserved and grouped correctly
        let examples = vec![
            RawChessExample {
                fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                active_won: 0, // Ignored when new fields are present
                win_prob: Some(0.8),
                draw_prob: Some(0.15),
                loss_prob: Some(0.05),
            },
            RawChessExample {
                fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                active_won: 0,
                win_prob: Some(0.6),
                draw_prob: Some(0.3),
                loss_prob: Some(0.1),
            },
        ];

        let config = ModelConfig::default();
        let dataset = OXIDataset::from_examples(examples, config);

        let example = &dataset.examples[0];

        // Average of [0.8, 0.6] = 0.7
        assert!(
            (example.win_prob - 0.7).abs() < 0.001,
            "Win prob should be 0.7, got {}",
            example.win_prob
        );
        // Average of [0.15, 0.3] = 0.225
        assert!(
            (example.draw_prob - 0.225).abs() < 0.001,
            "Draw prob should be 0.225, got {}",
            example.draw_prob
        );
        // Average of [0.05, 0.1] = 0.075
        assert!(
            (example.loss_prob - 0.075).abs() < 0.001,
            "Loss prob should be 0.075, got {}",
            example.loss_prob
        );
    }

    #[test]
    fn test_item_value_probabilities() {
        // Test that ChessItem correctly stores win/draw/loss probabilities
        let example = ChessExample {
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            move_probabilities: vec![("e2e4".to_string(), 1.0)].into_iter().collect(),
            avg_elo_self: 1500.0,
            avg_elo_oppo: 1500.0,
            avg_outcome: 0.5,
            win_prob: 0.45,
            draw_prob: 0.35,
            loss_prob: 0.20,
            count: 10,
        };

        let config = ModelConfig::default();
        let dataset = OXIDataset::new(vec![example], config);

        let item = dataset.get(0).expect("Failed to get item");

        assert!(
            (item.win_prob - 0.45).abs() < 0.001,
            "Item win prob should be 0.45, got {}",
            item.win_prob
        );
        assert!(
            (item.draw_prob - 0.35).abs() < 0.001,
            "Item draw prob should be 0.35, got {}",
            item.draw_prob
        );
        assert!(
            (item.loss_prob - 0.20).abs() < 0.001,
            "Item loss prob should be 0.20, got {}",
            item.loss_prob
        );
    }
}

