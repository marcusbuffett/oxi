use std::io::Write;
use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use pgn_reader::{San, SanPlus};
use rand::Rng;
use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, Chess, Position, Square};
use statrs::distribution::{Beta, ContinuousCDF};

use crate::config::{
    get_global_config, should_log_item, FEATURES_PER_TOKEN, LEGAL_MOVES, MISC_FEATURES,
    PIECE_IDENTITY_FEATURES, POSITIONAL_FEATURES, PREVIOUS_POSITIONS, RECENCY_FEATURES,
    TACTICAL_FEATURES,
};
use crate::dataset::{ChessBatch, OXIDataset};
use crate::move_encoding::decode_move;

/// Input type for model prediction logger
#[derive(Clone)]
pub struct ModelPredictionLoggerInput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub batch: Option<ChessBatch<B>>,
}

impl<B: Backend> ModelPredictionLoggerInput<B> {
    pub fn new(policy_logits: Tensor<B, 2>, batch: Option<ChessBatch<B>>) -> Self {
        Self {
            policy_logits,
            batch,
        }
    }
}

/// Metric for logging detailed model predictions for debugging
#[derive(Default)]
pub struct ModelPredictionLogger<B: Backend> {
    _backend: PhantomData<B>,
}

/// Simple logging function that can be called directly from the model
pub fn log_model_predictions<B: Backend>(
    policy_logits: &Tensor<B, 2>,
    value_logits: &Tensor<B, 2>,
    time_usage_logits: &Tensor<B, 2>,
    batch: &crate::dataset::ChessBatch<B>,
) {
    if !should_log_item() {
        return;
    }

    if batch.items.is_empty() {
        return;
    }

    // Log up to 2 positions
    let num_positions = batch.items.len().min(2);

    for pos_idx in 0..num_positions {
        log_single_position(
            policy_logits,
            value_logits,
            time_usage_logits,
            batch,
            pos_idx,
        );
    }
}

fn log_single_position<B: Backend>(
    policy_logits: &Tensor<B, 2>,
    value_logits: &Tensor<B, 2>,
    time_usage_logits: &Tensor<B, 2>,
    batch: &crate::dataset::ChessBatch<B>,
    pos_idx: usize,
) {
    let item = &batch.items[pos_idx];

    // Prominent header with position number
    tracing::info!("");
    tracing::info!("╔═══════════════════════════════════════════════════════════════╗");
    tracing::info!(
        "║           MODEL PREDICTION LOG - POSITION {}                  ║",
        pos_idx + 1
    );
    tracing::info!("╚═══════════════════════════════════════════════════════════════╝");
    tracing::info!("");
    tracing::info!("FEN: {}", item.fen);
    tracing::info!(
        "Player ELOs: Self {}, Opponent {}",
        item.elo_self,
        item.elo_oppo
    );

    // Log global features (normalized)
    let global_features_data = batch.global_features.clone().to_data();
    if let Ok(global_features_slice) = global_features_data.as_slice::<f32>() {
        use crate::config::NUM_GLOBALS;
        let offset = pos_idx * NUM_GLOBALS;
        // NUM_GLOBALS is currently 7: [time_self_norm, time_self_ratio, time_oppo_norm,
        // time_oppo_ratio, increment_ratio, move_count_norm, elo_norm]
        // Material imbalance, momentum, and volatility are not included yet
        if offset + NUM_GLOBALS <= global_features_slice.len() {
            tracing::info!(
                "Global features (normalized): time_self={:.3}, time_self_ratio={:.3}, time_oppo={:.3}, time_oppo_ratio={:.3}, increment_ratio={:.3}, move_count={:.3}, elo={:.3}",
                global_features_slice[offset],
                global_features_slice[offset + 1],
                global_features_slice[offset + 2],
                global_features_slice[offset + 3],
                global_features_slice[offset + 4],
                global_features_slice[offset + 5],
                global_features_slice[offset + 6],
            );
        } else {
            tracing::warn!(
                "Global feature slice too short: expected at least {} entries, have {}",
                offset + NUM_GLOBALS,
                global_features_slice.len()
            );
        }
    }

    // Log match outcome
    let outcome_str = match item.outcome {
        1.0 => "Win",
        0.5 => "Draw",
        0.0 => "Loss",
        _ => "Unknown",
    };

    // Parse position and log board representation
    if let Ok(fen) = item.fen.parse::<Fen>() {
        if let Ok(pos) = fen.into_position(shakmaty::CastlingMode::Standard) {
            let move_uci: UciMove = item.move_uci.parse().unwrap();
            let full_move = move_uci
                .to_move(&pos)
                .unwrap_or_else(|_| panic!("Invalid move: {}", item.move_uci));
            let san = San::from_move(&pos, full_move);
            tracing::info!("Correct move: {}", san.to_string());
            let chess_pos: Chess = pos;
            tracing::info!("Board:\n{:?}", chess_pos.board());

            // Log encoded board per square
            // tracing::info!("Encoded board per square:");
            // log_encoded_board(&item.board_encoded);

            // Log top predicted moves
            tracing::info!("Top model predictions:");
            let probs = log_softmax(policy_logits.clone(), 1);
            let prob_data = probs.to_data();

            if let Ok(prob_slice) = prob_data.as_slice::<f32>() {
                let batch_start = pos_idx * LEGAL_MOVES;
                let batch_end = batch_start + LEGAL_MOVES;
                let batch_probs = &prob_slice[batch_start..batch_end];
                let mut indexed_probs: Vec<(usize, f32)> = batch_probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();

                // dbg!(
                //     "Batch probs: {:?}, len: {}, indexed probs: {:?}, len: {}",
                //     &batch_probs,
                //     batch_probs.len(),
                //     &indexed_probs,
                //     indexed_probs.len()
                // );
                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Log top 10 moves
                for (rank, (move_idx, prob)) in indexed_probs.iter().take(10).enumerate() {
                    // check if -inf
                    if prob.is_infinite() {
                        continue;
                    }
                    let decoded_move = decode_move((move_idx / 76) as u8, (move_idx % 76) as u8);
                    if let Some(uci_move) = decoded_move {
                        // Convert UCI to SAN using the same method as line 67
                        if let Ok(full_move) = uci_move.to_move(&chess_pos) {
                            let san = San::from_move(&chess_pos, full_move);
                            let san_str = format!("{:6}", san.to_string()); // Ensure 6 characters
                            tracing::info!("{}: {} ({:.4})", rank + 1, san_str, prob);
                        } else {
                            let uci_str = format!("{:6}", uci_move.to_string()); // Fallback to UCI if conversion fails
                            tracing::info!("{}: {} ({:.4})", rank + 1, uci_str, prob);
                        }
                    } else {
                        tracing::info!("{}: Illegal index: {}", rank + 1, move_idx);
                    }
                }
            }

            // Log WDL prediction
            log_wdl_prediction(value_logits, item.outcome, pos_idx);

            // Log time usage distribution
            tracing::info!("Time usage distribution:");
            log_time_usage_distribution(
                time_usage_logits,
                item.time_used_for_move as f32,
                item.global_features.time_remaining_self as f32,
                pos_idx,
            );
        }
    }
}

pub fn log_encoded_board(board_encoded: &[f32]) {
    let formatted = format_encoded_board(board_encoded);
    for line in formatted.lines() {
        tracing::info!("{}", line);
    }
}

pub fn format_encoded_board(board_encoded: &[f32]) -> String {
    let mut output = String::new();
    output.push_str("Encoded board per square:\n");

    for square_idx in 0..64 {
        let square_output = format_square_encoding(square_idx, board_encoded);
        output.push_str(&square_output);
    }

    output
}

fn format_square_encoding(square_idx: usize, encoded_board: &[f32]) -> String {
    let mut output = String::new();

    let start_idx = square_idx * FEATURES_PER_TOKEN;
    let end_idx = start_idx + FEATURES_PER_TOKEN;
    let square_features = &encoded_board[start_idx..end_idx];

    output.push_str(&format!("Square {}:\n", Square::new(square_idx as u32)));

    let mut idx = 0;

    // Piece identity group
    let piece_features = &square_features[idx..idx + PIECE_IDENTITY_FEATURES];
    output.push_str(&format!(
        "\t\tPiece occupation: {}\n",
        piece_features
            .iter()
            .map(|f| format!("{f:.0}"))
            .collect::<Vec<String>>()
            .join(", ")
    ));
    idx += PIECE_IDENTITY_FEATURES;

    let fmt_slice = |slice: &[f32], precision: usize| -> String {
        slice
            .iter()
            .map(|f| format!("{f:.precision$}"))
            .collect::<Vec<String>>()
            .join(", ")
    };

    // Tactical group: attackers (8 white, 8 black), hanging, control
    let tactical_features = &square_features[idx..idx + TACTICAL_FEATURES];
    idx += TACTICAL_FEATURES;
    output.push_str(&format!(
        "\t\tWhite attackers: {}\n",
        fmt_slice(&tactical_features[0..8], 3)
    ));
    output.push_str(&format!(
        "\t\tBlack attackers: {}\n",
        fmt_slice(&tactical_features[8..16], 3)
    ));
    output.push_str(&format!("\t\tHanging flag: {:.0}\n", tactical_features[16]));
    output.push_str(&format!(
        "\t\tSquare control: {:.2}\n",
        tactical_features[17]
    ));
    debug_assert_eq!(TACTICAL_FEATURES, 18);

    // Positional group: mobility, rank one-hot, file one-hot
    let positional_features = &square_features[idx..idx + POSITIONAL_FEATURES];
    idx += POSITIONAL_FEATURES;
    output.push_str(&format!("\t\tLegal moves: {:.2}\n", positional_features[0]));
    output.push_str(&format!(
        "\t\tRank one-hot: {}\n",
        fmt_slice(&positional_features[1..9], 0)
    ));
    output.push_str(&format!(
        "\t\tFile one-hot: {}\n",
        fmt_slice(&positional_features[9..17], 0)
    ));
    debug_assert_eq!(POSITIONAL_FEATURES, 17);

    // Misc group: castling right
    let misc_features = &square_features[idx..idx + MISC_FEATURES];
    idx += MISC_FEATURES;
    output.push_str(&format!("\t\tCastling rights: {:.0}\n", misc_features[0]));

    // Recency channels
    let recency_features = &square_features[idx..idx + RECENCY_FEATURES];
    output.push_str(&format!(
        "\t\tRecency (white_from, white_to, black_from, black_to): {}\n",
        fmt_slice(recency_features, 3)
    ));
    idx += RECENCY_FEATURES;

    // History occupancy planes: 12 piece one-hots per past position
    for h in 0..PREVIOUS_POSITIONS {
        let planes = &square_features[idx..idx + PIECE_IDENTITY_FEATURES];
        idx += PIECE_IDENTITY_FEATURES;
        if planes.iter().any(|&f| f > 0.0) {
            output.push_str(&format!(
                "\t\tHistory t-{}: {}\n",
                h + 1,
                fmt_slice(planes, 0)
            ));
        }
    }

    debug_assert_eq!(idx, FEATURES_PER_TOKEN);

    // Raw embeddings: all features
    output.push_str(&format!(
        "\t\tRaw embeddings: {}\n",
        square_features
            .iter()
            .map(|f| format!("{f:.3}"))
            .collect::<Vec<String>>()
            .join(", ")
    ));

    output
}

fn log_wdl_prediction<B: Backend>(
    value_logits: &Tensor<B, 2>,
    actual_outcome: f32,
    pos_idx: usize,
) {
    let wdl_probs = softmax(value_logits.clone(), 1);
    let wdl_data = wdl_probs.to_data();

    if let Ok(wdl_slice) = wdl_data.as_slice::<f32>() {
        let offset = pos_idx * 3; // 3 WDL probabilities per example
        let win_prob = (wdl_slice[offset + 2] * 100.0).round() as i32; // Index 2 = win
        let draw_prob = (wdl_slice[offset + 1] * 100.0).round() as i32; // Index 1 = draw
        let loss_prob = (wdl_slice[offset + 0] * 100.0).round() as i32; // Index 0 = loss

        let actual_marker = match actual_outcome {
            1.0 => " (actual: Win)",
            0.5 => " (actual: Draw)",
            0.0 => " (actual: Loss)",
            _ => "",
        };

        tracing::info!(
            "WDL: {} / {} / {}{}",
            win_prob,
            draw_prob,
            loss_prob,
            actual_marker
        );
    } else {
        panic!("Failed to convert wdl_probs to slice");
    }
}

fn log_time_usage_distribution<B: Backend>(
    time_usage_logits: &Tensor<B, 2>,
    actual_time_usage: f32,
    time_remaining_self: f32,
    pos_idx: usize,
) {
    let formatted = format_time_usage_distribution(
        time_usage_logits,
        actual_time_usage,
        time_remaining_self,
        pos_idx,
    );
    for line in formatted.lines() {
        tracing::info!("{}", line);
    }
}

pub fn format_time_usage_distribution<B: Backend>(
    time_usage_logits: &Tensor<B, 2>,
    actual_time_usage: f32,
    time_remaining_self: f32,
    pos_idx: usize,
) -> String {
    let mut output = String::new();

    // Extract parameters for specified batch element: [alpha, beta]
    let params_data = time_usage_logits.to_data();
    if let Ok(params_slice) = params_data.as_slice::<f32>() {
        let offset = pos_idx * 2; // 2 parameters (alpha, beta) per example
        let alpha = params_slice[offset];
        let beta = params_slice[offset + 1];

        let concentration = alpha + beta;
        let mean_ratio = if concentration > 0.0 {
            alpha / concentration
        } else {
            0.5
        };
        let predicted_seconds = mean_ratio * time_remaining_self;
        let actual_ratio = if time_remaining_self > 0.0 {
            (actual_time_usage / time_remaining_self).clamp(0.0, 1.0)
        } else {
            0.0
        };

        output.push_str(&format!(
            "\nActual  time usage: {actual_time_usage:.3}s (ratio {actual_ratio:.3})\n"
        ));
        output.push_str(&format!(
            "Predicted Beta params: α={alpha:.5}, β={beta:.5} | mean ratio {mean_ratio:.5} (~{predicted_seconds:.3}s)\n"
        ));

        if let Ok(beta_dist) = Beta::new(alpha as f64, beta as f64) {
            if time_remaining_self > 0.0 {
                let buckets = [
                    (0.0_f32, 2.0_f32),
                    (2.0_f32, 6.0_f32),
                    (6.0_f32, 10.0_f32),
                    (10.0_f32, 20.0_f32),
                    (20.0_f32, f32::INFINITY),
                ];

                let max_bar_length = 40;
                let mut bucket_probs = Vec::new();
                let mut max_prob = 0.0f32;

                for (start_sec, end_sec) in buckets.iter() {
                    let lower_ratio = if *start_sec == 0.0 {
                        0.0
                    } else {
                        (*start_sec / time_remaining_self).clamp(0.0, 1.0)
                    } as f64;

                    let upper_ratio = if end_sec.is_infinite() {
                        1.0
                    } else {
                        (*end_sec / time_remaining_self).clamp(0.0, 1.0)
                    } as f64;

                    let prob = if upper_ratio <= lower_ratio {
                        0.0f32
                    } else {
                        (beta_dist.cdf(upper_ratio) - beta_dist.cdf(lower_ratio)) as f32
                    };

                    bucket_probs.push((*start_sec, *end_sec, prob));
                    max_prob = max_prob.max(prob);
                }

                for (start_sec, end_sec, prob) in bucket_probs {
                    let normalized = if max_prob > 0.0 { prob / max_prob } else { 0.0 };
                    let bar_length = (normalized * max_bar_length as f32) as usize;
                    let bar = create_unicode_bar(bar_length, max_bar_length);

                    let label = if end_sec.is_infinite() {
                        format!("{:.0}+s", start_sec)
                    } else {
                        format!("{:.0}-{:.0}s", start_sec, end_sec)
                    };

                    output.push_str(&format!("{:<7}: {:<40} ({:.3})\n", label, bar, prob));
                }
            } else {
                output.push_str("Time remaining is non-positive; skipping histogram.\n");
            }
        } else {
            output.push_str("Invalid beta distribution parameters");
        }
    } else {
        output.push_str("Failed to extract parameters from tensor");
    }

    output
}

pub fn format_elo_histogram(examples: &[crate::dataset::ChessExample]) -> String {
    let mut output = String::new();
    output.push_str("\nELO Distribution Histogram:\n");
    const BUCKETS: usize = 16;
    let mut counts = vec![0usize; BUCKETS];
    for ex in examples {
        let elo = ex.elo_self;
        if elo < 1000 {
            continue;
        }
        let idx = ((elo - 1000) / 100) as usize;
        let bucket = idx.min(BUCKETS - 1);
        counts[bucket] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&0) as f32;
    let max_bar_length = 40;
    for i in 0..BUCKETS - 1 {
        let start = 1000 + i * 100;
        let end = start + 99;
        let count = counts[i];
        let normalized = if max_count > 0.0 {
            count as f32 / max_count
        } else {
            0.0
        };
        let bar_length = (normalized * max_bar_length as f32) as usize;
        let bar = create_unicode_bar(bar_length, max_bar_length);
        output.push_str(&format!("{start}-{end}: {bar:<40} ({count})\n"));
    }
    let count = counts[BUCKETS - 1];
    let normalized = if max_count > 0.0 {
        count as f32 / max_count
    } else {
        0.0
    };
    let bar_length = (normalized * max_bar_length as f32) as usize;
    let bar = create_unicode_bar(bar_length, max_bar_length);
    output.push_str(&format!("2500+: {bar:<40} ({count})\n"));
    output
}

pub fn format_ply_histogram(examples: &[crate::dataset::ChessExample]) -> String {
    let mut output = String::new();
    output.push_str("\nPly Distribution Histogram:\n");

    // Define ply buckets: 0-5, 5-10, 10-20, 20-30, 30-40, 40-60, 60+
    let buckets = vec![
        (0, 5, "0-5"),
        (5, 10, "5-10"),
        (10, 20, "10-20"),
        (20, 30, "20-30"),
        (30, 40, "30-40"),
        (40, 60, "40-60"),
        (60, usize::MAX, "60+"),
    ];

    let mut counts = vec![0usize; buckets.len()];

    for ex in examples {
        let ply = ex.move_count;
        for (i, (start, end, _)) in buckets.iter().enumerate() {
            if ply >= *start && ply < *end {
                counts[i] += 1;
                break;
            }
        }
    }

    let max_count = *counts.iter().max().unwrap_or(&0) as f32;
    let max_bar_length = 40;

    for (i, (_, _, label)) in buckets.iter().enumerate() {
        let count = counts[i];
        let normalized = if max_count > 0.0 {
            count as f32 / max_count
        } else {
            0.0
        };
        let bar_length = (normalized * max_bar_length as f32) as usize;
        let bar = create_unicode_bar(bar_length, max_bar_length);
        output.push_str(&format!("{:>6}: {:<40} ({})\n", label, bar, count));
    }

    output
}

/// Create a Unicode bar with better granularity using block characters
fn create_unicode_bar(length: usize, max_length: usize) -> String {
    let full_blocks = length / 4;
    let remainder = length % 4;

    let mut bar = String::new();

    // Add full blocks
    bar.push_str(&"█".repeat(full_blocks));

    // Add partial block based on remainder
    match remainder {
        1 => bar.push('▎'),
        2 => bar.push('▌'),
        3 => bar.push('▊'),
        _ => {}
    }

    // Pad with spaces to maintain alignment
    let total_chars = full_blocks + if remainder > 0 { 1 } else { 0 };
    let max_chars = max_length.div_ceil(4); // Round up division
    if total_chars < max_chars {
        bar.push_str(&" ".repeat(max_chars - total_chars));
    }

    bar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend::{test_device, TestBackend};
    use burn::tensor::Tensor;

    #[test]
    fn test_log_wdl_prediction() {
        let device = test_device();

        // Create test tensor with value logits [loss, draw, win]
        let value_logits = Tensor::<TestBackend, 2>::from_data([[0.1, 0.2, 0.7]], &device);

        // Test with win outcome
        log_wdl_prediction(&value_logits, 1.0, 0);

        // Test with draw outcome
        log_wdl_prediction(&value_logits, 0.5, 0);

        // Test with loss outcome
        log_wdl_prediction(&value_logits, 0.0, 0);
    }

    #[test]
    fn test_format_time_usage_distribution_snapshot() {
        let device = test_device();

        // Create test tensor with gamma parameters that fit in 0-0.1 range
        // [alpha, beta]
        // Using smaller shape parameter and larger scale parameter for better fit
        let time_usage_logits = Tensor::<TestBackend, 2>::from_data([[2.0, 0.02]], &device);

        let actual_time_usage = 0.035;

        let formatted = format_time_usage_distribution(&time_usage_logits, 4.0, 60., 0);

        // Create snapshot
        insta::assert_snapshot!(formatted);
    }

    #[test]
    fn test_format_time_usage_distribution_edge_case() {
        let device = test_device();

        // Test with parameters that create distributions in the 0-0.1 range
        let time_usage_logits =
            Tensor::<TestBackend, 2>::from_data([[0.9, 0.1, 0.8, 1.2, 0.05, 0.04]], &device);

        let actual_time_usage = 0.0;

        let formatted =
            format_time_usage_distribution(&time_usage_logits, actual_time_usage, 60.0, 0);

        // Create snapshot for edge case
        insta::assert_snapshot!(formatted);
    }

    #[test]
    fn test_format_time_usage_distribution_high_variance() {
        let device = test_device();

        // Test with high variance scenario that still fits in 0-0.1 range
        let time_usage_logits =
            Tensor::<TestBackend, 2>::from_data([[0.3, 0.7, 0.5, 3.0, 0.08, 0.02]], &device);

        let actual_time_usage = 0.08;

        let formatted =
            format_time_usage_distribution(&time_usage_logits, actual_time_usage, 60.0, 0);

        // Create snapshot for high variance case
        insta::assert_snapshot!(formatted);
    }

    #[test]
    #[ignore]
    fn test_format_encoded_board_starting_position() {
        use crate::encoding::encode_position;
        use shakmaty::Chess;

        // Create starting position and encode it
        let starting_position = Chess::default();
        let board_encoded = encode_position(&starting_position, &[], &[]);

        // Format the encoded board
        let formatted = format_encoded_board(&board_encoded);

        // Create snapshot for testing
        insta::assert_snapshot!(formatted);
    }

    #[test]
    fn test_board_encoding_with_move_sequence() {
        use crate::encoding::encode_position;
        use shakmaty::{Chess, Position};

        // Play through: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 d6 5. d4 exd4
        let moves = vec![
            "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "d6", "d4", "exd4",
        ];

        let mut position = Chess::default();
        let mut previous_positions = Vec::new();
        let mut previous_moves = Vec::new();

        // Play all moves
        for move_san in &moves {
            let san_plus: SanPlus = move_san.parse().expect("valid move");
            let chess_move = san_plus.san.to_move(&position).expect("legal move");

            previous_positions.push(position.clone());
            previous_moves.push(move_san.to_string());

            position.play_unchecked(chess_move);
        }

        // Encode the final position
        let board_encoded = encode_position(&position, &previous_positions, &previous_moves);

        // Format the encoded board
        let formatted = format_encoded_board(&board_encoded);

        // Create snapshot
        insta::assert_snapshot!(
            "board_encoding_after_e4_e5_nf3_nc6_bb5_a6_ba4_d6_d4_exd4",
            formatted
        );
    }

    #[test]
    fn test_board_encoding_morra_gambit() {
        use crate::encoding::encode_position;
        use shakmaty::{Chess, Position};

        // Play through: 1. e4 c5 2. d4 cxd4 3. c3 Nc6 4. cxd4 d5 5. exd5 Qxd5 6. Nf3 e5 7. Nc3 Bb4
        let moves = vec![
            "e4", "c5", "d4", "cxd4", "c3", "Nc6", "cxd4", "d5", "exd5", "Qxd5", "Nf3", "e5",
            "Nc3", "Bb4",
        ];

        let mut position = Chess::default();
        let mut previous_positions = Vec::new();
        let mut previous_moves = Vec::new();

        // Play all moves
        for move_san in &moves {
            let san_plus: SanPlus = move_san.parse().expect("valid move");
            let chess_move = san_plus.san.to_move(&position).expect("legal move");

            previous_positions.push(position.clone());
            previous_moves.push(move_san.to_string());

            position.play_unchecked(chess_move);
        }

        // Encode the final position
        let board_encoded = encode_position(&position, &previous_positions, &previous_moves);

        // Format the encoded board
        let formatted = format_encoded_board(&board_encoded);

        // Create snapshot
        insta::assert_snapshot!("board_encoding_morra_gambit", formatted);
    }

    #[test]
    fn test_board_encoding_smith_morra_long() {
        use crate::encoding::encode_position;
        use shakmaty::{Chess, Position};

        // Play through: 1. e4 c5 2. d4 cxd4 3. c3 dxc3 4. Nxc3 Nc6 5. Nf3 e6 6. Bc4 Nf6 7. O-O Be7
        // 8. Qe2 O-O 9. Rd1 Qc7 10. h3 a6 11. b3 b5 12. Bd3 Bb7 13. Bb2 Rac8 14. Rac1 Qb8
        // 15. e5 Nh5 16. Ne4 Nf4 17. Qd2 Nxd3 18. Qxd3 Nb4 19. Qb1 Bxe4 20. Qxe4 Nxa2
        // 21. Rxc8 Rxc8 22. Nd4 Nc3 23. Bxc3 Rxc3 24. Ne2 Rxb3 25. Rxd7 Bf8 26. Qc2 Ra3
        // 27. Rd1 g6 28. f4 Qb6+ 29. Kf1 Bc5 30. Rc1 Be3 31. Re1 b4 32. Rb1 a5 33. Qb2
        let moves = vec![
            "e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3", "Nc6", "Nf3", "e6", "Bc4", "Nf6",
            "O-O", "Be7", "Qe2", "O-O", "Rd1", "Qc7", "h3", "a6", "b3", "b5", "Bd3", "Bb7", "Bb2",
            "Rac8", "Rac1", "Qb8", "e5", "Nh5", "Ne4", "Nf4", "Qd2", "Nxd3", "Qxd3", "Nb4", "Qb1",
            "Bxe4", "Qxe4", "Nxa2", "Rxc8", "Rxc8", "Nd4", "Nc3", "Bxc3", "Rxc3", "Ne2", "Rxb3",
            "Rxd7", "Bf8", "Qc2", "Ra3", "Rd1", "g6", "f4", "Qb6+", "Kf1", "Bc5", "Rc1", "Be3",
            "Re1", "b4", "Rb1", "a5", "Qb2",
        ];

        let mut position = Chess::default();
        let mut previous_positions = Vec::new();
        let mut previous_moves = Vec::new();

        // Play all moves
        for move_san in &moves {
            let san_plus: SanPlus = move_san.parse().expect("valid move");
            let chess_move = san_plus.san.to_move(&position).expect("legal move");

            previous_positions.push(position.clone());
            previous_moves.push(move_san.to_string());

            position.play_unchecked(chess_move);
        }

        // Encode the final position
        let board_encoded = encode_position(&position, &previous_positions, &previous_moves);

        // Format the encoded board
        let formatted = format_encoded_board(&board_encoded);

        // Create snapshot
        insta::assert_snapshot!("board_encoding_smith_morra_long", formatted);
    }
}
