use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use pgn_reader::San;
use rand::rng;
use rand::Rng as _;
use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, Chess, Position, Square};
use statrs::distribution::{Continuous, Gamma};

use crate::config::{get_global_config, FEATURES_PER_TOKEN, LEGAL_MOVES, PREVIOUS_POSITIONS};
use crate::dataset::ChessBatch;
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
    let config = get_global_config();
    let mut rng = rng();

    // Check if we should log this batch
    if rng.random::<f32>() > config.item_log_probability {
        return;
    }

    if batch.items.is_empty() {
        return;
    }

    let item = &batch.items[0];
    tracing::info!("=== Model Prediction Log ===");
    tracing::info!("FEN: {}", item.fen);

    // Log match outcome
    let outcome_str = match item.outcome {
        1.0 => "Win",
        0.5 => "Draw",
        0.0 => "Loss",
        _ => "Unknown",
    };
    tracing::info!("Match outcome: {} ({:.1})", outcome_str, item.outcome);

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
                let batch_probs = &prob_slice[0..LEGAL_MOVES]; // First batch element
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
            log_wdl_prediction(value_logits, item.outcome);

            // Log time usage distribution
            tracing::info!("Time usage distribution:");
            log_time_usage_distribution(
                time_usage_logits,
                item.time_used_for_move as f32,
                item.time_remaining_self as f32,
            );
        }
    }
}

pub fn log_encoded_board(board_encoded: &[f32]) {
    for square_idx in 0..64 {
        log_square_encoding(square_idx, board_encoded);
    }
}

fn log_square_encoding(square_idx: usize, encoded_board: &[f32]) {
    let start_idx = square_idx * FEATURES_PER_TOKEN;
    let end_idx = start_idx + FEATURES_PER_TOKEN;
    let square_features = &encoded_board[start_idx..end_idx];
    tracing::info!("Square {}:", Square::new(square_idx as u32));
    for position_idx in 0..(PREVIOUS_POSITIONS + 1) {
        let pos_start = position_idx * crate::config::FEATURES_PER_SQUARE_POSITION;
        let pos_end = pos_start + crate::config::FEATURES_PER_SQUARE_POSITION;
        tracing::info!(
            "\t\tPosition {}: {}, control: {:.2}",
            position_idx,
            square_features[pos_start..pos_end - 1]
                .iter()
                .map(|f| format!("{f:.0}"))
                .collect::<Vec<String>>()
                .join(", "),
            square_features[pos_end - 1]
        );
    }
    let global_start = (PREVIOUS_POSITIONS + 1) * crate::config::FEATURES_PER_SQUARE_POSITION;
    tracing::info!(
        "\tGlobal features: {}",
        square_features[global_start..]
            .iter()
            .map(|f| format!("{f:.1}"))
            .collect::<Vec<String>>()
            .join(", ")
    );
}

fn log_wdl_prediction<B: Backend>(value_logits: &Tensor<B, 2>, actual_outcome: f32) {
    let wdl_probs = softmax(value_logits.clone(), 1);
    let wdl_data = wdl_probs.to_data();

    if let Ok(wdl_slice) = wdl_data.as_slice::<f32>() {
        let win_prob = (wdl_slice[2] * 100.0).round() as i32; // Index 2 = win
        let draw_prob = (wdl_slice[1] * 100.0).round() as i32; // Index 1 = draw
        let loss_prob = (wdl_slice[0] * 100.0).round() as i32; // Index 0 = loss

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
) {
    let formatted =
        format_time_usage_distribution(time_usage_logits, actual_time_usage, time_remaining_self);
    for line in formatted.lines() {
        tracing::info!("{}", line);
    }
}

pub fn format_time_usage_distribution<B: Backend>(
    time_usage_logits: &Tensor<B, 2>,
    actual_time_usage: f32,
    time_remaining_self: f32,
) -> String {
    let mut output = String::new();

    // Extract parameters for first batch element: [w1, w2, alpha1, alpha2, theta1, theta2]
    let params_data = time_usage_logits.to_data();
    if let Ok(params_slice) = params_data.as_slice::<f32>() {
        let w1 = params_slice[0];
        let w2 = params_slice[1];
        let alpha1 = params_slice[2];
        let alpha2 = params_slice[3];
        let theta1 = params_slice[4];
        let theta2 = params_slice[5];

        output.push_str(&format!(
            "\nActual  time usage: {actual_time_usage:.3}s | Predicted parameters: w1={w1:.5}, w2={w2:.5}, α1={alpha1:.5}, α2={alpha2:.5}, θ1={theta1:.5}, θ2={theta2:.5}\n"
        ));

        // Create gamma distributions
        let gamma1_result = Gamma::new(alpha1 as f64, theta1 as f64);
        let gamma2_result = Gamma::new(alpha2 as f64, theta2 as f64);

        if let (Ok(gamma1), Ok(gamma2)) = (gamma1_result, gamma2_result) {
            // Create histogram for range 0.0 to 0.1 with 10 buckets
            let num_buckets = 10;
            let max_range = 0.1;
            let bucket_width = max_range / num_buckets as f32;
            let max_bar_length = 40; // Double the width

            let mut histogram = Vec::new();
            let mut max_density = 0.0f32;

            // Calculate density for each bucket
            for i in 0..num_buckets {
                let bucket_start = i as f32 * bucket_width;
                let bucket_mid = bucket_start + bucket_width / 2.0;

                let pdf1 = gamma1.pdf(bucket_mid as f64) as f32;
                let pdf2 = gamma2.pdf(bucket_mid as f64) as f32;
                let mixture_density = w1 * pdf1 + w2 * pdf2;

                histogram.push((bucket_start, bucket_start + bucket_width, mixture_density));
                max_density = max_density.max(mixture_density);
            }

            // Display histogram
            for (start, end, density) in histogram {
                let normalized_density = if max_density > 0.0 {
                    density / max_density
                } else {
                    0.0
                };
                let bar_length = (normalized_density * max_bar_length as f32) as usize;
                // Use Unicode block characters for better granularity
                let bar = create_unicode_bar(bar_length, max_bar_length);

                output.push_str(&format!(
                    "{:.1}s-{:.1}s: {:<40} ({:.6})\n",
                    start * time_remaining_self,
                    end * time_remaining_self,
                    bar,
                    density
                ));
            }

            // Show where actual value falls
            let actual_pdf1 = gamma1.pdf(actual_time_usage as f64) as f32;
            let actual_pdf2 = gamma2.pdf(actual_time_usage as f64) as f32;
            let _actual_mixture_density = w1 * actual_pdf1 + w2 * actual_pdf2;
        } else {
            output.push_str("Invalid gamma distribution parameters");
        }
    } else {
        output.push_str("Failed to extract parameters from tensor");
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
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_log_wdl_prediction() {
        let device = NdArrayDevice::Cpu;

        // Create test tensor with value logits [loss, draw, win]
        let value_logits = Tensor::<TestBackend, 2>::from_data([[0.1, 0.2, 0.7]], &device);

        // Test with win outcome
        log_wdl_prediction(&value_logits, 1.0);

        // Test with draw outcome
        log_wdl_prediction(&value_logits, 0.5);

        // Test with loss outcome
        log_wdl_prediction(&value_logits, 0.0);
    }

    #[test]
    fn test_format_time_usage_distribution_snapshot() {
        let device = NdArrayDevice::Cpu;

        // Create test tensor with gamma mixture parameters that fit in 0-0.1 range
        // [w1, w2, alpha1, alpha2, theta1, theta2]
        // Using smaller shape parameters and larger scale parameters for better fit
        let time_usage_logits =
            Tensor::<TestBackend, 2>::from_data([[0.6, 0.4, 3.0, 0.01, 10.0, 0.0001]], &device);

        let actual_time_usage = 0.035;

        let formatted = format_time_usage_distribution(&time_usage_logits, 4.0, 60.);

        // Create snapshot
        insta::assert_snapshot!(formatted);
    }

    #[test]
    fn test_format_time_usage_distribution_edge_case() {
        let device = NdArrayDevice::Cpu;

        // Test with parameters that create distributions in the 0-0.1 range
        let time_usage_logits =
            Tensor::<TestBackend, 2>::from_data([[0.9, 0.1, 0.8, 1.2, 0.05, 0.04]], &device);

        let actual_time_usage = 0.0;

        let formatted = format_time_usage_distribution(&time_usage_logits, actual_time_usage, 60.0);

        // Create snapshot for edge case
        insta::assert_snapshot!(formatted);
    }

    #[test]
    fn test_format_time_usage_distribution_high_variance() {
        let device = NdArrayDevice::Cpu;

        // Test with high variance scenario that still fits in 0-0.1 range
        let time_usage_logits =
            Tensor::<TestBackend, 2>::from_data([[0.3, 0.7, 0.5, 3.0, 0.08, 0.02]], &device);

        let actual_time_usage = 0.08;

        let formatted = format_time_usage_distribution(&time_usage_logits, actual_time_usage, 60.0);

        // Create snapshot for high variance case
        insta::assert_snapshot!(formatted);
    }
}
