use std::path::PathBuf;

use oxi::config::{get_global_config, set_global_config, Config};
use oxi::dataset::OXIDataset;
use oxi::inference::GlobalFeaturesInternal;
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess};

fn fen_board_to_ascii(board_part: &str) -> String {
    let ranks: Vec<String> = board_part
        .split('/')
        .map(|rank| {
            let mut expanded = String::new();
            for ch in rank.chars() {
                if let Some(spaces) = ch.to_digit(10) {
                    for _ in 0..spaces {
                        expanded.push('.');
                    }
                } else {
                    expanded.push(ch);
                }
            }
            expanded
        })
        .collect();
    ranks.join("\n")
}

// #[test]
// fn snapshot_real_game_positions_20_and_21() {
//     let mut config = Config::default();
//     config.enable_ply_sampling = false;
//     config.enable_elo_sampling = false;
//     config.single_legal_move_only = false;
//     config.checkmate_only = false;
//     config.seed = 42;
//     if set_global_config(config.clone()).is_err() {
//         let existing = get_global_config();
//         assert!(
//             existing.enable_ply_sampling == config.enable_ply_sampling
//                 && existing.enable_elo_sampling == config.enable_elo_sampling
//                 && existing.single_legal_move_only == config.single_legal_move_only
//                 && existing.checkmate_only == config.checkmate_only,
//             "Global config was initialized differently; ensure tests run with the expected settings"
//         );
//     }
//
//     let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//     let pgn_path = manifest_dir.join("tests/data/pgn/checkmate_long_time.pgn");
//
//     let dataset = OXIDataset::from_pgn_with_limit(&pgn_path, config.clone(), None)
//         .expect("failed to process PGN");
//
//     assert!(
//         dataset.examples.len() >= 90,
//         "expected at least 90 positions, found {}",
//         dataset.examples.len()
//     );
//
//     let mut snapshot_output = String::new();
//
//     for index in [15usize, 16usize] {
//         let example = dataset
//             .examples
//             .get(index)
//             .unwrap_or_else(|| panic!("missing example at index {index}"));
//         let item = dataset
//             .process_example(example)
//             .unwrap_or_else(|err| panic!("failed to process example {index}: {err}"));
//
//         let fen_str = item.fen.clone();
//         let board_part = fen_str
//             .split(' ')
//             .next()
//             .expect("fen must contain board part");
//         let board_ascii = fen_board_to_ascii(board_part);
//
//         let fen: Fen = fen_str.parse().expect("invalid fen");
//         let chess: Chess = fen
//             .clone()
//             .into_position(CastlingMode::Standard)
//             .expect("failed to convert fen into position");
//
//         let uci_move: UciMove = item.move_uci.parse().expect("invalid uci move");
//         let san = uci_move
//             .to_move(&chess)
//             .map(|mv| San::from_move(&chess, mv).to_string())
//             .unwrap_or_else(|_| "<illegal>".to_string());
//
//         let gf_internal = GlobalFeaturesInternal {
//             time_remaining_self: item.time_remaining_self,
//             time_remaining_oppo: item.time_remaining_oppo,
//             base_time: item.original_time_control.0,
//             material_imbalance: item.material_imbalance,
//             major_piece_imbalance: 0,
//             minor_piece_imbalance: 0,
//             pawn_imbalance: 0,
//             move_count: item.move_count,
//             elo_self: item.elo_self,
//             material_imbalance_history: item.material_imbalance_history.clone(),
//         };
//         let gf = gf_internal.to_normalized();
//
//         let time_usage_ratio = if item.time_remaining_self > 0 {
//             item.time_used_for_move as f32 / item.time_remaining_self as f32
//         } else {
//             0.0
//         };
//
//         snapshot_output.push_str(&format!(
//             "Position {} (dataset index {}):\n",
//             index + 1,
//             index
//         ));
//         snapshot_output.push_str(&format!("fen: {}\n", fen_str));
//         snapshot_output.push_str("board:\n");
//         snapshot_output.push_str(&board_ascii);
//         snapshot_output.push('\n');
//         snapshot_output.push_str(&format!("move_uci: {}\n", item.move_uci));
//         snapshot_output.push_str(&format!("move_san: {}\n", san));
//         snapshot_output.push_str(&format!("outcome: {}\n", item.outcome));
//         snapshot_output.push_str(&format!("elo_self: {}\n", item.elo_self));
//         snapshot_output.push_str(&format!("elo_oppo: {}\n", item.elo_oppo));
//         snapshot_output.push_str(&format!(
//             "time_remaining_self: {}\n",
//             item.time_remaining_self
//         ));
//         snapshot_output.push_str(&format!(
//             "time_remaining_oppo: {}\n",
//             item.time_remaining_oppo
//         ));
//         snapshot_output.push_str(&format!(
//             "time_used_for_move: {}\n",
//             item.time_used_for_move
//         ));
//         snapshot_output.push_str(&format!("time_usage_ratio: {:.6}\n", time_usage_ratio));
//         snapshot_output.push_str("global_features_normalized:\n");
//         snapshot_output.push_str(&format!(
//             "  time_self_normalized: {:.6}\n",
//             gf.time_self_normalized
//         ));
//         snapshot_output.push_str(&format!("  time_self_ratio: {:.6}\n", gf.time_self_ratio));
//         snapshot_output.push_str(&format!(
//             "  time_oppo_normalized: {:.6}\n",
//             gf.time_oppo_normalized
//         ));
//         snapshot_output.push_str(&format!("  time_oppo_ratio: {:.6}\n", gf.time_oppo_ratio));
//         snapshot_output.push_str(&format!(
//             "  move_count_normalized: {:.6}\n",
//             gf.move_count_normalized
//         ));
//         snapshot_output.push_str(&format!("  elo_normalized: {:.6}\n", gf.elo_normalized));
//         snapshot_output.push_str(&format!(
//             "  material_imbalance_normalized: {:.6}\n",
//             gf.material_imbalance_normalized
//         ));
//         snapshot_output.push_str(&format!("  momentum: {:.6}\n", gf.momentum));
//         snapshot_output.push_str(&format!("  volatility: {:.6}\n", gf.volatility));
//         snapshot_output.push_str(&format!("move_count: {}\n", item.move_count));
//         snapshot_output.push_str(&format!(
//             "material_imbalance_history: {:?}\n",
//             item.material_imbalance_history
//         ));
//
//         if example.previous_fens.is_empty() {
//             snapshot_output.push_str("previous_fens: (none)\n");
//         } else {
//             snapshot_output.push_str("previous_fens:\n");
//             for fen in &example.previous_fens {
//                 snapshot_output.push_str(&format!("  {}\n", fen));
//             }
//         }
//
//         if example.previous_moves.is_empty() {
//             snapshot_output.push_str("previous_moves: (none)\n");
//         } else {
//             snapshot_output.push_str("previous_moves:\n");
//             for mv in &example.previous_moves {
//                 snapshot_output.push_str(&format!("  {}\n", mv));
//             }
//         }
//
//         snapshot_output.push('\n');
//     }
//
//     insta::assert_snapshot!("blitz_single_game_positions_20_21", snapshot_output);
// }
