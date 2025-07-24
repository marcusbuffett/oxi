use shakmaty::{Bitboard, Chess, Color, EnPassantMode, File, Position, Role, Square};

use crate::config::{FEATURES_PER_SQUARE_POSITION, FEATURES_PER_TOKEN, PREVIOUS_POSITIONS};

pub fn encode_position(pos: &Chess, previous_positions: &[Chess]) -> Vec<f32> {
    let mut tokens = vec![0f32; 64 * FEATURES_PER_TOKEN];

    let mut all_positions = vec![Some(pos)];
    all_positions.extend(previous_positions.iter().map(Some));
    assert!(all_positions.len() <= PREVIOUS_POSITIONS + 1);
    all_positions.resize(PREVIOUS_POSITIONS + 1, None);

    for square in Square::ALL {
        let mut feature_idx = 0;
        let square_idx = square as usize * FEATURES_PER_TOKEN;
        for pos in &all_positions {
            if let Some(pos) = pos {
                let ep_square = pos.ep_square(EnPassantMode::Legal);
                let board = pos.board();
                if let Some(piece) = board.piece_at(square) {
                    let piece_offset = piece.role as usize - 1;
                    let color_offset = match piece.color {
                        Color::White => 0,
                        Color::Black => 6,
                    };
                    tokens[square_idx + feature_idx + color_offset + piece_offset] = 1.0;
                }
                feature_idx += 12;

                if Some(square) == ep_square {
                    tokens[square_idx + feature_idx] = 1.0;
                }
                feature_idx += 1;

                let castling = pos.castles().castling_rights();
                if castling.contains(square) {
                    tokens[square_idx + feature_idx] = 1.0;
                }
                feature_idx += 1;

                for color in [Color::White, Color::Black] {
                    let mut attackers = pos.king_attackers(square, color, board.occupied());
                    while let Some(square) = attackers.pop_front() {
                        if let Some(piece) = board.piece_at(square) {
                            tokens[square_idx + feature_idx + piece.role as usize - 1] = 1.0;
                        }
                    }
                    feature_idx += 6;
                }

                // Number of legal moves for the piece on this square, normalized and clamped
                let legal_moves_norm = normalized_legal_moves_for_square(pos, square);
                tokens[square_idx + feature_idx] = legal_moves_norm;
                feature_idx += 1;

                // Pawn structure features
                let (isolated, backward, doubled) = pawn_structure_features(pos, square);
                tokens[square_idx + feature_idx] = if isolated { 1.0 } else { 0.0 };
                feature_idx += 1;
                tokens[square_idx + feature_idx] = if backward { 1.0 } else { 0.0 };
                feature_idx += 1;
                tokens[square_idx + feature_idx] = if doubled { 1.0 } else { 0.0 };
                feature_idx += 1;

                // Add square control value as the last feature for this position
                let control_value = calculate_square_control(pos, square);
                tokens[square_idx + feature_idx] = control_value;
                feature_idx += 1;
            } else {
                feature_idx += FEATURES_PER_SQUARE_POSITION;
            }
            assert!(
                feature_idx % FEATURES_PER_SQUARE_POSITION == 0,
                "feature_idx % FEATURES_PER_SQUARE_POSITION == 0 {} % {} = {}",
                feature_idx,
                FEATURES_PER_SQUARE_POSITION,
                feature_idx % FEATURES_PER_SQUARE_POSITION
            );
        }

        tokens[square_idx + feature_idx] = square.is_dark() as usize as f32;
        feature_idx += 1;

        if feature_idx != FEATURES_PER_TOKEN {
            assert!(
                feature_idx == FEATURES_PER_TOKEN,
                "feature_idx == FEATURES_PER_TOKEN {feature_idx} should be {FEATURES_PER_TOKEN}"
            );
        }
        // if square_idx == 0 {
        //     tracing::info!("Position board after a1: {:?}", pos.board());
        //     log_encoded_board(&tokens);
        // }
    }
    // tracing::info!("Position board: {:?}", pos.board());
    // log_encoded_board(&tokens);
    // panic!();
    tokens
}
// let en_passant_offset = 12;
// if let Some(ep_square) = pos.ep_square(EnPassantMode::Legal) {
//     let idx = square_to_index(ep_square);
//     planes[en_passant_offset * 64 + idx] = 1.0;
// }
//
// let castling_offset = 13;
// let castling = pos.castles();
//
// // Mark corners based on castling rights
// if castling.has(Color::White, shakmaty::CastlingSide::KingSide) {
//     planes[castling_offset * 64 + 7] = 1.0; // h1
// }
// if castling.has(Color::White, shakmaty::CastlingSide::QueenSide) {
//     planes[castling_offset * 64 + 0] = 1.0; // a1
// }
// if castling.has(Color::Black, shakmaty::CastlingSide::KingSide) {
//     planes[castling_offset * 64 + 63] = 1.0; // h8
// }
// if castling.has(Color::Black, shakmaty::CastlingSide::QueenSide) {
//     planes[castling_offset * 64 + 56] = 1.0; // a8
// }

/// Get ELO bin index for a given rating
pub fn get_elo_bin(elo: i32, bins: &[i32]) -> usize {
    bins.iter().position(|&bin| elo < bin).unwrap_or(bins.len())
}

/// Get the piece value for square control calculation
fn get_piece_value(role: Role) -> f32 {
    match role {
        Role::Pawn => 1.0,
        Role::Knight => 3.0,
        Role::Bishop => 3.0,
        Role::Rook => 5.0,
        Role::Queen => 9.0,
        Role::King => 10.0, // King is valuable but not as much as queen for control
    }
}

/// Calculate square control for a given position and square
/// Returns a value from 0.0 to 1.0 where:
/// - 1.0 = total white control
/// - 0.0 = total black control  
/// - 0.5 = equal control
pub fn calculate_square_control(pos: &Chess, square: Square) -> f32 {
    let board = pos.board();
    let occupied = board.occupied();

    let mut white_control = 0.0f32;
    let mut black_control = 0.0f32;
    for color in [Color::White, Color::Black] {
        let mut occupied_clone = occupied;
        let mut board = pos.board().clone();
        loop {
            let Some(attacker) = board
                .attacks_to(square, color, occupied_clone)
                .into_iter()
                .min_by_key(|sq| sq.distance(square))
            else {
                break;
            };
            occupied_clone.toggle(attacker);
            let piece = board.remove_piece_at(attacker).unwrap();
            let piece_value = get_piece_value(piece.role);
            let control_contribution = 1.0 / piece_value.powf(2.0);

            match color {
                Color::White => white_control += control_contribution,
                Color::Black => black_control += control_contribution,
            }
        }
    }

    let total_control = white_control + black_control;
    if total_control == 0.0 {
        0.5 // No control from either side, neutral
    } else {
        white_control / total_control
    }
}

/// Compute number of legal moves available for the piece on a given square, normalized by 20
fn normalized_legal_moves_for_square(pos: &Chess, square: Square) -> f32 {
    let board = pos.board();
    let Some(_piece) = board.piece_at(square) else {
        return 0.0;
    };
    let mut count = 0usize;
    for m in pos.legal_moves() {
        if let Some(from_sq) = m.from() {
            if from_sq == square {
                count += 1;
            }
        }
    }
    ((count as f32) / 20.0).clamp(0.0, 1.0)
}

/// Determine pawn structure features for a pawn on the given square
/// Returns (isolated, backward, doubled)
fn pawn_structure_features(pos: &Chess, square: Square) -> (bool, bool, bool) {
    let board = pos.board();
    let Some(piece) = board.piece_at(square) else {
        return (false, false, false);
    };
    if piece.role != Role::Pawn {
        return (false, false, false);
    }
    let friendly_pawns = board.pawns() & board.by_color(piece.color);

    let file_idx = square.file() as i32; // 0..7
    let rank_idx = square.rank() as i32; // 0..7

    // Isolated: no pawns on adjacent files (any rank) of the same color
    let mut has_adjacent_file_pawn_same_color = false;
    for adj_file in [file_idx - 1, file_idx + 1] {
        if adj_file < 0 || adj_file > 7 {
            continue;
        }
        if (friendly_pawns & Bitboard::from_file(File::new(adj_file as u32))).count() > 0 {
            has_adjacent_file_pawn_same_color = true;
            break;
        }
    }
    let isolated = !has_adjacent_file_pawn_same_color;

    // Doubled: another pawn of same color on same file (different rank)
    let has_same_file_pawn_same_color_other_rank =
        (friendly_pawns & Bitboard::from_file(File::new(file_idx as u32))).count() > 1;
    let doubled = has_same_file_pawn_same_color_other_rank;

    // Backward: no friendly pawn on adjacent files behind this pawn (towards own home rank)
    // Build mask of adjacent files and intersect with friendly pawns, then scan ranks
    let adj_files_mask = if file_idx > 0 && file_idx < 7 {
        Bitboard::from_file(File::new((file_idx - 1) as u32))
            | Bitboard::from_file(File::new((file_idx + 1) as u32))
    } else if file_idx > 0 {
        Bitboard::from_file(File::new((file_idx - 1) as u32))
    } else {
        Bitboard::from_file(File::new((file_idx + 1) as u32))
    };
    let adjacent_friendly_pawns = friendly_pawns & adj_files_mask;
    let mut found_support = false;
    for sq in adjacent_friendly_pawns.into_iter() {
        let r = sq.rank() as i32;
        match piece.color {
            Color::White => {
                if r < rank_idx {
                    found_support = true;
                    break;
                }
            }
            Color::Black => {
                if r > rank_idx {
                    found_support = true;
                    break;
                }
            }
        }
    }
    let backward = !found_support;

    (isolated, backward, doubled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use std::str::FromStr;

    #[test]
    fn test_square_control_f6_and_d4() {
        // Test position: r1bq1rk1/pp2ppbp/3p2p1/3N2B1/8/1P1Q4/1P3PPP/R4RK1 b - - 1 15
        let fen = "7r/ppp2kpr/2n2p2/2bnp3/8/3P4/PPPN1PPP/R1B2RK1 w - - 2 17";
        let fen_parsed: Fen = fen.parse().expect("Valid FEN");
        let pos: Chess = fen_parsed
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");
        let mut formatted_output = String::new();
        for square in ["a3", "h2", "h1"] {
            let square = Square::from_str(square).expect("Valid square");
            let control = calculate_square_control(&pos, square);
            formatted_output += &format!("{}: {:.6}\n", square, control);
        }

        insta::assert_snapshot!(formatted_output);
    }
}
