use shakmaty::{Bitboard, Chess, Color, EnPassantMode, File, Position, Role, Square};

use crate::config::{
    BOARD_FEATURES_PER_TOKEN, FEATURES_PER_TOKEN, HISTORY_DECAY, MISC_FEATURES,
    PIECE_IDENTITY_FEATURES, POSITIONAL_FEATURES, PREVIOUS_POSITIONS, RECENCY_FEATURES,
    TACTICAL_FEATURES,
};

const OPEN_DIRECTIONS: [(i32, i32); 8] = [
    (0, 1),   // N
    (0, -1),  // S
    (1, 0),   // E
    (-1, 0),  // W
    (1, 1),   // NE
    (-1, 1),  // NW
    (1, -1),  // SE
    (-1, -1), // SW
];

pub fn encode_position(
    pos: &Chess,
    previous_positions: &[Chess],
    previous_moves: &[String],
) -> Vec<f32> {
    let mut tokens = vec![0f32; 64 * FEATURES_PER_TOKEN];

    // Precompute recency heatmaps with decay over up to PREVIOUS_POSITIONS positions
    let mut white_from = vec![0f32; 64];
    let mut white_to = vec![0f32; 64];
    let mut black_from = vec![0f32; 64];
    let mut black_to = vec![0f32; 64];
    let steps = PREVIOUS_POSITIONS
        .min(previous_positions.len())
        .min(previous_moves.len());
    for i in 0..steps {
        if let Ok(uci_move) = previous_moves[i].parse::<shakmaty::uci::UciMove>() {
            let w = HISTORY_DECAY.powi(i as i32);

            // Determine whose move it was by looking at the piece color in the previous position
            let prev_pos = &previous_positions[i];

            // For now, only handle normal moves. Castling and other special moves
            // will be handled by the normal move logic since UCI represents them as king moves
            match uci_move {
                shakmaty::uci::UciMove::Normal { from, to, .. } => {
                    // Determine mover by checking the piece color at the from square
                    if let Some(piece) = prev_pos.board().piece_at(from) {
                        match piece.color {
                            Color::White => {
                                white_from[from as usize] += w;
                                white_to[to as usize] += w;
                            }
                            Color::Black => {
                                black_from[from as usize] += w;
                                black_to[to as usize] += w;
                            }
                        }
                    }
                }
                _ => {
                    // this is only null and Put, neither of which we care about
                }
            }
        }
    }

    let ep_square = pos.ep_square(EnPassantMode::Legal);
    let board = pos.board();
    let mut pinned_to: [Option<Square>; 64] = [None; 64];
    let mut pin_targets = [false; 64];
    for square in Square::ALL {
        let pinned = is_piece_pinned(pos, square);
        pinned_to[square as usize] = pinned;
        if let Some(target_sq) = pinned {
            pin_targets[target_sq as usize] = true;
        }
    }

    for square in Square::ALL {
        let mut feature_idx = 0;
        let square_idx = square as usize * FEATURES_PER_TOKEN;

        // Current position features
        let piece = board.piece_at(square);
        let ep_active = if Some(square) == ep_square { 1.0 } else { 0.0 };
        let castling_right = if pos.castles().castling_rights().contains(square) {
            1.0
        } else {
            0.0
        };
        let pinned_to_square = pinned_to[square as usize];
        let absolute_pin = pinned_to_square
            .and_then(|target_sq| board.piece_at(target_sq))
            .map_or(0.0, |target_piece| {
                if target_piece.role == Role::King {
                    1.0
                } else {
                    0.0
                }
            });
        let is_pin_target = pin_targets[square as usize];
        let is_hanging = is_piece_hanging(pos, square);
        let has_pinned_def = has_pinned_defender(pos, square, &pinned_to);
        let is_passed = is_passed_pawn(pos, square);
        let legal_moves_norm = normalized_legal_moves_for_square(pos, square);
        let (isolated, backward, doubled) = pawn_structure_features(pos, square);
        let weak_white_square = is_weak_square(pos, square, Color::White);
        let weak_black_square = is_weak_square(pos, square, Color::Black);
        let open_file = is_open_file(pos, square);
        let control_value = calculate_square_control(pos, square);

        let mut attacker_features = [[0.0f32; 8]; 2];
        for (color_idx, color) in [Color::White, Color::Black].iter().enumerate() {
            let occupied_mask = board.occupied();
            let attackers = board.attacks_to(square, *color, occupied_mask);
            let mut material_sum = 0.0f32;
            let mut num_attackers = 0usize;
            for attacker_sq in attackers {
                if let Some(attacker_piece) = board.piece_at(attacker_sq) {
                    let role_idx = attacker_piece.role as usize - 1;
                    attacker_features[color_idx][role_idx] += 1.0;
                    material_sum += get_piece_value(attacker_piece.role);
                    num_attackers += 1;
                }
            }
            for i in 0..6 {
                attacker_features[color_idx][i] =
                    (attacker_features[color_idx][i] / 2.0).clamp(0.0, 1.0);
            }
            attacker_features[color_idx][6] = (material_sum / 24.0).clamp(0.0, 1.0);
            attacker_features[color_idx][7] = ((num_attackers as f32) / 6.0).clamp(0.0, 1.0);
        }

        // Piece identity group
        if let Some(piece) = piece {
            let piece_offset = piece.role as usize - 1;
            let color_offset = match piece.color {
                Color::White => 0,
                Color::Black => 6,
            };
            tokens[square_idx + feature_idx + color_offset + piece_offset] = 1.0;
        }
        feature_idx += PIECE_IDENTITY_FEATURES;
        debug_assert_eq!(
            feature_idx, PIECE_IDENTITY_FEATURES,
            "Piece identity group offset mismatch"
        );

        // Tactical group
        let mut tactical_offset = 0;
        for color_idx in 0..2 {
            for i in 0..8 {
                tokens[square_idx + feature_idx + tactical_offset + i] =
                    attacker_features[color_idx][i];
            }
            tactical_offset += 8;
        }
        tokens[square_idx + feature_idx + tactical_offset] =
            if pinned_to_square.is_some() { 1.0 } else { 0.0 };
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = absolute_pin;
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = if is_pin_target { 1.0 } else { 0.0 };
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = if is_hanging { 1.0 } else { 0.0 };
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = if has_pinned_def { 1.0 } else { 0.0 };
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = control_value;
        tactical_offset += 1;

        debug_assert_eq!(
            tactical_offset, TACTICAL_FEATURES,
            "Tactical group offset mismatch"
        );
        feature_idx += TACTICAL_FEATURES;
        debug_assert_eq!(
            feature_idx,
            PIECE_IDENTITY_FEATURES + TACTICAL_FEATURES,
            "Piece + tactical group offset mismatch"
        );

        // Positional group
        let mut positional_offset = 0;
        tokens[square_idx + feature_idx + positional_offset] = legal_moves_norm;
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] = if isolated { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] = if backward { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] = if doubled { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] =
            if weak_white_square { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] =
            if weak_black_square { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] = if open_file { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] = if is_passed { 1.0 } else { 0.0 };
        positional_offset += 1;
        tokens[square_idx + feature_idx + positional_offset] =
            if square.is_dark() { 1.0 } else { 0.0 };
        positional_offset += 1;

        let rank_base = square_idx + feature_idx + positional_offset;
        let rank_idx = square.rank() as usize;
        tokens[rank_base + rank_idx] = 1.0;
        positional_offset += 8;

        let file_base = square_idx + feature_idx + positional_offset;
        let file_idx = square.file() as usize;
        tokens[file_base + file_idx] = 1.0;
        positional_offset += 8;
        debug_assert_eq!(
            positional_offset, POSITIONAL_FEATURES,
            "Positional group offset mismatch"
        );
        feature_idx += POSITIONAL_FEATURES;
        debug_assert_eq!(
            feature_idx,
            PIECE_IDENTITY_FEATURES + TACTICAL_FEATURES + POSITIONAL_FEATURES,
            "Piece + tactical + positional group offset mismatch"
        );

        // Misc group
        let mut misc_offset = 0;
        tokens[square_idx + feature_idx + misc_offset] = ep_active;
        misc_offset += 1;
        tokens[square_idx + feature_idx + misc_offset] = castling_right;
        misc_offset += 1;
        debug_assert_eq!(misc_offset, MISC_FEATURES, "Misc group offset mismatch");
        feature_idx += MISC_FEATURES;
        debug_assert_eq!(
            feature_idx, BOARD_FEATURES_PER_TOKEN,
            "Main feature groups did not fill expected slots"
        );

        // Recency channels (clamped to [0,1])
        let sidx = square as usize;
        tokens[square_idx + feature_idx] = white_from[sidx].clamp(0.0, 1.0);
        feature_idx += 1;
        tokens[square_idx + feature_idx] = white_to[sidx].clamp(0.0, 1.0);
        feature_idx += 1;
        tokens[square_idx + feature_idx] = black_from[sidx].clamp(0.0, 1.0);
        feature_idx += 1;
        tokens[square_idx + feature_idx] = black_to[sidx].clamp(0.0, 1.0);
        feature_idx += 1;
        debug_assert_eq!(RECENCY_FEATURES, 4);

        if feature_idx != FEATURES_PER_TOKEN {
            assert!(
                feature_idx == FEATURES_PER_TOKEN,
                "feature_idx == FEATURES_PER_TOKEN {feature_idx} should be {FEATURES_PER_TOKEN}"
            );
        }
    }
    tokens
}

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

    // Backward: no friendly pawn on adjacent files at the same rank or behind
    // this pawn (towards own home rank). A same-rank neighbor counts as
    // support, so pawns in their initial structure are not flagged.
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
                if r <= rank_idx {
                    found_support = true;
                    break;
                }
            }
            Color::Black => {
                if r >= rank_idx {
                    found_support = true;
                    break;
                }
            }
        }
    }
    let backward = !found_support;

    (isolated, backward, doubled)
}

/// Determine if a square is a "weak square" for the given color.
/// A square is weak for a color IFF for the file on either side,
/// there is (no pawn OR the pawn is further advanced than the square).
/// "Further advanced" for white means lower rank (closer to rank 1).
/// "Further advanced" for black means higher rank (closer to rank 8).
fn is_weak_square(pos: &Chess, square: Square, color: Color) -> bool {
    let board = pos.board();
    let pawns = board.pawns() & board.by_color(color);
    let square_rank = square.rank() as i32;
    let square_file = square.file() as i32;

    // Do not count trivial weak squares on unreachable-support ranks:
    // - For White: ranks 1 and 2 (indices 0,1)
    // - For Black: ranks 7 and 8 (indices 6,7)
    match color {
        Color::White if square_rank <= 1 => return false,
        Color::Black if square_rank >= 6 => return false,
        _ => {}
    }

    // Check adjacent files (left and right)
    let adjacent_files = [square_file - 1, square_file + 1];

    for adj_file in adjacent_files {
        if adj_file < 0 || adj_file > 7 {
            continue; // Skip invalid files
        }

        // Get all pawns of the given color on this adjacent file
        let file_mask = Bitboard::from_file(File::new(adj_file as u32));
        let pawns_on_file = pawns & file_mask;

        let has_defending_pawn = pawns_on_file.into_iter().any(|pawn_square| {
            let pawn_rank = pawn_square.rank() as i32;
            match color {
                Color::White => pawn_rank < square_rank, // White pawn is more advanced (lower rank)
                Color::Black => pawn_rank > square_rank, // Black pawn is more advanced (higher rank)
            }
        });

        if has_defending_pawn {
            return false;
        }
    }

    // If we get here, there are defending pawns on both adjacent files
    true
}

/// Determine if the file of the given square is open (no pawns of either color on it)
fn is_open_file(pos: &Chess, square: Square) -> bool {
    let board = pos.board();
    let file_mask = Bitboard::from_file(square.file());
    ((board.pawns() & file_mask).count()) == 0
}

fn opposite_color(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

fn attacker_can_move_to(board: &shakmaty::Board, attacker_sq: Square, target_sq: Square) -> bool {
    let mut board_clone = board.clone();
    let Some(attacker_piece) = board_clone.remove_piece_at(attacker_sq) else {
        return false;
    };

    board_clone.remove_piece_at(target_sq);
    board_clone.set_piece_at(target_sq, attacker_piece);

    let king_bb = board_clone.by_role(Role::King) & board_clone.by_color(attacker_piece.color);
    let Some(king_sq) = king_bb.into_iter().next() else {
        return false;
    };

    let enemy_color = opposite_color(attacker_piece.color);
    let occupied = board_clone.occupied();
    let enemy_attacks = board_clone.attacks_to(king_sq, enemy_color, occupied);

    enemy_attacks.is_empty()
}

fn is_piece_hanging(pos: &Chess, square: Square) -> bool {
    let board = pos.board();
    let Some(piece) = board.piece_at(square) else {
        return false;
    };

    let piece_value = get_piece_value(piece.role);
    let enemy_color = opposite_color(piece.color);
    let attackers = board.attacks_to(square, enemy_color, board.occupied());

    for attacker_sq in attackers {
        if let Some(attacker_piece) = board.piece_at(attacker_sq) {
            if get_piece_value(attacker_piece.role) < piece_value
                && attacker_can_move_to(board, attacker_sq, square)
            {
                return true;
            }
        }
    }

    false
}

/// Detects pieces with illusory defense: defended by a pinned piece that can't recapture.
fn has_pinned_defender(pos: &Chess, square: Square, pinned_squares: &[Option<Square>; 64]) -> bool {
    let board = pos.board();
    let Some(piece) = board.piece_at(square) else {
        return false;
    };

    let defenders = board.attacks_to(square, piece.color, board.occupied());

    for defender_sq in defenders {
        if defender_sq == square {
            continue;
        }

        if let Some(pinned_to) = pinned_squares[defender_sq as usize] {
            let (defender_file, defender_rank) =
                (defender_sq.file() as i32, defender_sq.rank() as i32);
            let (target_file, target_rank) = (square.file() as i32, square.rank() as i32);
            let (pinned_to_file, pinned_to_rank) =
                (pinned_to.file() as i32, pinned_to.rank() as i32);

            let pin_direction = (
                (pinned_to_file - defender_file).signum(),
                (pinned_to_rank - defender_rank).signum(),
            );
            let capture_direction = (
                (target_file - defender_file).signum(),
                (target_rank - defender_rank).signum(),
            );

            let capture_along_pin_ray = pin_direction == capture_direction
                || pin_direction == (-capture_direction.0, -capture_direction.1);

            if !capture_along_pin_ray {
                return true;
            }
        }
    }

    false
}

fn is_passed_pawn(pos: &Chess, square: Square) -> bool {
    let board = pos.board();
    let Some(piece) = board.piece_at(square) else {
        return false;
    };

    if piece.role != Role::Pawn {
        return false;
    }

    let enemy_color = opposite_color(piece.color);
    let enemy_pawns = board.pawns() & board.by_color(enemy_color);

    let file_idx = square.file() as i32;
    let rank_idx = square.rank() as i32;

    for df in -1..=1 {
        let file = file_idx + df;
        if file < 0 || file > 7 {
            continue;
        }

        let file_mask = Bitboard::from_file(File::new(file as u32));
        let pawns_on_file = enemy_pawns & file_mask;

        for pawn_sq in pawns_on_file.into_iter() {
            let pawn_rank = pawn_sq.rank() as i32;
            match piece.color {
                Color::White if pawn_rank > rank_idx => return false,
                Color::Black if pawn_rank < rank_idx => return false,
                _ => {}
            }
        }
    }

    true
}

/// Determine if the piece on the given square is pinned to its own king.
/// A piece is pinned if, along a straight line from its king, there is exactly
/// one friendly piece (the candidate), and beyond it on the same ray there is
/// an enemy sliding piece (rook/bishop/queen) compatible with that ray.
/// The king is never considered a pinned piece.
fn is_piece_pinned(pos: &Chess, square: Square) -> Option<Square> {
    let board = pos.board();
    let Some(piece) = board.piece_at(square) else {
        return None;
    };

    if piece.role == Role::King {
        return None;
    }

    let color = piece.color;

    for &(dx, dy) in OPEN_DIRECTIONS.iter() {
        let Some(pinned_to_sq) = first_piece_along(board, square, dx, dy) else {
            continue;
        };

        let Some(pinned_to_piece) = board.piece_at(pinned_to_sq) else {
            continue;
        };

        if pinned_to_piece.color != color {
            continue;
        }

        if pinned_to_piece.role != Role::King {
            // Only treat this as a pin if moving exposes a strictly more valuable piece.
            let pinned_value = get_piece_value(piece.role);
            let shield_value = get_piece_value(pinned_to_piece.role);
            if shield_value <= pinned_value {
                continue;
            }
        }

        if role_moves_in_direction(piece.role, (dx, dy)) {
            continue;
        }

        let Some(pinner_sq) = first_piece_along(board, square, -dx, -dy) else {
            continue;
        };

        let Some(pinner_piece) = board.piece_at(pinner_sq) else {
            continue;
        };

        if pinner_piece.color == color {
            continue;
        }

        if !pinner_can_attack_direction(pinner_piece.role, (dx, dy)) {
            continue;
        }

        return Some(pinned_to_sq);
    }

    None
}

fn first_piece_along(board: &shakmaty::Board, origin: Square, dx: i32, dy: i32) -> Option<Square> {
    let mut file = origin.file() as i32 + dx;
    let mut rank = origin.rank() as i32 + dy;

    while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
        let idx = (rank * 8 + file) as u32;
        let sq = Square::new(idx);
        if board.piece_at(sq).is_some() {
            return Some(sq);
        }
        file += dx;
        rank += dy;
    }

    None
}

fn role_moves_in_direction(role: Role, direction: (i32, i32)) -> bool {
    let (dx, dy) = direction;
    let is_cardinal = dx == 0 || dy == 0;
    let is_diagonal = dx != 0 && dy != 0;

    match role {
        Role::Bishop => is_diagonal,
        Role::Rook => is_cardinal,
        Role::Queen => true,
        Role::King => true,
        Role::Pawn => false,
        Role::Knight => false,
    }
}

fn pinner_can_attack_direction(role: Role, direction: (i32, i32)) -> bool {
    let (dx, dy) = direction;
    let is_cardinal = dx == 0 || dy == 0;
    let is_diagonal = dx != 0 && dy != 0;

    match role {
        Role::Queen => true,
        Role::Rook => is_cardinal,
        Role::Bishop => is_diagonal,
        _ => false,
    }
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

    #[test]
    fn test_weak_square() {
        let fen = "r4rk1/pppqb1pp/1nn1bp2/4p3/8/2NP1NP1/PP1BPPBP/2RQR1K1 w - - 0 12";
        let fen_parsed: Fen = fen.parse().expect("Valid FEN");
        let pos: Chess = fen_parsed
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");
        let mut formatted_output = String::new();
        for square in ["d7", "d2"] {
            for color in [Color::White, Color::Black] {
                let square = Square::from_str(square).expect("Valid square");
                let is_weak_square = is_weak_square(&pos, square, color);
                let color_str = match color {
                    Color::White => "White",
                    Color::Black => "Black",
                };
                formatted_output +=
                    &format!("{} weak for {}: {}\n", square, color_str, is_weak_square);
            }
        }

        insta::assert_snapshot!(formatted_output);
    }

    #[test]
    #[ignore]
    fn test_move_history_decay() {
        // Test that move history decay works correctly with stored moves
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0
"#;

        // Write to a temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test_decay.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        // Process the PGN file
        let examples = crate::pgn_processor::process_pgn_file(&pgn_path).unwrap();

        // Get the final position (after Re1)
        // PGN: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5
        // That's 12 moves total, so Re1 is at index 11
        let final_example = &examples[11]; // Re1 move
        assert_eq!(final_example.move_uci, "f1e1");

        // Parse the position and previous positions
        let current_pos: shakmaty::Chess = shakmaty::fen::Fen::from_str(&final_example.fen)
            .unwrap()
            .into_position(shakmaty::CastlingMode::Standard)
            .unwrap();

        let mut previous_positions = Vec::new();
        for fen_str in &final_example.previous_fens {
            let fen: shakmaty::fen::Fen = fen_str.parse().unwrap();
            let pos: shakmaty::Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
            previous_positions.push(pos);
        }

        // Encode the position with move history
        let encoded = encode_position(
            &current_pos,
            &previous_positions,
            &final_example.previous_moves,
        );

        // Test that the encoding includes recency information
        // The encoding should have non-zero values in the recency channels
        let features_per_square = FEATURES_PER_TOKEN;
        let mut found_recency_data = false;

        for square_idx in 0..64 {
            let base_idx = square_idx * features_per_square;

            // Check recency channels (white_from, white_to, black_from, black_to)
            let white_from = encoded[base_idx + features_per_square - 4];
            let white_to = encoded[base_idx + features_per_square - 3];
            let black_from = encoded[base_idx + features_per_square - 2];
            let black_to = encoded[base_idx + features_per_square - 1];

            if white_from > 0.0 || white_to > 0.0 || black_from > 0.0 || black_to > 0.0 {
                found_recency_data = true;
            }
        }

        assert!(
            found_recency_data,
            "Should have found recency data in encoding"
        );

        // Test decay pattern - more recent moves should have higher weights
        // Find squares that have recency data and check the decay pattern
        // Chess square indexing: a1=0, b1=1, ..., h1=7, a2=8, etc.
        let e4_square = Square::from_str("e4").unwrap() as usize; // e4 = 4th file, 4th rank = 3*8 + 4 = 28
        let b5_square = Square::from_str("b5").unwrap() as usize; // b5 = 1st file, 5th rank = 4*8 + 1 = 33
        let e1_square = Square::from_str("e1").unwrap() as usize; // e1 = 4th file, 1st rank = 0*8 + 4 = 4

        // e4 should have decayed weight (oldest move)
        let e4_base = e4_square * features_per_square;
        let e4_white_from = encoded[e4_base + features_per_square - 4];

        // b5 should have higher weight (more recent)
        let b5_base = b5_square * features_per_square;
        let b5_black_from = encoded[b5_base + features_per_square - 2];

        // e1 (rook move to e1) should have highest weight (most recent)
        let e1_base = e1_square * features_per_square;
        let e1_white_to = encoded[e1_base + features_per_square - 3];

        println!("Move history decay test:");
        println!("e4 white_from: {:.6}", e4_white_from);
        println!("b5 black_from: {:.6}", b5_black_from);
        println!("e1 white_to: {:.6}", e1_white_to);

        // The most recent move should have the highest weight
        assert!(
            e1_white_to >= b5_black_from,
            "Most recent move should have highest weight"
        );
        assert!(
            b5_black_from >= e4_white_from,
            "More recent moves should have higher weights"
        );

        println!("Move decay pattern test passed!");
    }

    #[test]
    fn test_empty_move_history() {
        // Test that encoding works with empty move history
        let starting_pos = shakmaty::Chess::default();
        let encoded = encode_position(&starting_pos, &[], &[]);

        // Should still produce valid encoding
        assert_eq!(encoded.len(), 64 * FEATURES_PER_TOKEN);

        // Should have no recency data
        let features_per_square = FEATURES_PER_TOKEN;
        for square_idx in 0..64 {
            let base_idx = square_idx * features_per_square;

            // Check recency channels should be zero
            let white_from = encoded[base_idx + features_per_square - 4];
            let white_to = encoded[base_idx + features_per_square - 3];
            let black_from = encoded[base_idx + features_per_square - 2];
            let black_to = encoded[base_idx + features_per_square - 1];

            assert_eq!(
                white_from, 0.0,
                "Empty history should have no white_from recency"
            );
            assert_eq!(
                white_to, 0.0,
                "Empty history should have no white_to recency"
            );
            assert_eq!(
                black_from, 0.0,
                "Empty history should have no black_from recency"
            );
            assert_eq!(
                black_to, 0.0,
                "Empty history should have no black_to recency"
            );
        }

        println!("Empty move history test passed!");
    }

    #[test]
    fn test_pinned_logic_and_king_not_pinned() {
        // Position: White king on c1, white knight on b2, black bishop on a3 -> knight on b2 is pinned
        // FEN components:
        // - h8 black king
        // - a3 black bishop
        // - b2 white knight
        // - c1 white king
        let fen = "7k/8/8/8/8/b7/1N6/2K5 w - - 0 1";
        let fen_parsed: Fen = fen.parse().expect("Valid FEN");
        let pos: Chess = fen_parsed
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");

        let e2 = Square::from_str("b2").unwrap();
        let e1 = Square::from_str("c1").unwrap();

        let pinned_target = is_piece_pinned(&pos, e2);
        let expected_target = Square::from_str("c1").unwrap();
        assert_eq!(
            pinned_target,
            Some(expected_target),
            "Knight on b2 should be pinned to c1"
        );
        assert!(
            is_piece_pinned(&pos, e1).is_none(),
            "King should never be considered pinned"
        );

        // Now test a case where the side is in check by a knight; unrelated pieces should not be marked pinned
        // Position: White king e1, white rook a1, black knight c2 (checking e1)
        let fen2 = "8/1k6/8/8/8/8/2n5/R3K3 w - - 0 1";
        let fen_parsed2: Fen = fen2.parse().expect("Valid FEN");
        let pos2: Chess = fen_parsed2
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");

        let a1 = Square::from_str("a1").unwrap();
        assert!(
            is_piece_pinned(&pos2, a1).is_none(),
            "Rook on a1 should not be marked pinned just because king is in check"
        );
    }

    #[test]
    #[ignore] // Run with: cargo test bench_encode_position --release -- --ignored --nocapture
    fn bench_encode_position() {
        use rand::prelude::IndexedRandom;
        use rand::SeedableRng;
        use std::time::Instant;

        const NUM_POSITIONS: usize = 1024;
        const MOVES_PER_GAME: usize = 30;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut positions = Vec::with_capacity(NUM_POSITIONS);

        while positions.len() < NUM_POSITIONS {
            let mut pos = Chess::default();
            for _ in 0..MOVES_PER_GAME {
                let legal_moves: Vec<_> = pos.legal_moves().into_iter().collect();
                if legal_moves.is_empty() {
                    break;
                }
                let chosen_move = legal_moves.choose(&mut rng).unwrap();
                pos = pos.play(chosen_move.clone()).unwrap();
            }
            positions.push(pos);
        }

        let mut total_encoded = 0usize;

        for _ in 0..100 {
            let _ = encode_position(&positions[0], &[], &[]);
        }

        let start = Instant::now();
        for pos in &positions {
            let encoded = encode_position(pos, &[], &[]);
            total_encoded += encoded.len();
        }
        let elapsed = start.elapsed();

        let per_position = elapsed / NUM_POSITIONS as u32;
        let positions_per_sec = NUM_POSITIONS as f64 / elapsed.as_secs_f64();

        println!("\n=== Encoding Benchmark ===");
        println!("Positions encoded: {}", NUM_POSITIONS);
        println!("Total time: {:?}", elapsed);
        println!("Per position: {:?}", per_position);
        println!("Positions/sec: {:.0}", positions_per_sec);
        println!("Features per position: {}", total_encoded / NUM_POSITIONS);
    }

    #[test]
    #[ignore] // Run with: cargo test bench_full_pipeline --release -- --ignored --nocapture
    fn bench_full_pipeline() {
        use crate::config::{set_global_config, Config, LEGAL_MOVES};
        use crate::inference::compute_material_imbalance;
        use crate::move_encoding::encode_move;
        use crate::moves::get_side_info;
        use rand::prelude::IndexedRandom;
        use rand::SeedableRng;
        use shakmaty::uci::UciMove;
        use std::time::Instant;

        let _ = set_global_config(Config::default());

        const NUM_POSITIONS: usize = 1024;
        const MOVES_PER_GAME: usize = 30;
        const PREVIOUS_POSITIONS_COUNT: usize = 5;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        struct TestExample {
            pos: Chess,
            previous_positions: Vec<Chess>,
            previous_moves: Vec<String>,
            move_uci: String,
        }

        let mut examples = Vec::with_capacity(NUM_POSITIONS);

        while examples.len() < NUM_POSITIONS {
            let mut pos = Chess::default();
            let mut history: Vec<(Chess, String)> = Vec::new();

            for _ in 0..MOVES_PER_GAME {
                let legal_moves: Vec<_> = pos.legal_moves().into_iter().collect();
                if legal_moves.is_empty() {
                    break;
                }
                let chosen_move = legal_moves.choose(&mut rng).unwrap();
                let uci = chosen_move
                    .to_uci(shakmaty::CastlingMode::Standard)
                    .to_string();
                history.push((pos.clone(), uci));
                pos = pos.play(chosen_move.clone()).unwrap();
            }

            let legal_moves: Vec<_> = pos.legal_moves().into_iter().collect();
            if legal_moves.is_empty() {
                continue;
            }
            let final_move = legal_moves.choose(&mut rng).unwrap();
            let move_uci = final_move
                .to_uci(shakmaty::CastlingMode::Standard)
                .to_string();

            let prev_count = PREVIOUS_POSITIONS_COUNT.min(history.len());
            let previous_positions: Vec<Chess> = history
                .iter()
                .rev()
                .take(prev_count)
                .map(|(p, _)| p.clone())
                .collect();
            let previous_moves: Vec<String> = history
                .iter()
                .rev()
                .take(prev_count)
                .map(|(_, m)| m.clone())
                .collect();

            examples.push(TestExample {
                pos,
                previous_positions,
                previous_moves,
                move_uci,
            });
        }

        for _ in 0..10 {
            let ex = &examples[0];
            let _ = encode_position(&ex.pos, &ex.previous_positions, &ex.previous_moves);
        }

        let start = Instant::now();
        for ex in &examples {
            let _board_encoded =
                encode_position(&ex.pos, &ex.previous_positions, &ex.previous_moves);

            let mut move_distribution = vec![0.0f32; LEGAL_MOVES];
            let move_uci: UciMove = ex.move_uci.parse().unwrap();
            let (from_idx, to_idx) = encode_move(&move_uci).unwrap();
            let move_idx = from_idx as usize * 76 + to_idx as usize;
            move_distribution[move_idx] = 1.0;

            let _side_info = get_side_info(&ex.pos, &ex.move_uci);

            let mut legal_moves = [0f32; LEGAL_MOVES];
            ex.pos.legal_moves().iter().for_each(|m| {
                let (from, to) = encode_move(&m.to_uci(shakmaty::CastlingMode::Standard)).unwrap();
                let idx = from as usize * 76 + to as usize;
                legal_moves[idx] = 1.0;
            });

            let _material = compute_material_imbalance(&ex.pos);
        }
        let elapsed = start.elapsed();

        let per_position = elapsed / NUM_POSITIONS as u32;
        let positions_per_sec = NUM_POSITIONS as f64 / elapsed.as_secs_f64();

        println!("\n=== Full Pipeline Benchmark ===");
        println!("Positions processed: {}", NUM_POSITIONS);
        println!("Total time: {:?}", elapsed);
        println!("Per position: {:?}", per_position);
        println!("Positions/sec: {:.0}", positions_per_sec);
    }

    #[test]
    #[ignore] // Run with: cargo test bench_with_fen_parsing --release -- --ignored --nocapture
    fn bench_with_fen_parsing() {
        use crate::config::{set_global_config, Config, LEGAL_MOVES};
        use crate::inference::compute_material_imbalance;
        use crate::move_encoding::encode_move;
        use crate::moves::get_side_info;
        use rand::prelude::IndexedRandom;
        use rand::SeedableRng;
        use shakmaty::fen::Epd;
        use shakmaty::uci::UciMove;
        use std::time::Instant;

        let _ = set_global_config(Config::default());

        const NUM_POSITIONS: usize = 1024;
        const MOVES_PER_GAME: usize = 30;
        const PREVIOUS_POSITIONS_COUNT: usize = 5;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        struct TestExample {
            fen: String,
            previous_fens: Vec<String>,
            previous_moves: Vec<String>,
            move_uci: String,
        }

        let mut examples = Vec::with_capacity(NUM_POSITIONS);

        while examples.len() < NUM_POSITIONS {
            let mut pos = Chess::default();
            let mut history: Vec<(String, String)> = Vec::new();

            for _ in 0..MOVES_PER_GAME {
                let legal_moves: Vec<_> = pos.legal_moves().into_iter().collect();
                if legal_moves.is_empty() {
                    break;
                }
                let chosen_move = legal_moves.choose(&mut rng).unwrap();
                let uci = chosen_move
                    .to_uci(shakmaty::CastlingMode::Standard)
                    .to_string();
                let fen_str = Epd::from_position(&pos, shakmaty::EnPassantMode::Legal).to_string();
                history.push((fen_str, uci));
                pos = pos.play(chosen_move.clone()).unwrap();
            }

            let legal_moves: Vec<_> = pos.legal_moves().into_iter().collect();
            if legal_moves.is_empty() {
                continue;
            }
            let final_move = legal_moves.choose(&mut rng).unwrap();
            let move_uci = final_move
                .to_uci(shakmaty::CastlingMode::Standard)
                .to_string();
            let fen = Epd::from_position(&pos, shakmaty::EnPassantMode::Legal).to_string();

            let prev_count = PREVIOUS_POSITIONS_COUNT.min(history.len());
            let previous_fens: Vec<String> = history
                .iter()
                .rev()
                .take(prev_count)
                .map(|(f, _)| f.clone())
                .collect();
            let previous_moves: Vec<String> = history
                .iter()
                .rev()
                .take(prev_count)
                .map(|(_, m)| m.clone())
                .collect();

            examples.push(TestExample {
                fen,
                previous_fens,
                previous_moves,
                move_uci,
            });
        }

        for _ in 0..10 {
            let ex = &examples[0];
            let fen: Fen = ex.fen.parse().unwrap();
            let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
            let _ = encode_position(&pos, &[], &[]);
        }

        let start = Instant::now();
        for ex in &examples {
            let fen: Fen = ex.fen.parse().unwrap();
            let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();

            let previous_positions: Vec<Chess> = ex
                .previous_fens
                .iter()
                .map(|f| f.parse::<Fen>().unwrap())
                .map(|f| f.into_position(shakmaty::CastlingMode::Standard).unwrap())
                .collect();

            let _board_encoded = encode_position(&pos, &previous_positions, &ex.previous_moves);

            let mut move_distribution = vec![0.0f32; LEGAL_MOVES];
            let move_uci: UciMove = ex.move_uci.parse().unwrap();
            let (from_idx, to_idx) = encode_move(&move_uci).unwrap();
            let move_idx = from_idx as usize * 76 + to_idx as usize;
            move_distribution[move_idx] = 1.0;

            let _side_info = get_side_info(&pos, &ex.move_uci);

            let mut legal_moves = [0f32; LEGAL_MOVES];
            pos.legal_moves().iter().for_each(|m| {
                let (from, to) = encode_move(&m.to_uci(shakmaty::CastlingMode::Standard)).unwrap();
                let idx = from as usize * 76 + to as usize;
                legal_moves[idx] = 1.0;
            });

            let _material = compute_material_imbalance(&pos);
        }
        let elapsed = start.elapsed();

        let per_position = elapsed / NUM_POSITIONS as u32;
        let positions_per_sec = NUM_POSITIONS as f64 / elapsed.as_secs_f64();

        println!("\n=== Full Pipeline with FEN Parsing ===");
        println!("Positions processed: {}", NUM_POSITIONS);
        println!("Total time: {:?}", elapsed);
        println!("Per position: {:?}", per_position);
        println!("Positions/sec: {:.0}", positions_per_sec);
    }

    #[test]
    fn test_has_pinned_defender() {
        // Position: Black rook a3 "defended" by pawn b4, but pawn is pinned to queen b6 by rook b1
        // Tactic: Qxa3, bxa3??, Rxb6 wins the queen
        let fen = "7k/8/1q6/8/1p6/r7/1Q6/1R2K3 w - - 0 1";
        let fen_parsed: Fen = fen.parse().expect("Valid FEN");
        let pos: Chess = fen_parsed
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");

        let mut pinned_to: [Option<Square>; 64] = [None; 64];
        for square in Square::ALL {
            pinned_to[square as usize] = is_piece_pinned(&pos, square);
        }

        let mut output = String::new();

        let b4 = Square::from_str("b4").unwrap();
        let b6 = Square::from_str("b6").unwrap();
        let a3 = Square::from_str("a3").unwrap();
        let b2 = Square::from_str("b2").unwrap();
        let b1 = Square::from_str("b1").unwrap();

        output += &format!("b4 pawn pinned to: {:?}\n", pinned_to[b4 as usize]);
        output += &format!(
            "a3 rook has_pinned_defender: {}\n",
            has_pinned_defender(&pos, a3, &pinned_to)
        );
        output += &format!(
            "b6 queen has_pinned_defender: {}\n",
            has_pinned_defender(&pos, b6, &pinned_to)
        );
        output += &format!(
            "b2 queen has_pinned_defender: {}\n",
            has_pinned_defender(&pos, b2, &pinned_to)
        );
        output += &format!(
            "b1 rook has_pinned_defender: {}\n",
            has_pinned_defender(&pos, b1, &pinned_to)
        );

        insta::assert_snapshot!(output);
    }
}
