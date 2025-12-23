use std::f32::consts::FRAC_1_SQRT_2;

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

const DIRECTION_UNIT_VECTORS: [(f32, f32); 8] = [
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
    (-FRAC_1_SQRT_2, FRAC_1_SQRT_2),
    (FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
    (-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
];

const RAY_DISTANCE_DECAY: f32 = 0.5;

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
        let is_passed = is_passed_pawn(pos, square);
        let legal_moves_norm = normalized_legal_moves_for_square(pos, square);
        let (isolated, backward, doubled) = pawn_structure_features(pos, square);
        let weak_white_square = is_weak_square(pos, square, Color::White);
        let weak_black_square = is_weak_square(pos, square, Color::Black);
        let open_file = is_open_file(pos, square);
        let control_value = calculate_square_control(pos, square);
        let (ray_piece_weights, ray_pin_flag) = compute_ray_features(board, square);

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

        for dir_weights in ray_piece_weights.iter() {
            for &weight in dir_weights.iter() {
                tokens[square_idx + feature_idx + tactical_offset] = weight;
                tactical_offset += 1;
            }
        }

        tokens[square_idx + feature_idx + tactical_offset] = ray_pin_flag;
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

fn ray_piece_group(role: Role) -> usize {
    match role {
        Role::Knight | Role::Bishop => 0, // Minors
        Role::Rook | Role::Queen => 1,    // Majors
        Role::Pawn => 2,
        Role::King => 3,
    }
}

fn ray_piece_index(color: Color, role: Role) -> usize {
    let color_offset = match color {
        Color::White => 0,
        Color::Black => 4,
    };
    color_offset + ray_piece_group(role)
}

fn compute_ray_features(board: &shakmaty::Board, origin: Square) -> ([[f32; 8]; 8], f32) {
    let mut directional_weights = [[0.0f32; 8]; 8];
    let mut any_pin = 0.0f32;
    let origin_piece = board.piece_at(origin);
    let origin_color = origin_piece.map(|piece| piece.color);
    let origin_role = origin_piece.map(|piece| piece.role);

    for (dir_idx, &(dx, dy)) in OPEN_DIRECTIONS.iter().enumerate() {
        let mut file = origin.file() as i32 + dx;
        let mut rank = origin.rank() as i32 + dy;
        let mut distance = 1;
        let mut candidate_role: Option<Role> = None;
        let mut candidate_value = 0.0f32;
        let mut can_pin = origin_role
            .map(|role| pinner_can_attack_direction(role, (dx, dy)))
            .unwrap_or(false);

        while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
            let target_sq = Square::new((rank * 8 + file) as u32);
            if let Some(piece) = board.piece_at(target_sq) {
                let idx = ray_piece_index(piece.color, piece.role);
                let contribution = RAY_DISTANCE_DECAY.powi((distance - 1) as i32);
                directional_weights[dir_idx][idx] += contribution;

                if can_pin {
                    match origin_color {
                        Some(color) => {
                            if piece.color == color {
                                can_pin = false;
                            } else if let Some(c_role) = candidate_role {
                                let behind_value = get_piece_value(piece.role);
                                if behind_value > candidate_value
                                    && !role_moves_in_direction(c_role, (dx, dy))
                                {
                                    any_pin = 1.0;
                                }
                                can_pin = false;
                            } else {
                                candidate_role = Some(piece.role);
                                candidate_value = get_piece_value(piece.role);
                            }
                        }
                        None => can_pin = false,
                    }
                }
            }

            file += dx;
            rank += dy;
            distance += 1;
        }
    }

    for weights in directional_weights.iter_mut() {
        for weight in weights.iter_mut() {
            *weight = (*weight / 2.0).clamp(0.0, 1.0);
        }
    }

    (directional_weights, any_pin)
}

fn normalize_vec(vector: (f32, f32)) -> (f32, f32) {
    let magnitude = (vector.0 * vector.0 + vector.1 * vector.1).sqrt();
    if magnitude == 0.0 {
        (0.0, 0.0)
    } else {
        (vector.0 / magnitude, vector.1 / magnitude)
    }
}

fn best_direction_bucket(vector: (f32, f32)) -> usize {
    let (vx, vy) = normalize_vec(vector);
    let mut best_idx = 0;
    let mut best_dot = -1.0;
    for (idx, &(dx, dy)) in DIRECTION_UNIT_VECTORS.iter().enumerate() {
        let dot = vx * dx + vy * dy;
        if dot > best_dot {
            best_dot = dot;
            best_idx = idx;
        }
    }
    best_idx
}

fn accumulate_ray_open_count(
    board: &shakmaty::Board,
    origin: Square,
    direction: (i32, i32),
    color: Color,
    max_range: f32,
    limit_single_step: bool,
) -> f32 {
    let (dx, dy) = direction;
    let mut cx = origin.file() as i32 + dx;
    let mut cy = origin.rank() as i32 + dy;
    let mut steps = 0.0f32;

    while cx >= 0 && cx < 8 && cy >= 0 && cy < 8 {
        let target_sq = Square::new((cy * 8 + cx) as u32);
        if let Some(target_piece) = board.piece_at(target_sq) {
            if target_piece.color != color {
                steps += 1.0;
            }
            break;
        } else {
            steps += 1.0;
        }

        if steps >= max_range || limit_single_step {
            break;
        }

        cx += dx;
        cy += dy;
    }

    steps
}

fn accumulate_knight_open_counts(
    board: &shakmaty::Board,
    origin: Square,
    color: Color,
    open_counts: &mut [f32; 8],
) {
    let knight_deltas = [
        (1, 2),
        (2, 1),
        (-1, 2),
        (-2, 1),
        (1, -2),
        (2, -1),
        (-1, -2),
        (-2, -1),
    ];

    let mut bucket_totals = [0.0f32; 8];

    for &(kdx, kdy) in &knight_deltas {
        let kx = origin.file() as i32 + kdx;
        let ky = origin.rank() as i32 + kdy;
        if kx < 0 || kx >= 8 || ky < 0 || ky >= 8 {
            continue;
        }

        let bucket = best_direction_bucket((kdx as f32, kdy as f32));
        let target_sq = Square::new((ky * 8 + kx) as u32);
        let is_open = match board.piece_at(target_sq) {
            Some(target_piece) => target_piece.color != color,
            None => true,
        };

        bucket_totals[bucket] += 1.0;
        if is_open {
            open_counts[bucket] += 1.0;
        }
    }

    for idx in 0..8 {
        if bucket_totals[idx] > 0.0 {
            open_counts[idx] = (open_counts[idx] / bucket_totals[idx]).clamp(0.0, 1.0);
        }
    }
}

fn accumulate_pawn_open_counts(
    board: &shakmaty::Board,
    origin: Square,
    color: Color,
    open_counts: &mut [f32; 8],
) {
    let forward_delta = if color == Color::White {
        (0, 1)
    } else {
        (0, -1)
    };
    let capture_deltas = if color == Color::White {
        [(-1, 1), (1, 1)]
    } else {
        [(-1, -1), (1, -1)]
    };

    for (dir_idx, &(dx, dy)) in OPEN_DIRECTIONS.iter().enumerate() {
        let tx = origin.file() as i32 + dx;
        let ty = origin.rank() as i32 + dy;
        if tx < 0 || tx >= 8 || ty < 0 || ty >= 8 {
            continue;
        }

        let target_sq = Square::new((ty * 8 + tx) as u32);
        if (dx, dy) == forward_delta {
            if board.piece_at(target_sq).is_none() {
                open_counts[dir_idx] = 1.0;
            }
        } else if capture_deltas.contains(&(dx, dy)) {
            if let Some(target_piece) = board.piece_at(target_sq) {
                if target_piece.color != color {
                    open_counts[dir_idx] = 1.0;
                }
            }
        }
    }
}

fn accumulate_slider_open_counts(
    board: &shakmaty::Board,
    origin: Square,
    color: Color,
    role: Role,
    open_counts: &mut [f32; 8],
) {
    let (max_range, limit_to_single_step) = if role == Role::King {
        (1.0f32, true)
    } else {
        (7.0f32, false)
    };

    for (dir_idx, &(dx, dy)) in OPEN_DIRECTIONS.iter().enumerate() {
        let is_cardinal = dx == 0 || dy == 0;
        let is_diagonal = dx != 0 && dy != 0;
        let relevant = match role {
            Role::Bishop => is_diagonal,
            Role::Rook => is_cardinal,
            Role::Queen => true,
            Role::King => true,
            _ => false,
        };
        if !relevant {
            continue;
        }

        let steps = accumulate_ray_open_count(
            board,
            origin,
            (dx, dy),
            color,
            max_range,
            limit_to_single_step,
        );
        open_counts[dir_idx] = (steps / max_range).clamp(0.0, 1.0);
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
    fn test_weak_square_rank_gating() {
        // Any white weak-square marking on ranks 1-2 should be false
        // Minimal valid board: place both kings
        let fen = "7k/8/8/8/8/8/8/K7 w - - 0 1";
        let fen_parsed: Fen = fen.parse().expect("Valid FEN");
        let pos: Chess = fen_parsed
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("Valid position");

        let d2 = Square::from_str("d2").unwrap();
        let e1 = Square::from_str("e1").unwrap();
        assert!(!is_weak_square(&pos, d2, Color::White));
        assert!(!is_weak_square(&pos, e1, Color::White));

        // Any black weak-square marking on ranks 7-8 should be false
        let e7 = Square::from_str("e7").unwrap();
        let h8 = Square::from_str("h8").unwrap();
        assert!(!is_weak_square(&pos, e7, Color::Black));
        assert!(!is_weak_square(&pos, h8, Color::Black));
    }
}
