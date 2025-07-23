use once_cell::sync::Lazy;
use shakmaty::attacks::{knight_attacks, queen_attacks};
use shakmaty::uci::UciMove;
use shakmaty::{Bitboard, Chess, Move, Position, Role, Square};
use std::collections::HashMap;

/// Convert a square to its index (0-63)
pub fn square_to_index(square: Square) -> usize {
    square as usize
}

/// Convert UCI move to from-to indices for 64x64 representation
pub fn encode_move_spatial(uci_move: &str) -> Option<(usize, usize)> {
    let uci = uci_move.parse::<UciMove>().ok()?;

    match &uci {
        UciMove::Normal { from, to, .. } => {
            let from_idx = square_to_index(*from);
            let to_idx = square_to_index(*to);
            Some((from_idx, to_idx))
        }
        UciMove::Put { to, .. } => {
            // For drop moves, use the to square for both from and to
            let to_idx = square_to_index(*to);
            Some((to_idx, to_idx))
        }
        UciMove::Null => None,
    }
}

/// Convert from-to indices back to UCI string
pub fn decode_move_spatial(from_idx: usize, to_idx: usize) -> String {
    let from = Square::new(from_idx as u32);
    let to = Square::new(to_idx as u32);
    format!("{}{}", from, to)
}

/// Generate all possible moves following Oxi's approach
/// Returns exactly 1880 moves in UCI format
pub fn get_all_possible_moves() -> Vec<String> {
    let mut all_moves = Vec::new();

    // Generate queen and knight moves from each square
    let empty_board = Bitboard::EMPTY;

    for from_square in Square::ALL {
        // Queen moves
        let queen_targets = queen_attacks(from_square, empty_board);
        for to_square in queen_targets {
            all_moves.push(format!("{}{}", from_square, to_square));
        }

        // Knight moves
        let knight_targets = knight_attacks(from_square);
        for to_square in knight_targets {
            all_moves.push(format!("{}{}", from_square, to_square));
        }
    }

    // Add pawn promotions
    let pawn_promotions = generate_pawn_promotions();
    all_moves.extend(pawn_promotions);

    all_moves
}

/// Generate all pawn promotion moves
fn generate_pawn_promotions() -> Vec<String> {
    let mut promotions = Vec::new();
    let promotion_pieces = ['q', 'r', 'b', 'n'];

    // Only white promotions (as in Oxi)
    let row = '7';
    let target_row = '8';

    for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] {
        // Direct promotion
        for piece in promotion_pieces {
            promotions.push(format!("{}{}{}{}{}", file, row, file, target_row, piece));
        }

        // Capturing to the left
        if file != 'a' {
            let left_file = (file as u8 - 1) as char;
            for piece in promotion_pieces {
                promotions.push(format!(
                    "{}{}{}{}{}",
                    file, row, left_file, target_row, piece
                ));
            }
        }

        // Capturing to the right
        if file != 'h' {
            let right_file = (file as u8 + 1) as char;
            for piece in promotion_pieces {
                promotions.push(format!(
                    "{}{}{}{}{}",
                    file, row, right_file, target_row, piece
                ));
            }
        }
    }

    promotions
}

/// Global move dictionary for fast lookup
pub static MOVE_DICT: Lazy<HashMap<String, usize>> = Lazy::new(|| {
    let moves = get_all_possible_moves();
    moves.into_iter().enumerate().map(|(i, m)| (m, i)).collect()
});

/// Global reverse move dictionary for decoding
pub static MOVE_DICT_REVERSED: Lazy<HashMap<usize, String>> =
    Lazy::new(|| MOVE_DICT.iter().map(|(k, v)| (*v, k.clone())).collect());

/// Encode a UCI move string to an index
pub fn encode_move_uci(uci_move: &str) -> Option<usize> {
    MOVE_DICT.get(uci_move).copied()
}

/// Decode a move index to UCI string
pub fn decode_move_index(index: usize) -> Option<String> {
    MOVE_DICT_REVERSED.get(&index).cloned()
}

/// Mirror a move for black's perspective (as in Oxi)
pub fn mirror_move(uci_move: &str) -> String {
    if uci_move.len() < 4 {
        return uci_move.to_string();
    }

    let bytes = uci_move.as_bytes();
    let from_file = bytes[0];
    let from_rank = bytes[1];
    let to_file = bytes[2];
    let to_rank = bytes[3];

    let mut mirrored = String::new();
    mirrored.push(from_file as char);
    mirrored.push((b'1' + b'8' - from_rank) as char);
    mirrored.push(to_file as char);
    mirrored.push((b'1' + b'8' - to_rank) as char);

    // Handle promotion
    if uci_move.len() > 4 {
        mirrored.push_str(&uci_move[4..]);
    }

    mirrored
}

/// Get side info for a move (following Oxi's format)
/// Returns (legal_moves_mask_64x64, side_info)
pub fn get_side_info(pos: &Chess, uci_move: &str) -> (Vec<f32>, Vec<i32>) {
    let uci = match uci_move.parse::<UciMove>() {
        Ok(u) => u,
        Err(_) => {
            tracing::warn!("Warning: Invalid UCI move: {}", uci_move);
            return (vec![0f32; 4096], vec![0i32; 13]);
        }
    };
    let mv = match uci.to_move(pos) {
        Ok(m) => m,
        Err(_) => {
            tracing::warn!("Warning: Illegal move {} for position", uci_move);
            return (vec![0f32; 4096], vec![0i32; 13]);
        }
    };

    // Create legal moves mask for 64x64 representation
    let mut legal_moves = vec![0f32; 4096]; // 64x64 flattened
    for legal_mv in pos.legal_moves() {
        let legal_uci = legal_mv
            .to_uci(shakmaty::CastlingMode::Standard)
            .to_string();
        if let Some((from_idx, to_idx)) = encode_move_spatial(&legal_uci) {
            // Convert 2D indices to flat index
            let flat_idx = from_idx * 64 + to_idx;
            legal_moves[flat_idx] = 1.0;
        }
    }

    // Create side info vector
    let mut side_info = vec![0i32; 6 + 6 + 1]; // pieces + captured + check

    match &mv {
        Move::Normal {
            from, to, capture, ..
        } => {
            // Moving piece type
            if let Some(piece) = pos.board().piece_at(*from) {
                let piece_idx = match piece.role {
                    Role::Pawn => 0,
                    Role::Knight => 1,
                    Role::Bishop => 2,
                    Role::Rook => 3,
                    Role::Queen => 4,
                    Role::King => 5,
                };
                side_info[piece_idx] = 1;
            }

            // Special handling for castling moves
            if uci_move == "e1g1" || uci_move == "e1c1" {
                side_info[3] = 1; // Rook
            }

            // Captured piece (if any)
            if let Some(captured) = capture {
                let captured_idx = match captured {
                    Role::Pawn => 0,
                    Role::Knight => 1,
                    Role::Bishop => 2,
                    Role::Rook => 3,
                    Role::Queen => 4,
                    Role::King => 5,
                };
                side_info[6 + captured_idx] = 1;
            }

            // From/to square encoding
            // let from_idx = 13 + from.file() as usize * 8 + from.rank() as usize;
            // let to_idx = 13 + 64 + to.file() as usize * 8 + to.rank() as usize;
            // side_info[from_idx] = 1;
            // side_info[to_idx] = 1;

            // // Castling rook moves
            // if uci_move == "e1g1" {
            //     side_info[13 + 7] = 1; // h1
            //     side_info[13 + 64 + 5] = 1; // f1
            // } else if uci_move == "e1c1" {
            //     side_info[13 + 0] = 1; // a1
            //     side_info[13 + 64 + 3] = 1; // d1
            // }
        }
        Move::EnPassant { .. } => {
            // Moving piece type (always pawn for en passant)
            side_info[0] = 1; // Pawn

            // Captured piece (always pawn for en passant)
            side_info[6] = 1; // Captured pawn
        }
        Move::Castle { .. } => {
            // This shouldn't happen with UCI notation
            side_info[5] = 1; // King
            side_info[3] = 1; // Rook
        }
        _ => {}
    }

    // Check if move gives check
    let mut new_pos = pos.clone();
    new_pos.play_unchecked(mv);
    if new_pos.is_check() {
        side_info[12] = 1;
    }

    (legal_moves, side_info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;

    #[test]
    fn test_move_count() {
        let moves = get_all_possible_moves();
        assert_eq!(moves.len(), 1880, "Should have exactly 1880 moves");
    }

    #[test]
    fn test_mirror_move() {
        assert_eq!(mirror_move("e2e4"), "e7e5");
        assert_eq!(mirror_move("a1a8"), "a8a1");
        assert_eq!(mirror_move("e7e8q"), "e2e1q");
    }

    #[test]
    fn test_move_encoding() {
        let moves = get_all_possible_moves();

        // Test that encoding and decoding are inverses
        for (i, mv) in moves.iter().enumerate() {
            assert_eq!(encode_move_uci(mv), Some(i));
            assert_eq!(decode_move_index(i), Some(mv.clone()));
        }
    }

    #[test]
    fn test_get_side_info_nf3() {
        // Test g1f3 (Nf3) from starting position
        let starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen: Fen = starting_pos.parse().unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();

        let (legal_moves, side_info) = get_side_info(&pos, "g1f3");

        // Verify legal moves mask
        assert_eq!(legal_moves.len(), 4096);

        // Count legal moves - should be 20 in starting position
        let legal_count = legal_moves.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(legal_count, 20);

        // Verify side info vector
        assert_eq!(side_info.len(), 141);

        // Check piece type - should be knight (index 1)
        assert_eq!(side_info[1], 1);
        assert_eq!(side_info[0], 0); // Not a pawn
        assert_eq!(side_info[2], 0); // Not a bishop
        assert_eq!(side_info[3], 0); // Not a rook
        assert_eq!(side_info[4], 0); // Not a queen
        assert_eq!(side_info[5], 0); // Not a king

        // No capture
        for i in 6..12 {
            assert_eq!(side_info[i], 0);
        }

        // Not a check
        assert_eq!(side_info[12], 0);

        // From square g1 - file 6, rank 0 -> index 13 + 6*8 + 0 = 13 + 48 = 61
        assert_eq!(side_info[61], 1);

        // To square f3 - file 5, rank 2 -> index 13 + 64 + 5*8 + 2 = 77 + 40 + 2 = 119
        assert_eq!(side_info[119], 1);

        // Verify all other positions are 0
        let mut non_zero_count = 0;
        for (i, &val) in side_info.iter().enumerate() {
            if val != 0 {
                non_zero_count += 1;
                assert!(
                    i == 1 || i == 61 || i == 119,
                    "Unexpected non-zero at index {}",
                    i
                );
            }
        }
        assert_eq!(non_zero_count, 3); // Knight piece, from square, to square
    }
}
