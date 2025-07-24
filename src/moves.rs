use shakmaty::uci::UciMove;
use shakmaty::{Chess, Move, Position, Role};

/// Mirror a move for black's perspective (as in Oxi)
pub fn mirror_move(uci_move: &str) -> String {
    if uci_move.len() < 4 {
        panic!("Invalid UCI move: {uci_move}");
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
pub fn get_side_info(pos: &Chess, uci_move: &str) -> Vec<i32> {
    let uci = match uci_move.parse::<UciMove>() {
        Ok(u) => u,
        Err(_) => {
            tracing::warn!("Warning: Invalid UCI move: {}", uci_move);
            return vec![0i32; 13];
        }
    };
    let mv = match uci.to_move(pos) {
        Ok(m) => m,
        Err(_) => {
            tracing::warn!("Warning: Illegal move {} for position", uci_move);
            return vec![0i32; 13];
        }
    };

    // Create side info vector
    let mut side_info = vec![0i32; 6 + 6 + 1]; // pieces + captured + check

    match &mv {
        Move::Normal {
            from,  capture, ..
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

    side_info
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;

    #[test]
    fn test_mirror_move() {
        assert_eq!(mirror_move("e2e4"), "e7e5");
        assert_eq!(mirror_move("a1a8"), "a8a1");
        assert_eq!(mirror_move("e7e8q"), "e2e1q");
    }

    #[test]
    fn test_get_side_info_nf3() {
        // Test g1f3 (Nf3) from starting position
        let starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen: Fen = starting_pos.parse().unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();

        let side_info = get_side_info(&pos, "g1f3");

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
