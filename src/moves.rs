use shakmaty::uci::UciMove;
use shakmaty::{Chess, Move, Position, Role};

/// Mirror a FEN position for black's perspective (flip board and colors)
pub fn mirror_fen(fen: &str) -> String {
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
            let file = parts[3].chars().next().unwrap();
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
        Move::Normal { from, capture, .. } => {
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
    fn test_mirror_fen_castling_rights() {
        // Test the specific FEN that was problematic
        let original_fen = "r2qk1nr/2p3pp/p1p1p3/2bpP3/6b1/2NQ1N2/PPP2PPP/R1B2RK1 b kq - 3 10";
        let mirrored = mirror_fen(original_fen);

        // Expected result: colors flipped, board mirrored, castling rights flipped
        // Original: Black to move with black castling rights (kq)
        // Expected: White to move with white castling rights (KQ)
        let expected = "r1b2rk1/ppp2ppp/2nq1n2/6B1/2BPp3/P1P1P3/2P3PP/R2QK1NR w KQ - 3 10";

        assert_eq!(
            mirrored,
            expected,
            "mirror_fen failed for FEN with castling rights.\nOriginal: {}\nActual:   {}\nExpected: {}",
            original_fen,
            mirrored,
            expected,
        );
    }

    #[test]
    fn test_mirror_fen_mixed_castling() {
        // Test mixed castling rights
        let original_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kq - 0 1";
        let mirrored = mirror_fen(original_fen);

        // White K + black q should become black k + white Q
        let expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b kQ - 0 1";

        assert_eq!(mirrored, expected);
    }
}
