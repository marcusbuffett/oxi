use shakmaty::{Bitboard, Chess, Color, Position, Role, Square};

use crate::config::{
    BOARD_FEATURES_PER_TOKEN, FEATURES_PER_TOKEN, HISTORY_DECAY, MISC_FEATURES,
    PIECE_IDENTITY_FEATURES, POSITIONAL_FEATURES, PREVIOUS_POSITIONS, RECENCY_FEATURES,
    TACTICAL_FEATURES,
};

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

    let board = pos.board();

    for square in Square::ALL {
        let mut feature_idx = 0;
        let square_idx = square as usize * FEATURES_PER_TOKEN;

        // Current position features
        let piece = board.piece_at(square);
        let castling_right = if pos.castles().castling_rights().contains(square) {
            1.0
        } else {
            0.0
        };
        let is_hanging = is_piece_hanging(pos, square);
        let legal_moves_norm = normalized_legal_moves_for_square(pos, square);
        let control_value = calculate_square_control(pos, square);
        let see_white_initiates = static_exchange_eval(pos, square, Color::White);
        let see_black_initiates = static_exchange_eval(pos, square, Color::Black);
        let (xray_white_count, xray_white_material) = xray_attackers(pos, square, Color::White);
        let (xray_black_count, xray_black_material) = xray_attackers(pos, square, Color::Black);

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
        tokens[square_idx + feature_idx + tactical_offset] = if is_hanging { 1.0 } else { 0.0 };
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] = control_value;
        tactical_offset += 1;
        // SEE: graded capture outcome (capturer's perspective, pawns / 9)
        tokens[square_idx + feature_idx + tactical_offset] =
            (see_white_initiates.unwrap_or(0) as f32 / 9.0).clamp(-1.0, 1.0);
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] =
            (see_black_initiates.unwrap_or(0) as f32 / 9.0).clamp(-1.0, 1.0);
        tactical_offset += 1;
        // X-ray attackers (count, material) per side, same normalization as
        // the direct attacker channels
        tokens[square_idx + feature_idx + tactical_offset] =
            (xray_white_count as f32 / 6.0).clamp(0.0, 1.0);
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] =
            (xray_white_material / 24.0).clamp(0.0, 1.0);
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] =
            (xray_black_count as f32 / 6.0).clamp(0.0, 1.0);
        tactical_offset += 1;
        tokens[square_idx + feature_idx + tactical_offset] =
            (xray_black_material / 24.0).clamp(0.0, 1.0);
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

        // History occupancy planes (Maia-3 style): 12 piece one-hots for each
        // of the past PREVIOUS_POSITIONS positions, most recent first.
        // Positions are in the same (possibly mirrored) frame as the current
        // board. Plies before the game start are all-zero.
        for h in 0..PREVIOUS_POSITIONS {
            let base = square_idx + feature_idx + h * PIECE_IDENTITY_FEATURES;
            if let Some(prev) = previous_positions.get(h) {
                if let Some(prev_piece) = prev.board().piece_at(square) {
                    let piece_offset = prev_piece.role as usize - 1;
                    let color_offset = match prev_piece.color {
                        Color::White => 0,
                        Color::Black => 6,
                    };
                    tokens[base + color_offset + piece_offset] = 1.0;
                }
            }
        }
        feature_idx += crate::config::HISTORY_OCCUPANCY_FEATURES;

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

fn opposite_color(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// Standard piece values for static exchange evaluation, in pawns.
fn see_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight => 3,
        Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 100,
    }
}

/// Static exchange evaluation: net material gain, in pawns, for
/// `attacker_color` initiating the capture sequence on `square`. Each side
/// recaptures with its least valuable attacker; removing a capturer from the
/// occupancy reveals x-ray backup automatically (`attacks_to` is
/// occupancy-parameterized); either side may stand pat. The king joins only
/// when the opponent has no defender left. Returns None when there is no
/// enemy piece on the square or no attacker.
pub fn static_exchange_eval(pos: &Chess, square: Square, attacker_color: Color) -> Option<i32> {
    let board = pos.board();
    let target = board.piece_at(square)?;
    if target.color == attacker_color {
        return None;
    }

    let mut occupied = board.occupied();
    let mut captured_value = see_value(target.role);
    let mut side = attacker_color;
    let mut gains: Vec<i32> = Vec::with_capacity(8);

    loop {
        let attackers = board.attacks_to(square, side, occupied) & occupied;
        let mut least: Option<(Square, Role)> = None;
        for sq in attackers {
            if let Some(piece) = board.piece_at(sq) {
                if least.map_or(true, |(_, role)| see_value(piece.role) < see_value(role)) {
                    least = Some((sq, piece.role));
                }
            }
        }
        let Some((from_sq, role)) = least else { break };

        if role == Role::King {
            let mut without_king = occupied;
            without_king.toggle(from_sq);
            let defenders =
                board.attacks_to(square, opposite_color(side), without_king) & without_king;
            if !defenders.is_empty() {
                break;
            }
        }

        gains.push(captured_value);
        captured_value = see_value(role);
        occupied.toggle(from_sq);
        side = opposite_color(side);
    }

    if gains.is_empty() {
        return None;
    }

    // Speculative running totals, then backward negamax with standing pat.
    let mut spec = gains.clone();
    for i in 1..spec.len() {
        spec[i] = gains[i] - spec[i - 1];
    }
    for i in (1..spec.len()).rev() {
        spec[i - 1] = -std::cmp::max(-spec[i - 1], spec[i]);
    }
    Some(spec[0])
}

/// Count and material of `color`'s sliders that attack `square` through
/// exactly one intervening piece: batteries (own piece in front), pins and
/// skewers (enemy piece in front), and discovered-attack potential. Direct
/// attackers (zero blockers) are excluded — they live in the regular
/// attacker channels.
pub fn xray_attackers(pos: &Chess, square: Square, color: Color) -> (usize, f32) {
    use shakmaty::attacks;
    let board = pos.board();
    let occupied = board.occupied();
    let by_color = board.by_color(color);
    let rook_like = (board.by_role(Role::Rook) | board.by_role(Role::Queen)) & by_color;
    let bishop_like = (board.by_role(Role::Bishop) | board.by_role(Role::Queen)) & by_color;
    let snipers = (attacks::rook_attacks(square, Bitboard::EMPTY) & rook_like)
        | (attacks::bishop_attacks(square, Bitboard::EMPTY) & bishop_like);

    let mut count = 0usize;
    let mut material = 0.0f32;
    for sniper in snipers {
        if (attacks::between(square, sniper) & occupied).count() == 1 {
            if let Some(piece) = board.piece_at(sniper) {
                count += 1;
                material += get_piece_value(piece.role);
            }
        }
    }
    (count, material)
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

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use std::str::FromStr;

    fn parse_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("valid FEN")
            .into_position(shakmaty::CastlingMode::Standard)
            .expect("valid position")
    }

    #[test]
    fn test_see_simple_hanging_pawn() {
        // White queen attacks an undefended black pawn on d5.
        let pos = parse_fen("4k3/8/8/3p4/8/8/3Q4/4K3 w - - 0 1");
        assert_eq!(
            static_exchange_eval(&pos, Square::D5, Color::White),
            Some(1),
            "undefended pawn: clean win of 1"
        );
        assert_eq!(
            static_exchange_eval(&pos, Square::D5, Color::Black),
            None,
            "black can't capture its own pawn"
        );
    }

    #[test]
    fn test_see_defended_pawn_bad_for_queen() {
        // Black pawn on d5 defended by a pawn on e6; only white attacker is the queen.
        let pos = parse_fen("4k3/8/4p3/3p4/8/8/3Q4/4K3 w - - 0 1");
        assert_eq!(
            static_exchange_eval(&pos, Square::D5, Color::White),
            Some(-8),
            "QxP, PxQ: win 1, lose 9"
        );
    }

    #[test]
    fn test_see_knight_takes_defended_queen() {
        // Black queen on d5 defended by pawn e6; white knight on c3 can take.
        let pos = parse_fen("4k3/8/4p3/3q4/8/2N5/8/4K3 w - - 0 1");
        assert_eq!(
            static_exchange_eval(&pos, Square::D5, Color::White),
            Some(6),
            "NxQ, PxN: win 9, lose 3"
        );
    }

    #[test]
    fn test_see_xray_recapture_through_battery() {
        // The Complex Sicilian Probe blunder square: white Qb2 with Rb1
        // behind it. A black rook landing on b3 is captured by the queen,
        // and the rook's recapture is backed by the x-ray rook.
        let pos = parse_fen("6k1/5p1p/1q2p1p1/p3P3/1p3P2/1r2b2P/1Q2N1P1/1R3K2 w - - 2 34");
        assert_eq!(
            static_exchange_eval(&pos, Square::B3, Color::White),
            Some(5),
            "Qxb3 wins the rook; Qxb3 recapture is covered by Rb1 x-ray"
        );
    }

    #[test]
    fn test_xray_battery_and_blocked_ray() {
        // Probe position: Qb2 directly covers b3, Rb1 covers it only by x-ray.
        let pos = parse_fen("6k1/5p1p/1q2p1p1/p3P3/1p3P2/r3b2P/1Q2N1P1/1R3K2 b - - 1 33");
        let (count, material) = xray_attackers(&pos, Square::B3, Color::White);
        assert_eq!(count, 1, "Rb1 x-rays b3 through Qb2");
        assert_eq!(material as i32, 5);

        // No black slider x-rays b3 (queen on b6 attacks it directly through
        // nothing; b4 pawn blocks the file with two pieces between for b8).
        let (b_count, _) = xray_attackers(&pos, Square::G2, Color::Black);
        assert_eq!(b_count, 0, "no black x-ray onto g2");
    }

    /// Square control across every square of complex positions, snapshotted so
    /// performance work on the feature can't silently change its semantics.
    #[test]
    fn test_square_control_snapshots() {
        let positions = [
            (
                // Complex Sicilian Probe: queen+rook battery, loose pieces
                "complex_sicilian",
                "6k1/5p1p/1q2p1p1/p3P3/1p3P2/r3b2P/1Q2N1P1/1R3K2 b - - 1 33",
            ),
            (
                // Sharp Najdorf middlegame: many contested central squares
                "najdorf_middlegame",
                "r1b1k2r/1pq1bppp/p1nppn2/8/3NPP2/2N1B3/PPPQ2PP/2KR1B1R w kq - 3 10",
            ),
            (
                // Rook endgame: open files, king activity
                "rook_endgame",
                "8/5pk1/6p1/1R5p/2r4P/6P1/5PK1/8 w - - 0 40",
            ),
        ];
        for (name, fen) in positions {
            let pos = parse_fen(fen);
            let mut output = format!("{fen}\n");
            for rank in (0..8).rev() {
                for file in 0..8 {
                    let square = Square::new(rank * 8 + file);
                    output.push_str(&format!("{:.4} ", calculate_square_control(&pos, square)));
                }
                output.push('\n');
            }
            insta::assert_snapshot!(format!("square_control_{name}"), output);
        }
    }

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
}
