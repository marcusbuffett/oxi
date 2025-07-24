use std::sync::OnceLock;

use shakmaty::{uci::UciMove, Role, Square};

type Table = [[[i8; 5]; 64]; 64];
type DecodeTable = [[(u8, u8); 76]; 64];

static TABLES: OnceLock<(Table, DecodeTable)> = OnceLock::new();

fn build_tables() -> (Table, DecodeTable) {
    let mut table: Table = [[[-1; 5]; 64]; 64];

    for from in 0..64 {
        for to in 0..64 {
            if from == to {
                continue;
            }
            let rank_from = from / 8;
            let file_from = from % 8;
            let rank_to = to / 8;
            let file_to = to % 8;
            let dr = rank_to - rank_from;
            let df = file_to - file_from;

            // Slider moves (for None and Queen promotions)
            if let Some(idx) = compute_slider_idx(dr, df) {
                table[from as usize][to as usize][0] = idx; // None
            }

            // Knight moves (only for None)
            if let Some(idx) = compute_knight_idx(dr, df) {
                table[from as usize][to as usize][0] = idx;
            }

            // Underpromotions (Knight=1, Bishop=2, Rook=3)
            if dr == 1 && df.abs() <= 1 {
                let promo_dir_idx = match df {
                    -1 => 1, // queen-side (left)
                    0 => 0,  // straight
                    1 => 2,  // king-side (right)
                    _ => continue,
                };
                for role_idx in 0..4 {
                    let idx = 64 + promo_dir_idx * 4 + role_idx;
                    table[from as usize][to as usize][role_idx + 1] = idx as i8;
                }
            }
        }
    }

    let mut decode: DecodeTable = [[(255, 255); 76]; 64];
    for from in 0..64usize {
        for to in 0..64usize {
            for promo in 0..5usize {
                let idx = table[from][to][promo];
                if idx >= 0 {
                    let idx_us = idx as usize;
                    decode[from][idx_us] = (to as u8, promo as u8);
                }
            }
        }
    }

    (table, decode)
}

fn get_move_mapping() -> &'static Table {
    &TABLES.get_or_init(build_tables).0
}

fn get_move_decoding() -> &'static DecodeTable {
    &TABLES.get_or_init(build_tables).1
}

fn compute_slider_idx(dr: i32, df: i32) -> Option<i8> {
    if dr == 0 && df == 0 {
        return None;
    }
    let abs_dr = dr.abs();
    let abs_df = df.abs();
    let steps = abs_dr.max(abs_df);
    if !(1..=7).contains(&steps) {
        return None;
    }
    if (abs_dr != 0 && abs_df != 0) && abs_dr != abs_df {
        return None;
    }
    if dr != dr.signum() * steps || df != df.signum() * steps {
        return None;
    }
    let dir_idx = if abs_df == 0 {
        if dr > 0 {
            0
        } else {
            1
        } // N, S
    } else if abs_dr == 0 {
        if df > 0 {
            2
        } else {
            3
        } // E, W
    } else if dr > 0 && df > 0 {
        4 // NE
    } else if dr > 0 && df < 0 {
        5 // NW
    } else if dr < 0 && df > 0 {
        6 // SE
    } else {
        7 // SW
    };
    let idx = dir_idx * 7 + (steps - 1);
    Some(idx as i8)
}

fn compute_knight_idx(dr: i32, df: i32) -> Option<i8> {
    let knight_deltas: [(i32, i32); 8] = [
        (2, 1),
        (2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (-2, 1),
        (-2, -1),
    ];
    for (i, &(d_r, d_f)) in knight_deltas.iter().enumerate() {
        if dr == d_r && df == d_f {
            return Some((56 + i) as i8);
        }
    }
    None
}

pub fn encode_move(uci_move: &UciMove) -> Option<(u8, u8)> {
    match uci_move {
        UciMove::Normal {
            from,
            to,
            promotion,
        } => {
            let mapping = get_move_mapping();
            let promo_idx = match promotion {
                Some(Role::Knight) => 1,
                Some(Role::Bishop) => 2,
                Some(Role::Rook) => 3,
                Some(Role::Queen) => 4,
                _ => 0,
            };
            let idx = mapping[*from as usize][*to as usize][promo_idx];
            if idx == -1 {
                None
            } else {
                Some((*from as u32 as u8, idx as u8))
            }
        }
        _ => None,
    }
}

pub fn decode_move(from: u8, idx: u8) -> Option<UciMove> {
    let mapping = get_move_decoding();
    let (to, promo_idx) = mapping[from as usize][idx as usize];
    let role = match promo_idx {
        0 => None,
        1 => Some(Role::Knight),
        2 => Some(Role::Bishop),
        3 => Some(Role::Rook),
        4 => Some(Role::Queen),
        _ => None,
    };
    if to >= 64 {
        return None;
    }
    Some(UciMove::Normal {
        from: Square::new(from as u32),
        to: Square::new(to as u32),
        promotion: role,
    })
}

#[cfg(test)]
mod tests {
    use shakmaty::uci::UciMove;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_move_encoding() {
        // Initialize the mapping
        get_move_mapping();

        let test_cases = vec![
            // Pawn double push: e2e4 -> idx 1 (N direction, steps=2 -> 0*7 +1=1)
            ("e2e4", 12, 1),
            // Knight move: b1c3 -> idx 56 (first knight delta)
            ("b1c3", 1, 56),
            // Knight move: g1f3 -> idx 57 (second knight delta)
            ("g1f3", 6, 57),
            // Promotion to queen straight: e7e8q -> idx 0 (N, steps=1 -> 0)
            ("e7e8q", 52, 67),
            // Underpromotion to knight straight: e7e8n -> idx 64
            ("e7e8n", 52, 64),
            ("d1e1", 3, 14),
            // Promotion to queen capture left: e7d8q -> idx 35 (NW, steps=1 -> 5*7 +0=35)
            ("e7d8q", 52, 71),
            // Underpromotion to knight capture left: e7d8n -> idx 65
            ("e7d8n", 52, 68),
            // Underpromotion to bishop capture right: e7f8b -> idx 68 (dir=2, role=1 -> 64+3+2=69? wait)
            // role_idx: knight=0:64+0*3+dir, bishop=1:64+3+dir, rook=2:64+6+dir
            // For f8: df=5-4=1, dir=2
            // bishop: 64+3+2=69
            ("e7f8b", 52, 73),
            // Long slider: a1a8 -> idx 6 (N, steps=7 -> 0*7+6=6)
            ("a1a8", 0, 6),
            // Diagonal slider: a1b2 -> idx 28 (NE, steps=1 ->4*7+0=28)
            ("a1b2", 0, 28),
        ];

        for (uci, exp_from, exp_idx) in test_cases {
            let uci_move: UciMove = uci.parse().unwrap();
            dbg!(uci_move);
            let encoded = encode_move(&uci_move).expect(&format!("Encoding failed for {}", uci));
            dbg!(uci_move, encoded);
            assert_eq!(encoded, (exp_from, exp_idx), "Failed for UCI: {}", uci);
            let decoded = decode_move(encoded.0, encoded.1);
            assert_eq!(decoded, Some(uci_move));
        }
        dbg!(decode_move(3, 20));
    }

    #[test]
    fn test_all_possible_uci_moves_covered() {
        // Initialize the mapping
        get_move_mapping();

        let mut total_moves = 0;
        let mut encoding_failures = Vec::new();
        let mut decoding_failures = Vec::new();

        // Generate all possible UCI moves systematically
        for from_square in 0..64 {
            let from_file = from_square % 8;
            let from_rank = from_square / 8;

            // Generate all knight moves from this square
            let knight_deltas = [
                (2, 1),
                (2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
                (-2, 1),
                (-2, -1),
            ];

            for (dr, df) in knight_deltas {
                let to_rank = from_rank as i32 + dr;
                let to_file = from_file as i32 + df;

                if to_rank >= 0 && to_rank < 8 && to_file >= 0 && to_file < 8 {
                    let to_square = (to_rank * 8 + to_file) as u8;
                    let uci_move = UciMove::Normal {
                        from: Square::new(from_square as u32),
                        to: Square::new(to_square as u32),
                        promotion: None,
                    };

                    total_moves += 1;
                    test_move_encoding_decoding(
                        &uci_move,
                        &mut encoding_failures,
                        &mut decoding_failures,
                    );
                }
            }

            // Generate all sliding moves (8 directions, 1-7 steps)
            let directions = [
                (0, 1),   // North
                (0, -1),  // South
                (1, 0),   // East
                (-1, 0),  // West
                (1, 1),   // NE
                (1, -1),  // NW
                (-1, 1),  // SE
                (-1, -1), // SW
            ];

            for (dr, df) in directions {
                for steps in 1..8 {
                    let to_rank = from_rank as i32 + dr * steps;
                    let to_file = from_file as i32 + df * steps;

                    if to_rank >= 0 && to_rank < 8 && to_file >= 0 && to_file < 8 {
                        let to_square = (to_rank * 8 + to_file) as u32;
                        let uci_move = UciMove::Normal {
                            from: Square::new(from_square as u32),
                            to: Square::new(to_square),
                            promotion: None,
                        };

                        total_moves += 1;
                        test_move_encoding_decoding(
                            &uci_move,
                            &mut encoding_failures,
                            &mut decoding_failures,
                        );
                    }
                }
            }

            // Generate promotion moves if from 7th rank (white) or 2nd rank (black)
            if from_rank == 6 {
                // 7th rank for white
                // Generate moves to 8th rank with all promotion pieces
                for to_file_offset in [-1, 0, 1] {
                    // capture left, straight, capture right
                    let to_file = from_file as i32 + to_file_offset;
                    if to_file >= 0 && to_file < 8 {
                        let to_square = (7 * 8 + to_file) as u32; // 8th rank

                        for promotion_piece in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight]
                        {
                            let uci_move = UciMove::Normal {
                                from: Square::new(from_square as u32),
                                to: Square::new(to_square),
                                promotion: Some(promotion_piece),
                            };

                            total_moves += 1;
                            test_move_encoding_decoding(
                                &uci_move,
                                &mut encoding_failures,
                                &mut decoding_failures,
                            );
                        }
                    }
                }
            }
        }

        println!("Total UCI moves generated: {}", total_moves);
        println!("Encoding failures: {}", encoding_failures.len());
        println!("Decoding failures: {}", decoding_failures.len());

        if !encoding_failures.is_empty() {
            println!(
                "First 10 encoding failures: {:?}",
                &encoding_failures[..encoding_failures.len().min(10)]
            );
        }

        if !decoding_failures.is_empty() {
            println!(
                "First 10 decoding failures: {:?}",
                &decoding_failures[..decoding_failures.len().min(10)]
            );
        }

        // Assert no failures
        assert_eq!(encoding_failures.len(), 0, "Some moves failed to encode");
        assert_eq!(
            decoding_failures.len(),
            0,
            "Some moves failed to decode correctly"
        );

        println!(
            "✓ All {} generated UCI moves can be encoded and decoded correctly",
            total_moves
        );
    }

    fn test_move_encoding_decoding(
        uci_move: &UciMove,
        encoding_failures: &mut Vec<String>,
        decoding_failures: &mut Vec<String>,
    ) {
        match encode_move(uci_move) {
            Some((from, idx)) => {
                // Test decoding
                match decode_move(from, idx) {
                    Some(decoded_move) => {
                        if decoded_move != *uci_move {
                            decoding_failures.push(format!(
                                "Decode mismatch: {} -> ({}, {}) -> {}",
                                uci_move, from, idx, decoded_move
                            ));
                        }
                    }
                    None => {
                        decoding_failures.push(format!(
                            "Failed to decode: ({}, {}) from {}",
                            from, idx, uci_move
                        ));
                    }
                }
            }
            None => {
                encoding_failures.push(format!("Failed to encode: {}", uci_move));
            }
        }
    }
}
