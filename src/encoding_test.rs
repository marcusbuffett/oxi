#[cfg(test)]
mod tests {
    use crate::encoding::encode_position;
    use shakmaty::{fen::Fen, Chess, Position};

    #[test]
    fn test_encode_starting_position() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let parsed_fen: Fen = fen.parse().unwrap();
        let pos: Chess = parsed_fen
            .into_position(shakmaty::CastlingMode::Standard)
            .unwrap();

        let encoded = encode_position(&pos, &[]);

        // Should have FEATURES_PER_TOKEN * 64 squares
        assert_eq!(encoded.len(), crate::config::FEATURES_PER_TOKEN * 64);

        // Create a formatted representation for snapshot
        let mut output = String::new();
        output.push_str("Board encoding for starting position:\n");
        output.push_str("=====================================\n\n");

        let plane_names = [
            "White Pawns",
            "White Knights",
            "White Bishops",
            "White Rooks",
            "White Queens",
            "White Kings",
            "Black Pawns",
            "Black Knights",
            "Black Bishops",
            "Black Rooks",
            "Black Queens",
            "Black Kings",
            "En Passant",
            "Move Count",
            "Castling Rights",
            "Is Check",
        ];

        for (plane_idx, plane_name) in plane_names.iter().enumerate() {
            output.push_str(&format!("Plane {}: {}\n", plane_idx, plane_name));

            // Print board from white's perspective (rank 8 to rank 1)
            for rank in (0..8).rev() {
                for file in 0..8 {
                    let idx = plane_idx * 64 + rank * 8 + file;
                    if encoded[idx] == 1.0 {
                        output.push('1');
                    } else if encoded[idx] == 0.0 {
                        output.push('.');
                    } else {
                        output.push('*'); // For non-binary values
                    }
                    output.push(' ');
                }
                output.push('\n');
            }
            output.push('\n');
        }

        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_encode_position_with_en_passant() {
        let fen = "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPPP1P/RNBQKBNR w KQkq e6 0 2";
        let parsed_fen: Fen = fen.parse().unwrap();
        let pos: Chess = parsed_fen
            .into_position(shakmaty::CastlingMode::Standard)
            .unwrap();

        let encoded = encode_position(&pos, &[]);

        // Check en passant plane (plane 12)
        let en_passant_plane_start = 12 * 64;
        let e6_idx = 4 + 5 * 8; // e6 square

        assert_eq!(encoded[en_passant_plane_start + e6_idx], 1.0);

        // Verify only one square is marked for en passant
        let en_passant_count: f32 = (0..64).map(|i| encoded[en_passant_plane_start + i]).sum();
        assert_eq!(en_passant_count, 1.0);
    }

    #[test]
    fn test_encode_position_castling_rights() {
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let parsed_fen: Fen = fen.parse().unwrap();
        let pos: Chess = parsed_fen
            .into_position(shakmaty::CastlingMode::Standard)
            .unwrap();

        let encoded = encode_position(&pos, &[]);

        // Check castling plane (plane 14)
        let castling_plane_start = 14 * 64;

        // Should have corners marked: h1, a1, h8, a8
        assert_eq!(encoded[castling_plane_start + 7], 1.0); // h1
        assert_eq!(encoded[castling_plane_start + 0], 1.0); // a1
        assert_eq!(encoded[castling_plane_start + 63], 1.0); // h8
        assert_eq!(encoded[castling_plane_start + 56], 1.0); // a8

        // Verify only 4 squares are marked
        let castling_count: f32 = (0..64).map(|i| encoded[castling_plane_start + i]).sum();
        assert_eq!(castling_count, 4.0);
    }

    #[test]
    fn test_encode_position_in_check() {
        let fen = "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2";
        let parsed_fen: Fen = fen.parse().unwrap();
        let pos: Chess = parsed_fen
            .into_position(shakmaty::CastlingMode::Standard)
            .unwrap();

        let encoded = encode_position(&pos, &[]);

        // Check "is check" plane (plane 15)
        let is_check_plane_start = 15 * 64;

        // All squares should be 1.0 if in check
        let check_sum: f32 = (0..64).map(|i| encoded[is_check_plane_start + i]).sum();

        if pos.is_check() {
            assert_eq!(check_sum, 64.0);
        } else {
            assert_eq!(check_sum, 0.0);
        }
    }
}
