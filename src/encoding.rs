use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role, Square};

pub fn encode_position(pos: &Chess) -> Vec<f32> {
    let mut planes = vec![0f32; 16 * 8 * 8];

    // Encode pieces (6 planes per color, 12 total)
    let board = pos.board();
    for square in Square::ALL {
        let idx = square_to_index(square);

        if let Some(piece) = board.piece_at(square) {
            let piece_plane = match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };

            let plane_offset = match piece.color {
                Color::White => piece_plane,
                Color::Black => 6 + piece_plane,
            };

            planes[plane_offset * 64 + idx] = 1.0;
        }
    }

    // En passant square (1 plane)
    let en_passant_offset = 12;
    if let Some(ep_square) = pos.ep_square(EnPassantMode::Legal) {
        let idx = square_to_index(ep_square);
        planes[en_passant_offset * 64 + idx] = 1.0;
    }

    // Total move count (1 plane)
    let move_count_offset = 13;
    let move_count = (pos.fullmoves().get() - 1) as f32 / 100.0; // Normalize
    for i in 0..64 {
        planes[move_count_offset * 64 + i] = move_count;
    }

    // Castling rights (1 plane with corner markers)
    let castling_offset = 14;
    let castling = pos.castles();

    // Mark corners based on castling rights
    if castling.has(Color::White, shakmaty::CastlingSide::KingSide) {
        planes[castling_offset * 64 + 7] = 1.0; // h1
    }
    if castling.has(Color::White, shakmaty::CastlingSide::QueenSide) {
        planes[castling_offset * 64 + 0] = 1.0; // a1
    }
    if castling.has(Color::Black, shakmaty::CastlingSide::KingSide) {
        planes[castling_offset * 64 + 63] = 1.0; // h8
    }
    if castling.has(Color::Black, shakmaty::CastlingSide::QueenSide) {
        planes[castling_offset * 64 + 56] = 1.0; // a8
    }

    // Is check (1 plane)
    let is_check_offset = 15;
    if pos.is_check() {
        for i in 0..64 {
            planes[is_check_offset * 64 + i] = 1.0;
        }
    }

    planes
}

/// Convert square to flat index (0-63)
fn square_to_index(square: Square) -> usize {
    let file = square.file() as usize;
    let rank = square.rank() as usize;
    rank * 8 + file
}

/// Get ELO bin index for a given rating
pub fn get_elo_bin(elo: i32, bins: &[i32]) -> usize {
    bins.iter().position(|&bin| elo < bin).unwrap_or(bins.len())
}
