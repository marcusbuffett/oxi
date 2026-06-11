use crate::dataset::ChessExample;
use crate::moves::{mirror_fen, mirror_move};
use anyhow::Result;
use rand::Rng;
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Color, EnPassantMode, Position};
use std::path::Path;

const PUZZLE_TIME_CONTROL: (u32, u32) = (600, 0);
const PUZZLE_TIME_REMAINING: u32 = 600;
const PUZZLE_TIME_USED: u32 = 30;

fn puzzle_rating_to_elo(puzzle_rating: i32) -> i32 {
    (puzzle_rating as f64 * 0.75) as i32
}

pub struct PuzzleRecord {
    pub puzzle_id: String,
    pub fen: String,
    pub moves: Vec<String>,
    pub rating: i32,
    pub themes: Vec<String>,
}

pub fn parse_puzzle_line(line: &str) -> Option<PuzzleRecord> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 8 {
        return None;
    }

    Some(PuzzleRecord {
        puzzle_id: fields[0].to_string(),
        fen: fields[1].to_string(),
        moves: fields[2].split_whitespace().map(String::from).collect(),
        rating: fields[3].parse().ok()?,
        themes: fields[7].split_whitespace().map(String::from).collect(),
    })
}

pub fn puzzle_to_example(puzzle: &PuzzleRecord) -> Option<ChessExample> {
    if puzzle.moves.len() < 2 {
        return None;
    }

    let fen: Fen = puzzle.fen.parse().ok()?;
    let mut position: Chess = fen.into_position(CastlingMode::Standard).ok()?;

    let setup_move_uci: UciMove = puzzle.moves[0].parse().ok()?;
    let setup_move = setup_move_uci.to_move(&position).ok()?;
    position = position.play(setup_move).ok()?;

    let position_fen =
        shakmaty::fen::Fen::from_position(&position, EnPassantMode::Legal).to_string();

    let target_move = &puzzle.moves[1];
    let is_black_to_move = position.turn() == Color::Black;

    let (final_fen, final_move, prev_fen, prev_move) = if is_black_to_move {
        (
            mirror_fen(&position_fen),
            mirror_move(target_move),
            mirror_fen(&puzzle.fen),
            mirror_move(&puzzle.moves[0]),
        )
    } else {
        (
            position_fen,
            target_move.clone(),
            puzzle.fen.clone(),
            puzzle.moves[0].clone(),
        )
    };

    let outcome = 1.0;

    let elo = puzzle_rating_to_elo(puzzle.rating);

    Some(ChessExample {
        fen: final_fen,
        move_uci: final_move,
        elo_self: elo,
        elo_oppo: elo,
        outcome,
        previous_fens: vec![prev_fen],
        previous_moves: vec![prev_move],
        time_remaining_self: PUZZLE_TIME_REMAINING,
        time_remaining_oppo: PUZZLE_TIME_REMAINING,
        time_used_for_move: PUZZLE_TIME_USED,
        original_time_control: PUZZLE_TIME_CONTROL,
        move_count: 20,
        material_imbalance_history: vec![],
        is_puzzle: true,
    })
}

pub fn process_puzzle_file_iter(path: &Path) -> Result<impl Iterator<Item = ChessExample>> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)?;

    let reader: Box<dyn std::io::Read + Send> = if path.to_string_lossy().ends_with(".zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else {
        Box::new(file)
    };

    let buf_reader = BufReader::new(reader);
    let mut lines = buf_reader.lines();

    let _ = lines.next();

    Ok(lines
        .filter_map(|line| line.ok())
        .filter_map(|line| parse_puzzle_line(&line))
        .filter_map(|puzzle| puzzle_to_example(&puzzle)))
}

pub struct MixedExampleIterator<P, Z, R>
where
    P: Iterator<Item = ChessExample>,
    Z: Iterator<Item = ChessExample>,
    R: Rng,
{
    pgn_iter: P,
    puzzle_iter: Z,
    puzzle_ratio: f64,
    rng: R,
    pgn_exhausted: bool,
    puzzle_exhausted: bool,
}

impl<P, Z, R> MixedExampleIterator<P, Z, R>
where
    P: Iterator<Item = ChessExample>,
    Z: Iterator<Item = ChessExample>,
    R: Rng,
{
    pub fn new(pgn_iter: P, puzzle_iter: Z, puzzle_ratio: f64, rng: R) -> Self {
        Self {
            pgn_iter,
            puzzle_iter,
            puzzle_ratio: puzzle_ratio.clamp(0.0, 1.0),
            rng,
            pgn_exhausted: false,
            puzzle_exhausted: false,
        }
    }
}

impl<P, Z, R> Iterator for MixedExampleIterator<P, Z, R>
where
    P: Iterator<Item = ChessExample>,
    Z: Iterator<Item = ChessExample>,
    R: Rng,
{
    type Item = ChessExample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pgn_exhausted && self.puzzle_exhausted {
            return None;
        }

        let random_val: f64 = self.rng.gen();
        let should_use_puzzle =
            !self.puzzle_exhausted && (self.pgn_exhausted || random_val < self.puzzle_ratio);

        if should_use_puzzle {
            match self.puzzle_iter.next() {
                Some(example) => Some(example),
                None => {
                    self.puzzle_exhausted = true;
                    if !self.pgn_exhausted {
                        self.pgn_iter.next().or_else(|| {
                            self.pgn_exhausted = true;
                            None
                        })
                    } else {
                        None
                    }
                }
            }
        } else {
            match self.pgn_iter.next() {
                Some(example) => Some(example),
                None => {
                    self.pgn_exhausted = true;
                    if !self.puzzle_exhausted {
                        self.puzzle_iter.next().or_else(|| {
                            self.puzzle_exhausted = true;
                            None
                        })
                    } else {
                        None
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_puzzle_line() {
        let line = "00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,1852,76,97,1436140,advantage hangingPiece long middlegame,https://lichess.org/787zsVup/black#48,Italian_Game Italian_Game_Classical_Variation";

        let record = parse_puzzle_line(line).unwrap();

        assert_eq!(record.puzzle_id, "00008");
        assert_eq!(
            record.fen,
            "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
        );
        assert_eq!(
            record.moves,
            vec!["f2g3", "e6e7", "b2b1", "b3c1", "b1c1", "h6c1"]
        );
        assert_eq!(record.rating, 1852);
        assert!(record.themes.contains(&"advantage".to_string()));
    }

    #[test]
    fn test_puzzle_to_example() {
        let puzzle = PuzzleRecord {
            puzzle_id: "test".to_string(),
            fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".to_string(),
            moves: vec!["e7e5".to_string(), "g1f3".to_string()],
            rating: 1500,
            themes: vec!["opening".to_string()],
        };

        let example = puzzle_to_example(&puzzle).unwrap();

        assert_eq!(example.move_uci, "g1f3");
        assert_eq!(example.elo_self, puzzle_rating_to_elo(1500));
        assert_eq!(example.outcome, 1.0);
        assert_eq!(example.previous_fens.len(), 1);
        assert_eq!(example.previous_moves.len(), 1);
    }

    #[test]
    fn test_puzzle_rating_to_elo() {
        assert_eq!(puzzle_rating_to_elo(1000), 750);
        assert_eq!(puzzle_rating_to_elo(2000), 1500);
        assert_eq!(puzzle_rating_to_elo(3000), 2250);
    }

    #[test]
    fn test_puzzle_with_insufficient_moves() {
        let puzzle = PuzzleRecord {
            puzzle_id: "test".to_string(),
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            moves: vec!["e2e4".to_string()],
            rating: 1500,
            themes: vec![],
        };

        assert!(puzzle_to_example(&puzzle).is_none());
    }

    #[test]
    fn test_mixed_iterator_ratio() {
        use rand::SeedableRng;

        let pgn_examples: Vec<ChessExample> = (0..80)
            .map(|i| ChessExample {
                fen: format!("pgn_{}", i),
                move_uci: "e2e4".to_string(),
                elo_self: 1500,
                elo_oppo: 1500,
                outcome: 0.5,
                previous_fens: vec![],
                previous_moves: vec![],
                time_remaining_self: 600,
                time_remaining_oppo: 600,
                time_used_for_move: 30,
                original_time_control: (600, 0),
                move_count: 10,
                material_imbalance_history: vec![],
                is_puzzle: false,
            })
            .collect();

        let puzzle_examples: Vec<ChessExample> = (0..20)
            .map(|i| ChessExample {
                fen: format!("puzzle_{}", i),
                move_uci: "e2e4".to_string(),
                elo_self: 2600,
                elo_oppo: 2600,
                outcome: 1.0,
                previous_fens: vec![],
                previous_moves: vec![],
                time_remaining_self: 600,
                time_remaining_oppo: 600,
                time_used_for_move: 30,
                original_time_control: (600, 0),
                move_count: 20,
                material_imbalance_history: vec![],
                is_puzzle: true,
            })
            .collect();

        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mixed = MixedExampleIterator::new(
            pgn_examples.into_iter(),
            puzzle_examples.into_iter(),
            0.2,
            rng,
        );

        let results: Vec<_> = mixed.collect();
        assert_eq!(results.len(), 100);

        let puzzle_count = results
            .iter()
            .filter(|e| e.fen.starts_with("puzzle_"))
            .count();
        let pgn_count = results.iter().filter(|e| e.fen.starts_with("pgn_")).count();

        assert_eq!(puzzle_count, 20);
        assert_eq!(pgn_count, 80);
    }

    #[test]
    #[ignore]
    fn test_load_real_puzzles() {
        let puzzle_path = std::path::Path::new("../data/puzzles/lichess_db_puzzle.csv.zst");
        if !puzzle_path.exists() {
            println!("Skipping test: puzzle file not found at {:?}", puzzle_path);
            return;
        }

        let iter = process_puzzle_file_iter(puzzle_path).expect("Failed to open puzzle file");
        let examples: Vec<_> = iter.take(100).collect();

        assert!(!examples.is_empty(), "Should have loaded some puzzles");
        println!("Loaded {} puzzle examples", examples.len());

        for (i, ex) in examples.iter().take(5).enumerate() {
            println!(
                "Puzzle {}: elo={}, move={}, fen={}",
                i,
                ex.elo_self,
                ex.move_uci,
                &ex.fen[..50.min(ex.fen.len())]
            );
            assert!(
                ex.elo_self >= 375 && ex.elo_self <= 2625,
                "Elo {} should be in range 375-2625 (puzzle rating 500-3500 * 0.75)",
                ex.elo_self
            );
        }
    }
}
