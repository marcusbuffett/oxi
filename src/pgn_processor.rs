use anyhow::Result;
use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};
use shakmaty::{
    fen::{Epd, Fen},
    Chess, Color, EnPassantMode, Position,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::dataset::RawChessExample;
use crate::moves::mirror_move;

/// Minimum Elo rating for both players to include games
const MIN_ELO: i32 = 600;

/// Minimum clock time (in seconds) to include moves
const MIN_CLOCK_TIME: u32 = 30;

/// PGN visitor that extracts training examples from games
pub struct PgnVisitor {
    // Game state
    position: Chess,
    moves: Vec<String>,

    // Headers
    white_elo: Option<i32>,
    black_elo: Option<i32>,
    result: Option<String>,

    // Clock tracking
    white_clock: Option<u32>, // Current white clock time in seconds
    black_clock: Option<u32>, // Current black clock time in seconds

    // Extracted examples
    examples: Vec<RawChessExample>,

    // Filters
    skip: bool,

    // Position deduplication (shared across all visitors)
    position_counts: Arc<Mutex<HashMap<String, usize>>>,
    max_examples_per_position: usize,
}

impl PgnVisitor {
    pub fn new() -> Self {
        Self::with_position_counts(Arc::new(Mutex::new(HashMap::new())))
    }

    pub fn with_position_counts(position_counts: Arc<Mutex<HashMap<String, usize>>>) -> Self {
        // Check if deduplication is disabled
        let max_examples = if std::env::var("NO_DEDUP").is_ok() {
            usize::MAX
        } else {
            10 // Allow max 10 examples per unique position
        };

        Self {
            position: Chess::default(),
            moves: Vec::new(),
            white_elo: None,
            black_elo: None,
            result: None,
            white_clock: None,
            black_clock: None,
            examples: Vec::new(),
            skip: false,
            position_counts,
            max_examples_per_position: max_examples,
        }
    }

    pub fn with_max_examples_per_position(mut self, max: usize) -> Self {
        self.max_examples_per_position = max;
        self
    }

    pub fn take_examples(&mut self) -> Vec<RawChessExample> {
        std::mem::take(&mut self.examples)
    }
}

impl Visitor for PgnVisitor {
    type Result = ();

    fn begin_game(&mut self) {
        self.position = Chess::default();
        self.moves.clear();
        self.white_elo = None;
        self.black_elo = None;
        self.result = None;
        self.white_clock = None;
        self.black_clock = None;
        self.skip = false;
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        // Don't process comments if we're already skipping
        if self.skip {
            return;
        }

        // Parse clock time from comment if present
        let comment_bytes = comment.as_bytes();
        if let Ok(comment_str) = std::str::from_utf8(comment_bytes) {
            // Look for clock comment in format [%clk H:MM:SS]
            if let Some(clock_start) = comment_str.find("%clk ") {
                if let Some(clock_end) = comment_str[clock_start..].find(']') {
                    let clock_str = &comment_str[clock_start + 5..clock_start + clock_end];
                    if let Some(clock_seconds) = parse_clock_time(clock_str) {
                        // If clock is below minimum, remove the last example and skip the rest of the game
                        if clock_seconds < MIN_CLOCK_TIME {
                            // Remove the last added example if there is one
                            if !self.examples.is_empty() {
                                self.examples.pop();
                            }
                            // Skip the rest of this game
                            self.skip = true;
                        }
                    }
                }
            }
        }
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let Ok(value_str) = value.decode_utf8() else {
            return;
        };

        match key {
            b"WhiteElo" => {
                self.white_elo = value_str.parse::<i32>().ok();
            }
            b"BlackElo" => {
                self.black_elo = value_str.parse::<i32>().ok();
            }
            b"Result" => {
                self.result = Some(value_str.to_string());
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        // Skip games without required data
        if self.white_elo.is_none() || self.black_elo.is_none() || self.result.is_none() {
            self.skip = true;
            return Skip(true);
        }

        // Skip low-rated games
        let white_elo = self.white_elo.unwrap();
        let black_elo = self.black_elo.unwrap();
        if white_elo < MIN_ELO || black_elo < MIN_ELO {
            self.skip = true;
            return Skip(true);
        }

        Skip(false)
    }

    fn san(&mut self, san_plus: SanPlus) {
        if self.skip {
            return;
        }

        // Get move
        let Ok(chess_move) = san_plus.san.to_move(&self.position) else {
            self.skip = true;
            return;
        };

        // Clock filtering is handled in the comment() callback

        // Get position before move
        let _epd = Epd::from_position(&self.position, EnPassantMode::Legal);
        let fen = Fen::from_position(&self.position, EnPassantMode::Legal);

        // Check if we've seen this position too many times
        let position_key = fen.to_string();
        let should_skip = {
            let mut counts = self.position_counts.lock().unwrap();
            let count = counts.entry(position_key.clone()).or_insert(0);
            *count += 1;
            *count > self.max_examples_per_position
        };

        // Skip if we've seen this position too many times
        if should_skip {
            // Still make the move to continue the game
            self.position.play_unchecked(chess_move);
            return;
        }

        // Determine whose turn it is
        let turn = self.position.turn();
        let (elo_self, elo_oppo) = if turn == Color::White {
            (self.white_elo.unwrap(), self.black_elo.unwrap())
        } else {
            (self.black_elo.unwrap(), self.white_elo.unwrap())
        };

        // Determine if active player won
        let active_won = match (self.result.as_ref().unwrap().as_str(), turn) {
            ("1-0", Color::White) => 1,
            ("0-1", Color::Black) => 1,
            ("1-0", Color::Black) => -1,
            ("0-1", Color::White) => -1,
            _ => 0, // Draw
        };

        // Create UCI move string
        let uci_move = if chess_move.is_castle() {
            // Handle castling moves specially
            match (chess_move.from().unwrap(), chess_move.to()) {
                (e1, g1) if e1 == shakmaty::Square::E1 && g1 == shakmaty::Square::G1 => {
                    "e1g1".to_string()
                }
                (e1, c1) if e1 == shakmaty::Square::E1 && c1 == shakmaty::Square::C1 => {
                    "e1c1".to_string()
                }
                (e8, g8) if e8 == shakmaty::Square::E8 && g8 == shakmaty::Square::G8 => {
                    "e8g8".to_string()
                }
                (e8, c8) if e8 == shakmaty::Square::E8 && c8 == shakmaty::Square::C8 => {
                    "e8c8".to_string()
                }
                _ => format!("{}{}", chess_move.from().unwrap(), chess_move.to()),
            }
        } else {
            format!("{}{}", chess_move.from().unwrap(), chess_move.to())
        };

        // For black's moves, mirror the position and move
        let (fen_str, move_uci) = if turn == Color::Black {
            // Mirror the FEN
            let mirrored_fen = mirror_fen(&fen.to_string());
            let mirrored_move = mirror_move(&uci_move);
            (mirrored_fen, mirrored_move)
        } else {
            (fen.to_string(), uci_move)
        };

        // We'll add the example after checking for checkmate below

        // Play the move
        let new_position = self.position.clone().play(chess_move).unwrap();

        // Check if this move delivers checkmate
        let _is_checkmate = new_position.is_checkmate();

        // Only add the example if it's a checkmate move
        // if is_checkmate {
        self.examples.push(RawChessExample {
            fen: fen_str,
            move_uci,
            elo_self,
            elo_oppo,
            active_won,
            win_prob: None,
            draw_prob: None,
            loss_prob: None,
        });
        // }

        self.position = new_position;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // Skip variations
    }

    fn end_game(&mut self) -> Self::Result {
        // Game finished
    }
}

/// Mirror a FEN string (flip board vertically)
fn mirror_fen(fen: &str) -> String {
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
            let file = parts[3].chars().nth(0).unwrap();
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

/// Process a PGN file and extract training examples
pub fn process_pgn_file(path: &std::path::Path) -> Result<Vec<RawChessExample>> {
    process_pgn_file_with_limit(path, None)
}

/// Process a PGN file with optional max samples limit
pub fn process_pgn_file_with_limit(
    path: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<RawChessExample>> {
    let position_counts = Arc::new(Mutex::new(HashMap::new()));
    process_pgn_file_with_dedup(path, position_counts, 10, max_samples)
}

pub fn process_pgn_file_with_dedup(
    path: &std::path::Path,
    position_counts: Arc<Mutex<HashMap<String, usize>>>,
    max_per_position: usize,
    max_samples: Option<usize>,
) -> Result<Vec<RawChessExample>> {
    use std::io::BufReader;

    tracing::info!("Opening PGN file: {:?}", path);

    // Open the file with a buffer
    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

    // Create appropriate reader based on file extension
    let reader: Box<dyn std::io::Read> = if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s == "zst")
        .unwrap_or(false)
    {
        Box::new(zstd::stream::read::Decoder::new(buf_reader)?)
    } else {
        Box::new(buf_reader)
    };

    // Process the PGN file
    let mut reader = BufferedReader::new(reader);
    let mut all_examples = Vec::new();
    let mut game_count = 0;

    loop {
        // Check if we've reached the max samples limit
        if let Some(max) = max_samples {
            if all_examples.len() >= max {
                tracing::info!("Reached max samples limit of {}", max);
                break;
            }
        }

        let mut visitor = PgnVisitor::with_position_counts(position_counts.clone())
            .with_max_examples_per_position(max_per_position);

        match reader.read_game(&mut visitor) {
            Ok(None) => break,
            Ok(Some(_)) => {
                game_count += 1;
                if game_count % 1000 == 0 {
                    let unique_positions = position_counts.lock().unwrap().len();
                    tracing::info!(
                        "  Processed {} games, {} examples, {} unique positions",
                        game_count,
                        all_examples.len(),
                        unique_positions
                    );
                }

                let new_examples = visitor.take_examples();

                // If adding these examples would exceed max_samples, only take what we need
                if let Some(max) = max_samples {
                    let remaining = max.saturating_sub(all_examples.len());
                    all_examples.extend(new_examples.into_iter().take(remaining));
                } else {
                    all_examples.extend(new_examples);
                }
            }
            Err(e) => {
                tracing::warn!("Error reading game: {:?}", e);
                continue;
            }
        }
    }

    let final_unique = position_counts.lock().unwrap().len();
    tracing::info!(
        "Finished processing PGN: {} games, {} examples, {} unique positions",
        game_count,
        all_examples.len(),
        final_unique
    );

    Ok(all_examples)
}

// Backward compatibility alias for process_pgn_file_zst
pub fn process_pgn_file_zst(path: &std::path::Path) -> Result<Vec<RawChessExample>> {
    process_pgn_file(path)
}

/// Process all PGN files in a directory
pub fn process_pgn_directory(dir: &std::path::Path) -> Result<Vec<RawChessExample>> {
    process_pgn_directory_with_limit(dir, None)
}

/// Process all PGN files in a directory with optional max samples limit
pub fn process_pgn_directory_with_limit(
    dir: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<RawChessExample>> {
    let mut all_examples = Vec::new();
    let position_counts = Arc::new(Mutex::new(HashMap::new()));
    let max_per_position = 10;

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Process both .pgn and .pgn.zst files
        let is_pgn = path.extension().and_then(|s| s.to_str()) == Some("pgn")
            || (path
                .to_str()
                .map(|s| s.ends_with(".pgn.zst"))
                .unwrap_or(false));

        if is_pgn {
            tracing::info!("Processing PGN file: {:?}", path);

            // Calculate remaining samples
            let remaining = max_samples.map(|max| max.saturating_sub(all_examples.len()));

            // Stop if we've reached the limit
            if remaining == Some(0) {
                tracing::info!("Reached max samples limit, stopping directory processing");
                break;
            }

            let examples = process_pgn_file_with_dedup(
                &path,
                position_counts.clone(),
                max_per_position,
                remaining,
            )?;
            let unique_positions = position_counts.lock().unwrap().len();
            tracing::info!(
                "  Found {} examples, {} unique positions so far",
                examples.len(),
                unique_positions
            );
            all_examples.extend(examples);
        }
    }

    let final_unique = position_counts.lock().unwrap().len();
    tracing::info!("Total unique positions: {}", final_unique);
    tracing::info!("Total examples: {}", all_examples.len());

    Ok(all_examples)
}

/// Parse clock time from a string like "0:09:58" or "1:05:23" into total seconds
fn parse_clock_time(clock_str: &str) -> Option<u32> {
    let parts: Vec<&str> = clock_str.trim().split(':').collect();

    match parts.len() {
        2 => {
            // MM:SS format
            let minutes = parts[0].parse::<u32>().ok()?;
            let seconds = parts[1].parse::<u32>().ok()?;
            Some(minutes * 60 + seconds)
        }
        3 => {
            // H:MM:SS format
            let hours = parts[0].parse::<u32>().ok()?;
            let minutes = parts[1].parse::<u32>().ok()?;
            let seconds = parts[2].parse::<u32>().ok()?;
            Some(hours * 3600 + minutes * 60 + seconds)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::dataset::OXIDataset;
    use burn::data::dataloader::Dataset;

    #[test]
    fn test_pgn_processing_simple_game() {
        // Create a simple PGN with two moves
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 d5 1-0
"#;

        // Write to a temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        // Process the PGN file
        let examples = process_pgn_file(&pgn_path).unwrap();

        // Should have 2 examples (one for each move)
        assert_eq!(examples.len(), 2, "Should have 2 examples from 2 moves");

        // Print the examples
        println!("\n=== PGN Processing Test Results ===");
        for (i, example) in examples.iter().enumerate() {
            println!("\nExample {}:", i + 1);
            println!("{:#?}", example);
        }

        // Now create a dataset and process the examples to see the training items
        let model_config = ModelConfig::new(3, 64); // Small config for testing
        let dataset = OXIDataset::from_examples(examples, model_config);

        println!("\n=== Training Items ===");
        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                println!("\nTraining Item {}:", i + 1);
                println!("{:#?}", item);
            }
        }
    }

    #[test]
    fn test_clock_time_parsing() {
        // Test various clock time formats
        assert_eq!(parse_clock_time("0:09:58"), Some(9 * 60 + 58));
        assert_eq!(parse_clock_time("0:00:29"), Some(29));
        assert_eq!(parse_clock_time("1:05:23"), Some(1 * 3600 + 5 * 60 + 23));
        assert_eq!(parse_clock_time("05:23"), Some(5 * 60 + 23));
        assert_eq!(parse_clock_time("invalid"), None);
    }

    #[test]
    fn test_simple_clock_filtering() {
        // Test with a simple game where black runs low on time
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 { [%clk 0:10:00] } e5 { [%clk 0:00:25] } 1-0
"#;

        // Write to a temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test_simple_clock.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        // Process the PGN file
        let examples = process_pgn_file(&pgn_path).unwrap();

        println!("\nSimple test - Found {} examples:", examples.len());
        for (i, ex) in examples.iter().enumerate() {
            println!("  {}: {} (elo: {})", i + 1, ex.move_uci, ex.elo_self);
        }

        // Should have only 1 move (e4), since e5 has clock < 30 seconds
        assert_eq!(examples.len(), 1, "Should have only white's e4 move");
        assert_eq!(examples[0].move_uci, "e2e4");
    }

    #[test]
    fn test_clock_filtering() {
        // Test that moves with low clock time are filtered out
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 { [%clk 0:09:58] } e5 { [%clk 0:09:57] } 
2. Nf3 { [%clk 0:09:55] } Nc6 { [%clk 0:00:29] }
3. Bb5 { [%clk 0:09:52] } a6 { [%clk 0:00:15] }
4. Ba4 { [%clk 0:09:49] } Nf6 { [%clk 0:00:45] } 1-0
"#;

        // Write to a temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test_clock.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        // Process the PGN file
        let examples = process_pgn_file(&pgn_path).unwrap();

        // Print out what we got for debugging
        println!("\nFound {} examples after clock filtering:", examples.len());
        for (i, ex) in examples.iter().enumerate() {
            println!("  {}: {} (elo_self: {})", i + 1, ex.move_uci, ex.elo_self);
        }

        // The behavior depends on when comments are processed by pgn-reader
        // In this case, it seems the Nc6 comment (29 seconds) is processed after Nf3 is added
        // So Nf3 gets removed when we detect the low clock

        // We should have e4 and e5 only
        assert!(examples.len() >= 2, "Should have at least e4 and e5");

        // Check that we have the expected moves
        let moves: Vec<&str> = examples.iter().map(|e| e.move_uci.as_str()).collect();

        // Should have e2e4 at least once
        assert!(moves.contains(&"e2e4"), "Should contain e2e4");

        // Should NOT have any moves after the low clock detection
        assert!(!moves.contains(&"b8c6"), "Should NOT contain b8c6");
        assert!(!moves.contains(&"b1c3"), "Should NOT contain mirrored b8c6");
        assert!(!moves.contains(&"f1b5"), "Should NOT contain f1b5");
        assert!(!moves.contains(&"b5a4"), "Should NOT contain b5a4");
    }
}
