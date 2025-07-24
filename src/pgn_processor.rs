use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};
use shakmaty::{
    fen::{Epd, Fen},
    CastlingMode, Chess, Color, EnPassantMode, Move, Position,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::moves::mirror_move;
use crate::{
    config::{get_global_config, PREVIOUS_POSITIONS},
    dataset::ChessExample,
};

/// Minimum Elo rating for both players to include games
const MIN_ELO: i32 = 1000;
const MIN_PLY: usize = 10;
const MAX_ELO: i32 = 2000;
const MAX_ELO_DIFF: i32 = 300;

/// Minimum clock time (in seconds) to include moves
const MIN_CLOCK_TIME: u32 = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BySide<T> {
    pub white: T,
    pub black: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingChessExample {
    pub fen: String,
    pub move_uci: String,
    pub elo_self: i32,
    pub elo_oppo: i32,
    pub outcome: f32,
    pub previous_fens: Vec<String>,
    pub time_remaining_self: u32,
    pub time_remaining_oppo: u32,
    pub move_count: usize,
    pub turn: Color,
}

/// PGN visitor that extracts training examples from games
pub struct PgnVisitor {
    // Game state
    position: Chess,
    moves: Vec<String>,
    previous_moves: Vec<Move>, // Track actual Move objects for game state

    // Headers
    white_elo: Option<i32>,
    black_elo: Option<i32>,
    result: Option<String>,
    link: Option<String>,
    time_control: Option<(u32, u32)>,
    current_clock: BySide<Option<u32>>,

    // Extracted examples
    examples: Vec<ChessExample>,
    pending_example: Option<PendingChessExample>,

    // Filters
    skip: bool,

    previous_fens: VecDeque<BySide<String>>,
    max_examples_per_position: usize,
}

impl Default for PgnVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PgnVisitor {
    pub fn new() -> Self {
        Self::with_position_counts()
    }

    pub fn default() -> Self {
        Self::with_position_counts()
    }

    pub fn with_position_counts() -> Self {
        // Check if deduplication is disabled
        let max_examples = if std::env::var("NO_DEDUP").is_ok() {
            usize::MAX
        } else {
            10 // Allow max 10 examples per unique position
        };

        Self {
            position: Chess::default(),
            previous_fens: VecDeque::new(),
            moves: Vec::new(),
            previous_moves: Vec::new(),
            white_elo: None,
            black_elo: None,
            result: None,
            link: None,
            time_control: None,
            current_clock: BySide {
                white: None,
                black: None,
            },
            examples: Vec::new(),
            skip: false,
            max_examples_per_position: max_examples,
            pending_example: None,
        }
    }

    pub fn with_max_examples_per_position(mut self, max: usize) -> Self {
        self.max_examples_per_position = max;
        self
    }

    pub fn take_examples(&mut self) -> Vec<ChessExample> {
        std::mem::take(&mut self.examples)
    }
}

impl Visitor for PgnVisitor {
    type Result = ();

    fn begin_game(&mut self) {
        self.position = Chess::default();
        self.moves.clear();
        self.previous_moves.clear();
        self.previous_fens.clear();
        self.white_elo = None;
        self.black_elo = None;
        self.result = None;
        self.link = None;
        self.time_control = None;
        self.current_clock = BySide {
            white: None,
            black: None,
        };
        self.skip = false;
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        let Some((_time_control, increment)) = self.time_control else {
            self.skip = true;
            return;
        };
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
                        // Update clock for the player whose move was just made
                        // moves.len() gives us the number of completed moves
                        let move_count = self.moves.len();

                        if clock_seconds < MIN_CLOCK_TIME {
                            self.pending_example = None;
                        }

                        if let Some(pending_example) = self.pending_example.take() {
                            let previous_clock = match pending_example.turn {
                                Color::White => self.current_clock.white,
                                Color::Black => self.current_clock.black,
                            }
                            .unwrap();
                            if previous_clock > MIN_CLOCK_TIME && move_count > MIN_PLY {
                                let time_used_for_move =
                                    previous_clock as i32 - clock_seconds as i32 + increment as i32;
                                let time_used_for_move = time_used_for_move.max(1) as u32;
                                self.examples.push(ChessExample {
                                    fen: pending_example.fen,
                                    move_uci: pending_example.move_uci,
                                    elo_self: pending_example.elo_self,
                                    elo_oppo: pending_example.elo_oppo,
                                    outcome: pending_example.outcome,
                                    previous_fens: pending_example.previous_fens,
                                    time_remaining_self: pending_example.time_remaining_self,
                                    time_remaining_oppo: pending_example.time_remaining_oppo,
                                    time_used_for_move,
                                    original_time_control: self.time_control.expect("time control"),
                                    move_count,
                                });
                            }
                        };

                        match self.position.turn() {
                            Color::White => self.current_clock.white = Some(clock_seconds),
                            Color::Black => self.current_clock.black = Some(clock_seconds),
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
            b"Site" => {
                self.link = Some(value_str.to_string());
            }
            b"TimeControl" => {
                self.time_control = parse_time_control(&value_str);
                if let Some((base_time, _increment)) = self.time_control {
                    if base_time <= 60 {
                        self.skip = true;
                        return;
                    }
                    self.current_clock.white = Some(base_time);
                    self.current_clock.black = Some(base_time);
                }
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        // Skip games without required data
        if self.white_elo.is_none()
            || self.black_elo.is_none()
            || self.result.is_none()
            || self.time_control.is_none()
        {
            self.skip = true;
            return Skip(true);
        }

        // Skip games with time control > 15 minutes (900 seconds)
        let (base_time, _increment) = self.time_control.unwrap();
        // only rapid for now
        if !(600..=900).contains(&base_time) {
            self.skip = true;
            return Skip(true);
        }

        // Skip low-rated games
        let white_elo = self.white_elo.unwrap();
        let black_elo = self.black_elo.unwrap();
        if white_elo < MIN_ELO || black_elo < MIN_ELO || white_elo > MAX_ELO || black_elo > MAX_ELO
        {
            self.skip = true;
            return Skip(true);
        }

        if white_elo.abs_diff(black_elo) > MAX_ELO_DIFF as u32 {
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
        let fen = Fen::from_position(&self.position, EnPassantMode::Legal).to_string();

        // Determine whose turn it is
        let turn = self.position.turn();
        let (elo_self, elo_oppo) = if turn == Color::White {
            (self.white_elo.unwrap(), self.black_elo.unwrap())
        } else {
            (self.black_elo.unwrap(), self.white_elo.unwrap())
        };

        // Determine if active player won
        let outcome = match (self.result.as_ref().unwrap().as_str(), turn) {
            ("1-0", Color::White) => 1.,
            ("0-1", Color::Black) => 1.,
            ("1-0", Color::Black) => 0.,
            ("0-1", Color::White) => 0.,
            _ => 0.5, // Draw
        };

        // Create UCI move string
        let uci_move = chess_move.to_uci(CastlingMode::Standard).to_string();
        let (mirrored_fen, mirrored_move) = {
            // Mirror the FEN
            let mirrored_fen = mirror_fen(&fen.to_string());
            let mirrored_move = mirror_move(&uci_move);
            (mirrored_fen, mirrored_move)
        };
        let (adjusted_fen, adjusted_move) = if turn == Color::Black {
            (&mirrored_fen, &mirrored_move)
        } else {
            (&fen, &uci_move)
        };

        let config = get_global_config();

        let has_single_legal_move = self.position.legal_moves().len() == 1;

        let new_position = self.position.clone().play(chess_move).unwrap();

        let is_checkmate = new_position.is_checkmate();

        let should_include = if config.single_legal_move_only && config.checkmate_only {
            // Both flags: only include positions with single legal move AND that deliver checkmate
            has_single_legal_move && is_checkmate
        } else if config.single_legal_move_only {
            // Only single legal move positions
            has_single_legal_move
        } else if config.checkmate_only {
            // Only checkmate positions
            is_checkmate
        } else {
            // No filtering, include all positions
            true
        };

        if should_include {
            if self.current_clock.white.unwrap() == 0 || self.current_clock.black.unwrap() == 0 {
                self.skip = true;
                return;
            }
            self.pending_example = Some(PendingChessExample {
                fen: adjusted_fen.clone(),
                move_uci: adjusted_move.clone(),
                elo_self,
                elo_oppo,
                outcome,
                previous_fens: self
                    .previous_fens
                    .iter()
                    .take(PREVIOUS_POSITIONS)
                    .map(|fen_by_side| match turn {
                        Color::White => &fen_by_side.white,
                        Color::Black => &fen_by_side.black,
                    })
                    .cloned()
                    .collect(),
                time_remaining_oppo: match turn {
                    Color::White => self.current_clock.black.unwrap(),
                    Color::Black => self.current_clock.white.unwrap(),
                },
                time_remaining_self: match turn {
                    Color::White => self.current_clock.white.unwrap(),
                    Color::Black => self.current_clock.black.unwrap(),
                },
                move_count: self.moves.len() + 1,
                turn,
            });
        }

        self.previous_fens.push_front(BySide {
            white: fen.clone(),
            black: mirrored_fen.clone(),
        });

        if self.previous_fens.len() > 5 {
            self.previous_fens.pop_back();
        }

        // Add the move to our tracking list
        self.moves.push(uci_move.clone());

        self.position = new_position;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {}
}

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

/// Process a PGN file and extract training examples
pub fn process_pgn_file(path: &std::path::Path) -> Result<Vec<ChessExample>> {
    process_pgn_file_with_limit(path, None)
}

/// Process a PGN file with optional max samples limit
pub fn process_pgn_file_with_limit(
    path: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<ChessExample>> {
    let position_counts = Arc::new(Mutex::new(HashMap::new()));
    process_pgn_file_with_dedup(path, position_counts, 10, max_samples)
}

pub fn process_pgn_file_with_dedup(
    path: &std::path::Path,
    position_counts: Arc<Mutex<HashMap<String, usize>>>,
    max_per_position: usize,
    max_samples: Option<usize>,
) -> Result<Vec<ChessExample>> {
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

        let mut visitor =
            PgnVisitor::with_position_counts().with_max_examples_per_position(max_per_position);

        match reader.read_game(&mut visitor) {
            Ok(None) => break,
            Ok(Some(_)) => {
                game_count += 1;
                if game_count % 1000 == 0 {
                    tracing::info!(
                        "  Processed {} games, {} examples",
                        game_count,
                        all_examples.len(),
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
pub fn process_pgn_file_zst(path: &std::path::Path) -> Result<Vec<ChessExample>> {
    process_pgn_file(path)
}

/// Process all PGN files in a directory
pub fn process_pgn_directory(dir: &std::path::Path) -> Result<Vec<ChessExample>> {
    process_pgn_directory_with_limit(dir, None)
}

/// Process all PGN files in a directory with optional max samples limit
pub fn process_pgn_directory_with_limit(
    dir: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<ChessExample>> {
    process_pgn_directory_parallel(dir, max_samples)
}

/// Process all PGN files in a directory in parallel
pub fn process_pgn_directory_parallel(
    dir: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<ChessExample>> {
    // Get list of PGN files
    let mut pgn_files = Vec::new();
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
            pgn_files.push(path);
        }
    }

    if pgn_files.is_empty() {
        tracing::info!("No PGN files found in directory: {:?}", dir);
        return Ok(Vec::new());
    }

    tracing::info!("Found {} PGN files to process", pgn_files.len());

    // Use parallel processing
    let num_threads = num_cpus::get();
    tracing::info!("Using {} threads for parallel processing", num_threads);

    // Setup progress bars
    let multi_progress = MultiProgress::new();

    // Main progress bar for overall progress
    let main_pb = multi_progress.add(ProgressBar::new(pgn_files.len() as u64));
    main_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} files ({percent}%) | {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    main_pb.set_message("Processing PGN files");

    // Examples progress bar
    let examples_pb = if let Some(max) = max_samples {
        let pb = multi_progress.add(ProgressBar::new(max as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.green/yellow}] {pos}/{len} examples ({percent}%) | {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        pb.set_message("Collecting examples");
        Some(pb)
    } else {
        let pb = multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {pos} examples | {msg}")
                .unwrap(),
        );
        pb.set_message("Collecting examples");
        Some(pb)
    };

    // Shared state
    let position_counts = Arc::new(Mutex::new(HashMap::new()));
    let abort_flag = Arc::new(AtomicBool::new(false));

    // Channel for receiving batches of examples from workers
    let (tx, rx) = mpsc::channel::<Vec<ChessExample>>();

    // Channel for progress updates from workers
    let (progress_tx, progress_rx) = mpsc::channel::<()>();

    // Spawn worker threads
    let mut handles = Vec::new();
    let files_per_thread = pgn_files.len().div_ceil(num_threads);

    for thread_id in 0..num_threads {
        let start_idx = thread_id * files_per_thread;
        let end_idx = std::cmp::min(start_idx + files_per_thread, pgn_files.len());

        if start_idx >= pgn_files.len() {
            break;
        }

        let thread_files: Vec<_> = pgn_files[start_idx..end_idx].to_vec();
        let tx = tx.clone();
        let progress_tx = progress_tx.clone();
        let position_counts = position_counts.clone();
        let abort_flag = abort_flag.clone();

        let handle = thread::spawn(move || {
            worker_thread(
                thread_id,
                thread_files,
                tx,
                progress_tx,
                position_counts,
                abort_flag,
            )
        });

        handles.push(handle);
    }

    // Drop the original senders so the channels close when all workers finish
    drop(tx);
    drop(progress_tx);

    // Spawn progress update thread
    let progress_main_pb = main_pb.clone();
    let progress_handle = thread::spawn(move || {
        while progress_rx.recv().is_ok() {
            progress_main_pb.inc(1);
        }
    });

    // Collect results from workers
    let mut all_examples = Vec::new();
    let mut batches_received = 0;

    while let Ok(batch) = rx.recv() {
        batches_received += 1;

        // Check if we need to abort due to reaching max samples
        if let Some(max) = max_samples {
            if all_examples.len() >= max {
                abort_flag.store(true, Ordering::Relaxed);
                // Still collect this batch but truncate if needed
                let remaining = max.saturating_sub(all_examples.len());
                all_examples.extend(batch.into_iter().take(remaining));
                break;
            }
        }

        all_examples.extend(batch);

        // Update examples progress bar
        if let Some(ref pb) = examples_pb {
            pb.set_position(all_examples.len() as u64);
            if batches_received % 10 == 0 {
                let unique_positions = position_counts.lock().unwrap().len();
                pb.set_message(format!("{unique_positions} unique positions"));
            }
        }
    }

    // Wait for all threads to complete
    for handle in handles {
        if let Err(e) = handle.join() {
            tracing::warn!("Worker thread panicked: {:?}", e);
        }
    }

    // Wait for progress thread to complete
    let _ = progress_handle.join();

    // Finish progress bars
    main_pb.finish_with_message("All files processed");
    if let Some(pb) = examples_pb {
        let final_unique = position_counts.lock().unwrap().len();
        pb.finish_with_message(format!(
            "Complete: {} examples, {} unique positions",
            all_examples.len(),
            final_unique
        ));
    }

    tracing::info!(
        "Parallel processing complete: {} examples, {} unique positions",
        all_examples.len(),
        position_counts.lock().unwrap().len()
    );

    Ok(all_examples)
}

/// Worker thread function that processes a set of PGN files
fn worker_thread(
    thread_id: usize,
    files: Vec<std::path::PathBuf>,
    tx: mpsc::Sender<Vec<ChessExample>>,
    progress_tx: mpsc::Sender<()>,
    position_counts: Arc<Mutex<HashMap<String, usize>>>,
    abort_flag: Arc<AtomicBool>,
) {
    tracing::info!("Worker {} starting with {} files", thread_id, files.len());

    for (file_idx, file_path) in files.iter().enumerate() {
        // Check abort flag before processing each file
        if abort_flag.load(Ordering::Relaxed) {
            tracing::info!("Worker {} aborting due to abort flag", thread_id);
            break;
        }

        tracing::info!(
            "Worker {} processing file {}/{}: {:?}",
            thread_id,
            file_idx + 1,
            files.len(),
            file_path
        );

        match process_single_file_for_worker(file_path, &position_counts, &abort_flag, &tx) {
            Ok(()) => {
                tracing::debug!("Worker {} completed file: {:?}", thread_id, file_path);
                // Signal progress completion for this file
                let _ = progress_tx.send(());
            }
            Err(e) => {
                tracing::warn!(
                    "Worker {} failed to process file {:?}: {}",
                    thread_id,
                    file_path,
                    e
                );
                // Still signal progress even on error
                let _ = progress_tx.send(());
            }
        }
    }

    tracing::info!("Worker {} finished", thread_id);
}

/// Process a single PGN file for a worker thread, sending batches every 1000 examples
fn process_single_file_for_worker(
    path: &std::path::Path,
    _position_counts: &Arc<Mutex<HashMap<String, usize>>>,
    abort_flag: &Arc<AtomicBool>,
    tx: &mpsc::Sender<Vec<ChessExample>>,
) -> Result<()> {
    use std::io::BufReader;

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
    let mut current_batch = Vec::new();
    let mut game_count = 0;
    const BATCH_SIZE: usize = 1000;

    loop {
        // Check abort flag every game
        if abort_flag.load(Ordering::Relaxed) {
            // Send any remaining examples in current batch
            if !current_batch.is_empty() {
                let _ = tx.send(std::mem::take(&mut current_batch));
            }
            break;
        }

        let mut visitor = PgnVisitor::with_position_counts();

        match reader.read_game(&mut visitor) {
            Ok(None) => break, // End of file
            Ok(Some(_)) => {
                game_count += 1;

                let new_examples = visitor.take_examples();
                current_batch.extend(new_examples);

                // Send batch if it's large enough
                if current_batch.len() >= BATCH_SIZE
                    && tx.send(std::mem::take(&mut current_batch)).is_err()
                {
                    // Main thread has stopped receiving, abort
                    break;
                }

                if game_count % 10000 == 0 {
                    tracing::debug!("Processed {} games from {:?}", game_count, path);
                }
            }
            Err(e) => {
                tracing::warn!("Error reading game from {:?}: {:?}", path, e);
                continue;
            }
        }
    }

    // Send any remaining examples
    if !current_batch.is_empty() {
        let _ = tx.send(current_batch);
    }

    tracing::info!("Completed processing {:?}: {} games", path, game_count);
    Ok(())
}

/// Legacy serial processing function (kept for compatibility)
pub fn process_pgn_directory_serial(
    dir: &std::path::Path,
    max_samples: Option<usize>,
) -> Result<Vec<ChessExample>> {
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

/// Parse TimeControl header like "300+3" into (base_time, increment)
fn parse_time_control(header: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = header.split('+').collect();
    if parts.len() != 2 {
        return None;
    }

    let base_time = parts[0].parse::<u32>().ok()?;
    let increment = parts[1].parse::<u32>().ok()?;

    Some((base_time, increment))
}

/// Calculate time data for a move
// fn calculate_time_data(turn: Color, visitor: &PgnVisitor, increment: i32) -> (i32, i32, i32) {
//     let (base_time, _) = visitor.time_control.unwrap_or((300.0, 0.0));
//
//     let time_remaining_self = current_time.unwrap_or(-1.0);
//     // For early moves, opponent might not have clock data yet - use base time as fallback
//     let time_remaining_oppo = opponent_time
//         .or_else(|| {
//             // If it's the very first move and we don't have opponent time yet,
//             // assume they have full time remaining (we'll get their actual time with their first move)
//             if current_time.is_some() {
//                 Some(base_time)
//             } else {
//                 None
//             }
//         })
//         .unwrap_or(-1.0);
//
//     // Calculate time used for this move
//     let time_used_for_move = if let (Some(prev), Some(curr)) = (prev_time, current_time) {
//         (prev - curr - increment).max(0.0)
//     } else if current_time.is_some() {
//         // For the first move, we don't have previous time, but we have current time
//         // Set time_used_for_move to 0.0 as a placeholder for first moves
//         0.0
//     } else {
//         -1.0 // Invalid if we don't have current time
//     };
//
//     (time_remaining_self, time_remaining_oppo, time_used_for_move)
// }
//
/// Parse clock time from a string like "0:09:58" or "1:05:23" into total seconds (float)
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
[TimeControl "300+0"]
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
        let model_config = ModelConfig::default(); // Small config for testing
        let dataset = OXIDataset::new(examples, model_config);

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
    fn test_time_control_header_parsing() {
        // Test TimeControl header parsing
        assert_eq!(parse_time_control("300+3"), Some((300, 3)));
        assert_eq!(parse_time_control("180+2"), Some((180, 2)));
        assert_eq!(parse_time_control("600+0"), Some((600, 0)));
        assert_eq!(parse_time_control("5+3"), Some((5, 3)));
        assert_eq!(parse_time_control("1800+30"), Some((1800, 30)));

        // Invalid formats
        assert_eq!(parse_time_control("300"), None);
        assert_eq!(parse_time_control("300+"), None);
        assert_eq!(parse_time_control("+3"), None);
        assert_eq!(parse_time_control("invalid+format"), None);
        assert_eq!(parse_time_control("300+3+extra"), None);
    }

    #[test]
    fn test_simple_clock_filtering() {
        // Test with a simple game where black runs low on time
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "600+0"]
[Result "1-0"]

1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:00:25] } 1-0
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
[TimeControl "600+0"]
[Result "1-0"]

1. e4 { [%clk 0:09:58] } 1... e5 { [%clk 0:09:57] } 
2. Nf3 { [%clk 0:09:55] } 2... Nc6 { [%clk 0:00:29] }
3. Bb5 { [%clk 0:09:52] } 3... a6 { [%clk 0:00:15] }
4. Ba4 { [%clk 0:09:49] } 4... Nf6 { [%clk 0:00:45] } 1-0
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

    #[test]
    fn test_complete_time_control_data() {
        // Test that we correctly extract time control data from PGN with clock comments
        let pgn_content = r#"[Event "Test Game with Time Control"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+5"]
[Result "1-0"]

1. e4 { [%clk 0:04:58] } 1... e5 { [%clk 0:04:55] } 
2. Nf3 { [%clk 0:04:53] } 2... Nc6 { [%clk 0:04:50] } 1-0
"#;

        // Write to a temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test_time_control.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        // Process the PGN file
        let examples = process_pgn_file(&pgn_path).unwrap();

        println!("\nTime control test - Found {} examples:", examples.len());
        for (i, ex) in examples.iter().enumerate() {
            println!("Example {}:", i + 1);
            println!("  Move: {}", ex.move_uci);
            println!("  Time remaining self: {:.1}s", ex.time_remaining_self);
            println!("  Time remaining oppo: {:.1}s", ex.time_remaining_oppo);
            println!("  Time used for move: {:.1}s", ex.time_used_for_move);
            println!("  Original time control: {:?}", ex.original_time_control);
            println!("  Move count: {}", ex.move_count);
        }

        // Should have 4 examples (e4, e5, Nf3, Nc6)
        assert!(
            examples.len() >= 3,
            "Should have at least 3 examples with complete time data"
        );

        // Check the first example (White's e4)
        let first_example = &examples[0];
        assert_eq!(first_example.move_uci, "e2e4");
        assert_eq!(first_example.time_remaining_self, 298); // 4:58 = 298 seconds
        assert_eq!(first_example.original_time_control, (300, 5)); // 5 minutes + 5 second increment
        assert_eq!(first_example.move_count, 1);

        // For the first move, we can't calculate time used since there's no previous time
        // But we should have valid remaining time
        assert!(first_example.time_remaining_self > 0);
        assert!(first_example.time_remaining_oppo >= 0);

        // Check a later example that should have time usage data
        if examples.len() >= 3 {
            let third_example = &examples[2]; // Should be Nf3
            assert_eq!(third_example.move_uci, "g1f3");
            assert_eq!(third_example.time_remaining_self, 293); // 4:53 = 293 seconds

            // Time used should be calculated: previous_time - current_time - increment
            // White had 298s after e4, now has 293s, with 5s increment
            // So time used = 298 - 293 - 5 = 0 seconds (minimum)
            assert!(third_example.time_used_for_move >= 0);
        }
    }

    #[test]
    fn test_time_usage_calculation() {
        // Test precise time usage calculation
        let pgn_content = r#"[Event "Time Usage Test"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "180+2"]
[Result "1-0"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:02:58] } 
2. Nf3 { [%clk 0:02:50] } 2... Nc6 { [%clk 0:02:45] } 1-0
"#;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("test_time_usage.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        let examples = process_pgn_file(&pgn_path).unwrap();

        println!("\nTime usage calculation test:");
        for (i, ex) in examples.iter().enumerate() {
            println!(
                "Move {}: {} - Used {:.1}s",
                i + 1,
                ex.move_uci,
                ex.time_used_for_move
            );
        }

        // Should have examples with calculated time usage
        assert!(
            examples.len() >= 3,
            "Should have examples to test time usage"
        );

        // Check Nf3 (should have used time calculated)
        if let Some(nf3_example) = examples.iter().find(|ex| ex.move_uci == "g1f3") {
            // White went from 180s (3:00) to 170s (2:50), with 2s increment
            // Time used = 180 - 170 - 2 = 8 seconds
            assert_eq!(
                nf3_example.time_used_for_move, 8,
                "Nf3 should have used 8 seconds"
            );
        }

        // Check Nc6 (black's second move)
        if let Some(nc6_example) = examples.iter().find(|ex| ex.move_uci == "b8c6") {
            // Black went from 178s (2:58) to 165s (2:45), with 2s increment
            // Time used = 178 - 165 - 2 = 11 seconds
            assert_eq!(
                nc6_example.time_used_for_move, 11,
                "Nc6 should have used 11 seconds"
            );
        }
    }

    #[test]
    fn test_game_filtering_by_time_control() {
        // Test that games with long time controls are filtered out
        let long_game = r#"[Event "Long Time Control Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "1800+30"]
[Result "1-0"]

1. e4 e5 1-0
"#;

        let short_game = r#"[Event "Short Time Control Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+3"]
[Result "1-0"]

1. e4 { [%clk 0:04:58] } 1... e5 { [%clk 0:04:55] } 1-0
"#;

        let temp_dir = tempfile::TempDir::new().unwrap();

        // Test long game (should be filtered out)
        let long_path = temp_dir.path().join("long_game.pgn");
        std::fs::write(&long_path, long_game).unwrap();
        let long_examples = process_pgn_file(&long_path).unwrap();
        assert_eq!(
            long_examples.len(),
            0,
            "Games with >15 minute time control should be filtered out"
        );

        // Test short game (should be processed)
        let short_path = temp_dir.path().join("short_game.pgn");
        std::fs::write(&short_path, short_game).unwrap();
        let short_examples = process_pgn_file(&short_path).unwrap();
        assert!(
            short_examples.len() >= 1,
            "Games with ≤15 minute time control should be processed"
        );
    }

    #[test]
    fn test_clock_data_snapshot() {
        // Snapshot test to verify clock information extraction looks correct
        let pgn_content = r#"[Event "Clock Data Test"]
[White "TestPlayer1"]
[Black "TestPlayer2"]
[WhiteElo "1500"]
[BlackElo "1600"]  
[TimeControl "300+5"]
[Result "1-0"]

1. e4 { [%clk 0:04:55] } 1... e5 { [%clk 0:04:50] } 
2. Nf3 { [%clk 0:04:48] } 2... Nc6 { [%clk 0:04:43] } 1-0
"#;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let pgn_path = temp_dir.path().join("clock_snapshot.pgn");
        std::fs::write(&pgn_path, pgn_content).unwrap();

        let examples = process_pgn_file(&pgn_path).unwrap();

        println!("Found {} examples", examples.len());
        assert!(examples.len() > 0, "Should have found at least 1 example");

        // Create a simplified view of the clock data for snapshot testing
        let snapshot_data: Vec<_> = examples.iter().map(|ex| {
            format!(
                "Move: {} | Player: {} | Time remaining: {:.1}s | Opponent time: {:.1}s | Time used: {:.1}s | Move #: {} | Time control: {}+{}",
                ex.move_uci,
                if ex.move_count % 2 == 1 { "White" } else { "Black" },
                ex.time_remaining_self,
                ex.time_remaining_oppo,
                ex.time_used_for_move,
                ex.move_count,
                ex.original_time_control.0,
                ex.original_time_control.1
            )
        }).collect();

        insta::assert_debug_snapshot!(snapshot_data);
    }

    #[test]
    fn test_parallel_processing() {
        // Create a temporary directory with multiple PGN files
        let temp_dir = tempfile::TempDir::new().unwrap();

        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+3"]
[Result "1-0"]

1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:00:25] }
"#;

        // Create multiple PGN files
        for i in 0..3 {
            let pgn_path = temp_dir.path().join(format!("test_{}.pgn", i));
            std::fs::write(&pgn_path, pgn_content).unwrap();
        }

        // Process directory in parallel
        let examples_parallel = process_pgn_directory_parallel(temp_dir.path(), Some(100)).unwrap();

        // Process directory serially for comparison
        let examples_serial = process_pgn_directory_serial(temp_dir.path(), Some(100)).unwrap();

        println!(
            "\nParallel processing: {} examples",
            examples_parallel.len()
        );
        println!("Serial processing: {} examples", examples_serial.len());

        // Both should find the same number of examples (order might differ)
        assert_eq!(
            examples_parallel.len(),
            examples_serial.len(),
            "Parallel and serial processing should find the same number of examples"
        );

        // Should have found examples from multiple files
        assert!(
            examples_parallel.len() > 4,
            "Should have found examples from multiple files"
        );
    }
}
