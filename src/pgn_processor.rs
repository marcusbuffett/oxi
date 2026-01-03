use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::{
    fen::{Epd, Fen},
    CastlingMode, Chess, Color, EnPassantMode, Move, Position,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::MIN_TIME_CONTROL;
use crate::{
    config::{
        get_global_config, should_keep_game_by_elo, should_keep_position_by_ply, PREVIOUS_POSITIONS,
    },
    dataset::ChessExample,
};
use crate::{
    config::{MAX_ELO, MAX_ELO_DIFF, MIN_CLOCK_TIME, MIN_ELO, MIN_PLY},
    moves::{mirror_fen, mirror_move},
};

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
    pub previous_moves: Vec<String>,
    pub time_remaining_self: u32,
    pub time_remaining_oppo: u32,
    pub move_count: usize,
    pub turn: Color,
    pub material_imbalance_history: Vec<i32>,
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
    tcec_mode: bool,

    previous_fens: VecDeque<BySide<String>>,
    previous_moves_uci: VecDeque<BySide<String>>, // UCI moves corresponding to previous_fens
    previous_material_imbalances: VecDeque<i32>,  // Material imbalance history for momentum
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

        let config = get_global_config();
        let rng = StdRng::seed_from_u64(config.seed);

        Self {
            position: Chess::default(),
            previous_fens: VecDeque::new(),
            previous_moves_uci: VecDeque::new(),
            previous_material_imbalances: VecDeque::new(),
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
            tcec_mode: false,
            max_examples_per_position: max_examples,
            pending_example: None,
        }
    }

    pub fn with_max_examples_per_position(mut self, max: usize) -> Self {
        self.max_examples_per_position = max;
        self
    }

    pub fn with_tcec_mode(mut self, enabled: bool) -> Self {
        self.tcec_mode = enabled;
        self
    }

    pub fn take_examples(&mut self) -> Vec<ChessExample> {
        std::mem::take(&mut self.examples)
    }

    pub fn moves_seen(&self) -> usize {
        self.moves.len()
    }

    pub fn is_skipped(&self) -> bool {
        self.skip
    }

    pub fn has_pending_example(&self) -> bool {
        self.pending_example.is_some()
    }
}

impl Visitor for PgnVisitor {
    type Result = ();

    fn begin_game(&mut self) {
        self.position = Chess::default();
        self.moves.clear();
        self.previous_moves.clear();
        self.previous_fens.clear();
        self.previous_moves_uci.clear();
        self.previous_material_imbalances.clear();
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
        if self.skip {
            return;
        }

        let Some((_time_control, increment)) = self.time_control else {
            if !self.tcec_mode {
                self.skip = true;
            }
            return;
        };

        let comment_bytes = comment.as_bytes();
        if let Ok(comment_str) = std::str::from_utf8(comment_bytes) {
            if let Some(clock_start) = comment_str.find("%clk ") {
                if let Some(clock_end) = comment_str[clock_start..].find(']') {
                    let clock_str = &comment_str[clock_start + 5..clock_start + clock_end];
                    if let Some(clock_seconds) = parse_clock_time(clock_str) {
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
                                    previous_moves: pending_example.previous_moves,
                                    time_remaining_self: pending_example.time_remaining_self,
                                    time_remaining_oppo: pending_example.time_remaining_oppo,
                                    time_used_for_move,
                                    original_time_control: self.time_control.expect("time control"),
                                    move_count,
                                    material_imbalance_history: pending_example
                                        .material_imbalance_history,
                                });
                            }
                        };

                        match self.position.turn() {
                            Color::White => self.current_clock.black = Some(clock_seconds),
                            Color::Black => self.current_clock.white = Some(clock_seconds),
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
                    if base_time < MIN_TIME_CONTROL {
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
        // TCEC mode: assign default values for missing headers
        if self.tcec_mode {
            if self.white_elo.is_none() {
                self.white_elo = Some(3500);
            }
            if self.black_elo.is_none() {
                self.black_elo = Some(3500);
            }
            if self.time_control.is_none() {
                self.time_control = Some((7200, 30));
                self.current_clock.white = Some(7200);
                self.current_clock.black = Some(7200);
            }
            if self.result.is_none() {
                self.skip = true;
                return Skip(true);
            }
            return Skip(false);
        }

        // Skip games without required data
        if self.white_elo.is_none()
            || self.black_elo.is_none()
            || self.result.is_none()
            || self.time_control.is_none()
        {
            self.skip = true;
            tracing::debug!("skipping!");
            return Skip(true);
        }

        let (base_time, _increment) = self.time_control.unwrap();

        // Skip low-rated games
        let white_elo = self.white_elo.unwrap();
        let black_elo = self.black_elo.unwrap();
        if white_elo < MIN_ELO || black_elo < MIN_ELO || white_elo > MAX_ELO || black_elo > MAX_ELO
        {
            tracing::debug!("skipping!");
            self.skip = true;
            return Skip(true);
        }

        if white_elo.abs_diff(black_elo) > MAX_ELO_DIFF as u32 {
            self.skip = true;
            tracing::debug!("skipping!");
            return Skip(true);
        }

        // Apply Elo-based game sampling
        let rng_value: f64 = rand::thread_rng().gen();
        if !should_keep_game_by_elo(white_elo, black_elo, rng_value) {
            self.skip = true;
            tracing::debug!("skipping!");
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

        // Compute material imbalance for momentum tracking
        let material_imbalance = crate::inference::compute_material_imbalance(&self.position);

        let should_include = if config.single_legal_move_only.unwrap_or(false)
            && config.checkmate_only.unwrap_or(false)
        {
            // Both flags: only include positions with single legal move AND that deliver checkmate
            has_single_legal_move && is_checkmate
        } else if config.single_legal_move_only.unwrap_or(false) {
            // Only single legal move positions
            has_single_legal_move
        } else if config.checkmate_only.unwrap_or(false) {
            // Only checkmate positions
            is_checkmate
        } else {
            // No filtering, include all positions
            true
        };

        // Apply ply-based position sampling
        let current_ply = self.moves.len(); // This gives us the ply number (0-based)
        let rng_value: f64 = rand::thread_rng().gen();
        let should_include = should_include && should_keep_position_by_ply(current_ply, rng_value);

        if should_include {
            if self.current_clock.white.unwrap() == 0 || self.current_clock.black.unwrap() == 0 {
                self.skip = true;
                return;
            }

            let move_count = self.moves.len();
            let previous_fens: Vec<String> = self
                .previous_fens
                .iter()
                .take(PREVIOUS_POSITIONS)
                .map(|fen_by_side| match turn {
                    Color::White => &fen_by_side.white,
                    Color::Black => &fen_by_side.black,
                })
                .cloned()
                .collect();
            let previous_moves: Vec<String> = self
                .previous_moves_uci
                .iter()
                .take(PREVIOUS_POSITIONS)
                .map(|move_by_side| match turn {
                    Color::White => &move_by_side.white,
                    Color::Black => &move_by_side.black,
                })
                .cloned()
                .collect();
            let time_remaining_self = match turn {
                Color::White => self.current_clock.white.unwrap(),
                Color::Black => self.current_clock.black.unwrap(),
            };
            let time_remaining_oppo = match turn {
                Color::White => self.current_clock.black.unwrap(),
                Color::Black => self.current_clock.white.unwrap(),
            };
            let material_imbalance_history: Vec<i32> = self
                .previous_material_imbalances
                .iter()
                .take(PREVIOUS_POSITIONS)
                .cloned()
                .collect();

            if self.tcec_mode {
                if move_count > MIN_PLY {
                    self.examples.push(ChessExample {
                        fen: adjusted_fen.clone(),
                        move_uci: adjusted_move.clone(),
                        elo_self,
                        elo_oppo,
                        outcome,
                        previous_fens,
                        previous_moves,
                        time_remaining_self,
                        time_remaining_oppo,
                        time_used_for_move: 30,
                        original_time_control: self.time_control.expect("time control"),
                        move_count,
                        material_imbalance_history,
                    });
                }
            } else {
                self.pending_example = Some(PendingChessExample {
                    fen: adjusted_fen.clone(),
                    move_uci: adjusted_move.clone(),
                    elo_self,
                    elo_oppo,
                    outcome,
                    previous_fens,
                    previous_moves,
                    time_remaining_oppo,
                    time_remaining_self,
                    move_count,
                    turn,
                    material_imbalance_history,
                });
            }
        }

        self.previous_fens.push_front(BySide {
            white: fen.clone(),
            black: mirrored_fen.clone(),
        });

        // Track the moves that led to these positions
        self.previous_moves_uci.push_front(BySide {
            white: uci_move.clone(),
            black: mirror_move(&uci_move),
        });

        // Track material imbalance for momentum calculation
        self.previous_material_imbalances
            .push_front(material_imbalance);

        if self.previous_fens.len() > 5 {
            self.previous_fens.pop_back();
            self.previous_moves_uci.pop_back();
            self.previous_material_imbalances.pop_back();
        }

        // Add the move to our tracking list
        self.moves.push(uci_move.clone());

        // Store the move that led to this position (for history tracking)
        let move_for_history = if turn == Color::Black {
            // Mirror the move for black's perspective
            mirror_move(&uci_move)
        } else {
            uci_move.clone()
        };

        self.position = new_position;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {}
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
    let mut remaining_skip = get_global_config().skip.unwrap_or(0);

    while let Ok(batch) = rx.recv() {
        batches_received += 1;

        // Handle skipping of initial samples by dropping full batches or trimming head
        let mut batch = batch;
        if remaining_skip > 0 {
            if remaining_skip >= batch.len() {
                remaining_skip -= batch.len();
                // Drop entire batch
                continue;
            } else {
                // Trim the head of the batch by remaining_skip
                batch.drain(0..remaining_skip);
                remaining_skip = 0;
            }
        }

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

/// Process all PGN files in a directory, returning an iterator
/// This is a non-parallel version that yields examples one at a time
pub fn process_pgn_directory_iter(
    dir: &std::path::Path,
) -> Result<impl Iterator<Item = ChessExample>> {
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
    } else {
        tracing::info!("Found {} PGN files to process", pgn_files.len());
    }

    Ok(PgnDirectoryIterator {
        files: pgn_files,
        current_file_index: 0,
        current_reader: None,
        current_visitor: PgnVisitor::with_position_counts(),
        pending_examples: Vec::new(),
    })
}

/// Iterator that yields ChessExamples from a directory of PGN files
struct PgnDirectoryIterator {
    files: Vec<std::path::PathBuf>,
    current_file_index: usize,
    current_reader: Option<BufferedReader<Box<dyn std::io::Read>>>,
    current_visitor: PgnVisitor,
    pending_examples: Vec<ChessExample>,
}

impl Iterator for PgnDirectoryIterator {
    type Item = ChessExample;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Return pending examples first
            if let Some(example) = self.pending_examples.pop() {
                return Some(example);
            }

            // Try to read from current file
            if let Some(reader) = &mut self.current_reader {
                match reader.read_game(&mut self.current_visitor) {
                    Ok(Some(_)) => {
                        // Game read successfully, collect examples
                        let examples = self.current_visitor.take_examples();
                        if !examples.is_empty() {
                            self.pending_examples.extend(examples);
                            continue;
                        }
                    }
                    Ok(None) => {
                        // End of file, move to next
                        self.current_reader = None;
                    }
                    Err(e) => {
                        tracing::warn!("Error reading game: {:?}", e);
                        continue;
                    }
                }
            }

            // Open next file if needed
            if self.current_reader.is_none() {
                if self.current_file_index >= self.files.len() {
                    return None; // No more files
                }

                let path = &self.files[self.current_file_index];
                self.current_file_index += 1;

                tracing::info!("Processing PGN file: {:?}", path);

                // Open the file
                match open_pgn_file(path) {
                    Ok(reader) => {
                        self.current_reader = Some(BufferedReader::new(reader));
                        self.current_visitor = PgnVisitor::with_position_counts();
                    }
                    Err(e) => {
                        tracing::warn!("Failed to open file {:?}: {}", path, e);
                        continue;
                    }
                }
            }
        }
    }
}

/// Helper function to open a PGN file (handles both .pgn and .pgn.zst)
fn open_pgn_file(path: &std::path::Path) -> Result<Box<dyn std::io::Read>> {
    use std::io::BufReader;

    let file = std::fs::File::open(path)?;
    let buf_reader = BufReader::with_capacity(1024 * 1024, file);

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

    Ok(reader)
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

pub fn process_tcec_directory_iter(
    dir: &std::path::Path,
) -> Result<impl Iterator<Item = ChessExample>> {
    let mut pgn_files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

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
        tracing::info!("No TCEC PGN files found in directory: {:?}", dir);
    } else {
        tracing::info!("Found {} TCEC PGN files to process", pgn_files.len());
    }

    Ok(TcecDirectoryIterator {
        files: pgn_files,
        current_file_index: 0,
        current_reader: None,
        current_visitor: PgnVisitor::with_position_counts().with_tcec_mode(true),
        pending_examples: Vec::new(),
        games_processed: 0,
        games_with_examples: 0,
        games_skipped: 0,
        games_no_moves: 0,
        total_examples: 0,
    })
}

struct TcecDirectoryIterator {
    files: Vec<std::path::PathBuf>,
    current_file_index: usize,
    current_reader: Option<BufferedReader<Box<dyn std::io::Read>>>,
    current_visitor: PgnVisitor,
    pending_examples: Vec<ChessExample>,
    games_processed: usize,
    games_with_examples: usize,
    games_skipped: usize,
    games_no_moves: usize,
    total_examples: usize,
}

impl Iterator for TcecDirectoryIterator {
    type Item = ChessExample;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(example) = self.pending_examples.pop() {
                return Some(example);
            }

            if let Some(reader) = &mut self.current_reader {
                match reader.read_game(&mut self.current_visitor) {
                    Ok(Some(_)) => {
                        self.games_processed += 1;
                        let was_skipped = self.current_visitor.is_skipped();
                        let moves_seen = self.current_visitor.moves_seen();
                        let examples = self.current_visitor.take_examples();
                        let example_count = examples.len();

                        if was_skipped {
                            self.games_skipped += 1;
                        } else if moves_seen == 0 {
                            self.games_no_moves += 1;
                        }

                        if !examples.is_empty() {
                            self.games_with_examples += 1;
                            self.total_examples += example_count;
                            self.pending_examples.extend(examples);
                        }

                        if self.games_processed % 1000 == 0 {
                            let pct_with_examples = if self.games_processed > 0 {
                                100.0 * self.games_with_examples as f64
                                    / self.games_processed as f64
                            } else {
                                0.0
                            };
                            tracing::info!(
                                "TCEC: {} games | {} with examples ({:.1}%) | {} skipped | {} no moves | {} examples",
                                self.games_processed,
                                self.games_with_examples,
                                pct_with_examples,
                                self.games_skipped,
                                self.games_no_moves,
                                self.total_examples
                            );
                        }

                        if example_count > 0 {
                            continue;
                        }
                    }
                    Ok(None) => {
                        self.current_reader = None;
                    }
                    Err(e) => {
                        tracing::warn!("Error reading TCEC game: {:?}", e);
                        continue;
                    }
                }
            }

            if self.current_reader.is_none() {
                if self.current_file_index >= self.files.len() {
                    return None;
                }

                let path = &self.files[self.current_file_index];
                self.current_file_index += 1;

                tracing::info!("Processing TCEC PGN file: {:?}", path);

                match open_pgn_file(path) {
                    Ok(reader) => {
                        self.current_reader = Some(BufferedReader::new(reader));
                        self.current_visitor =
                            PgnVisitor::with_position_counts().with_tcec_mode(true);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to open TCEC file {:?}: {}", path, e);
                        continue;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{set_global_config, Config, ModelConfig};
    use crate::dataset::OXIDataset;
    use crate::model_prediction_logger::format_encoded_board;
    use burn::data::dataloader::Dataset;

    /// Create a test configuration that doesn't filter based on ply/elo
    fn create_test_config() -> Config {
        let mut config = Config::default();
        config.enable_ply_sampling = Some(false);
        config.enable_elo_sampling = Some(false);
        config.single_legal_move_only = Some(false);
        config.checkmate_only = Some(false);
        set_global_config(config.clone()).unwrap();
        config
    }

    #[test]
    #[ignore]
    fn test_pgn_processing_simple_game() {
        let config = create_test_config();
        // Create a simple PGN with two moves
        let pgn_content = r#"[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:01:35] }
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
                insta::assert_snapshot!(format_encoded_board(&item.board_encoded));
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

    //     #[test]
    //     fn test_simple_clock_filtering() {
    //         // Test with a simple game where black runs low on time
    //         let pgn_content = r#"[Event "Test Game"]
    // [White "Player1"]
    // [Black "Player2"]
    // [WhiteElo "1500"]
    // [BlackElo "1600"]
    // [TimeControl "600+0"]
    // [Result "1-0"]
    //
    // 1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:00:25] } 1-0
    // "#;
    //
    //         // Write to a temporary file
    //         let temp_dir = tempfile::TempDir::new().unwrap();
    //         let pgn_path = temp_dir.path().join("test_simple_clock.pgn");
    //         std::fs::write(&pgn_path, pgn_content).unwrap();
    //
    //         // Process the PGN file
    //         let examples = process_pgn_file(&pgn_path).unwrap();
    //
    //         println!("\nSimple test - Found {} examples:", examples.len());
    //         for (i, ex) in examples.iter().enumerate() {
    //             println!("  {}: {} (elo: {})", i + 1, ex.move_uci, ex.elo_self);
    //         }
    //
    //         // Should have only 1 move (e4), since e5 has clock < 30 seconds
    //         assert_eq!(examples.len(), 1, "Should have only white's e4 move");
    //         assert_eq!(examples[0].move_uci, "e2e4");
    //     }

    //     #[test]
    //     fn test_clock_filtering() {
    //         // Test that moves with low clock time are filtered out
    //         let pgn_content = r#"[Event "Test Game"]
    // [White "Player1"]
    // [Black "Player2"]
    // [WhiteElo "1500"]
    // [BlackElo "1600"]
    // [TimeControl "600+0"]
    // [Result "1-0"]
    //
    // 1. e4 { [%clk 0:09:58] } 1... e5 { [%clk 0:09:57] }
    // 2. Nf3 { [%clk 0:09:55] } 2... Nc6 { [%clk 0:00:29] }
    // 3. Bb5 { [%clk 0:09:52] } 3... a6 { [%clk 0:00:15] }
    // 4. Ba4 { [%clk 0:09:49] } 4... Nf6 { [%clk 0:00:45] } 1-0
    // "#;
    //
    //         // Write to a temporary file
    //         let temp_dir = tempfile::TempDir::new().unwrap();
    //         let pgn_path = temp_dir.path().join("test_clock.pgn");
    //         std::fs::write(&pgn_path, pgn_content).unwrap();
    //
    //         // Process the PGN file
    //         let examples = process_pgn_file(&pgn_path).unwrap();
    //
    //         // Print out what we got for debugging
    //         println!("\nFound {} examples after clock filtering:", examples.len());
    //         for (i, ex) in examples.iter().enumerate() {
    //             println!("  {}: {} (elo_self: {})", i + 1, ex.move_uci, ex.elo_self);
    //         }
    //
    //         // The behavior depends on when comments are processed by pgn-reader
    //         // In this case, it seems the Nc6 comment (29 seconds) is processed after Nf3 is added
    //         // So Nf3 gets removed when we detect the low clock
    //
    //         // We should have e4 and e5 only
    //         assert!(examples.len() >= 2, "Should have at least e4 and e5");
    //
    //         // Check that we have the expected moves
    //         let moves: Vec<&str> = examples.iter().map(|e| e.move_uci.as_str()).collect();
    //
    //         // Should have e2e4 at least once
    //         assert!(moves.contains(&"e2e4"), "Should contain e2e4");
    //
    //         // Should NOT have any moves after the low clock detection
    //         assert!(!moves.contains(&"b8c6"), "Should NOT contain b8c6");
    //         assert!(!moves.contains(&"b1c3"), "Should NOT contain mirrored b8c6");
    //         assert!(!moves.contains(&"f1b5"), "Should NOT contain f1b5");
    //         assert!(!moves.contains(&"b5a4"), "Should NOT contain b5a4");
    //     }

    #[test]
    fn test_move_history_structure() {
        // Test that ChessExample properly stores move history
        let example = ChessExample {
            fen: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3".to_string(),
            move_uci: "f1b5".to_string(),
            elo_self: 1500,
            elo_oppo: 1600,
            outcome: 1.0,
            previous_fens: vec![
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2".to_string(),
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".to_string(),
            ],
            previous_moves: vec![
                "g1f3".to_string(), // Nf3
                "e7e5".to_string(), // e5
                "e2e4".to_string(), // e4
            ],
            time_remaining_self: 300,
            time_remaining_oppo: 300,
            time_used_for_move: 5,
            original_time_control: (300, 0),
            move_count: 5,
            material_imbalance_history: vec![0, 0, 0], // Starting position material imbalance is 0
        };

        // Verify the structure
        assert_eq!(example.move_uci, "f1b5");
        assert_eq!(example.previous_moves.len(), 3);
        assert_eq!(example.previous_fens.len(), 3);

        // Verify moves are in correct order (most recent first)
        assert_eq!(example.previous_moves[0], "g1f3"); // Most recent
        assert_eq!(example.previous_moves[2], "e2e4"); // Oldest

        println!("Move history structure test passed!");
    }

    #[test]
    #[ignore]
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
