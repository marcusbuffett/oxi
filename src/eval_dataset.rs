//! Evaluation dataset: positions annotated with Stockfish evaluations of all legal moves.
//!
//! Used to measure model strength beyond top-1 accuracy:
//! - ECL (Expected Centipawn Loss) under the model's policy
//! - Sharpness (how punishing a position is for random play)
//! - Blunder rate (fraction of positions where model's top move loses >N centipawns)
//! - Comparison to human ECL at the same Elo bracket

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::stockfish::{MoveEval, PositionEval, StockfishEngine};

/// A single position in the evaluation dataset, annotated with Stockfish evals
/// and the human move that was actually played.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalPosition {
    /// FEN of the position (original, not mirrored)
    pub fen: String,
    /// Stockfish evaluation of every legal move
    pub move_evals: Vec<MoveEval>,
    /// Best move's eval in centipawns
    pub best_eval_cp: i32,
    /// The UCI move the human player actually made
    pub human_move: String,
    /// Elo of the player whose turn it is
    pub player_elo: i32,
    /// Elo of the opponent
    pub opponent_elo: i32,
    /// Half-move number in the game (0 = first move)
    pub ply: u32,
    /// Game result from the side-to-move's perspective: 1.0=win, 0.5=draw, 0.0=loss
    pub game_result: f32,
    /// Stockfish search depth used
    pub stockfish_depth: u32,

    // Precomputed metrics
    /// Sharpness: mean ECL under uniform prior (how punishing is random play)
    pub sharpness: f32,
    /// ECL of the human's actual move
    pub human_ecl: f32,
}

/// The full evaluation dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDataset {
    pub positions: Vec<EvalPosition>,
    pub metadata: EvalDatasetMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDatasetMetadata {
    pub created_at: String,
    pub stockfish_depth: u32,
    pub num_positions: usize,
    pub elo_range: (i32, i32),
    pub source_description: String,
}

/// Results of evaluating a model on the eval dataset
#[derive(Debug, Clone, Serialize)]
pub struct EvalResults {
    /// Overall metrics across all positions
    pub overall: EvalBucketResults,
    /// Metrics broken down by Elo bucket (bucket key = lower bound, e.g. 1000 for 1000-1200)
    pub by_elo: BTreeMap<i32, EvalBucketResults>,
    /// Metrics broken down by sharpness bucket (bucket key = lower bound in cp)
    pub by_sharpness: BTreeMap<i32, EvalBucketResults>,
    /// Metrics broken down by game stage
    pub by_stage: BTreeMap<String, EvalBucketResults>,
}

/// Metrics for a single bucket of positions
#[derive(Debug, Clone, Serialize)]
pub struct EvalBucketResults {
    /// Mean ECL under the model's policy
    pub model_ecl: f32,
    /// Mean ECL of the human moves
    pub human_ecl: f32,
    /// ECL ratio: model_ecl / human_ecl (< 1.0 means model is better than humans)
    pub ecl_ratio: f32,
    /// Fraction of positions where model's top move loses >100cp
    pub blunder_rate: f32,
    /// Fraction of positions where model's top move loses >300cp (serious blunders)
    pub serious_blunder_rate: f32,
    /// Fraction of positions where the human's actual move loses >100cp
    pub human_blunder_rate: f32,
    /// Fraction of positions where the human's actual move loses >300cp
    pub human_serious_blunder_rate: f32,
    /// Top-1 accuracy vs Stockfish (fraction where model's top move = Stockfish's best move)
    pub top1_accuracy: f32,
    /// Top-1 accuracy vs human (fraction where model's top move = the human's actual move)
    pub human_top1_accuracy: f32,
    /// Mean sharpness of positions in this bucket
    pub mean_sharpness: f32,
    /// Number of positions in this bucket
    pub count: usize,
}

/// Config for creating an eval dataset
#[derive(Debug, Clone)]
pub struct CreateEvalSetConfig {
    pub pgn_dir: std::path::PathBuf,
    pub output_path: std::path::PathBuf,
    pub stockfish_path: Option<String>,
    pub stockfish_depth: u32,
    pub stockfish_threads: u32,
    pub num_positions: usize,
    /// Elo buckets to sample from, e.g. [(1000,1200), (1200,1400), ...]
    pub elo_buckets: Vec<(i32, i32)>,
}

impl Default for CreateEvalSetConfig {
    fn default() -> Self {
        Self {
            pgn_dir: std::path::PathBuf::from("./data/pgn"),
            output_path: std::path::PathBuf::from("./data/eval_set.json"),
            stockfish_path: None,
            stockfish_depth: 12,
            stockfish_threads: 1,
            num_positions: 10_000,
            elo_buckets: vec![
                (1000, 1200),
                (1200, 1400),
                (1400, 1600),
                (1600, 1800),
                (1800, 2000),
                (2000, 2200),
                (2200, 2400),
                (2400, 2700),
            ],
        }
    }
}

/// Lightweight representation of a sampled position before Stockfish eval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledPosition {
    pub fen: String,
    pub human_move: String,
    pub player_elo: i32,
    pub opponent_elo: i32,
    pub ply: u32,
    pub game_result: f32,
}

impl EvalDataset {
    /// Load an eval dataset from a JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open eval dataset at {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let dataset: EvalDataset = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse eval dataset from {}", path.display()))?;
        Ok(dataset)
    }

    /// Save the eval dataset to a JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create eval dataset at {}", path.display()))?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .with_context(|| "Failed to serialize eval dataset")?;
        Ok(())
    }

    /// Create an eval dataset from sampled positions by running Stockfish on each.
    pub fn from_sampled_positions(
        positions: Vec<SampledPosition>,
        stockfish_path: Option<&str>,
        depth: u32,
        threads: u32,
    ) -> Result<Self> {
        let mut engine = StockfishEngine::new(stockfish_path, depth, threads)?;
        let total = positions.len();
        let mut eval_positions = Vec::with_capacity(total);

        println!(
            "Evaluating {} positions with Stockfish at depth {}...",
            total, depth
        );

        for (i, pos) in positions.iter().enumerate() {
            if (i + 1) % 100 == 0 || i + 1 == total {
                println!("  [{}/{}] positions evaluated", i + 1, total);
            }

            match engine.evaluate_position_at_depth(&pos.fen, depth) {
                Ok(pos_eval) => {
                    let best_eval_cp = pos_eval.best_eval_cp();
                    let sharpness = pos_eval.sharpness();
                    let human_ecl = pos_eval.ecl_of_move(&pos.human_move).unwrap_or(f32::NAN);

                    eval_positions.push(EvalPosition {
                        fen: pos.fen.clone(),
                        move_evals: pos_eval.move_evals,
                        best_eval_cp,
                        human_move: pos.human_move.clone(),
                        player_elo: pos.player_elo,
                        opponent_elo: pos.opponent_elo,
                        ply: pos.ply,
                        game_result: pos.game_result,
                        stockfish_depth: depth,
                        sharpness,
                        human_ecl,
                    });
                }
                Err(e) => {
                    eprintln!("Warning: Failed to evaluate position {}: {}", pos.fen, e);
                    continue;
                }
            }
        }

        let elo_range = if eval_positions.is_empty() {
            (0, 0)
        } else {
            let min_elo = eval_positions.iter().map(|p| p.player_elo).min().unwrap();
            let max_elo = eval_positions.iter().map(|p| p.player_elo).max().unwrap();
            (min_elo, max_elo)
        };

        let metadata = EvalDatasetMetadata {
            created_at: chrono_now_string(),
            stockfish_depth: depth,
            num_positions: eval_positions.len(),
            elo_range,
            source_description: format!("Sampled {} positions from PGN data", total),
        };

        Ok(Self {
            positions: eval_positions,
            metadata,
        })
    }

    /// Evaluate a model's policy against this dataset.
    ///
    /// `model_policies` maps position FEN -> HashMap of (uci_move -> probability).
    /// The map must contain an entry for every position in the dataset.
    pub fn evaluate_model(
        &self,
        model_policies: &HashMap<String, HashMap<String, f32>>,
    ) -> EvalResults {
        let mut all_entries: Vec<EvalEntry> = Vec::new();

        for pos in &self.positions {
            let model_ecl = match model_policies.get(&pos.fen) {
                Some(policy) => pos_eval_from_evals(pos).ecl_under_policy(policy),
                None => {
                    eprintln!("Warning: No model policy for position {}", pos.fen);
                    continue;
                }
            };

            let policy = model_policies.get(&pos.fen).unwrap();
            // Find model's top move
            let top_move = policy
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(m, _)| m.as_str())
                .unwrap_or("");

            let top_move_ecl = pos_eval_from_evals(pos)
                .ecl_of_move(top_move)
                .unwrap_or(f32::NAN);

            let is_top1 = pos
                .move_evals
                .first()
                .map(|e| e.uci_move == top_move)
                .unwrap_or(false);

            let is_human_top1 = top_move == pos.human_move;

            all_entries.push(EvalEntry {
                model_ecl: model_ecl.min(ECL_CAP_CP),
                human_ecl: if pos.human_ecl.is_nan() {
                    pos.human_ecl
                } else {
                    pos.human_ecl.min(ECL_CAP_CP)
                },
                top_move_ecl: if top_move_ecl.is_nan() {
                    top_move_ecl
                } else {
                    top_move_ecl.min(ECL_CAP_CP)
                },
                is_top1,
                is_human_top1,
                sharpness: pos.sharpness,
                player_elo: pos.player_elo,
                ply: pos.ply,
            });
        }

        let overall = compute_bucket_results(&all_entries);

        // By Elo bucket (200-point buckets)
        let mut by_elo: BTreeMap<i32, Vec<&EvalEntry>> = BTreeMap::new();
        for entry in &all_entries {
            let bucket = (entry.player_elo / 200) * 200;
            by_elo.entry(bucket).or_default().push(entry);
        }
        let by_elo = by_elo
            .into_iter()
            .map(|(k, entries)| (k, compute_bucket_results_from_refs(&entries)))
            .collect();

        // By sharpness bucket
        let sharpness_boundaries = [0, 20, 50, 100, 200, 500];
        let mut by_sharpness: BTreeMap<i32, Vec<&EvalEntry>> = BTreeMap::new();
        for entry in &all_entries {
            let bucket = sharpness_boundaries
                .iter()
                .rev()
                .find(|&&b| entry.sharpness >= b as f32)
                .copied()
                .unwrap_or(0);
            by_sharpness.entry(bucket).or_default().push(entry);
        }
        let by_sharpness = by_sharpness
            .into_iter()
            .map(|(k, entries)| (k, compute_bucket_results_from_refs(&entries)))
            .collect();

        // By game stage
        let mut by_stage: BTreeMap<String, Vec<&EvalEntry>> = BTreeMap::new();
        for entry in &all_entries {
            let stage = if entry.ply < 30 {
                "opening"
            } else if entry.ply < 70 {
                "middlegame"
            } else {
                "endgame"
            };
            by_stage.entry(stage.to_string()).or_default().push(entry);
        }
        let by_stage = by_stage
            .into_iter()
            .map(|(k, entries)| (k, compute_bucket_results_from_refs(&entries)))
            .collect();

        EvalResults {
            overall,
            by_elo,
            by_sharpness,
            by_stage,
        }
    }

    /// Print a summary report of evaluation results
    pub fn print_report(results: &EvalResults) {
        println!("\n{}", "=".repeat(70));
        println!("  MODEL EVALUATION REPORT");
        println!("{}\n", "=".repeat(70));

        println!("OVERALL ({} positions)", results.overall.count);
        print_bucket(&results.overall);

        println!("\nBY ELO BRACKET:");
        println!(
            "{:<12} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>6}",
            "Elo",
            "ModelECL",
            "HumanECL",
            "Ratio",
            "MBlunder%",
            "HBlunder%",
            "SFTop1",
            "HuTop1",
            "Count"
        );
        println!("{}", "-".repeat(90));
        for (elo, bucket) in &results.by_elo {
            println!(
                "{:<12} {:>8.1} {:>8.1} {:>8.2} {:>8.1}% {:>8.1}% {:>7.1}% {:>7.1}% {:>6}",
                format!("{}-{}", elo, elo + 200),
                bucket.model_ecl,
                bucket.human_ecl,
                bucket.ecl_ratio,
                bucket.blunder_rate * 100.0,
                bucket.human_blunder_rate * 100.0,
                bucket.top1_accuracy * 100.0,
                bucket.human_top1_accuracy * 100.0,
                bucket.count,
            );
        }

        println!("\nBY SHARPNESS (mean uniform ECL, cp):");
        println!(
            "{:<12} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>6}",
            "Sharpness",
            "ModelECL",
            "HumanECL",
            "Ratio",
            "MBlunder%",
            "HBlunder%",
            "SFTop1",
            "HuTop1",
            "Count"
        );
        println!("{}", "-".repeat(90));
        for (sharp, bucket) in &results.by_sharpness {
            println!(
                "{:<12} {:>8.1} {:>8.1} {:>8.2} {:>8.1}% {:>8.1}% {:>7.1}% {:>7.1}% {:>6}",
                format!(">={}cp", sharp),
                bucket.model_ecl,
                bucket.human_ecl,
                bucket.ecl_ratio,
                bucket.blunder_rate * 100.0,
                bucket.human_blunder_rate * 100.0,
                bucket.top1_accuracy * 100.0,
                bucket.human_top1_accuracy * 100.0,
                bucket.count,
            );
        }

        println!("\nBY GAME STAGE:");
        println!(
            "{:<12} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>6}",
            "Stage",
            "ModelECL",
            "HumanECL",
            "Ratio",
            "MBlunder%",
            "HBlunder%",
            "SFTop1",
            "HuTop1",
            "Count"
        );
        println!("{}", "-".repeat(90));
        for (stage, bucket) in &results.by_stage {
            println!(
                "{:<12} {:>8.1} {:>8.1} {:>8.2} {:>8.1}% {:>8.1}% {:>7.1}% {:>7.1}% {:>6}",
                stage,
                bucket.model_ecl,
                bucket.human_ecl,
                bucket.ecl_ratio,
                bucket.blunder_rate * 100.0,
                bucket.human_blunder_rate * 100.0,
                bucket.top1_accuracy * 100.0,
                bucket.human_top1_accuracy * 100.0,
                bucket.count,
            );
        }

        println!();
    }
}

// ---------- Internal helpers ----------

/// Cap for per-position centipawn loss. Anything beyond this is "decisive mistake"
/// and the exact magnitude doesn't matter. Without this, mate positions (scored at
/// 30,000 cp) would dominate averages in small buckets.
const ECL_CAP_CP: f32 = 1000.0;

struct EvalEntry {
    model_ecl: f32,
    human_ecl: f32,
    top_move_ecl: f32,
    /// Model's top move matches Stockfish's best move
    is_top1: bool,
    /// Model's top move matches the human's actual move
    is_human_top1: bool,
    sharpness: f32,
    player_elo: i32,
    ply: u32,
}

fn pos_eval_from_evals(pos: &EvalPosition) -> PositionEval {
    PositionEval {
        fen: pos.fen.clone(),
        move_evals: pos.move_evals.clone(),
        depth: pos.stockfish_depth,
    }
}

fn compute_bucket_results(entries: &[EvalEntry]) -> EvalBucketResults {
    compute_bucket_results_from_refs(&entries.iter().collect::<Vec<_>>())
}

fn compute_bucket_results_from_refs(entries: &[&EvalEntry]) -> EvalBucketResults {
    if entries.is_empty() {
        return EvalBucketResults {
            model_ecl: 0.0,
            human_ecl: 0.0,
            ecl_ratio: 0.0,
            blunder_rate: 0.0,
            serious_blunder_rate: 0.0,
            human_blunder_rate: 0.0,
            human_serious_blunder_rate: 0.0,
            top1_accuracy: 0.0,
            human_top1_accuracy: 0.0,
            mean_sharpness: 0.0,
            count: 0,
        };
    }

    let n = entries.len() as f32;
    let model_ecl = entries.iter().map(|e| e.model_ecl).sum::<f32>() / n;
    let human_ecl = entries
        .iter()
        .filter(|e| !e.human_ecl.is_nan())
        .map(|e| e.human_ecl)
        .sum::<f32>()
        / entries
            .iter()
            .filter(|e| !e.human_ecl.is_nan())
            .count()
            .max(1) as f32;
    let ecl_ratio = if human_ecl > 0.0 {
        model_ecl / human_ecl
    } else {
        0.0
    };
    let blunder_rate = entries.iter().filter(|e| e.top_move_ecl > 100.0).count() as f32 / n;
    let serious_blunder_rate = entries.iter().filter(|e| e.top_move_ecl > 300.0).count() as f32 / n;
    let valid_human = entries
        .iter()
        .filter(|e| !e.human_ecl.is_nan())
        .collect::<Vec<_>>();
    let nh = valid_human.len().max(1) as f32;
    let human_blunder_rate = valid_human.iter().filter(|e| e.human_ecl > 100.0).count() as f32 / nh;
    let human_serious_blunder_rate =
        valid_human.iter().filter(|e| e.human_ecl > 300.0).count() as f32 / nh;
    let top1_accuracy = entries.iter().filter(|e| e.is_top1).count() as f32 / n;
    let human_top1_accuracy = entries.iter().filter(|e| e.is_human_top1).count() as f32 / n;
    let mean_sharpness = entries.iter().map(|e| e.sharpness).sum::<f32>() / n;

    EvalBucketResults {
        model_ecl,
        human_ecl,
        ecl_ratio,
        blunder_rate,
        serious_blunder_rate,
        human_blunder_rate,
        human_serious_blunder_rate,
        top1_accuracy,
        human_top1_accuracy,
        mean_sharpness,
        count: entries.len(),
    }
}

fn print_bucket(b: &EvalBucketResults) {
    println!("  Model ECL:               {:.1} cp", b.model_ecl);
    println!("  Human ECL:               {:.1} cp", b.human_ecl);
    println!("  ECL Ratio (model/human): {:.2}", b.ecl_ratio);
    println!(
        "  Blunder rate (>100cp):    {:.1}% model, {:.1}% human",
        b.blunder_rate * 100.0,
        b.human_blunder_rate * 100.0
    );
    println!(
        "  Serious blunders (>300cp): {:.1}% model, {:.1}% human",
        b.serious_blunder_rate * 100.0,
        b.human_serious_blunder_rate * 100.0
    );
    println!("  Top-1 vs Stockfish:      {:.1}%", b.top1_accuracy * 100.0);
    println!(
        "  Top-1 vs human:          {:.1}%",
        b.human_top1_accuracy * 100.0
    );
    println!("  Mean sharpness:          {:.1} cp", b.mean_sharpness);
}

fn chrono_now_string() -> String {
    // Simple timestamp without chrono dependency
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", duration.as_secs())
}

/// Sample positions from PGN files for evaluation.
///
/// Uses a lightweight PGN parser that extracts original (non-mirrored) positions
/// with the human move, Elo, and game result. Stratifies by Elo bucket.
pub fn sample_positions_from_pgn(
    pgn_dir: &Path,
    num_positions: usize,
    elo_buckets: &[(i32, i32)],
) -> Result<Vec<SampledPosition>> {
    use pgn_reader::{BufferedReader, RawHeader, SanPlus, Skip, Visitor};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use shakmaty::{Chess, Color, EnPassantMode, Position};

    let per_bucket = num_positions / elo_buckets.len();

    /// Minimal PGN visitor that collects positions without mirroring
    struct EvalSampler {
        position: Chess,
        white_elo: Option<i32>,
        black_elo: Option<i32>,
        result: Option<String>,
        ply: u32,
        skip: bool,
        time_control: Option<(u32, u32)>,
        collected: Vec<SampledPosition>,
        rng: rand::rngs::ThreadRng,
        /// Probability of keeping each position (reservoir-like sampling)
        keep_prob: f64,
    }

    impl EvalSampler {
        fn new(keep_prob: f64) -> Self {
            Self {
                position: Chess::default(),
                white_elo: None,
                black_elo: None,
                result: None,
                ply: 0,
                skip: false,
                time_control: None,
                collected: Vec::new(),
                rng: rand::rng(),
                keep_prob,
            }
        }
    }

    impl Visitor for EvalSampler {
        type Result = ();

        fn begin_game(&mut self) {
            self.position = Chess::default();
            self.white_elo = None;
            self.black_elo = None;
            self.result = None;
            self.ply = 0;
            self.skip = false;
            self.time_control = None;
        }

        fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
            match key {
                b"WhiteElo" => {
                    self.white_elo = std::str::from_utf8(value.as_bytes())
                        .ok()
                        .and_then(|s| s.parse().ok());
                }
                b"BlackElo" => {
                    self.black_elo = std::str::from_utf8(value.as_bytes())
                        .ok()
                        .and_then(|s| s.parse().ok());
                }
                b"Result" => {
                    self.result = std::str::from_utf8(value.as_bytes()).ok().map(String::from);
                }
                b"TimeControl" => {
                    if let Ok(tc) = std::str::from_utf8(value.as_bytes()) {
                        if let Some((base, inc)) = tc.split_once('+') {
                            if let (Ok(b), Ok(i)) = (base.parse::<u32>(), inc.parse::<u32>()) {
                                self.time_control = Some((b, i));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        fn end_headers(&mut self) -> Skip {
            // Filter: need both Elos, standard time control
            let dominated = match (self.white_elo, self.black_elo) {
                (Some(w), Some(b)) => {
                    let diff = (w - b).abs();
                    w >= 1000 && w <= 3000 && b >= 1000 && b <= 3000 && diff <= 200
                }
                _ => false,
            };
            // Filter for classical/rapid (>60s base time)
            let good_tc = self
                .time_control
                .map(|(base, _)| base >= 61)
                .unwrap_or(false);
            self.skip = !dominated || !good_tc;
            Skip(self.skip)
        }

        fn san(&mut self, san_plus: SanPlus) {
            if self.skip {
                return;
            }

            let chess_move = match san_plus.san.to_move(&self.position) {
                Ok(m) => m,
                Err(_) => {
                    self.skip = true;
                    return;
                }
            };

            // Skip very early moves (opening book territory)
            if self.ply >= 10 {
                // Randomly sample this position
                if (self.rng.next_u64() as f64 / u64::MAX as f64) < self.keep_prob {
                    let is_white = self.position.turn() == Color::White;
                    let (player_elo, opponent_elo) = if is_white {
                        (
                            self.white_elo.unwrap_or(1500),
                            self.black_elo.unwrap_or(1500),
                        )
                    } else {
                        (
                            self.black_elo.unwrap_or(1500),
                            self.white_elo.unwrap_or(1500),
                        )
                    };

                    let game_result = match self.result.as_deref() {
                        Some("1-0") => {
                            if is_white {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        Some("0-1") => {
                            if is_white {
                                0.0
                            } else {
                                1.0
                            }
                        }
                        Some("1/2-1/2") => 0.5,
                        _ => 0.5,
                    };

                    let fen =
                        shakmaty::fen::Fen::from_position(&self.position, EnPassantMode::Legal)
                            .to_string();

                    let uci_move = chess_move
                        .to_uci(shakmaty::CastlingMode::Standard)
                        .to_string();

                    self.collected.push(SampledPosition {
                        fen,
                        human_move: uci_move,
                        player_elo,
                        opponent_elo,
                        ply: self.ply,
                        game_result,
                    });
                }
            }

            self.position.play_unchecked(chess_move.clone());
            self.ply += 1;
        }

        fn begin_variation(&mut self) -> Skip {
            Skip(true)
        }

        fn end_game(&mut self) -> Self::Result {}
    }

    // Collect PGN files
    let mut pgn_files: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(pgn_dir)
        .with_context(|| format!("Failed to read PGN directory: {}", pgn_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let is_pgn = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s == "pgn")
            .unwrap_or(false)
            || path
                .to_str()
                .map(|s| s.ends_with(".pgn.zst"))
                .unwrap_or(false);
        if is_pgn {
            pgn_files.push(path);
        }
    }

    if pgn_files.is_empty() {
        anyhow::bail!("No PGN files found in {}", pgn_dir.display());
    }

    pgn_files.sort();
    println!(
        "Sampling positions from {} PGN files in {}",
        pgn_files.len(),
        pgn_dir.display()
    );

    // Use a low keep_prob to sample sparsely across many games.
    // We'll oversample and then downsample per Elo bucket.
    let target_oversample = num_positions * 5;
    // Rough estimate: ~40 positions per game, ~100k games per file
    let estimated_positions = pgn_files.len() as f64 * 100_000.0 * 30.0;
    let keep_prob = (target_oversample as f64 / estimated_positions)
        .min(1.0)
        .max(0.0001);

    println!(
        "Sampling with keep_prob={:.6} (targeting ~{} raw samples)",
        keep_prob, target_oversample
    );

    let mut all_sampled: Vec<SampledPosition> = Vec::new();

    for (file_idx, path) in pgn_files.iter().enumerate() {
        println!(
            "  [{}/{}] Processing {}",
            file_idx + 1,
            pgn_files.len(),
            path.file_name().unwrap_or_default().to_string_lossy()
        );

        let file = std::fs::File::open(path)?;
        let buf_reader = std::io::BufReader::with_capacity(1024 * 1024, file);

        let reader: Box<dyn std::io::Read> = if path
            .to_str()
            .map(|s| s.ends_with(".pgn.zst"))
            .unwrap_or(false)
        {
            Box::new(zstd::stream::read::Decoder::new(buf_reader)?)
        } else {
            Box::new(buf_reader)
        };

        let mut pgn_reader = BufferedReader::new(reader);
        let mut visitor = EvalSampler::new(keep_prob);

        loop {
            match pgn_reader.read_game(&mut visitor) {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(e) => {
                    let err_str = format!("{:?}", e);
                    if err_str.contains("incomplete frame") || err_str.contains("UnexpectedEof") {
                        break;
                    }
                    continue;
                }
            }

            // Early exit if we have way more than enough
            if visitor.collected.len() >= target_oversample * 2 {
                break;
            }
        }

        println!("    Collected {} positions", visitor.collected.len());
        all_sampled.extend(visitor.collected);

        // Early exit if we have enough across all files
        if all_sampled.len() >= target_oversample * 2 {
            println!("  Enough positions collected, stopping early");
            break;
        }
    }

    println!(
        "Total raw samples: {}. Now stratifying by Elo...",
        all_sampled.len()
    );

    // Stratify by Elo bucket
    let mut by_bucket: HashMap<(i32, i32), Vec<SampledPosition>> = HashMap::new();
    for pos in all_sampled {
        for &(lo, hi) in elo_buckets {
            if pos.player_elo >= lo && pos.player_elo < hi {
                by_bucket.entry((lo, hi)).or_default().push(pos.clone());
                break;
            }
        }
    }

    let mut final_positions: Vec<SampledPosition> = Vec::new();
    let mut rng = rand::rng();

    for &(lo, hi) in elo_buckets {
        let available = by_bucket.get(&(lo, hi)).map(|b| b.len()).unwrap_or(0);
        let take = per_bucket.min(available);

        if let Some(bucket) = by_bucket.get_mut(&(lo, hi)) {
            bucket.shuffle(&mut rng);
            final_positions.extend(bucket.drain(..take));
        }

        println!(
            "  Elo {}-{}: {} available, taking {}",
            lo, hi, available, take
        );
    }

    println!("Final eval set: {} positions", final_positions.len());
    Ok(final_positions)
}
