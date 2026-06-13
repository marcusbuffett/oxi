//! Move-matching evaluation on the Allie test set.
//!
//! Replicates the offline evaluation protocol from "Human-Aligned Chess With
//! a Bit of Search" (Zhang et al., ICLR 2025) so Oxi's top-1 move-matching
//! accuracy is directly comparable to the paper's reported numbers (Allie
//! policy: 55.7%, Maia: 51.6%).
//!
//! Test data: `yimingzhang/allie-data` on HuggingFace,
//! `lichess-2022-blitz-test/2022-test-annotated.jsonl` — 18k Lichess 2022
//! blitz games, downsampled to roughly uniform 100-Elo bins. Protocol:
//! discard the first 5 moves (10 plies) of each game and any move made with
//! under 30 seconds on the mover's clock, then score argmax-vs-human over the
//! remaining positions (884,049 in the paper).
//!
//! Clocks are reconstructed from `moves-seconds` (per-move think time derived
//! from Lichess `%clk` annotations): a player's remaining time before their
//! k-th move is `base - sum(their prior think times) + increment * (k - 1)`,
//! mirroring how the training pipeline reads `%clk` directly.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};

use crate::config::PREVIOUS_POSITIONS;
use crate::inference::{BatchItem, GlobalFeatures, InferenceEngine};

#[derive(Debug, Deserialize)]
struct AllieGame {
    #[serde(rename = "game-id")]
    game_id: String,
    #[serde(rename = "moves-uci")]
    moves_uci: String,
    #[serde(rename = "moves-seconds")]
    moves_seconds: Vec<i64>,
    #[serde(rename = "white-elo")]
    white_elo: i32,
    #[serde(rename = "black-elo")]
    black_elo: i32,
    #[serde(rename = "time-control")]
    time_control: String,
}

#[derive(Debug, Clone)]
pub struct AllieEvalParams {
    pub dataset: PathBuf,
    pub batch_size: usize,
    /// Plies excluded at the start of each game (paper: first 5 moves = 10 plies).
    pub skip_plies: usize,
    /// Moves made with less than this many seconds on the mover's clock are excluded.
    pub min_clock: i64,
    /// Cap on number of games (smoke tests).
    pub limit: Option<usize>,
    /// Optional JSONL dump of per-position results.
    pub dump: Option<PathBuf>,
}

/// One scoreable position, paired with everything needed to attribute the result.
struct EvalItem {
    batch_item: BatchItem,
    target_uci: String,
    mover_elo: i32,
    is_castling: bool,
    is_en_passant: bool,
    is_promotion: bool,
    game_id: String,
    ply: usize,
}

#[derive(Default)]
struct BucketStats {
    total: usize,
    matched: usize,
}

#[derive(Default)]
struct EvalStats {
    games: usize,
    games_skipped: usize,
    plies_total: usize,
    excluded_opening: usize,
    excluded_clock: usize,
    total: usize,
    matched: usize,
    by_elo: BTreeMap<i32, BucketStats>,
    castling: BucketStats,
    en_passant: BucketStats,
    promotion: BucketStats,
}

#[derive(Serialize)]
struct DumpRecord<'a> {
    game_id: &'a str,
    ply: usize,
    mover_elo: i32,
    target: &'a str,
    predicted: &'a str,
    probability: f32,
    matched: bool,
}

fn parse_time_control(tc: &str) -> Option<(i64, i64)> {
    let (base, inc) = tc.split_once('+')?;
    Some((base.parse().ok()?, inc.parse().ok()?))
}

/// Build the scoreable positions for one game, applying the paper's
/// exclusions. Returns None if the game can't be replayed.
fn build_game_items(
    game: &AllieGame,
    params: &AllieEvalParams,
    stats: &mut EvalStats,
) -> Option<Vec<EvalItem>> {
    let (base_time, increment) = parse_time_control(&game.time_control)?;
    let moves: Vec<&str> = game.moves_uci.split_whitespace().collect();
    if moves.len() != game.moves_seconds.len() {
        return None;
    }

    let mut items = Vec::new();
    // positions[k] = position after k plies, most recent LAST (reversed per item).
    let mut positions: Vec<Chess> = vec![Chess::default()];
    let mut uci_moves: Vec<String> = Vec::new();
    // Remaining clock for each side before their next move.
    let mut clocks = [base_time, base_time];

    for (ply, (&mv_str, &seconds)) in moves.iter().zip(&game.moves_seconds).enumerate() {
        stats.plies_total += 1;
        let pos = positions.last().unwrap().clone();
        let side = ply % 2; // 0 = white
        let mover_clock = clocks[side];
        let oppo_clock = clocks[1 - side];

        let uci: UciMove = mv_str.parse().ok()?;
        let chess_move = uci.to_move(&pos).ok()?;

        let included = if ply < params.skip_plies {
            stats.excluded_opening += 1;
            false
        } else if mover_clock < params.min_clock {
            stats.excluded_clock += 1;
            false
        } else {
            true
        };

        if included {
            let (mover_elo, oppo_elo) = if side == 0 {
                (game.white_elo, game.black_elo)
            } else {
                (game.black_elo, game.white_elo)
            };

            // `positions` ends with the position the mover faces, so reversing
            // puts the current position first (the BatchItem convention).
            let item_positions: Vec<Chess> = positions
                .iter()
                .rev()
                .take(PREVIOUS_POSITIONS + 1)
                .cloned()
                .collect();
            let previous_moves: Vec<String> = uci_moves
                .iter()
                .rev()
                .take(PREVIOUS_POSITIONS)
                .cloned()
                .collect();

            let globals = GlobalFeatures {
                time_remaining_self: mover_clock.max(0) as u32,
                time_remaining_oppo: oppo_clock.max(0) as u32,
                base_time: base_time.max(0) as u32,
                increment: increment.max(0) as u32,
                move_count: ply,
                elo_self: mover_elo,
                elo_oppo: oppo_elo,
                is_puzzle: false,
                is_in_check: pos.is_check(),
                total_pieces: pos.board().occupied().count() as u32,
            };

            items.push(EvalItem {
                batch_item: BatchItem {
                    positions: item_positions,
                    previous_moves,
                    globals,
                    temperature: 1.0,
                    top_k: 1,
                },
                target_uci: chess_move.to_uci(CastlingMode::Standard).to_string(),
                mover_elo,
                is_castling: chess_move.is_castle(),
                is_en_passant: chess_move.is_en_passant(),
                is_promotion: chess_move.promotion().is_some(),
                game_id: game.game_id.clone(),
                ply,
            });
        }

        let next = pos.play(chess_move).ok()?;
        positions.push(next);
        uci_moves.push(mv_str.to_string());
        clocks[side] = mover_clock - seconds + increment;
    }

    Some(items)
}

fn record_result(stats: &mut EvalStats, item: &EvalItem, matched: bool) {
    stats.total += 1;
    let bucket = (item.mover_elo / 100) * 100;
    let entry = stats.by_elo.entry(bucket).or_default();
    entry.total += 1;
    for (flag, bucket_stats) in [
        (item.is_castling, &mut stats.castling),
        (item.is_en_passant, &mut stats.en_passant),
        (item.is_promotion, &mut stats.promotion),
    ] {
        if flag {
            bucket_stats.total += 1;
            if matched {
                bucket_stats.matched += 1;
            }
        }
    }
    if matched {
        stats.matched += 1;
        stats.by_elo.get_mut(&bucket).unwrap().matched += 1;
    }
}

fn print_report(stats: &EvalStats, params: &AllieEvalParams) {
    let pct = |s: &BucketStats| {
        if s.total == 0 {
            f64::NAN
        } else {
            100.0 * s.matched as f64 / s.total as f64
        }
    };

    println!("\n=== Allie test set move-matching ===");
    println!(
        "Games: {} evaluated, {} skipped (parse/replay failures)",
        stats.games, stats.games_skipped
    );
    println!(
        "Plies: {} total, {} excluded (first {} plies), {} excluded (clock < {}s)",
        stats.plies_total,
        stats.excluded_opening,
        params.skip_plies,
        stats.excluded_clock,
        params.min_clock
    );
    let acc = 100.0 * stats.matched as f64 / stats.total.max(1) as f64;
    let stderr = (acc / 100.0 * (1.0 - acc / 100.0) / stats.total.max(1) as f64).sqrt() * 100.0;
    println!(
        "\nTop-1 accuracy: {:.2}% ± {:.2} ({}/{} positions)",
        acc, stderr, stats.matched, stats.total
    );
    println!("(paper reference on this protocol: Allie 55.7%, Maia 51.6%)");

    println!("\n{:<12} {:>10} {:>10}", "Elo bucket", "Accuracy", "Count");
    println!("{}", "-".repeat(34));
    for (bucket, s) in &stats.by_elo {
        println!(
            "{:<12} {:>9.2}% {:>10}",
            format!("{}-{}", bucket, bucket + 99),
            pct(s),
            s.total
        );
    }

    println!("\nSpecial moves:");
    for (name, s) in [
        ("castling", &stats.castling),
        ("en passant", &stats.en_passant),
        ("promotion", &stats.promotion),
    ] {
        println!("  {:<12} {:>6.1}% ({})", name, pct(s), s.total);
    }
}

pub fn run_allie_eval<B: Backend>(
    engine: &InferenceEngine<B>,
    params: &AllieEvalParams,
) -> Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    let file = File::open(&params.dataset)
        .with_context(|| format!("opening dataset {:?}", params.dataset))?;
    let reader = BufReader::new(file);

    let mut dump_writer = params
        .dump
        .as_ref()
        .map(|p| -> Result<BufWriter<File>> {
            Ok(BufWriter::new(
                File::create(p).with_context(|| format!("creating dump file {p:?}"))?,
            ))
        })
        .transpose()?;

    let mut stats = EvalStats::default();
    let mut pending: Vec<EvalItem> = Vec::new();
    let started = std::time::Instant::now();

    let mut flush = |pending: &mut Vec<EvalItem>,
                     stats: &mut EvalStats,
                     dump_writer: &mut Option<BufWriter<File>>|
     -> Result<()> {
        for chunk in pending.chunks(params.batch_size) {
            let batch: Vec<BatchItem> = chunk.iter().map(|it| it.batch_item.clone()).collect();
            let predictions = engine.predict_batch(&batch)?;
            for (item, prediction) in chunk.iter().zip(&predictions) {
                let top = prediction
                    .moves
                    .first()
                    .context("prediction returned no moves")?;
                let matched = top.uci_move == item.target_uci;
                record_result(stats, item, matched);
                if let Some(w) = dump_writer {
                    serde_json::to_writer(
                        &mut *w,
                        &DumpRecord {
                            game_id: &item.game_id,
                            ply: item.ply,
                            mover_elo: item.mover_elo,
                            target: &item.target_uci,
                            predicted: &top.uci_move,
                            probability: top.probability,
                            matched,
                        },
                    )?;
                    w.write_all(b"\n")?;
                }
            }
        }
        pending.clear();
        Ok(())
    };

    for (line_idx, line) in reader.lines().enumerate() {
        if let Some(limit) = params.limit {
            if stats.games >= limit {
                break;
            }
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let game: AllieGame = match serde_json::from_str(&line) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("line {}: failed to parse game: {e}", line_idx + 1);
                stats.games_skipped += 1;
                continue;
            }
        };
        match build_game_items(&game, params, &mut stats) {
            Some(items) => {
                stats.games += 1;
                pending.extend(items);
            }
            None => {
                stats.games_skipped += 1;
            }
        }

        if pending.len() >= params.batch_size {
            flush(&mut pending, &mut stats, &mut dump_writer)?;
        }

        if stats.games > 0 && stats.games % 500 == 0 && pending.is_empty() {
            let acc = 100.0 * stats.matched as f64 / stats.total.max(1) as f64;
            println!(
                "  {} games, {} positions, running accuracy {:.2}% ({:.0}s elapsed)",
                stats.games,
                stats.total,
                acc,
                started.elapsed().as_secs_f64()
            );
        }
    }
    flush(&mut pending, &mut stats, &mut dump_writer)?;

    if let Some(mut w) = dump_writer {
        w.flush()?;
    }

    print_report(&stats, params);
    Ok(())
}
