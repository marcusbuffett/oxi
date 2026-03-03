//! Stockfish UCI protocol wrapper for position evaluation.
//!
//! Spawns a Stockfish process and communicates via UCI to evaluate
//! all legal moves in a position at a given depth.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

/// Large centipawn value used for mate scores.
/// Mate in N is scored as +/-(MATE_SCORE - N) so mates sort correctly.
const MATE_SCORE: i32 = 30_000;

/// Per-move centipawn loss cap. Anything beyond this is "decisive mistake" —
/// the difference between losing 1000cp and 30,000cp is meaningless for model eval.
/// Without this, mate positions would dominate averages.
const MOVE_LOSS_CAP: f32 = 1000.0;

/// Evaluation of a single move in a position
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoveEval {
    /// UCI move string (e.g., "e2e4", "g1f3")
    pub uci_move: String,
    /// Score in centipawns from the side-to-move's perspective.
    /// For mate scores, uses +/-(MATE_SCORE - mate_distance).
    pub score_cp: i32,
    /// Whether this is a mate score
    pub is_mate: bool,
    /// Signed mate distance: positive = winning in N, negative = losing in N
    pub mate_in: Option<i32>,
}

/// Evaluation result for all legal moves in a position
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionEval {
    pub fen: String,
    pub move_evals: Vec<MoveEval>,
    pub depth: u32,
}

impl PositionEval {
    /// Best move's eval in centipawns
    pub fn best_eval_cp(&self) -> i32 {
        self.move_evals
            .iter()
            .map(|e| e.score_cp)
            .max()
            .unwrap_or(0)
    }

    /// Sharpness: mean centipawn loss under a uniform prior over legal moves.
    /// This measures "how punishing is this position if you pick randomly."
    /// High sharpness = narrow ridge of good moves, cliff everywhere else.
    pub fn sharpness(&self) -> f32 {
        if self.move_evals.is_empty() {
            return 0.0;
        }
        let best = self.best_eval_cp() as f32;
        let n = self.move_evals.len() as f32;
        let total_loss: f32 = self
            .move_evals
            .iter()
            .map(|e| (best - e.score_cp as f32).max(0.0).min(MOVE_LOSS_CAP))
            .sum();
        total_loss / n
    }

    /// ECL (expected centipawn loss) under an arbitrary policy distribution.
    /// `move_probs` maps UCI move strings to probabilities.
    /// Returns the probability-weighted centipawn loss.
    pub fn ecl_under_policy(&self, move_probs: &HashMap<String, f32>) -> f32 {
        let best = self.best_eval_cp() as f32;
        let mut ecl = 0.0f32;
        for eval in &self.move_evals {
            let prob = move_probs.get(&eval.uci_move).copied().unwrap_or(0.0);
            let loss = (best - eval.score_cp as f32).max(0.0).min(MOVE_LOSS_CAP);
            ecl += prob * loss;
        }
        ecl
    }

    /// ECL of a specific move (the "human ECL" when given the move a player made)
    pub fn ecl_of_move(&self, uci_move: &str) -> Option<f32> {
        let best = self.best_eval_cp() as f32;
        self.move_evals
            .iter()
            .find(|e| e.uci_move == uci_move)
            .map(|e| (best - e.score_cp as f32).max(0.0).min(MOVE_LOSS_CAP))
    }
}

/// Wraps a Stockfish process for evaluating positions.
///
/// Keeps the process alive across multiple evaluations for efficiency.
/// Uses MultiPV to get evaluations of all legal moves in a single search.
pub struct StockfishEngine {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    depth: u32,
}

impl StockfishEngine {
    /// Spawn a new Stockfish process.
    ///
    /// # Arguments
    /// * `stockfish_path` - Path to Stockfish binary, defaults to "stockfish"
    /// * `depth` - Default search depth for evaluations
    /// * `threads` - Number of threads for Stockfish to use
    pub fn new(stockfish_path: Option<&str>, depth: u32, threads: u32) -> Result<Self> {
        let path = stockfish_path.unwrap_or("stockfish");
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!("Failed to spawn Stockfish at '{}'. Is it installed?", path)
            })?;

        let stdin = child
            .stdin
            .take()
            .context("Failed to get Stockfish stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("Failed to get Stockfish stdout")?;
        let reader = BufReader::new(stdout);

        let mut engine = Self {
            child,
            stdin,
            reader,
            depth,
        };

        // Initialize UCI protocol
        engine.send("uci")?;
        engine.wait_for_line("uciok")?;

        // Configure: high MultiPV to get all legal moves, configurable threads
        engine.send(&format!("setoption name MultiPV value 256"))?;
        engine.send(&format!("setoption name Threads value {}", threads))?;
        // Limit hash to avoid excessive memory use during eval
        engine.send("setoption name Hash value 128")?;

        engine.send("isready")?;
        engine.wait_for_line("readyok")?;

        Ok(engine)
    }

    fn send(&mut self, cmd: &str) -> Result<()> {
        writeln!(self.stdin, "{}", cmd)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn wait_for_line(&mut self, prefix: &str) -> Result<()> {
        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line)?;
            if line.trim().starts_with(prefix) {
                return Ok(());
            }
        }
    }

    /// Evaluate all legal moves in a position.
    ///
    /// Returns evaluations sorted by score (best first).
    /// Uses the default depth configured at construction time.
    pub fn evaluate_position(&mut self, fen: &str) -> Result<PositionEval> {
        self.evaluate_position_at_depth(fen, self.depth)
    }

    /// Evaluate all legal moves at a specific depth.
    pub fn evaluate_position_at_depth(&mut self, fen: &str, depth: u32) -> Result<PositionEval> {
        // Reset state between positions for clean evaluation
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.wait_for_line("readyok")?;

        self.send(&format!("position fen {}", fen))?;
        self.send(&format!("go depth {}", depth))?;

        // Collect info lines. For each (multipv, depth) pair, keep the latest.
        // We want the results at the target depth.
        let mut best_by_pv: HashMap<usize, MoveEval> = HashMap::new();
        let mut best_depth_seen: HashMap<usize, u32> = HashMap::new();

        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line)?;
            let trimmed = line.trim();

            if trimmed.starts_with("bestmove") {
                break;
            }

            if trimmed.starts_with("info")
                && trimmed.contains("multipv")
                && trimmed.contains(" pv ")
            {
                if let Some((multipv, info_depth, eval)) = Self::parse_info_line(trimmed) {
                    // Keep the highest depth we've seen for each PV
                    let prev_depth = best_depth_seen.get(&multipv).copied().unwrap_or(0);
                    if info_depth >= prev_depth {
                        best_depth_seen.insert(multipv, info_depth);
                        best_by_pv.insert(multipv, eval);
                    }
                }
            }
        }

        // Collect results sorted by multipv index (which is score-ordered by Stockfish)
        let mut results: Vec<(usize, MoveEval)> = best_by_pv.into_iter().collect();
        results.sort_by_key(|(idx, _)| *idx);
        let move_evals: Vec<MoveEval> = results.into_iter().map(|(_, eval)| eval).collect();

        Ok(PositionEval {
            fen: fen.to_string(),
            move_evals,
            depth,
        })
    }

    /// Parse a single Stockfish info line.
    /// Returns (multipv_index, depth, MoveEval) if the line is valid.
    fn parse_info_line(line: &str) -> Option<(usize, u32, MoveEval)> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        let mut depth: Option<u32> = None;
        let mut multipv: Option<usize> = None;
        let mut score_cp: Option<i32> = None;
        let mut is_mate = false;
        let mut mate_in: Option<i32> = None;
        let mut pv_move: Option<String> = None;

        let mut i = 0;
        while i < parts.len() {
            match parts[i] {
                "depth" if i + 1 < parts.len() => {
                    depth = parts[i + 1].parse().ok();
                    i += 2;
                }
                "multipv" if i + 1 < parts.len() => {
                    multipv = parts[i + 1].parse().ok();
                    i += 2;
                }
                "score" if i + 2 < parts.len() => {
                    match parts[i + 1] {
                        "cp" => {
                            score_cp = parts[i + 2].parse().ok();
                            i += 3;
                        }
                        "mate" => {
                            let m: i32 = parts[i + 2].parse().ok()?;
                            is_mate = true;
                            mate_in = Some(m);
                            // Convert to centipawn equivalent preserving ordering
                            score_cp = Some(if m > 0 {
                                MATE_SCORE - m.abs()
                            } else {
                                -(MATE_SCORE - m.abs())
                            });
                            i += 3;
                        }
                        _ => {
                            i += 2;
                        }
                    }
                }
                "pv" if i + 1 < parts.len() => {
                    pv_move = Some(parts[i + 1].to_string());
                    break; // PV is always at the end
                }
                _ => {
                    i += 1;
                }
            }
        }

        match (multipv, depth, score_cp, pv_move) {
            (Some(mpv), Some(d), Some(cp), Some(mv)) => Some((
                mpv,
                d,
                MoveEval {
                    uci_move: mv,
                    score_cp: cp,
                    is_mate,
                    mate_in,
                },
            )),
            _ => None,
        }
    }
}

impl Drop for StockfishEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_info_line_cp() {
        let line = "info depth 12 seldepth 18 multipv 1 score cp 35 nodes 15234 nps 500000 time 30 pv e2e4 e7e5";
        let result = StockfishEngine::parse_info_line(line);
        assert!(result.is_some());
        let (mpv, depth, eval) = result.unwrap();
        assert_eq!(mpv, 1);
        assert_eq!(depth, 12);
        assert_eq!(eval.score_cp, 35);
        assert_eq!(eval.uci_move, "e2e4");
        assert!(!eval.is_mate);
    }

    #[test]
    fn test_parse_info_line_mate() {
        let line = "info depth 10 multipv 1 score mate 3 pv e1g1";
        let result = StockfishEngine::parse_info_line(line);
        assert!(result.is_some());
        let (_, _, eval) = result.unwrap();
        assert!(eval.is_mate);
        assert_eq!(eval.mate_in, Some(3));
        assert!(eval.score_cp > 0);
        assert_eq!(eval.uci_move, "e1g1");
    }

    #[test]
    fn test_parse_info_line_negative_mate() {
        let line = "info depth 10 multipv 2 score mate -2 pv a1a2";
        let result = StockfishEngine::parse_info_line(line);
        assert!(result.is_some());
        let (_, _, eval) = result.unwrap();
        assert!(eval.is_mate);
        assert_eq!(eval.mate_in, Some(-2));
        assert!(eval.score_cp < 0);
    }

    #[test]
    fn test_position_eval_sharpness() {
        let eval = PositionEval {
            fen: "startpos".to_string(),
            move_evals: vec![
                MoveEval {
                    uci_move: "e2e4".into(),
                    score_cp: 30,
                    is_mate: false,
                    mate_in: None,
                },
                MoveEval {
                    uci_move: "d2d4".into(),
                    score_cp: 25,
                    is_mate: false,
                    mate_in: None,
                },
                MoveEval {
                    uci_move: "a2a3".into(),
                    score_cp: -10,
                    is_mate: false,
                    mate_in: None,
                },
            ],
            depth: 12,
        };
        // best=30, losses: 0, 5, 40 -> mean = 45/3 = 15
        assert!((eval.sharpness() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_position_eval_ecl() {
        let eval = PositionEval {
            fen: "startpos".to_string(),
            move_evals: vec![
                MoveEval {
                    uci_move: "e2e4".into(),
                    score_cp: 100,
                    is_mate: false,
                    mate_in: None,
                },
                MoveEval {
                    uci_move: "d2d4".into(),
                    score_cp: 50,
                    is_mate: false,
                    mate_in: None,
                },
                MoveEval {
                    uci_move: "a2a3".into(),
                    score_cp: -100,
                    is_mate: false,
                    mate_in: None,
                },
            ],
            depth: 12,
        };

        let mut policy = HashMap::new();
        policy.insert("e2e4".to_string(), 0.7);
        policy.insert("d2d4".to_string(), 0.2);
        policy.insert("a2a3".to_string(), 0.1);

        // ECL = 0.7*0 + 0.2*50 + 0.1*200 = 0 + 10 + 20 = 30
        assert!((eval.ecl_under_policy(&policy) - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_ecl_of_move() {
        let eval = PositionEval {
            fen: "startpos".to_string(),
            move_evals: vec![
                MoveEval {
                    uci_move: "e2e4".into(),
                    score_cp: 100,
                    is_mate: false,
                    mate_in: None,
                },
                MoveEval {
                    uci_move: "a2a3".into(),
                    score_cp: -50,
                    is_mate: false,
                    mate_in: None,
                },
            ],
            depth: 12,
        };
        assert_eq!(eval.ecl_of_move("e2e4"), Some(0.0));
        assert_eq!(eval.ecl_of_move("a2a3"), Some(150.0));
        assert_eq!(eval.ecl_of_move("b2b3"), None);
    }
}
