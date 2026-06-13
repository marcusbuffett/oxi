//! Input-channel ablation against a trained checkpoint.
//!
//! Measures how much the current model relies on each hand-crafted board
//! feature by zeroing channel groups at inference time and scoring the
//! degradation on Allie test-set positions (top-1 move match and mean
//! log-probability of the human move). No retraining required.
//!
//! Interpretation caveat: this measures reliance of THIS trained model, not
//! the value of a feature for retraining. A channel can score near zero here
//! because it's redundant with other inputs (the trunk reconstructs it), and
//! a high-reliance channel might still be learnable from raw planes if
//! removed before training. Use these scores to shortlist prune candidates,
//! then confirm with one retrain of the pruned feature set.

use anyhow::Result;
use burn::tensor::backend::Backend;

use crate::allie_eval::{load_eval_items, AllieEvalParams, EvalItem};
use crate::config::FEATURES_PER_TOKEN;
use crate::inference::InferenceEngine;

/// Keep every legal move in the prediction list so the human move's
/// probability is always available (max legal moves in chess is 218).
const ABLATION_TOP_K: usize = 256;

/// Floor for missing/zero probabilities when computing mean log-prob.
const PROB_FLOOR: f32 = 1e-9;

/// Named channel ranges within the per-square feature vector. Must stay in
/// sync with `encoding::encode_position`.
pub const CHANNEL_GROUPS: &[(&str, std::ops::Range<usize>)] = &[
    // Coarse groups (post-2026-06 pruning layout + SEE/x-ray)
    ("piece_identity (ALL)", 0..12),
    ("tactical (ALL)", 12..36),
    ("positional (ALL)", 36..53),
    ("recency (ALL)", 54..58),
    ("history_occupancy (ALL)", 58..142),
    // Tactical subgroups
    ("attackers_white", 12..20),
    ("attackers_black", 20..28),
    ("hanging", 28..29),
    ("square_control", 29..30),
    ("see (both sides)", 30..32),
    ("xray_white (count/material)", 32..34),
    ("xray_black (count/material)", 34..36),
    // Positional subgroups
    ("mobility (legal_moves_norm)", 36..37),
    ("rank_onehot", 37..45),
    ("file_onehot", 45..53),
    // Misc
    ("castling_right", 53..54),
    // Recency subgroups
    ("recency_white (from/to)", 54..56),
    ("recency_black (from/to)", 56..58),
    // History occupancy by age (12 channels per past position)
    ("history_t-1", 58..70),
    ("history_t-2", 70..82),
    ("history_t-3..7", 82..142),
];

struct SweepResult {
    name: &'static str,
    channels: usize,
    top1: f64,
    mean_logprob: f64,
}

fn score_items<B: Backend>(
    engine: &InferenceEngine<B>,
    items: &[EvalItem],
    batch_size: usize,
    mask: Option<&[f32]>,
) -> Result<(f64, f64)>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    let mut matched = 0usize;
    let mut logprob_sum = 0f64;
    for chunk in items.chunks(batch_size) {
        let batch: Vec<_> = chunk.iter().map(|it| it.batch_item.clone()).collect();
        let predictions = engine.predict_batch_with_channel_mask(&batch, mask)?;
        for (item, prediction) in chunk.iter().zip(&predictions) {
            if let Some(top) = prediction.moves.first() {
                if top.uci_move == item.target_uci {
                    matched += 1;
                }
            }
            let target_prob = prediction
                .moves
                .iter()
                .find(|m| m.uci_move == item.target_uci)
                .map(|m| m.probability)
                .unwrap_or(0.0)
                .max(PROB_FLOOR);
            logprob_sum += (target_prob as f64).ln();
        }
    }
    let n = items.len().max(1) as f64;
    Ok((100.0 * matched as f64 / n, logprob_sum / n))
}

pub fn run_feature_ablation<B: Backend>(
    engine: &InferenceEngine<B>,
    params: &AllieEvalParams,
    batch_size: usize,
) -> Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    let mut items = load_eval_items(params)?;
    anyhow::ensure!(!items.is_empty(), "no eval items loaded");
    for item in &mut items {
        item.batch_item.top_k = ABLATION_TOP_K;
    }
    println!(
        "Loaded {} positions; scoring baseline + {} channel groups...",
        items.len(),
        CHANNEL_GROUPS.len()
    );

    let started = std::time::Instant::now();
    let (base_top1, base_logprob) = score_items(engine, &items, batch_size, None)?;
    println!(
        "Baseline: top-1 {:.2}%, mean ln p(human move) {:.4}  ({:.0}s)",
        base_top1,
        base_logprob,
        started.elapsed().as_secs_f64()
    );

    let mut results = Vec::new();
    for (name, range) in CHANNEL_GROUPS {
        let mut mask = vec![1.0f32; FEATURES_PER_TOKEN];
        for c in range.clone() {
            mask[c] = 0.0;
        }
        let (top1, logprob) = score_items(engine, &items, batch_size, Some(&mask))?;
        println!(
            "  {:38} Δtop-1 {:+6.2}pt  Δln p {:+8.4}",
            name,
            top1 - base_top1,
            logprob - base_logprob
        );
        results.push(SweepResult {
            name,
            channels: range.len(),
            top1,
            mean_logprob: logprob,
        });
    }

    results.sort_by(|a, b| {
        (a.mean_logprob - base_logprob)
            .partial_cmp(&(b.mean_logprob - base_logprob))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n=== Reliance ranking (most load-bearing first) ===");
    println!(
        "{:<38} {:>4} {:>10} {:>12}",
        "group", "ch", "Δtop-1", "Δ ln p"
    );
    println!("{}", "-".repeat(68));
    for r in &results {
        println!(
            "{:<38} {:>4} {:>+9.2}pt {:>+12.4}",
            r.name,
            r.channels,
            r.top1 - base_top1,
            r.mean_logprob - base_logprob
        );
    }
    println!(
        "\nBaseline top-1 {:.2}% over {} positions. Near-zero Δ = redundant for\nthis model (prune candidate); large negative Δ = load-bearing input.",
        base_top1,
        items.len()
    );
    Ok(())
}
