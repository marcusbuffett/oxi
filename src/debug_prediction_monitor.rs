use std::borrow::Cow;
use std::time::Instant;

use anyhow::{Context, Result};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use shakmaty::fen::Fen;
use shakmaty::san::{San, SanPlus};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position};

use crate::config::PREVIOUS_POSITIONS;
use crate::dataset::{ChessBatcher, ChessExample, ChessItem, OXIDataset};
use crate::inference::compute_material_imbalance;
use crate::metrics_renderer::{MetricState, MetricsRenderer, PredictionEntry, PredictionMetric};
use crate::model::OXIModel;
use crate::move_encoding::decode_move;
use crate::moves::{mirror_fen, mirror_move};
const MAX_DEBUG_SLOTS: usize = 10;

#[derive(Debug, Clone)]
struct DebugPositionSpec {
    id: &'static str,
    name: &'static str,
    history_san: &'static [&'static str],
    white_time_ms: u32,
    black_time_ms: u32,
    initial_time_ms: u32,
    increment_ms: u32,
    bot_elo: i32,
    top_n: usize,
}

#[derive(Clone)]
struct DebugPosition {
    id: String,
    name: String,
    item: ChessItem,
    reference_pos: Chess,
    /// True when the example was mirrored into model frame (black to move);
    /// decoded moves must be mirrored back before display.
    mirrored: bool,
    display_slots: usize,
    query_top: usize,
}

pub struct DebugPredictionMonitor<B: AutodiffBackend>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    batcher: ChessBatcher<B>,
    device: B::Device,
    positions: Vec<DebugPosition>,
}

impl<B: AutodiffBackend> DebugPredictionMonitor<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    pub fn new(dataset: &OXIDataset, device: B::Device) -> Result<Option<Self>> {
        let specs = default_debug_specs();
        if specs.is_empty() {
            return Ok(None);
        }

        let mut positions = Vec::with_capacity(specs.len());
        for spec in specs {
            let position =
                build_debug_position(dataset, spec).with_context(|| spec_context(spec))?;
            positions.push(position);
        }

        let batcher = ChessBatcher::new(device.clone());

        Ok(Some(Self {
            batcher,
            device,
            positions,
        }))
    }

    pub fn evaluate(
        &mut self,
        iteration: usize,
        model: &OXIModel<B>,
        renderer: &mut dyn MetricsRenderer,
    ) -> Result<()> {
        let t_total = Instant::now();

        for position in &self.positions {
            let t_batch = Instant::now();
            let batch = self
                .batcher
                .batch(vec![position.item.clone()], &self.device)
                .to_device(&self.device);
            let batch_time = t_batch.elapsed();

            let t_forward = Instant::now();
            let output = model.forward_classification(batch.clone());
            let forward_time = t_forward.elapsed();

            let t_to_data = Instant::now();
            let policy_probs = softmax(output.policy_output.clone(), 1)
                .to_data()
                .convert::<f32>();
            let to_data_time = t_to_data.elapsed();

            let prob_slice = policy_probs
                .as_slice::<f32>()
                .expect("policy probabilities should be accessible");

            let legal_data = batch.legal_moves.to_data();
            let legal_slice = legal_data
                .as_slice::<f32>()
                .expect("legal moves should be accessible");

            let t_select = Instant::now();
            let predictions = select_top_moves(
                prob_slice,
                legal_slice,
                position.query_top,
                &position.reference_pos,
                position.mirrored,
            );
            let select_time = t_select.elapsed();

            let entries = predictions
                .into_iter()
                .take(position.display_slots)
                .map(|(label, probability)| PredictionEntry {
                    label,
                    probability: probability as f64,
                })
                .collect::<Vec<_>>();

            let metric = PredictionMetric {
                name: format!("Debug Prediction {}", position.name),
                formatted: format!("Iteration {iteration}"),
                predictions: entries,
            };

            renderer.update_train(MetricState::Predictions(metric));

            tracing::info!(
                "perf_debug_monitor_position: iter={} pos={} batch={:?} forward={:?} to_data={:?} select={:?}",
                iteration,
                position.name,
                batch_time,
                forward_time,
                to_data_time,
                select_time
            );
        }

        let total_time = t_total.elapsed();
        tracing::info!(
            "perf_debug_monitor_total: iter={} total={:?} positions={}",
            iteration,
            total_time,
            self.positions.len()
        );

        Ok(())
    }
}

fn select_top_moves(
    probabilities: &[f32],
    legal_mask: &[f32],
    top_n: usize,
    reference_pos: &Chess,
    mirrored: bool,
) -> Vec<(String, f32)> {
    let mut indexed = probabilities
        .iter()
        .enumerate()
        .filter(|(idx, _)| legal_mask.get(*idx).copied().unwrap_or(0.0) > 0.0)
        .map(|(idx, prob)| (idx, *prob))
        .collect::<Vec<_>>();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed
        .into_iter()
        .take(top_n)
        .filter_map(|(idx, prob)| {
            if prob <= 0.0 || !prob.is_finite() {
                return None;
            }
            let from = (idx / 76) as u8;
            let to = (idx % 76) as u8;
            // decode_move yields the move in model frame; for black-to-move
            // positions the item was mirrored, so mirror back before
            // rendering SAN against the real board.
            let mut uci_move = decode_move(from, to)?;
            if mirrored {
                uci_move = mirror_move(&uci_move.to_string()).parse().ok()?;
            }
            let san = uci_move
                .to_move(reference_pos)
                .ok()
                .map(|mv| SanPlus::from_move((*reference_pos).clone(), mv).to_string())
                .unwrap_or_else(|| uci_move.to_string());
            Some((san, prob))
        })
        .collect()
}

fn build_debug_position(dataset: &OXIDataset, spec: &DebugPositionSpec) -> Result<DebugPosition> {
    let mut positions = Vec::new();
    positions.push(Chess::default());

    let mut uci_history = Vec::new();

    for san_str in spec.history_san {
        let current = positions
            .last()
            .expect("at least starting position is present")
            .clone();
        let san: San = san_str
            .parse()
            .with_context(|| format!("failed to parse SAN move `{san_str}`"))?;
        let mv = san
            .to_move(&current)
            .with_context(|| format!("failed to convert SAN `{san_str}` to move"))?;
        let uci = mv.to_uci(CastlingMode::Standard).to_string().to_owned();
        let next = current
            .clone()
            .play(mv)
            .with_context(|| format!("failed to apply move `{san_str}`"))?;
        positions.push(next);
        uci_history.push(uci);
    }

    let final_pos = positions
        .last()
        .cloned()
        .context("no final position computed")?;
    let turn = final_pos.turn();
    // ChessExample is by convention in MODEL frame: pgn_processor mirrors the
    // board, move, and history whenever black is to move, so the model only
    // ever sees side-to-move-as-white. Feeding an unmirrored black-to-move
    // position here is out-of-distribution input and yields nonsense
    // predictions (this exact bug shipped: the probe showed random legal
    // moves for black while the production inference path was fine).
    let needs_mirror = turn == Color::Black;

    let mut previous_moves = Vec::new();
    let mut previous_fens = Vec::new();
    let mut material_history = Vec::new();

    let total_moves = uci_history.len();
    for i in 0..PREVIOUS_POSITIONS.min(total_moves) {
        let move_idx = total_moves - 1 - i;
        // positions[move_idx] is the position uci_history[move_idx] was played
        // from — the same snapshot pgn_processor stores per ply.
        let snapshot = &positions[move_idx];
        let fen = Fen::from_position(snapshot, EnPassantMode::Legal).to_string();
        if needs_mirror {
            previous_moves.push(mirror_move(&uci_history[move_idx]));
            previous_fens.push(mirror_fen(&fen));
        } else {
            previous_moves.push(uci_history[move_idx].clone());
            previous_fens.push(fen);
        }
    }

    // Raw (white minus black) imbalance of the before-move snapshots, most
    // recent first — matching pgn_processor's previous_material_imbalances.
    let before_move_positions = &positions[..positions.len().saturating_sub(1)];
    for snapshot in before_move_positions.iter().rev().take(PREVIOUS_POSITIONS) {
        material_history.push(compute_material_imbalance(snapshot));
    }
    if material_history.is_empty() {
        material_history.push(compute_material_imbalance(&final_pos));
    }

    let (time_self_ms, time_oppo_ms) = match turn {
        Color::White => (spec.white_time_ms, spec.black_time_ms),
        Color::Black => (spec.black_time_ms, spec.white_time_ms),
    };

    let time_remaining_self = ms_to_secs(time_self_ms);
    let time_remaining_oppo = ms_to_secs(time_oppo_ms);
    let base_time = ms_to_secs(spec.initial_time_ms);
    let increment = ms_to_secs(spec.increment_ms);

    let move_count = spec.history_san.len();
    let default_move = final_pos
        .legal_moves()
        .into_iter()
        .next()
        .context("debug position has no legal moves")?
        .to_uci(CastlingMode::Standard)
        .to_string();

    let raw_fen = Fen::from_position(&final_pos, EnPassantMode::Legal).to_string();
    let (fen, move_uci) = if needs_mirror {
        (mirror_fen(&raw_fen), mirror_move(&default_move))
    } else {
        (raw_fen, default_move)
    };

    let example = ChessExample {
        fen,
        move_uci,
        elo_self: spec.bot_elo,
        elo_oppo: spec.bot_elo,
        outcome: 0.5,
        previous_fens,
        previous_moves,
        time_remaining_self,
        time_remaining_oppo,
        time_used_for_move: 0,
        original_time_control: (base_time, increment),
        move_count,
        material_imbalance_history: material_history,
        is_puzzle: false,
    };

    let item = dataset
        .process_example(&example)
        .context("failed to convert debug example into dataset item")?;

    let display_slots = spec.top_n.min(MAX_DEBUG_SLOTS);
    let query_top = spec.top_n.max(display_slots);

    Ok(DebugPosition {
        id: spec.id.to_string(),
        name: spec.name.to_string(),
        item,
        reference_pos: final_pos,
        mirrored: needs_mirror,
        display_slots,
        query_top,
    })
}

fn ms_to_secs(ms: u32) -> u32 {
    (ms / 1000).max(1)
}

fn default_debug_specs() -> Vec<&'static DebugPositionSpec> {
    vec![&DebugPositionSpec {
        id: "complex_sicilian_probe",
        name: "Complex Sicilian Probe",
        history_san: &[
            "e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3", "Nc6", "Nf3", "e6", "Bc4", "Nf6",
            "O-O", "Be7", "Qe2", "O-O", "Rd1", "Qc7", "h3", "a6", "b3", "b5", "Bd3", "Bb7", "Bb2",
            "Rac8", "Rac1", "Qb8", "e5", "Nh5", "Ne4", "Nf4", "Qd2", "Nxd3", "Qxd3", "Nb4", "Qb1",
            "Bxe4", "Qxe4", "Nxa2", "Rxc8", "Rxc8", "Nd4", "Nc3", "Bxc3", "Rxc3", "Ne2", "Rxb3",
            "Rxd7", "Bf8", "Qc2", "Ra3", "Rd1", "g6", "f4", "Qb6+", "Kf1", "Bc5", "Rc1", "Be3",
            "Re1", "b4", "Rb1", "a5", "Qb2",
        ],
        white_time_ms: 200_000,
        black_time_ms: 150_000,
        initial_time_ms: 300_000,
        increment_ms: 3_000,
        bot_elo: 2045,
        top_n: 5,
    }]
}

fn spec_context(spec: &DebugPositionSpec) -> Cow<'static, str> {
    Cow::Owned(format!("while constructing debug position {}", spec.id))
}
