//! Rating-sensitivity probe: sweep fixed positions across conditioned Elo
//! values and dump the move distributions, one JSON line per (epd, elo).
//!
//! Built to measure whether conditioning changes (soft elo bins, flattened
//! sampler) actually decompress predictions across rating bands — compare the
//! per-position cross-elo disparity of two checkpoints on the same EPD set.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

use crate::inference::{BatchItem, GlobalFeatures, InferenceEngine};

pub struct EloProbeParams {
    pub epds: PathBuf,
    pub elos: Vec<i32>,
    pub top_n: usize,
    pub out: PathBuf,
    /// Legacy serving metadata (fake 5+0 clocks) instead of declaring the
    /// clock/history missing. Use for models trained before the missing
    /// indicators existed.
    pub legacy_metadata: bool,
}

pub fn run_elo_probe<B: Backend>(engine: &InferenceEngine<B>, params: &EloProbeParams) -> Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    let file = std::fs::File::open(&params.epds)
        .with_context(|| format!("open epd list {:?}", params.epds))?;
    let epds: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .map(|l| l.split('|').next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    println!("Probing {} positions x {:?} elos", epds.len(), params.elos);

    let mut out = std::io::BufWriter::new(std::fs::File::create(&params.out)?);
    let mut items: Vec<(String, i32, BatchItem)> = Vec::new();
    for epd in &epds {
        let fen: Fen = match epd.parse() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("skipping invalid EPD {epd}: {e}");
                continue;
            }
        };
        let pos: Chess = match fen.into_position(CastlingMode::Standard) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping illegal EPD {epd}: {e}");
                continue;
            }
        };
        for &elo in &params.elos {
            let globals = GlobalFeatures {
                time_remaining_self: 300,
                time_remaining_oppo: 300,
                base_time: 300,
                increment: 3,
                move_count: 0,
                elo_self: elo,
                elo_oppo: elo,
                is_puzzle: false,
                is_in_check: pos.is_check(),
                total_pieces: pos.board().occupied().count() as u32,
                clock_missing: !params.legacy_metadata,
                history_missing: !params.legacy_metadata,
            };
            items.push((
                epd.clone(),
                elo,
                BatchItem {
                    positions: vec![pos.clone()],
                    previous_moves: Vec::new(),
                    globals,
                    temperature: 1.0,
                    top_k: params.top_n,
                },
            ));
        }
    }

    for chunk in items.chunks(64) {
        let batch: Vec<BatchItem> = chunk.iter().map(|(_, _, it)| it.clone()).collect();
        let predictions = engine.predict_batch(&batch)?;
        for ((epd, elo, _), prediction) in chunk.iter().zip(&predictions) {
            let moves: Vec<(String, f32)> = prediction
                .moves
                .iter()
                .map(|m| (m.uci_move.clone(), m.probability))
                .collect();
            let line = serde_json::json!({ "epd": epd, "elo": elo, "moves": moves });
            writeln!(out, "{line}")?;
        }
    }
    println!("Wrote {:?}", params.out);
    Ok(())
}
