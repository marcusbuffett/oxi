use anyhow::Result;
use std::path::Path;

use crate::config::Config;
use crate::dataset::ChessExample;
use crate::eval_dataset::SampledPosition;
use crate::pgn_processor::process_pgn_directory_iter;

pub fn calibration_stream_config() -> Config {
    let mut config = Config::default();
    config.enable_ply_sampling = Some(false);
    config.enable_elo_sampling = Some(false);
    config
}

pub fn build_human_training_stream(
    data_path: &Path,
) -> Result<Box<dyn Iterator<Item = ChessExample>>> {
    if data_path.is_dir() {
        tracing::info!(
            "Setting up shared human training stream from {:?}",
            data_path
        );
        // Parallel readers: a single sequential reader can't keep a batch-512
        // 768/8 H100 run fed once it reaches modern (slower-parsing) months,
        // and the mini at batch 4096 needs ~5x more samples/s than that.
        // Default 8 suits the full model; OXI_PGN_READER_THREADS overrides
        // (the mini needs ~20 on a 26-core box).
        let reader_threads = std::env::var("OXI_PGN_READER_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        Ok(Box::new(
            crate::pgn_processor::process_pgn_directory_iter_parallel(data_path, reader_threads)?,
        ))
    } else {
        anyhow::bail!(
            "Human training stream requires a directory path, got file: {:?}",
            data_path
        );
    }
}

pub fn sample_positions_from_human_training_stream(
    data_path: &Path,
    num_positions: usize,
) -> Result<Vec<SampledPosition>> {
    Ok(build_human_training_stream(data_path)?
        .take(num_positions)
        .map(|example| SampledPosition {
            fen: example.fen,
            human_move: example.move_uci,
            player_elo: example.elo_self,
            opponent_elo: example.elo_oppo,
            ply: example.move_count as u32,
            game_result: example.outcome,
        })
        .collect())
}
