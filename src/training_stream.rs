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
        Ok(Box::new(process_pgn_directory_iter(data_path)?))
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
