use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    cursor, execute,
    terminal::{disable_raw_mode, LeaveAlternateScreen},
};
use futures_util::StreamExt;
#[cfg(feature = "train")]
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use shakmaty::Position;
#[cfg(feature = "train")]
use std::collections::HashSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::{io::AsyncWriteExt, task};

use oxi::calibration::{label_sampled_position, CalibrationDb, RegretBin};
use oxi::config::{set_global_config, Config, ConfigOverrides, ModelSize};
use oxi::constants::{LICHESS_PUZZLE_URL, TCEC_DOWNLOAD_URL};
#[cfg(feature = "train")]
use oxi::custom_training::train_custom;
#[cfg(feature = "train")]
use oxi::dataset::ChessExample;
use oxi::eval_dataset::{sample_positions_from_pgn, EvalDataset};
use oxi::inference::{GlobalFeatures, InferenceEngine};
#[cfg(feature = "train")]
use oxi::pgn_processor::{process_pgn_directory_with_limit, process_pgn_file_with_limit};
use oxi::stockfish::StockfishEngine;
#[cfg(feature = "train")]
use oxi::training_stream::{
    calibration_stream_config, sample_positions_from_human_training_stream,
};

#[cfg(all(target_os = "linux", feature = "backend-cuda"))]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(all(target_os = "linux", feature = "backend-candle"))]
use burn_candle::{Candle, CandleDevice};

#[cfg(all(target_os = "linux", feature = "backend-tch"))]
use burn::backend::LibTorch;
#[cfg(all(target_os = "linux", feature = "backend-tch"))]
use burn_tch::LibTorchDevice;

#[cfg(all(target_os = "macos", feature = "backend-tch"))]
use burn::backend::LibTorch;
#[cfg(all(target_os = "macos", feature = "backend-tch"))]
use burn_tch::LibTorchDevice;

#[derive(Parser, Debug)]
#[command(name = "oxi")]
#[command(about = "Oxi chess engine implementation in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a new Oxi model
    Train(ConfigOverrides),

    /// Run inference on chess positions
    Inference(InferenceConfig),

    /// Download pre-trained model
    Download {
        /// Model name (e.g., "blitz", "rapid")
        model: String,
    },

    /// Download PGN files from Lichess database
    DownloadPgn {
        /// Year to download (2018-2023)
        #[arg(long)]
        year: u32,

        /// Month to download (1-12)
        #[arg(long)]
        month: u32,

        /// Output directory for downloaded files
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Download only one file for testing
        #[arg(long)]
        local: bool,
    },

    /// Download all Lichess PGN files since 2022
    DownloadAll {
        /// Output directory for downloaded files
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },

    /// Process PGN files into training data
    ProcessPgn(ProcessPgnConfig),

    /// Evaluate model performance on an eval dataset (ECL, sharpness, blunder rate)
    Evaluate(EvaluateConfig),

    /// Create an evaluation dataset by sampling positions from PGNs and running Stockfish
    CreateEvalSet(CreateEvalSetCli),

    /// Quick one-shot evaluation: sample positions, run Stockfish, evaluate model, print report
    QuickEval(QuickEvalCli),

    /// Create a SQLite cache of Stockfish-labeled positions for calibrated-strength training
    CreateCalibrationDb(CreateCalibrationDbCli),

    /// Estimate a ZCA whitening transform for trunk-mean embeddings and write
    /// whitening.json next to the model (loaded automatically at inference time)
    ComputeWhitening(ComputeWhiteningCli),

    /// Move-matching accuracy on the Allie test set (Zhang et al., ICLR 2025),
    /// directly comparable to the paper's Allie/Maia numbers
    EvaluateAllie(AllieEvalCli),

    /// Measure how much a trained checkpoint relies on each hand-crafted
    /// input feature by zeroing channel groups at inference time (no
    /// retraining) and scoring degradation on Allie test-set positions
    FeatureAblation(FeatureAblationCli),

    /// Download TCEC (Top Chess Engine Championship) games for pretraining
    DownloadTcec {
        #[arg(long)]
        data_path: Option<PathBuf>,
    },

    /// Download Lichess puzzle database
    DownloadPuzzles {
        #[arg(long)]
        data_path: Option<PathBuf>,
    },
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct InferenceConfig {
    /// Path to model checkpoint
    #[arg(long)]
    model_path: PathBuf,

    /// FEN position(s) to analyze (can be multiple)
    #[arg(long)]
    fen: Vec<String>,

    /// Path to file containing FEN positions (one per line)
    #[arg(long)]
    fen_file: Option<PathBuf>,

    /// ELO rating for white player
    #[arg(long)]
    white_elo: Option<i32>,

    /// ELO rating for black player
    #[arg(long)]
    black_elo: Option<i32>,

    /// Temperature for move sampling
    #[arg(long)]
    temperature: Option<f32>,

    /// Number of top moves to show
    #[arg(long)]
    top_k: Option<usize>,
}

impl InferenceConfig {
    fn white_elo(&self) -> i32 {
        self.white_elo.unwrap_or(1500)
    }
    fn black_elo(&self) -> i32 {
        self.black_elo.unwrap_or(1500)
    }
    fn temperature(&self) -> f32 {
        self.temperature.unwrap_or(1.0)
    }
    fn top_k(&self) -> usize {
        self.top_k.unwrap_or(5)
    }
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct ProcessPgnConfig {
    /// Input PGN file(s) or directory
    #[arg(long)]
    input: Vec<PathBuf>,

    /// Output directory for processed data
    #[arg(long)]
    output_dir: PathBuf,

    /// Minimum ELO rating to include
    #[arg(long)]
    min_elo: Option<i32>,

    /// Maximum ELO rating to include
    #[arg(long)]
    max_elo: Option<i32>,

    /// Number of parallel processing threads
    #[arg(long)]
    num_threads: Option<usize>,

    /// Chunk size for processing
    #[arg(long)]
    chunk_size: Option<usize>,
}

impl ProcessPgnConfig {
    fn num_threads(&self) -> usize {
        self.num_threads.unwrap_or(4)
    }
    fn chunk_size(&self) -> usize {
        self.chunk_size.unwrap_or(10000)
    }
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct EvaluateConfig {
    /// Model directory (containing model.mpk and params.json)
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to evaluation dataset (created by create-eval-set)
    #[arg(long)]
    data_path: PathBuf,

    /// Batch size for evaluation
    #[arg(long)]
    batch_size: Option<usize>,

    /// Device to use (cpu, cuda)
    #[arg(long)]
    device: Option<String>,

    /// Elo to condition the model on (if not set, uses each position's player Elo)
    #[arg(long)]
    model_elo: Option<i32>,
}

impl EvaluateConfig {
    fn batch_size(&self) -> usize {
        self.batch_size.unwrap_or(256)
    }
    fn device(&self) -> &str {
        self.device.as_deref().unwrap_or("cpu")
    }
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct CreateEvalSetCli {
    /// Directory containing PGN files to sample from
    #[arg(long)]
    pgn_dir: PathBuf,

    /// Output path for the eval dataset JSON
    #[arg(long, default_value = "./data/eval_set.json")]
    output: PathBuf,

    /// Path to Stockfish binary (defaults to "stockfish" in PATH)
    #[arg(long)]
    stockfish_path: Option<String>,

    /// Stockfish search depth (higher = more accurate but slower)
    #[arg(long, default_value = "12")]
    depth: u32,

    /// Number of threads for Stockfish
    #[arg(long, default_value = "1")]
    threads: u32,

    /// Total number of positions to include in the eval set
    #[arg(long, default_value = "10000")]
    num_positions: usize,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct QuickEvalCli {
    /// Model directory (containing model.mpk and params.json)
    #[arg(long)]
    model_dir: PathBuf,

    /// PGN file or directory to sample positions from
    #[arg(long)]
    pgn: PathBuf,

    /// Number of positions to evaluate
    #[arg(long, default_value = "200")]
    num_positions: usize,

    /// Stockfish search depth (lower = faster)
    #[arg(long, default_value = "10")]
    depth: u32,

    /// Elo to condition the model on (if not set, uses each position's player Elo)
    #[arg(long)]
    model_elo: Option<i32>,

    /// Path to Stockfish binary
    #[arg(long)]
    stockfish_path: Option<String>,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct ComputeWhiteningCli {
    /// Model directory (containing model.mpk and params.json)
    #[arg(long)]
    model_dir: PathBuf,

    /// Training data directory to sample positions from
    #[arg(long, default_value = "../data")]
    data_path: PathBuf,

    /// Number of positions to estimate statistics from
    #[arg(long, default_value = "20000")]
    num_positions: usize,

    /// Output path (defaults to <model_dir>/whitening.json)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Inference batch size
    #[arg(long, default_value = "256")]
    batch_size: usize,

    /// Fixed Elo used for the embedding globals, so player metadata does not
    /// leak into similarity
    #[arg(long, default_value = "1500")]
    model_elo: i32,

    /// Number of close/far example pairs to print
    #[arg(long, default_value = "5")]
    examples: usize,
}

#[derive(Parser, Debug)]
struct AllieEvalCli {
    /// Model directory (containing model.mpk and params.json)
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to the Allie test set JSONL
    /// (yimingzhang/allie-data, lichess-2022-blitz-test/2022-test-annotated.jsonl)
    #[arg(long)]
    dataset: PathBuf,

    /// Inference batch size
    #[arg(long, default_value = "512")]
    batch_size: usize,

    /// Plies excluded at the start of each game (paper: first 5 moves)
    #[arg(long, default_value = "10")]
    skip_plies: usize,

    /// Moves with less than this many seconds on the mover's clock are excluded
    #[arg(long, default_value = "30")]
    min_clock: i64,

    /// Evaluate only the first N games (smoke tests)
    #[arg(long)]
    limit: Option<usize>,

    /// Device: cpu, mps, or cuda
    #[arg(long)]
    device: Option<String>,

    /// Write per-position results to this JSONL file
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct FeatureAblationCli {
    /// Model directory (containing model.mpk and params.json)
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to the Allie test set JSONL (positions to score on)
    #[arg(long)]
    dataset: PathBuf,

    /// Games to load (~50 positions each; 400 games ≈ 20k positions)
    #[arg(long, default_value = "400")]
    limit: usize,

    /// Inference batch size
    #[arg(long, default_value = "512")]
    batch_size: usize,

    /// Device: cpu, mps, or cuda
    #[arg(long)]
    device: Option<String>,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct CreateCalibrationDbCli {
    /// Directory containing PGN files to sample from
    #[arg(long)]
    pgn_dir: PathBuf,

    /// Output path for the SQLite DB
    #[arg(long, default_value = "./data/calibration.db")]
    output: PathBuf,

    /// Path to Stockfish binary (defaults to "stockfish" in PATH)
    #[arg(long)]
    stockfish_path: Option<String>,

    /// Stockfish search depth
    #[arg(long, default_value = "10")]
    depth: u32,

    /// Number of threads for Stockfish
    #[arg(long, default_value = "1")]
    threads: u32,

    /// Total number of positions to sample before labeling
    #[arg(long, default_value = "10000")]
    num_positions: usize,

    /// Preserve training-stream order instead of shuffling/stratifying sampled positions
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    preserve_order: Option<bool>,
}

#[cfg(feature = "train")]
fn sample_training_positions_for_calibration(
    pgn_path: &PathBuf,
    num_positions: usize,
    elo_buckets: &[(i32, i32)],
    preserve_order: bool,
) -> Result<Vec<oxi::eval_dataset::SampledPosition>> {
    if preserve_order {
        return sample_positions_from_human_training_stream(pgn_path.as_path(), num_positions);
    }

    let candidate_target = (num_positions * 8).max(num_positions);
    let mut examples = if pgn_path.is_dir() {
        process_pgn_directory_with_limit(pgn_path.as_path(), Some(candidate_target))?
    } else {
        process_pgn_file_with_limit(pgn_path.as_path(), Some(candidate_target))?
    };

    if examples.is_empty() {
        return Ok(Vec::new());
    }

    let mut rng = rand::rng();
    examples.shuffle(&mut rng);

    let per_bucket = (num_positions / elo_buckets.len()).max(1);
    let mut selected = Vec::with_capacity(num_positions);
    let mut remaining_examples: Vec<ChessExample> = Vec::new();

    for &(elo_min, elo_max) in elo_buckets {
        let mut bucket_examples: Vec<ChessExample> = Vec::new();
        for example in examples.drain(..) {
            if example.elo_self >= elo_min && example.elo_self < elo_max {
                bucket_examples.push(example);
            } else {
                remaining_examples.push(example);
            }
        }

        bucket_examples.shuffle(&mut rng);
        selected.extend(bucket_examples.into_iter().take(per_bucket).map(|example| {
            oxi::eval_dataset::SampledPosition {
                fen: example.fen,
                human_move: example.move_uci,
                player_elo: example.elo_self,
                opponent_elo: example.elo_oppo,
                ply: example.move_count as u32,
                game_result: example.outcome,
            }
        }));

        examples = remaining_examples;
        remaining_examples = Vec::new();
    }

    if selected.len() < num_positions {
        examples.shuffle(&mut rng);
        selected.extend(
            examples
                .into_iter()
                .take(num_positions - selected.len())
                .map(|example| oxi::eval_dataset::SampledPosition {
                    fen: example.fen,
                    human_move: example.move_uci,
                    player_elo: example.elo_self,
                    opponent_elo: example.elo_oppo,
                    ply: example.move_count as u32,
                    game_result: example.outcome,
                }),
        );
    }

    Ok(selected)
}

/// Minimal model architecture params — mirrors what the production bot uses.
/// Only reads the fields needed to construct the model, ignoring training-only config.
#[derive(Debug, Clone, serde::Deserialize)]
struct ModelParams {
    #[serde(default)]
    pub model_size: ModelSize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default = "default_smolgen_hidden")]
    pub smolgen_hidden: usize,
    #[serde(default = "default_smolgen_global_dim")]
    pub smolgen_global_dim: usize,
    #[serde(default = "default_smolgen_gen_size")]
    pub smolgen_gen_size: usize,
}

fn default_smolgen_hidden() -> usize {
    24
}
fn default_smolgen_global_dim() -> usize {
    128
}
fn default_smolgen_gen_size() -> usize {
    128
}
/// Load a Config from a model directory's params.json file.
/// Only reads architecture params, fills the rest with defaults — same approach as the production bot.
fn load_config_from_model_dir(model_dir: &std::path::Path) -> Result<Config> {
    let params_path = model_dir.join("params.json");
    if !params_path.exists() {
        anyhow::bail!(
            "No params.json found in {}. Expected model directory with model.mpk and params.json.",
            model_dir.display()
        );
    }
    let params_str = std::fs::read_to_string(&params_path)?;
    let params: ModelParams = serde_json::from_str(&params_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", params_path.display(), e))?;
    Ok(Config {
        model_size: params.model_size,
        embed_dim: params.embed_dim,
        num_heads: params.num_heads,
        num_layers: params.num_layers,
        smolgen_hidden: params.smolgen_hidden,
        smolgen_global_dim: params.smolgen_global_dim,
        smolgen_gen_size: params.smolgen_gen_size,
        ..Default::default()
    })
}

/// Stub for builds without train/backend-tch (the command needs the training
/// stream for position sampling and tch for inference).
#[cfg(not(all(feature = "train", feature = "backend-tch")))]
fn compute_whitening_command(_cli: &ComputeWhiteningCli) -> Result<()> {
    anyhow::bail!("compute-whitening requires the train and backend-tch features")
}

#[cfg(not(feature = "backend-tch"))]
fn evaluate_allie_command(_cli: &AllieEvalCli) -> Result<()> {
    anyhow::bail!("evaluate-allie requires the backend-tch feature")
}

#[cfg(not(feature = "backend-tch"))]
fn feature_ablation_command(_cli: &FeatureAblationCli) -> Result<()> {
    anyhow::bail!("feature-ablation requires the backend-tch feature")
}

#[cfg(feature = "backend-tch")]
fn feature_ablation_command(cli: &FeatureAblationCli) -> Result<()> {
    use oxi::allie_eval::AllieEvalParams;
    use oxi::feature_ablation::run_feature_ablation;

    type EvalBackend = burn::backend::LibTorch<f32>;
    let device = resolve_tch_device(cli.device.as_deref())?;

    let config = load_config_from_model_dir(&cli.model_dir)?;
    let _ = set_global_config(config.clone());
    let engine = InferenceEngine::<EvalBackend>::from_checkpoint(
        &cli.model_dir.join("model"),
        config,
        device,
    )?;

    run_feature_ablation(
        &engine,
        &AllieEvalParams {
            dataset: cli.dataset.clone(),
            batch_size: cli.batch_size,
            skip_plies: 10,
            min_clock: 30,
            limit: Some(cli.limit),
            dump: None,
        },
        cli.batch_size,
    )
}

#[cfg(feature = "backend-tch")]
fn resolve_tch_device(device: Option<&str>) -> Result<burn_tch::LibTorchDevice> {
    Ok(match device {
        Some("cpu") => burn_tch::LibTorchDevice::Cpu,
        Some("mps") => burn_tch::LibTorchDevice::Mps,
        Some("cuda") => burn_tch::LibTorchDevice::Cuda(0),
        Some(other) => anyhow::bail!("unknown device {other:?} (expected cpu, mps, or cuda)"),
        None => {
            if cfg!(target_os = "macos") {
                burn_tch::LibTorchDevice::Mps
            } else {
                burn_tch::LibTorchDevice::Cuda(0)
            }
        }
    })
}

#[cfg(feature = "backend-tch")]
fn evaluate_allie_command(cli: &AllieEvalCli) -> Result<()> {
    use oxi::allie_eval::{run_allie_eval, AllieEvalParams};

    type EvalBackend = burn::backend::LibTorch<f32>;
    let device = resolve_tch_device(cli.device.as_deref())?;

    let config = load_config_from_model_dir(&cli.model_dir)?;
    let _ = set_global_config(config.clone());
    println!(
        "Model: embed_dim={}, num_layers={}, num_heads={}",
        config.embed_dim, config.num_layers, config.num_heads
    );
    let engine = InferenceEngine::<EvalBackend>::from_checkpoint(
        &cli.model_dir.join("model"),
        config,
        device,
    )?;

    run_allie_eval(
        &engine,
        &AllieEvalParams {
            dataset: cli.dataset.clone(),
            batch_size: cli.batch_size,
            skip_plies: cli.skip_plies,
            min_clock: cli.min_clock,
            limit: cli.limit,
            dump: cli.dump.clone(),
        },
    )
}

#[cfg(all(feature = "train", feature = "backend-tch"))]
fn compute_whitening_command(cli: &ComputeWhiteningCli) -> Result<()> {
    use oxi::inference::BatchItem;
    use oxi::training_stream::sample_positions_from_human_training_stream;

    type WhiteningBackend = burn::backend::LibTorch<f32>;
    let device = if cfg!(target_os = "macos") {
        burn_tch::LibTorchDevice::Mps
    } else {
        burn_tch::LibTorchDevice::Cpu
    };

    let config = load_config_from_model_dir(&cli.model_dir)?;
    let _ = set_global_config(config.clone());
    let engine = InferenceEngine::<WhiteningBackend>::from_checkpoint(
        &cli.model_dir.join("model"),
        config,
        device,
    )?;

    println!(
        "Sampling {} positions from {:?}...",
        cli.num_positions, cli.data_path
    );
    let sampled = sample_positions_from_human_training_stream(&cli.data_path, cli.num_positions)?;
    anyhow::ensure!(
        !sampled.is_empty(),
        "no positions sampled from {:?}",
        cli.data_path
    );

    let mut fens: Vec<String> = Vec::with_capacity(sampled.len());
    let mut items: Vec<BatchItem> = Vec::with_capacity(sampled.len());
    for pos in &sampled {
        let Ok(parsed) = pos.fen.parse::<shakmaty::fen::Fen>() else {
            continue;
        };
        let Ok(chess) = parsed.into_position::<shakmaty::Chess>(shakmaty::CastlingMode::Standard)
        else {
            continue;
        };
        let globals = GlobalFeatures {
            time_remaining_self: 300,
            time_remaining_oppo: 300,
            base_time: 300,
            increment: 0,
            move_count: pos.ply as usize,
            elo_self: cli.model_elo,
            elo_oppo: cli.model_elo,
            is_puzzle: false,
            is_in_check: chess.is_check(),
            total_pieces: chess.board().occupied().count() as u32,
        };
        fens.push(pos.fen.clone());
        items.push(BatchItem {
            positions: vec![chess],
            previous_moves: vec![],
            globals,
            temperature: 1.0,
            top_k: 1,
        });
    }
    println!("Embedding {} valid positions...", items.len());

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(items.len());
    for (i, chunk) in items.chunks(cli.batch_size).enumerate() {
        embeddings.extend(engine.raw_trunk_mean_embeddings_batch(chunk)?);
        if (i + 1) % 10 == 0 {
            println!("  embedded {}/{}", embeddings.len(), items.len());
        }
    }

    let transform = oxi::whitening::compute_whitening(&embeddings)?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| cli.model_dir.join("whitening.json"));
    transform.save(&output)?;
    println!(
        "Wrote whitening transform (dim {}, {} samples) to {:?}",
        transform.dim, transform.samples, output
    );

    report_whitening_similarity(&embeddings, &fens, &transform, cli.examples);
    Ok(())
}

/// Print cosine-similarity distributions raw vs whitened, plus example
/// close/far position pairs under the whitened geometry.
#[cfg(all(feature = "train", feature = "backend-tch"))]
fn report_whitening_similarity(
    embeddings: &[Vec<f32>],
    fens: &[String],
    transform: &oxi::inference::WhiteningTransform,
    num_examples: usize,
) {
    use oxi::whitening::{cosine, percentiles};

    let n = embeddings.len();
    let cap = n.min(800);
    let stride = (n / cap).max(1);
    let idx: Vec<usize> = (0..n).step_by(stride).take(cap).collect();
    let whitened: Vec<Vec<f32>> = idx
        .iter()
        .map(|&i| transform.apply(&embeddings[i]))
        .collect();

    let mut raw_cos: Vec<f32> = Vec::new();
    let mut white_cos: Vec<f32> = Vec::new();
    // (whitened_sim, raw_sim, fen_index_a, fen_index_b)
    let mut pairs: Vec<(f32, f32, usize, usize)> = Vec::new();
    for a in 0..idx.len() {
        for b in (a + 1)..idx.len() {
            let raw = cosine(&embeddings[idx[a]], &embeddings[idx[b]]);
            let white = cosine(&whitened[a], &whitened[b]);
            raw_cos.push(raw);
            white_cos.push(white);
            pairs.push((white, raw, idx[a], idx[b]));
        }
    }

    let raw_p = percentiles(&mut raw_cos);
    let white_p = percentiles(&mut white_cos);
    println!(
        "\nPairwise cosine similarity ({} sampled pairs):",
        pairs.len()
    );
    println!("            min     p5     p25    p50    p75    p95    max");
    println!(
        "  raw      {:+.3} {:+.3} {:+.3} {:+.3} {:+.3} {:+.3} {:+.3}",
        raw_p[0], raw_p[1], raw_p[2], raw_p[3], raw_p[4], raw_p[5], raw_p[6]
    );
    println!(
        "  whitened {:+.3} {:+.3} {:+.3} {:+.3} {:+.3} {:+.3} {:+.3}",
        white_p[0], white_p[1], white_p[2], white_p[3], white_p[4], white_p[5], white_p[6]
    );

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nClosest pairs (whitened):");
    let mut shown = 0;
    for &(white, raw, a, b) in pairs.iter() {
        if fens[a] == fens[b] {
            continue;
        }
        println!("  whitened={:+.3} raw={:+.3}", white, raw);
        println!("    {}", lichess_analysis_link(&fens[a]));
        println!("    {}", lichess_analysis_link(&fens[b]));
        shown += 1;
        if shown >= num_examples {
            break;
        }
    }
    println!("\nFarthest pairs (whitened):");
    for &(white, raw, a, b) in pairs.iter().rev().take(num_examples) {
        println!("  whitened={:+.3} raw={:+.3}", white, raw);
        println!("    {}", lichess_analysis_link(&fens[a]));
        println!("    {}", lichess_analysis_link(&fens[b]));
    }
}

/// Lichess analysis-board link for a FEN (same scheme as the server's
/// `gen_lichess_link`; FENs only need their spaces encoded).
#[cfg(all(feature = "train", feature = "backend-tch"))]
fn lichess_analysis_link(fen: &str) -> String {
    format!("https://lichess.org/analysis/{}", fen.replace(' ', "%20"))
}

#[cfg(all(feature = "train", feature = "backend-tch"))]
fn compute_whitening_after_training(config: &Config) -> Result<()> {
    if !config.whiten_after_training() {
        println!("Skipping post-training whitening (--whiten-after-training=false)");
        return Ok(());
    }
    let Some(log_dir) = config.log_dir.as_ref() else {
        println!("Skipping post-training whitening because --log-dir is not set");
        return Ok(());
    };
    let Some(data_path) = config.data_path.clone() else {
        println!("Skipping post-training whitening because --data-path is not set");
        return Ok(());
    };
    let model_dir = log_dir.join("model");
    let cli = ComputeWhiteningCli {
        model_dir,
        data_path,
        num_positions: config.whitening_positions,
        output: None,
        batch_size: config.whitening_batch_size,
        model_elo: 1500,
        examples: 5,
    };
    println!(
        "Computing post-training whitening transform ({} positions)...",
        cli.num_positions
    );
    compute_whitening_command(&cli)
}

#[cfg(not(all(feature = "train", feature = "backend-tch")))]
fn compute_whitening_after_training(config: &Config) -> Result<()> {
    if config.whiten_after_training() {
        println!("Skipping post-training whitening; this build lacks backend-tch");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    install_panic_hook();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(overrides) => {
            #[cfg(feature = "train")]
            {
                let config = Config::with_overrides(overrides);
                if let Some(ref log_dir) = config.log_dir {
                    if log_dir.exists() {
                        // A checkpoint lives at log_dir/model/model.mpk.
                        let checkpoint = log_dir.join("model").join("model.mpk");
                        if config.resume.unwrap_or(false) {
                            // Resuming: NEVER clear — the checkpoint we are about
                            // to load lives in here. (This unconditional clear
                            // previously wiped the very run being resumed.)
                            tracing::info!(
                                "Resuming; preserving existing log directory: {}",
                                log_dir.display()
                            );
                        } else if checkpoint.exists() {
                            // Fresh run, but this dir already holds a real
                            // checkpoint. Refuse to destroy it silently.
                            anyhow::bail!(
                                "Refusing to clear log directory {} — it contains a \
                                 checkpoint ({}). Pass --resume to continue that run, \
                                 or choose a fresh --log-dir for a new one.",
                                log_dir.display(),
                                checkpoint.display(),
                            );
                        } else {
                            tracing::info!("Clearing log directory: {}", log_dir.display());
                            if let Err(e) = std::fs::remove_dir_all(log_dir) {
                                eprintln!("Warning: failed to clear log directory: {}", e);
                            }
                        }
                    }
                }
                let _guard = oxi::custom_training::init_train_logging(config.log_dir.as_deref());
                tracing::info!("Starting training with config: {:?}", config);
                if let Some(max_ply) = config.max_ply {
                    tracing::info!(
                        "Training data ply filter: max_ply={} (samples with ply > {} will be skipped)",
                        max_ply,
                        max_ply
                    );
                }
                set_global_config(config.clone()).unwrap();

                use burn::backend::Autodiff;

                #[cfg(target_os = "macos")]
                {
                    type Backend = Autodiff<LibTorch<f32>>;
                    let devices: Vec<burn::tensor::Device<Backend>> = (0..config.num_devices)
                        .map(|_| LibTorchDevice::Mps)
                        .collect();
                    train_custom::<Backend>(config.clone(), devices)?;
                }

                #[cfg(all(target_os = "linux", feature = "backend-cuda"))]
                {
                    type Backend = Autodiff<Cuda>;
                    let devices: Vec<burn::tensor::Device<Backend>> = (0..config.num_devices)
                        .map(|i| CudaDevice::new(i))
                        .collect();
                    println!("Using burn-cuda backend with fusion + autotune");
                    train_custom::<Backend>(config.clone(), devices)?;
                }

                #[cfg(all(target_os = "linux", feature = "backend-candle"))]
                {
                    type Backend = Autodiff<Candle<f32, i64>>;
                    let devices: Vec<burn::tensor::Device<Backend>> =
                        vec![CandleDevice::Cpu; config.num_devices];
                    println!(
                        "Using burn-candle backend (CPU for now - CUDA device construction TBD)"
                    );
                    train_custom::<Backend>(config.clone(), devices)?;
                }

                #[cfg(all(target_os = "linux", feature = "backend-tch"))]
                {
                    type Backend = Autodiff<LibTorch<f32>>;
                    let devices: Vec<burn::tensor::Device<Backend>> =
                        (0..config.num_devices).map(LibTorchDevice::Cuda).collect();
                    println!("Using burn-tch backend (LibTorch) - WARNING: No fusion support");
                    train_custom::<Backend>(config.clone(), devices)?;
                }

                compute_whitening_after_training(&config)?;
                Ok(())
            }

            #[cfg(not(feature = "train"))]
            {
                let _ = overrides;
                anyhow::bail!("Training requires the 'train' feature. Run with: cargo run --features \"train,backend-tch\"")
            }
        }

        Commands::Inference(config) => {
            tracing::info!("Running inference with config: {:?}", config);

            let mut positions = config.fen.clone();

            if let Some(fen_file) = &config.fen_file {
                let file = File::open(fen_file)?;
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    positions.push(line?);
                }
            }

            if positions.is_empty() {
                anyhow::bail!("No positions provided. Use --fen or --fen-file");
            }

            #[cfg(feature = "backend-ndarray")]
            {
                use burn_ndarray::{NdArray, NdArrayDevice};

                let device = NdArrayDevice::Cpu;
                let model_config = Config::default();
                let _engine = InferenceEngine::<NdArray<f32>>::from_checkpoint(
                    &config.model_path,
                    model_config,
                    device,
                )?;

                // TODO: Implement inference on positions using engine
                Ok(())
            }

            #[cfg(not(feature = "backend-ndarray"))]
            anyhow::bail!("Inference requires backend-ndarray feature. Run with: cargo run --features backend-ndarray")
        }

        Commands::Download { model } => {
            tracing::info!("Downloading model: {}", model);
            // TODO: Implement model download from remote storage
            println!("Model download not yet implemented");
            println!("Available models: blitz, rapid, classical");
            Ok(())
        }

        Commands::ProcessPgn(config) => {
            tracing::info!("Processing PGN files with config: {:?}", config);

            // Set global config for PGN processing
            let global_config = Config {
                ..Config::default()
            };
            let _ = set_global_config(global_config);

            // TODO: Implement PGN processing with proper Visitor trait
            println!("PGN processing not yet implemented - see PATCHES_AND_TODOS.md");
            println!("This requires implementing the pgn_reader::Visitor trait");
            Ok(())
        }

        Commands::Evaluate(config) => {
            tracing::info!("Evaluating model with config: {:?}", config);

            // Load eval dataset
            println!(
                "Loading eval dataset from {}...",
                config.data_path.display()
            );
            let eval_dataset = EvalDataset::load(&config.data_path)?;
            println!(
                "Loaded {} positions (depth {}, Elo range {}-{})",
                eval_dataset.metadata.num_positions,
                eval_dataset.metadata.stockfish_depth,
                eval_dataset.metadata.elo_range.0,
                eval_dataset.metadata.elo_range.1,
            );

            // Load model and run inference using NdArray backend (CPU, same as production bot)
            #[cfg(feature = "backend-ndarray")]
            {
                use burn_ndarray::{NdArray, NdArrayDevice};

                let device = NdArrayDevice::Cpu;
                let model_config = load_config_from_model_dir(&config.model_dir)?;
                let _ = set_global_config(model_config.clone());
                println!(
                    "Model: embed_dim={}, num_layers={}, num_heads={}",
                    model_config.embed_dim, model_config.num_layers, model_config.num_heads
                );
                let model_mpk = config.model_dir.join("model");
                let engine = InferenceEngine::<NdArray<f32>>::from_checkpoint(
                    &model_mpk,
                    model_config,
                    device,
                )?;

                println!(
                    "Model loaded. Running inference on {} positions...",
                    eval_dataset.positions.len()
                );

                let mut model_policies = std::collections::HashMap::new();
                let total = eval_dataset.positions.len();

                for (i, pos) in eval_dataset.positions.iter().enumerate() {
                    if (i + 1) % 100 == 0 || i + 1 == total {
                        println!("  [{}/{}] positions", i + 1, total);
                    }

                    let parsed_fen: shakmaty::fen::Fen = match pos.fen.parse() {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("Skipping invalid FEN {}: {}", pos.fen, e);
                            continue;
                        }
                    };
                    let chess_pos: shakmaty::Chess =
                        match parsed_fen.into_position(shakmaty::CastlingMode::Standard) {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("Skipping invalid position {}: {}", pos.fen, e);
                                continue;
                            }
                        };

                    // Use default global features for eval (no time pressure)
                    let elo = config.model_elo.unwrap_or(pos.player_elo);
                    let globals = GlobalFeatures {
                        time_remaining_self: 300,
                        time_remaining_oppo: 300,
                        base_time: 300,
                        increment: 3,
                        move_count: pos.ply as usize,
                        elo_self: elo,
                        elo_oppo: elo,
                        is_puzzle: false,
                        is_in_check: chess_pos.is_check(),
                        total_pieces: chess_pos.board().occupied().count() as u32,
                    };

                    match engine.predict_full_policy(&[chess_pos], &globals, 1.0) {
                        Ok(policy) => {
                            model_policies.insert(pos.fen.clone(), policy);
                        }
                        Err(e) => {
                            eprintln!("Inference failed for {}: {}", pos.fen, e);
                        }
                    }
                }

                println!("\nComputing metrics...\n");
                let results = eval_dataset.evaluate_model(&model_policies);

                // Print report
                EvalDataset::print_report(&results);

                // Save results to JSON
                let results_path = config.data_path.with_extension("results.json");
                let results_json = serde_json::to_string_pretty(&results)?;
                std::fs::write(&results_path, results_json)?;
                println!("Results saved to {}", results_path.display());
                Ok(())
            }

            #[cfg(not(feature = "backend-ndarray"))]
            anyhow::bail!("Evaluation requires backend-ndarray feature. Run with: cargo run --features backend-ndarray")
        }

        Commands::CreateEvalSet(config) => {
            println!("Creating evaluation dataset...");
            println!("  PGN directory: {}", config.pgn_dir.display());
            println!("  Output: {}", config.output.display());
            println!("  Stockfish depth: {}", config.depth);
            println!("  Target positions: {}", config.num_positions);

            // Default Elo buckets
            let elo_buckets = vec![
                (1000, 1200),
                (1200, 1400),
                (1400, 1600),
                (1600, 1800),
                (1800, 2000),
                (2000, 2200),
                (2200, 2400),
                (2400, 2700),
            ];

            // Need global config for PGN processing
            let mut global_config = Config::default();
            global_config.enable_ply_sampling = Some(false);
            global_config.enable_elo_sampling = Some(false);
            set_global_config(global_config).expect(
                "global config should not be initialized before create-calibration-db runs",
            );

            // Step 1: Sample positions from PGN files
            let sampled =
                sample_positions_from_pgn(&config.pgn_dir, config.num_positions, &elo_buckets)?;

            if sampled.is_empty() {
                anyhow::bail!("No positions were sampled. Check PGN directory and filters.");
            }

            // Step 2: Evaluate with Stockfish and build dataset
            let dataset = EvalDataset::from_sampled_positions(
                sampled,
                config.stockfish_path.as_deref(),
                config.depth,
                config.threads,
            )?;

            // Step 3: Save
            dataset.save(&config.output)?;
            println!(
                "\nEval dataset saved to {} ({} positions)",
                config.output.display(),
                dataset.positions.len()
            );

            // Print a quick summary of the human ECL baseline
            println!("\nHuman ECL baseline by Elo (from the sampled games):");
            let mut by_elo: std::collections::BTreeMap<i32, Vec<f32>> =
                std::collections::BTreeMap::new();
            for pos in &dataset.positions {
                let bucket = (pos.player_elo / 200) * 200;
                if !pos.human_ecl.is_nan() {
                    by_elo.entry(bucket).or_default().push(pos.human_ecl);
                }
            }
            println!(
                "{:<12} {:>10} {:>10} {:>8}",
                "Elo", "Mean ECL", "Median ECL", "Count"
            );
            println!("{}", "-".repeat(44));
            for (elo, mut ecls) in by_elo {
                ecls.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mean = ecls.iter().sum::<f32>() / ecls.len() as f32;
                let median = ecls[ecls.len() / 2];
                println!(
                    "{:<12} {:>10.1} {:>10.1} {:>8}",
                    format!("{}-{}", elo, elo + 200),
                    mean,
                    median,
                    ecls.len()
                );
            }

            Ok(())
        }

        Commands::QuickEval(config) => {
            println!(
                "Quick evaluation: {} positions, depth {}",
                config.num_positions, config.depth
            );

            // Load model config from params.json
            let model_config = load_config_from_model_dir(&config.model_dir)?;
            let _ = set_global_config(model_config.clone());
            println!(
                "Model: embed_dim={}, num_layers={}, num_heads={}",
                model_config.embed_dim, model_config.num_layers, model_config.num_heads
            );

            // Determine PGN source - file or directory
            let pgn_path = &config.pgn;
            let pgn_dir = if pgn_path.is_dir() {
                pgn_path.clone()
            } else {
                // If it's a single file, use its parent directory
                // but we need to handle this case - copy to a temp dir or handle directly
                pgn_path
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_path_buf()
            };

            // Step 1: Sample positions
            let elo_buckets = vec![(1000, 1400), (1400, 1800), (1800, 2200), (2200, 2700)];
            println!(
                "\nStep 1: Sampling {} positions from PGN...",
                config.num_positions
            );
            let sampled = sample_positions_from_pgn(&pgn_dir, config.num_positions, &elo_buckets)?;

            if sampled.is_empty() {
                anyhow::bail!("No positions sampled. Check PGN path and filters.");
            }
            println!("Sampled {} positions", sampled.len());

            // Step 2: Run Stockfish
            println!("\nStep 2: Running Stockfish at depth {}...", config.depth);
            let dataset = EvalDataset::from_sampled_positions(
                sampled,
                config.stockfish_path.as_deref(),
                config.depth,
                1, // single thread for simplicity
            )?;
            println!("Evaluated {} positions", dataset.positions.len());

            // Print human baseline
            println!("\nHuman ECL baseline:");
            let mut elo_ecls: std::collections::BTreeMap<i32, Vec<f32>> =
                std::collections::BTreeMap::new();
            for pos in &dataset.positions {
                let bucket = (pos.player_elo / 400) * 400;
                if !pos.human_ecl.is_nan() {
                    elo_ecls.entry(bucket).or_default().push(pos.human_ecl);
                }
            }
            println!("{:<12} {:>10} {:>8}", "Elo", "Mean ECL", "Count");
            println!("{}", "-".repeat(32));
            for (elo, ecls) in &elo_ecls {
                let mean = ecls.iter().sum::<f32>() / ecls.len() as f32;
                println!(
                    "{:<12} {:>10.1} {:>8}",
                    format!("{}-{}", elo, elo + 400),
                    mean,
                    ecls.len()
                );
            }

            // Step 3: Run model inference using NdArray backend (CPU, same as production bot)
            println!("\nStep 3: Running model inference...");

            #[cfg(feature = "backend-ndarray")]
            {
                use burn_ndarray::{NdArray, NdArrayDevice};

                let device = NdArrayDevice::Cpu;
                let model_mpk = config.model_dir.join("model");
                let engine = InferenceEngine::<NdArray<f32>>::from_checkpoint(
                    &model_mpk,
                    model_config,
                    device,
                )?;

                let mut model_policies = std::collections::HashMap::new();
                let total = dataset.positions.len();

                for (i, pos) in dataset.positions.iter().enumerate() {
                    if (i + 1) % 50 == 0 || i + 1 == total {
                        println!("  [{}/{}]", i + 1, total);
                    }

                    let parsed_fen: shakmaty::fen::Fen = match pos.fen.parse() {
                        Ok(f) => f,
                        Err(_) => continue,
                    };
                    let chess_pos: shakmaty::Chess =
                        match parsed_fen.into_position(shakmaty::CastlingMode::Standard) {
                            Ok(p) => p,
                            Err(_) => continue,
                        };

                    let elo = config.model_elo.unwrap_or(pos.player_elo);
                    let globals = GlobalFeatures {
                        time_remaining_self: 300,
                        time_remaining_oppo: 300,
                        base_time: 300,
                        increment: 3,
                        move_count: pos.ply as usize,
                        elo_self: elo,
                        elo_oppo: elo,
                        is_puzzle: false,
                        is_in_check: chess_pos.is_check(),
                        total_pieces: chess_pos.board().occupied().count() as u32,
                    };

                    match engine.predict_full_policy(&[chess_pos], &globals, 1.0) {
                        Ok(policy) => {
                            model_policies.insert(pos.fen.clone(), policy);
                        }
                        Err(e) => {
                            eprintln!("Inference failed for {}: {}", pos.fen, e);
                        }
                    }
                }

                println!("\nStep 4: Computing metrics...");
                let results = dataset.evaluate_model(&model_policies);
                EvalDataset::print_report(&results);
                Ok(())
            }

            #[cfg(not(feature = "backend-ndarray"))]
            anyhow::bail!("Quick eval requires backend-ndarray feature. Run with: cargo run --features backend-ndarray")
        }

        Commands::ComputeWhitening(config) => compute_whitening_command(&config),
        Commands::EvaluateAllie(config) => evaluate_allie_command(&config),
        Commands::FeatureAblation(config) => feature_ablation_command(&config),
        Commands::CreateCalibrationDb(config) => {
            println!("Creating calibration DB...");
            println!("  PGN path: {}", config.pgn_dir.display());
            println!("  Output: {}", config.output.display());
            println!("  Stockfish depth: {}", config.depth);
            println!("  Target positions: {}", config.num_positions);

            if let Some(parent) = config.output.parent() {
                std::fs::create_dir_all(parent)?;
            }

            #[cfg(feature = "train")]
            {
                let global_config = calibration_stream_config();
                let _ = set_global_config(global_config);
            }

            let elo_buckets = vec![
                (1000, 1200),
                (1200, 1400),
                (1400, 1600),
                (1600, 1800),
                (1800, 2000),
                (2000, 2200),
                (2200, 2400),
                (2400, 2700),
            ];

            #[cfg(feature = "train")]
            let sampled = sample_training_positions_for_calibration(
                &config.pgn_dir,
                config.num_positions,
                &elo_buckets,
                config.preserve_order.unwrap_or(false),
            )?;

            #[cfg(not(feature = "train"))]
            let sampled: Vec<oxi::eval_dataset::SampledPosition> = {
                anyhow::bail!(
                    "create-calibration-db now uses the training PGN pipeline and requires the `train` feature"
                );
            };

            if sampled.is_empty() {
                anyhow::bail!("No positions were sampled. Check PGN directory and filters.");
            }

            for (idx, sample) in sampled.iter().take(10).enumerate() {
                println!(
                    "  sample[{idx}] fen={} move={} elo_self={} elo_oppo={} ply={}",
                    sample.fen,
                    sample.human_move,
                    sample.player_elo,
                    sample.opponent_elo,
                    sample.ply
                );
            }

            let db = CalibrationDb::open(&config.output)?;
            let mut existing_keys: std::collections::HashSet<oxi::calibration::CalibrationKey> =
                db.load_existing_keys(config.depth)?;
            let existing_at_start = existing_keys.len();
            let mut engine = StockfishEngine::new(
                config.stockfish_path.as_deref(),
                config.depth,
                config.threads,
            )?;

            let total = sampled.len();
            let mut inserted = 0usize;
            let mut skipped = 0usize;
            let mut skipped_existing = 0usize;
            let mut regret_counts = [0usize; RegretBin::COUNT];

            println!(
                "  Found {} already-labeled positions at depth {}",
                existing_at_start, config.depth
            );

            for (idx, sampled_position) in sampled.into_iter().enumerate() {
                if (idx + 1) % 50 == 0 || idx + 1 == total {
                    println!(
                        "  [{}/{}] processed | ready={} (existing {} + inserted {}) | skipped_existing={} skipped_failed={}",
                        idx + 1,
                        total,
                        existing_at_start + inserted,
                        existing_at_start,
                        inserted,
                        skipped_existing,
                        skipped
                    );
                }

                let key = oxi::calibration::calibration_key_for_sample(
                    &sampled_position.fen,
                    &sampled_position.human_move,
                );
                if existing_keys.contains(&key) {
                    skipped_existing += 1;
                    continue;
                }

                match label_sampled_position(&mut engine, sampled_position, config.depth) {
                    Ok(labeled) => {
                        regret_counts[labeled.regret_bin.index() as usize] += 1;
                        db.insert_labeled_position(&labeled, config.depth)?;
                        existing_keys.insert(key);
                        inserted += 1;
                    }
                    Err(err) => {
                        eprintln!("Warning: failed to label position: {err}");
                        skipped += 1;
                    }
                }
            }

            let total_in_db = db.count_positions()?;
            println!("\nCalibration DB written to {}", config.output.display());
            println!("  Inserted this run: {}", inserted);
            println!("  Skipped existing: {}", skipped_existing);
            println!("  Skipped this run: {}", skipped);
            println!("  Total rows in DB: {}", total_in_db);
            println!("\nRegret-bin histogram:");
            for bin in [
                RegretBin::ExactZero,
                RegretBin::Cp1To10,
                RegretBin::Cp11To25,
                RegretBin::Cp26To50,
                RegretBin::Cp51To100,
                RegretBin::Cp101To200,
                RegretBin::Cp201To400,
                RegretBin::Cp400Plus,
            ] {
                println!(
                    "  {:>7}: {:>6}",
                    bin.label(),
                    regret_counts[bin.index() as usize]
                );
            }

            Ok(())
        }

        Commands::DownloadPgn {
            year,
            month,
            output_dir,
            local,
        } => {
            let output_dir = output_dir.unwrap_or_else(|| PathBuf::from("./data/pgn"));
            tracing::info!("Downloading PGN files for {}-{:02}", year, month);

            // Create output directory if it doesn't exist
            std::fs::create_dir_all(&output_dir)?;

            // Format the URL according to Lichess database naming
            let filename = format!("lichess_db_standard_rated_{year}-{month:02}.pgn.zst");
            let url = format!("https://database.lichess.org/standard/{filename}");
            let output_path = output_dir.join(&filename);

            // Check if file already exists
            if output_path.exists() {
                println!("File {filename} already exists, skipping download");
                return Ok(());
            }

            println!("Downloading {filename} from {url}");

            // Use reqwest to download the file
            let client = reqwest::Client::new();
            let response = client.get(&url).send().await?;

            if !response.status().is_success() {
                anyhow::bail!("Failed to download file: HTTP {}", response.status());
            }

            // Get the content length for progress tracking
            let total_size = response.content_length().unwrap_or(0);
            println!("File size: {} MB", total_size / 1_048_576);

            // Stream the download to file
            use futures_util::StreamExt;
            use tokio::io::AsyncWriteExt;

            let mut file = tokio::fs::File::create(&output_path).await?;
            let mut downloaded = 0u64;
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                // Print progress every 10MB
                if downloaded % (10 * 1_048_576) == 0 {
                    let progress = (downloaded as f64 / total_size as f64 * 100.0).min(100.0);
                    println!(
                        "Progress: {:.1}% ({} / {} MB)",
                        progress,
                        downloaded / 1_048_576,
                        total_size / 1_048_576
                    );
                }
            }

            println!("Download complete: {}", output_path.display());

            if local {
                println!("Local mode: Downloaded single file for testing");
            }

            Ok(())
        }

        Commands::DownloadAll { output_dir } => {
            let output_dir = output_dir.unwrap_or_else(|| PathBuf::from("/lambda/nfs/chessbook"));
            tracing::info!("Downloading all Lichess PGN files since 2022");
            download_all_lichess_files(&output_dir).await?;
            Ok(())
        }

        Commands::DownloadTcec { data_path } => {
            let data_path = data_path.unwrap_or_else(|| PathBuf::from("/lambda/nfs/chessbook"));
            let tcec_dir = data_path.join("tcec");
            tracing::info!("Downloading TCEC games to {:?}", tcec_dir);
            download_tcec_games(&tcec_dir).await?;
            Ok(())
        }

        Commands::DownloadPuzzles { data_path } => {
            let data_path = data_path.unwrap_or_else(|| PathBuf::from("/lambda/nfs/chessbook"));
            let puzzles_dir = data_path.join("puzzles");
            tracing::info!("Downloading Lichess puzzles to {:?}", puzzles_dir);
            download_puzzles(&puzzles_dir).await?;
            Ok(())
        }
    }
}

async fn download_all_lichess_files(output_dir: &PathBuf) -> Result<()> {
    let download_list_url = "https://database.lichess.org/standard/list.txt";
    let client = reqwest::Client::new();
    let response = client.get(download_list_url).send().await?;
    let body = response.text().await?;
    let files: Vec<String> = body
        .lines()
        .filter(|url| {
            // Filter for files from 2022-2025
            url.contains("2022")
                || url.contains("2023")
                || url.contains("2024")
                || url.contains("2025")
        })
        .map(|x| x.to_string())
        .collect();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    println!("Found {} files to download since 2022", files.len());

    let semaphore = Arc::new(Semaphore::new(4)); // Limit concurrent downloads
    let tasks = files.into_iter().map(|url| {
        let output_dir = output_dir.clone();
        let semaphore = Arc::clone(&semaphore);
        task::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap(); // Acquire permit
            let file_name = url.split('/').next_back().unwrap(); // Extract the file name from the URL
            let path = output_dir.join(file_name);

            if path.exists() {
                println!("Already downloaded {file_name}");
                return Ok::<(), anyhow::Error>(());
            }

            let client = reqwest::Client::new();
            let response = client.get(&url).send().await?;

            if !response.status().is_success() {
                anyhow::bail!(
                    "Failed to download {}: HTTP {}",
                    file_name,
                    response.status()
                );
            }

            let total_size = response.content_length().unwrap_or(0);
            println!(
                "Downloading {file_name} ({:.1} GB)",
                total_size as f64 / 1_073_741_824.0
            );

            let mut file = tokio::fs::File::create(&path).await?;
            let mut stream = response.bytes_stream();
            let mut downloaded: u64 = 0;
            let mut last_report: u64 = 0;

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                // Report progress every 100MB
                if downloaded - last_report >= 100 * 1_048_576 {
                    last_report = downloaded;
                    if total_size > 0 {
                        let progress = (downloaded as f64 / total_size as f64 * 100.0).min(100.0);
                        println!(
                            "  {file_name}: {:.1}% ({:.0} MB / {:.0} MB)",
                            progress,
                            downloaded as f64 / 1_048_576.0,
                            total_size as f64 / 1_048_576.0
                        );
                    } else {
                        println!(
                            "  {file_name}: {:.0} MB downloaded",
                            downloaded as f64 / 1_048_576.0
                        );
                    }
                }
            }

            file.flush().await?;
            println!("Completed {file_name}");
            Ok(())
        })
    });

    // Collect all tasks and wait for completion
    let tasks: Vec<_> = tasks.collect();
    let results = futures_util::future::join_all(tasks).await;

    for result in results {
        if let Err(e) = result {
            eprintln!("Task error: {e:?}");
        } else if let Err(e) = result.unwrap() {
            eprintln!("Download error: {e:?}");
        }
    }

    println!("All downloads completed!");
    Ok(())
}

async fn download_tcec_games(output_dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let zip_filename = "TCEC-everything-compact.zip";
    let zip_path = output_dir.join(zip_filename);

    if zip_path.exists() {
        println!("TCEC archive already downloaded: {}", zip_path.display());
    } else {
        println!("Downloading TCEC games from {}", TCEC_DOWNLOAD_URL);

        let client = reqwest::Client::new();
        let response = client.get(TCEC_DOWNLOAD_URL).send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download TCEC games: HTTP {}", response.status());
        }

        let total_size = response.content_length().unwrap_or(0);
        println!("File size: {} MB", total_size / 1_048_576);

        let mut file = tokio::fs::File::create(&zip_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if total_size > 0 && downloaded % (10 * 1_048_576) == 0 {
                let progress = (downloaded as f64 / total_size as f64 * 100.0).min(100.0);
                println!(
                    "Progress: {:.1}% ({} / {} MB)",
                    progress,
                    downloaded / 1_048_576,
                    total_size / 1_048_576
                );
            }
        }

        println!("Download complete: {}", zip_path.display());
    }

    let pgn_dir = output_dir.to_path_buf();
    let pgn_files: Vec<_> = std::fs::read_dir(&pgn_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "pgn")
                .unwrap_or(false)
        })
        .collect();

    if !pgn_files.is_empty() {
        println!(
            "Found {} PGN files already extracted, skipping extraction",
            pgn_files.len()
        );
    } else {
        println!("Extracting PGN files from archive...");

        let zip_file = std::fs::File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(zip_file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => output_dir.join(path),
                None => continue,
            };

            if file.name().ends_with('/') {
                std::fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        std::fs::create_dir_all(p)?;
                    }
                }
                let mut outfile = std::fs::File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
                println!("Extracted: {}", outpath.display());
            }
        }

        println!("Extraction complete!");
    }

    println!("\nTCEC games ready at: {}", output_dir.display());
    println!("Use --pretrain-samples N with train command to pretrain on these games");
    Ok(())
}

async fn download_puzzles(output_dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let zst_filename = "lichess_db_puzzle.csv.zst";
    let zst_path = output_dir.join(zst_filename);

    if zst_path.exists() {
        println!("Puzzle database already downloaded: {}", zst_path.display());
    } else {
        println!("Downloading Lichess puzzles from {}", LICHESS_PUZZLE_URL);

        let client = reqwest::Client::new();
        let response = client.get(LICHESS_PUZZLE_URL).send().await?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download puzzle database: HTTP {}",
                response.status()
            );
        }

        let total_size = response.content_length().unwrap_or(0);
        println!("File size: {} MB", total_size / 1_048_576);

        let mut file = tokio::fs::File::create(&zst_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if total_size > 0 && downloaded % (10 * 1_048_576) == 0 {
                let progress = (downloaded as f64 / total_size as f64 * 100.0).min(100.0);
                println!(
                    "Progress: {:.1}% ({} / {} MB)",
                    progress,
                    downloaded / 1_048_576,
                    total_size / 1_048_576
                );
            }
        }

        println!("Download complete: {}", zst_path.display());
    }

    println!("\nPuzzle database ready at: {}", zst_path.display());
    println!(
        "File is kept compressed (.csv.zst) - puzzle_processor handles decompression on the fly"
    );
    Ok(())
}

fn install_panic_hook() {
    let original_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen, cursor::Show);

        let panic_message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic payload".to_string()
        };

        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "unknown location".to_string());

        let log_message = format!("PANIC at {}: {}", location, panic_message);

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("train.log")
        {
            let _ = writeln!(file, "{}", log_message);
            let _ = file.flush();
        }

        original_hook(panic_info);
    }));
}
