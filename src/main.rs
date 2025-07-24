use anyhow::Result;
#[cfg(target_os = "linux")]
use burn::backend::cuda::CudaDevice;
#[cfg(target_os = "linux")]
use burn::backend::Cuda;
#[cfg(target_os = "macos")]
use burn::backend::Metal;

use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::{io::AsyncWriteExt, task};

use oxi::config::{set_global_config, Config};
use oxi::inference::InferenceEngine;
use oxi::training::train;

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
    Train(Config),

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
        #[arg(long, default_value = "./data/pgn")]
        output_dir: PathBuf,

        /// Download only one file for testing
        #[arg(long)]
        local: bool,
    },

    /// Download all Lichess PGN files since 2022
    DownloadAll {
        /// Output directory for downloaded files
        #[arg(long, default_value = "./data/pgn")]
        output_dir: PathBuf,
    },

    /// Process PGN files into training data
    ProcessPgn(ProcessPgnConfig),

    /// Evaluate model performance
    Evaluate(EvaluateConfig),
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
    #[arg(long, default_value = "1500")]
    white_elo: i32,

    /// ELO rating for black player
    #[arg(long, default_value = "1500")]
    black_elo: i32,

    /// Temperature for move sampling
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Number of top moves to show
    #[arg(long, default_value = "5")]
    top_k: usize,
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
    #[arg(long, default_value = "4")]
    num_threads: usize,

    /// Chunk size for processing
    #[arg(long, default_value = "10000")]
    chunk_size: usize,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
struct EvaluateConfig {
    /// Path to model checkpoint
    #[arg(long)]
    model_path: PathBuf,

    /// Path to evaluation dataset
    #[arg(long)]
    data_path: PathBuf,

    /// Batch size for evaluation
    #[arg(long, default_value = "256")]
    batch_size: usize,

    /// Device to use (cpu, cuda)
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with environment variable support

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(config) => {
            tracing::info!("Starting training with config: {:?}", config);
            set_global_config(config.clone()).unwrap();
            if config.disable_tui {
                tracing_subscriber::fmt().init();
            }

            // Validate that data_path is provided
            let _data_path = config
                .data_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--data-path is required for training"))?;

            #[cfg(target_os = "linux")]
            type Backend = Autodiff<Cuda>;
            #[cfg(target_os = "linux")]
            let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> =
                (0..config.num_devices).map(CudaDevice::new).collect();

            // #[cfg(target_os = "macos")]
            // type Backend = Autodiff<Metal>;
            // #[cfg(target_os = "macos")]
            // let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> =
            //     vec![<Backend as burn::tensor::backend::Backend>::Device::default()];
            // #[cfg(target_os = "macos")]
            // type Backend = Autodiff<Metal<f32, i32>>;
            // #[cfg(target_os = "macos")]
            // let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> =
            //     vec![<Backend as burn::tensor::backend::Backend>::Device::default()];
            #[cfg(target_os = "macos")]
            type Backend = Autodiff<burn_ndarray::NdArray<f32, i32>>;
            #[cfg(target_os = "macos")]
            let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> = vec![
                <Backend as burn::tensor::backend::Backend>::Device::default(),
                <Backend as burn::tensor::backend::Backend>::Device::default(),
            ];

            // Run training with unified config
            use burn::backend::Autodiff;
            train::<Backend>(config.clone(), devices)?;
            Ok(())
        }

        Commands::Inference(config) => {
            tracing::info!("Running inference with config: {:?}", config);

            // Collect FEN positions
            let mut positions = config.fen.clone();

            // Load from file if provided
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

            let device = burn_ndarray::NdArrayDevice::default();

            // Load model and create engine
            // TODO: We need to store model config with checkpoint to know num_blocks/channels
            // For now, use defaults
            let model_config = Config::default();
            let _engine = InferenceEngine::<burn_ndarray::NdArray>::from_checkpoint(
                &config.model_path,
                model_config,
                device,
            )?;

            // Run inference on positions
            // for position in &positions {
            //     // Determine player colors from FEN
            //     let parts: Vec<&str> = position.split(' ').collect();
            //     let (elo_self, elo_oppo) = if parts.len() > 1 && parts[1] == "w" {
            //         (config.white_elo, config.black_elo)
            //     } else {
            //         (config.black_elo, config.white_elo)
            //     };
            //
            //     let predictions = engine.predict(
            //         position,
            //         elo_self,
            //         elo_oppo,
            //         config.temperature,
            //         config.top_k,
            //     )?;
            //
            //     println!("\nPosition: {}", position);
            //     println!("Top {} moves:", config.top_k);
            //     for (i, pred) in predictions.iter().enumerate() {
            //         println!(
            //             "{}. {} ({:.2}%)",
            //             i + 1,
            //             pred.uci_move,
            //             pred.probability * 100.0,
            //         );
            //     }
            // }
            Ok(())
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
            // TODO: Implement full evaluation on test set
            println!("Evaluation not yet fully implemented");
            println!("This will calculate accuracy and loss metrics on the test dataset");
            Ok(())
        }

        Commands::DownloadPgn {
            year,
            month,
            output_dir,
            local,
        } => {
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
            tracing::info!("Downloading all Lichess PGN files since 2022");
            download_all_lichess_files(&output_dir).await?;
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

            println!("Downloading {file_name}");
            let client = reqwest::Client::new();
            let response = client.get(&url).send().await?;

            if !response.status().is_success() {
                anyhow::bail!(
                    "Failed to download {}: HTTP {}",
                    file_name,
                    response.status()
                );
            }

            let mut file = tokio::fs::File::create(&path).await?;
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                file.write_all(&chunk).await?;
            }

            file.flush().await?;
            println!("Downloaded {file_name}");
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
