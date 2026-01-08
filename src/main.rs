use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    cursor, execute,
    terminal::{disable_raw_mode, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::{io::AsyncWriteExt, task};

use oxi::config::{set_global_config, Config};
use oxi::constants::TCEC_DOWNLOAD_URL;
use oxi::custom_training::train_custom;
use oxi::inference::InferenceEngine;

#[cfg(all(target_os = "linux", feature = "backend-cuda"))]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(all(target_os = "linux", feature = "backend-candle"))]
use burn_candle::{Candle, CandleDevice};

#[cfg(all(target_os = "linux", feature = "backend-tch"))]
use burn::backend::LibTorch;
#[cfg(all(target_os = "linux", feature = "backend-tch"))]
use burn_tch::LibTorchDevice;

#[cfg(target_os = "macos")]
use burn::backend::LibTorch;
#[cfg(target_os = "macos")]
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
        #[arg(long, default_value = "/lambda/nfs/chessbook")]
        output_dir: PathBuf,
    },

    /// Process PGN files into training data
    ProcessPgn(ProcessPgnConfig),

    /// Evaluate model performance
    Evaluate(EvaluateConfig),

    /// Download TCEC (Top Chess Engine Championship) games for pretraining
    DownloadTcec {
        /// Data directory (TCEC files will be downloaded to <data-path>/tcec)
        #[arg(long, default_value = "/lambda/nfs/chessbook")]
        data_path: PathBuf,
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
    install_panic_hook();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(config) => {
            let _guard = oxi::custom_training::init_train_logging(config.log_dir.as_deref());
            tracing::info!("Starting training with config: {:?}", config);
            set_global_config(config.clone()).unwrap();

            use burn::backend::Autodiff;

            #[cfg(target_os = "macos")]
            {
                type Backend = Autodiff<LibTorch<f32>>;
                let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> = (0..config
                    .num_devices)
                    .map(|_| LibTorchDevice::Mps)
                    .collect();
                train_custom::<Backend>(config.clone(), devices)?;
            }

            #[cfg(all(target_os = "linux", feature = "backend-cuda"))]
            {
                type Backend = Autodiff<Cuda>;
                let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> = (0..config
                    .num_devices)
                    .map(|i| CudaDevice::new(i))
                    .collect();
                println!("Using burn-cuda backend with fusion + autotune");
                train_custom::<Backend>(config.clone(), devices)?;
            }

            #[cfg(all(target_os = "linux", feature = "backend-candle"))]
            {
                type Backend = Autodiff<Candle<f32, i64>>;
                let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> =
                    vec![CandleDevice::Cpu; config.num_devices];
                println!("Using burn-candle backend (CPU for now - CUDA device construction TBD)");
                train_custom::<Backend>(config.clone(), devices)?;
            }

            #[cfg(all(target_os = "linux", feature = "backend-tch"))]
            {
                type Backend = Autodiff<LibTorch<f32>>;
                let devices: Vec<<Backend as burn::tensor::backend::Backend>::Device> =
                    (0..config.num_devices).map(LibTorchDevice::Cuda).collect();
                println!("Using burn-tch backend (LibTorch) - WARNING: No fusion support");
                train_custom::<Backend>(config.clone(), devices)?;
            }

            Ok(())
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

            #[cfg(any(target_os = "macos", feature = "backend-tch"))]
            {
                #[cfg(target_os = "macos")]
                let device = LibTorchDevice::Mps;
                #[cfg(all(target_os = "linux", feature = "backend-tch"))]
                let device = LibTorchDevice::Cuda(0);

                let model_config = Config::default();
                let _engine = InferenceEngine::<LibTorch<f32>>::from_checkpoint(
                    &config.model_path,
                    model_config,
                    device,
                )?;
            }

            #[cfg(all(target_os = "linux", not(feature = "backend-tch")))]
            {
                anyhow::bail!("Inference currently only supported with backend-tch feature");
            }

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

        Commands::DownloadTcec { data_path } => {
            let tcec_dir = data_path.join("tcec");
            tracing::info!("Downloading TCEC games to {:?}", tcec_dir);
            download_tcec_games(&tcec_dir).await?;
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
