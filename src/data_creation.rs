use anyhow::Result;
use bincode::encode_into_std_write;
use burn::prelude::*;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use crate::config::{set_global_config, Config, LEGAL_MOVES};
use crate::constants::EASY_POSITIONS_PATH;
use crate::dataset::{ChessBatcher, ChessExample, OXIDataset};
use crate::pgn_processor::process_pgn_directory_iter;

#[derive(Debug, Clone)]
pub struct FilterConfidentConfig {
    pub model_path: PathBuf,
    pub data_path: PathBuf,
    pub physical_batch_size: usize,
    pub confidence_threshold: f32,
    pub max_examples: usize,
}

pub fn filter_confident_positions<B: Backend>(
    config: FilterConfidentConfig,
    device: Device<B>,
) -> Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    println!(
        "Filtering positions with >{}% confidence from {:?}",
        config.confidence_threshold * 100.0,
        config.data_path
    );

    // Set up global config for PGN processing
    let global_config = Config {
        ..Default::default()
    };
    let _ = set_global_config(global_config.clone());

    println!("Loading model from {:?}", config.model_path);

    // Check if model file exists
    if !config.model_path.exists() {
        anyhow::bail!("Model file not found: {:?}", config.model_path);
    }
    println!(
        "Model file exists, size: {} bytes",
        std::fs::metadata(&config.model_path)?.len()
    );

    // Create a new model and load weights
    use burn::record::CompactRecorder;

    println!("Initializing model architecture...");
    let model = crate::model::OXIModel::<B>::new(&device, &global_config);
    println!("Model initialized");

    println!("Loading model weights from checkpoint...");
    let recorder = CompactRecorder::new();
    let model = model
        .load_file(config.model_path.clone(), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    println!("Model loaded successfully!");

    // Create iterator over positions
    println!("Creating iterator over PGN files in {:?}", config.data_path);
    let examples_iter = process_pgn_directory_iter(&config.data_path)?;
    println!("PGN iterator created");

    // Create batcher
    println!("Creating batcher...");
    let batcher = ChessBatcher::<B>::new(device.clone());
    println!("Batcher created");

    // Buffer for collecting examples before writing
    let mut filtered_examples = Vec::with_capacity(100);
    let mut total_processed = 0usize;
    let mut total_kept = 0usize;

    // Create output file
    let output_file = std::fs::File::create(EASY_POSITIONS_PATH)?;
    let mut writer = BufWriter::new(output_file);
    let bincode_config = bincode::config::standard();

    // Process in batches
    let mut batch_buffer = Vec::with_capacity(config.physical_batch_size);

    println!("Starting to process positions...");
    println!(
        "Batch size: {}, Max examples: {}, Threshold: {}",
        config.physical_batch_size, config.max_examples, config.confidence_threshold
    );

    for example in examples_iter {
        batch_buffer.push(example);

        // Process batch when full
        if batch_buffer.len() >= config.physical_batch_size {
            let (kept, processed) = process_batch(
                &batch_buffer,
                &model,
                &global_config,
                &batcher,
                &device,
                config.confidence_threshold,
                &mut filtered_examples,
            )?;
            println!("From this batch: Processed {}, Kept {}", processed, kept);

            total_processed += processed;
            total_kept += kept;

            // Write buffered examples if buffer is full
            if filtered_examples.len() >= 100 {
                for ex in filtered_examples.drain(..) {
                    encode_into_std_write(&ex, &mut writer, bincode_config)?;
                }
                writer.flush()?;
            }

            // Log progress
            if total_processed % 10000 == 0 {
                let keep_rate = if total_processed > 0 {
                    (total_kept as f64 / total_processed as f64) * 100.0
                } else {
                    0.0
                };
                println!(
                    "Processed: {} | Kept: {} ({:.2}%)",
                    total_processed, total_kept, keep_rate
                );
            }

            batch_buffer.clear();

            // Check if we've collected enough
            if total_kept >= config.max_examples {
                println!("Reached target of {} examples", config.max_examples);
                break;
            }
        }
    }

    // Process remaining examples in batch_buffer
    if !batch_buffer.is_empty() {
        let (kept, processed) = process_batch(
            &batch_buffer,
            &model,
            &global_config,
            &batcher,
            &device,
            config.confidence_threshold,
            &mut filtered_examples,
        )?;
        total_processed += processed;
        total_kept += kept;
    }

    // Write any remaining filtered examples
    for ex in filtered_examples.drain(..) {
        encode_into_std_write(&ex, &mut writer, bincode_config)?;
    }

    writer.flush()?;

    let keep_rate = if total_processed > 0 {
        (total_kept as f64 / total_processed as f64) * 100.0
    } else {
        0.0
    };

    println!("\nFiltering complete!");
    println!("Total processed: {}", total_processed);
    println!("Total kept: {} ({:.2}%)", total_kept, keep_rate);
    println!("Saved to: {}", EASY_POSITIONS_PATH);

    Ok(())
}

fn process_batch<B: Backend>(
    examples: &[ChessExample],
    model: &crate::model::OXIModel<B>,
    config: &Config,
    batcher: &ChessBatcher<B>,
    device: &Device<B>,
    confidence_threshold: f32,
    filtered_examples: &mut Vec<ChessExample>,
) -> Result<(usize, usize)>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataloader::Dataset;

    // Create dataset for this batch
    let dataset = OXIDataset::new(examples.to_vec(), config.clone());

    // Process all examples in the batch
    let mut items = Vec::new();
    for i in 0..dataset.len() {
        if let Some(item) = dataset.get(i) {
            items.push(item);
        }
    }

    if items.is_empty() {
        return Ok((0, 0));
    }

    // Create batch
    let batch = batcher.batch(items.clone(), device);

    // Run inference
    let (policy_logits, _value_logits, _side_info_logits, _time_logits) =
        model.forward(batch.board_input, batch.global_features);

    // policy_logits is [batch, 64, LEGAL_MOVES/64], reshape to [batch, LEGAL_MOVES]
    let batch_size = policy_logits.dims()[0];
    let policy_logits_flat = policy_logits.reshape([batch_size, LEGAL_MOVES]);

    // Apply legal move masking (same as inference code)
    let mask = batch.legal_moves.clone().equal_elem(0.0);
    let masked_logits = policy_logits_flat.mask_fill(mask, f32::NEG_INFINITY);

    // Apply log_softmax (consistent with inference)
    let log_probs = burn::tensor::activation::log_softmax(masked_logits, 1);

    // Convert to probabilities
    let policy_probs = log_probs.exp();

    // Convert to data
    let policy_data = policy_probs.to_data();
    let move_dist_data = batch.move_distributions.to_data();

    // Check each example
    let mut kept = 0;
    for (idx, example) in examples.iter().enumerate() {
        // Get the correct move index from move_distribution
        let move_dist_slice =
            &move_dist_data.as_slice::<f32>().unwrap()[idx * LEGAL_MOVES..(idx + 1) * LEGAL_MOVES];

        // Find the correct move (should have probability 1.0 in move_distribution)
        if let Some(correct_move_idx) = move_dist_slice.iter().position(|&p| p > 0.5) {
            // Get predicted probability for that move
            let policy_slice =
                &policy_data.as_slice::<f32>().unwrap()[idx * LEGAL_MOVES..(idx + 1) * LEGAL_MOVES];
            let predicted_prob = policy_slice[correct_move_idx];

            // Keep if confidence exceeds threshold
            // Note: With 4864 moves, even confident predictions have low absolute probabilities
            // A threshold of 0.9 would filter out everything. Typical confident moves: 0.05-0.2
            if predicted_prob >= confidence_threshold {
                filtered_examples.push(example.clone());
                kept += 1;
            }
        }
    }

    Ok((kept, examples.len()))
}
