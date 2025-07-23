use anyhow::Result;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use oxi::config::{ModelConfig, TrainingConfig};
use oxi::training::train;
use std::path::PathBuf;

#[test]
fn smoke_test_training_single_epoch() -> Result<()> {
    println!("Starting smoke test with a4.pgn...");

    // Configuration - smaller model for testing
    let model_config = ModelConfig {
        num_blocks: 2, // Reduced from 15
        channels: 64,  // Reduced from 256
        ..Default::default()
    };

    let training_config = TrainingConfig {
        batch_size: 2,
        learning_rate: 0.001,
        num_epochs: 1, // Just one epoch
        ..Default::default()
    };

    // Use a4.pgn as test data
    let a4_pgn_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgn/a4.pgn");

    println!("Training with data from: {:?}", a4_pgn_path);

    // Call the existing train function
    train::<Autodiff<NdArray>>(
        model_config,
        training_config,
        &a4_pgn_path,
        0.8,  // 80% train, 20% validation split
        None, // No sample limit
    )?;

    println!("\nSmoke test passed - training completed without errors!");
    Ok(())
}

