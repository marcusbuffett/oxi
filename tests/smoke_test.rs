use anyhow::Result;
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use oxi::config::set_global_config;
use oxi::config::{ModelConfig, TrainingConfig};
use oxi::training::train;
use std::path::PathBuf;

#[test]
fn smoke_test_training_single_epoch() -> Result<()> {
    println!("Starting smoke test with a4.pgn...");

    tracing_subscriber::fmt().init();
    // Configuration - smaller model for testing
    let model_config = ModelConfig {
        data_path: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgn")),
        max_samples: Some(1000),
        num_layers: 2,
        batch_size: Some(2),
        disable_tui: true,
        learning_rate: 0.001,
        num_epochs: 1, // Just one epoch
        ..Default::default()
    };

    // Ensure global config is set for components that access it
    let _ = set_global_config(model_config.clone());

    let devices = vec![NdArrayDevice::default()];

    // Call the existing train function
    train::<Autodiff<NdArray<f32, i32, i8>>>(model_config, devices)?;

    println!("\nSmoke test passed - training completed without errors!");
    Ok(())
}
