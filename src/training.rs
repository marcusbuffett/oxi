#[cfg(target_os = "linux")]
use burn::backend::cuda::CudaDevice;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::Progress;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::module::AutodiffModule;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
use burn_train::checkpoint::ComposedCheckpointingStrategy;
use burn_train::checkpoint::KeepLastNCheckpoints;
use burn_train::metric::{Metric, MetricMetadata, Numeric};
use burn_train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;

use crate::config::Config;
use crate::config::LEGAL_MOVES;
use crate::dataset::{ChessBatcher, OXIDataset};
use crate::legal_move_probability_metric::LegalMoveProbabilityMetric;
use crate::model::OXIModel;
use crate::move_accuracy_metric::{MoveAccuracyInput, MoveAccuracyMetric, MoveTop1AccuracyMetric};
use crate::policy_loss_metric::PolicyLossMetric;
use crate::side_info_loss_metric::SideInfoLossMetric;
use crate::time_usage_loss_metric::TimeUsageLossMetric;
use crate::uncertainty_metric::UncertaintyMetric;
use crate::value_loss_metric::ValueLossMetric;
use crate::wdl_accuracy_metric::WdlAccuracyMetric;

struct CustomRenderer {}

impl MetricsRenderer for CustomRenderer {
    fn update_train(&mut self, _state: MetricState) {
        // dbg!(_state);
    }

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {
        // dbg!(item);
    }

    fn render_valid(&mut self, _item: TrainingProgress) {
        // dbg!(item);
    }
}

/// Train the Maia 2 model with dataset
pub fn train<B: AutodiffBackend>(config: Config, devices: Vec<B::Device>) -> anyhow::Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    // #[cfg(target_os = "linux")]
    // let devices: Vec<B::Device> = (0..num_devices).map(B::Device::default).collect();
    //
    // #[cfg(target_os = "macos")]
    // let devices: Vec<B::Device> = (0..1).map(|_| B::Device::default()).collect();

    // Create model
    let model: OXIModel<B> = OXIModel::new(&devices[0], &config);
    // dbg!(&model);
    // return Ok(());

    let path = config.data_path.clone().expect("Model path not set");
    let data_path = Path::new(&path);
    // Load dataset with max_samples limit
    let dataset: OXIDataset = if data_path.is_dir() {
        tracing::info!("Loading data from PGN directory: {:?}", data_path);
        OXIDataset::from_pgn_dir_with_limit(data_path, config.clone(), config.max_samples)?
    } else {
        // Handle both .pgn and .pgn.zst files
        tracing::info!("Loading data from PGN file: {:?}", data_path);
        OXIDataset::from_pgn_with_limit(data_path, config.clone(), config.max_samples)?
    };
    tracing::info!("Training with {} samples", dataset.examples.len());

    // Split into train and validation
    tracing::info!("Splitting dataset with train ratio: {}", config.train_ratio);
    let (train_dataset, valid_dataset) = dataset.split(config.train_ratio);

    // Create batchers
    let batcher_train: ChessBatcher<B> = ChessBatcher::new(devices[0].clone());
    let batcher_valid: ChessBatcher<B::InnerBackend> = ChessBatcher::new(devices[0].clone());

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.physical_batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.physical_batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Create learner
    let mut learner_builder = LearnerBuilder::new("checkpoints")
        .metric_train_numeric(MoveAccuracyMetric::new())
        .metric_valid_numeric(MoveAccuracyMetric::new())
        .metric_train_numeric(MoveTop1AccuracyMetric::new())
        .metric_valid_numeric(MoveTop1AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PolicyLossMetric::new())
        .metric_valid_numeric(PolicyLossMetric::new())
        .metric_train_numeric(ValueLossMetric::new())
        .metric_valid_numeric(ValueLossMetric::new())
        .metric_train_numeric(SideInfoLossMetric::new())
        .metric_valid_numeric(SideInfoLossMetric::new())
        .metric_train_numeric(TimeUsageLossMetric::new())
        .metric_valid_numeric(TimeUsageLossMetric::new())
        .metric_train_numeric(WdlAccuracyMetric::new())
        .metric_train_numeric(LegalMoveProbabilityMetric::default())
        .metric_valid_numeric(LegalMoveProbabilityMetric::default())
        .metric_train_numeric(burn_train::metric::LearningRateMetric::default())
        .metric_train(UncertaintyMetric::<B>::new())
        .metric_valid(UncertaintyMetric::<B::InnerBackend>::new())
        // Removed ModelPredictionLogger - now called directly in forward_classification
        // .metric_train(burn_train::metric::IterationSpeedMetric::new())
        .with_file_checkpointer(CompactRecorder::default())
        .with_checkpointing_strategy(
            ComposedCheckpointingStrategy::builder()
                .add(KeepLastNCheckpoints::new(2))
                .build(),
        )
        .devices(devices)
        .num_epochs(config.num_epochs);

    if let Some(checkpoint_idx) = config.checkpoint {
        learner_builder = learner_builder.checkpoint(checkpoint_idx);
    }
    if let Some(batch_size) = config.batch_size {
        println!(
            "Grad accumulation will be {}",
            batch_size / config.physical_batch_size
        );
        learner_builder =
            learner_builder.grads_accumulation(batch_size / config.physical_batch_size);
    }

    // Create interrupter for timeout support
    let interrupter = learner_builder.interrupter();

    // Start timeout task if specified
    if let Some(timeout) = config.timeout {
        tokio::spawn(async move {
            sleep(Duration::from_secs(timeout)).await;
            println!("Timeout reached, stopping training");
            interrupter.stop();
        });
    }

    let mut learner = learner_builder;
    if config.disable_tui {
        learner = learner.renderer(CustomRenderer {});
    }
    let learner = learner
        // .summary()
        .build(
            model,
            AdamConfig::new()
                // .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
                .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                    config.weight_decay as f32,
                )))
                .with_grad_clipping(if config.gradient_clip > 0.0 {
                    Some(GradientClippingConfig::Norm(config.gradient_clip as f32))
                } else {
                    None
                })
                .init(),
            NoamLrSchedulerConfig::new(config.learning_rate)
                .with_warmup_steps(config.warmup)
                .with_model_size(config.embed_dim)
                .init()
                .unwrap(),
        );

    // Train the model
    let trained_model = learner.fit(dataloader_train, dataloader_valid);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Should be able to save the model");
    // let inference_model = trained_model.valid();

    // // Print final validation accuracy for Optuna if validation_samples > 0
    // if let Some(_validation_samples) = config.validation_samples {
    //     let mut total_accuracy = 0.0;
    //     let mut total_samples = 0;
    //     let mut batch_count = 0;
    //
    //     // Evaluate on validation set
    //     let dataloader_iter = dataloader_valid_clone.iter();
    //     for batch in dataloader_iter {
    //         let batch_size = batch.board_input.shape().dims[0];
    //
    //         let (policy_logits, _value_logits, _side_info_logits, _time_usage_logits) =
    //             inference_model.forward(batch.board_input.clone(), batch.global_features.clone());
    //
    //         // Create input for MoveAccuracyMetric
    //
    //         let policy_logits_flat_original = policy_logits.reshape([batch_size, LEGAL_MOVES]);
    //
    //         let mask = batch.legal_moves.clone().equal_elem(0.0);
    //         let policy_logits_flat = policy_logits_flat_original
    //             .clone()
    //             .mask_fill(mask.clone(), f32::NEG_INFINITY);
    //         let targets = batch.move_distributions.clone().argmax(1).squeeze(1);
    //
    //         let accuracy_input = MoveAccuracyInput::new(policy_logits_flat, targets);
    //
    //         // Calculate accuracy for this batch
    //         let mut accuracy_metric = MoveAccuracyMetric::new();
    //         let mut top1_accuracy_metric = MoveTop1AccuracyMetric::new();
    //         let metadata = MetricMetadata {
    //             progress: Progress {
    //                 items_processed: batch_count,
    //                 items_total: batch_count + 1,
    //             },
    //             epoch: 0,
    //             epoch_total: 1,
    //             iteration: batch_count,
    //             lr: Some(config.learning_rate),
    //         };
    //         let _metric_entry = accuracy_metric.update(&accuracy_input, &metadata);
    //         let batch_accuracy = Numeric::value(&accuracy_metric);
    //         let _top1_metric_entry = top1_accuracy_metric.update(&accuracy_input, &metadata);
    //         let _batch_top1_accuracy = Numeric::value(&top1_accuracy_metric);
    //
    //         // Accumulate weighted accuracy
    //         total_accuracy += batch_accuracy * batch_size as f64;
    //         total_samples += batch_size;
    //         batch_count += 1;
    //
    //         // Limit evaluation to configured validation samples
    //         if total_samples >= config.validation_samples.expect("validation_samples") {
    //             break;
    //         }
    //     }
    //
    //     // Calculate final weighted accuracy
    //     let final_accuracy = if total_samples > 0 {
    //         total_accuracy / total_samples as f64
    //     } else {
    //         0.0
    //     };
    //
    //     println!("VALIDATION_ACCURACY:{final_accuracy:.4}");
    //     tracing::info!(
    //         "Validation accuracy: {:.4} (calculated on {} samples in {} batches, target: {})",
    //         final_accuracy,
    //         total_samples,
    //         batch_count,
    //         config.validation_samples.expect("validation_samples")
    //     );
    // }

    Ok(())
}

/// Load a trained model from checkpoint
pub fn load_model<B: Backend>(
    path: &Path,
    config: &Config,
    device: &Device<B>,
) -> anyhow::Result<OXIModel<B>> {
    let record = CompactRecorder::new().load(path.to_path_buf(), device)?;

    let model = OXIModel::new(device, config).load_record(record);

    Ok(model)
}

/// Save model checkpoint
pub fn save_model<B: Backend>(model: OXIModel<B>, path: &Path) -> anyhow::Result<()> {
    let recorder = CompactRecorder::new();
    model.save_file(path, &recorder)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceEngine;
    use burn::data::dataloader::Dataset;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use tempfile::TempDir;

    // #[test]
    // fn test_model_creation() {
    //     let device = NdArrayDevice::default();
    //     let config = Config::default();
    //     let model: OXIModel<NdArray> = OXIModel::new(&device, &config);
    //
    //     // Test forward pass with dummy data
    //     let batch_size = 2;
    //     let board = Tensor::zeros([batch_size, 16, 8, 8], &device);
    //     let elo_self = Tensor::zeros([batch_size, 1], &device);
    //     let elo_oppo = Tensor::zeros([batch_size, 1], &device);
    //
    //     let (policy, value, side_info) = model.forward(board, elo_self, elo_oppo);
    //
    //     assert_eq!(policy.dims(), [batch_size, 64, 64]);
    //     assert_eq!(value.dims(), [batch_size, 1]);
    //     assert_eq!(side_info.dims(), [batch_size, 13]);
    // }
    //
    // #[test]
    // fn test_minimal_save_load() {
    //     type Backend = NdArray;
    //     let device = NdArrayDevice::default();
    //
    //     // Create minimal model config
    //     let config = Config::new(32, 256, 1);
    //
    //     // Create model
    //     let model: OXIModel<Backend> = OXIModel::new(&device, &config);
    //
    //     // Create temporary directory for model
    //     let temp_dir = TempDir::new().unwrap();
    //     let model_path = temp_dir.path().join("test_model.mpk");
    //
    //     // Save the model
    //     save_model(model, &model_path).unwrap();
    //
    //     // Load the model
    //     let loaded_model: OXIModel<Backend> = load_model(&model_path, &config, &device).unwrap();
    //
    //     unimplemented!();
    //
    //     // Create inference engine
    //     // let engine = InferenceEngine::new(loaded_model, config, device);
    //     //
    //     // // Test inference
    //     // let test_fen = "5rk1/2p3p1/p2q1r1p/1p1p2p1/3P1nN1/2P1RP2/PP1Q2PP/4R1K1 w - - 0 31";
    //     // let predictions = engine.predict(test_fen, 1600, 1600, 1.0, 10).unwrap();
    //     //
    //     // // Verify we get predictions
    //     // assert!(
    //     //     !predictions.is_empty(),
    //     //     "Should get at least one move prediction"
    //     // );
    // }
    //
    // #[test]
    // fn test_save_load_inference() {
    //     use crate::dataset::{ChessBatcher, OXIDataset};
    //     use burn::data::dataloader::DataLoaderBuilder;
    //     use burn::optim::AdamConfig;
    //     use burn::train::metric::LossMetric;
    //     use burn::train::LearnerBuilder;
    //     use burn_autodiff::Autodiff;
    //     use std::path::PathBuf;
    //
    //     type Backend = Autodiff<NdArray>;
    //     let device = NdArrayDevice::default();
    //
    //     // Create minimal model config
    //     let mut config = Config::new(32, 256, 1);
    //     config.batch_size = 32;
    //     config.learning_rate = 0.001;
    //
    //     // Create temporary directory for model
    //     let temp_dir = TempDir::new().unwrap();
    //     let model_path = temp_dir.path().join("model.mpk");
    //
    //     // Load real PGN data
    //     let pgn_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgn");
    //
    //     let dataset = OXIDataset::from_pgn_dir_with_limit(&pgn_dir, config.clone(), None).unwrap();
    //     let dataset_len = dataset.len();
    //
    //     // Split dataset for train/valid (use same dataset for simplicity)
    //     let train_dataset = dataset;
    //     let valid_dataset =
    //         OXIDataset::from_pgn_dir_with_limit(&pgn_dir, config.clone(), None).unwrap();
    //
    //     // Create model
    //     let model: OXIModel<Backend> = OXIModel::new(&device, &config);
    //
    //     // Create batchers
    //     let batcher_train = ChessBatcher::<Backend>::new(device.clone());
    //     let batcher_valid = ChessBatcher::<NdArray>::new(device.clone());
    //
    //     // Create data loaders
    //     let dataloader_train = DataLoaderBuilder::new(batcher_train)
    //         .batch_size(config.batch_size)
    //         .shuffle(42)
    //         .num_workers(1)
    //         .build(train_dataset);
    //
    //     let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
    //         .batch_size(config.batch_size)
    //         .shuffle(42)
    //         .num_workers(1)
    //         .build(valid_dataset);
    //
    //     // Train the model
    //     let learner = LearnerBuilder::new(temp_dir.path())
    //         .metric_train_numeric(LossMetric::new())
    //         .metric_valid_numeric(LossMetric::new())
    //         .with_file_checkpointer(CompactRecorder::new())
    //         .devices(vec![device.clone()])
    //         .num_epochs(1) // Run one full epoch through the data
    //         .summary()
    //         .build(model, AdamConfig::new().init(), config.learning_rate);
    //
    //     // Limit the dataset size for faster training
    //     let _num_batches = (dataset_len / config.batch_size).min(10); // Limit to 10 batches for testing
    //
    //     // Train the model
    //     let trained_model = learner.fit(dataloader_train, dataloader_valid);
    //
    //     // Save the trained model
    //     save_model(trained_model, &model_path).unwrap();
    //
    //     // Load the model for inference
    //     let loaded_model: OXIModel<NdArray> = load_model(&model_path, &config, &device).unwrap();
    //
    //     // Create inference engine
    //     let engine = InferenceEngine::new(loaded_model, config, device);
    //
    //     // Test inference on the requested position
    //     let test_fen = "5rk1/2p3p1/p2q1r1p/1p1p2p1/3P1nN1/2P1RP2/PP1Q2PP/4R1K1 w - - 0 31";
    //     unimplemented!();
    //     // let predictions = engine.predict(test_fen, 1600, 1600, 1.0, 20).unwrap();
    //     //
    //     // // Verify we get predictions
    //     // assert!(
    //     //     !predictions.is_empty(),
    //     //     "Should get at least one move prediction"
    //     // );
    //     //
    //     // // Check if g4f6 is among the predictions
    //     // let _has_g4f6 = predictions.iter().any(|p| p.uci_move == "g4f6");
    //     //
    //     // // Verify the model can make legal predictions
    //     // for pred in &predictions {
    //     //     assert!(pred.probability > 0.0 && pred.probability <= 1.0);
    //     //     assert!(pred.win_prob >= 0.0 && pred.win_prob <= 1.0);
    //     //     assert!(pred.draw_prob >= 0.0 && pred.draw_prob <= 1.0);
    //     //     assert!(pred.loss_prob >= 0.0 && pred.loss_prob <= 1.0);
    //     //     let prob_sum = pred.win_prob + pred.draw_prob + pred.loss_prob;
    //     //     assert!(
    //     //         (prob_sum - 1.0).abs() < 0.01,
    //     //         "WDL probabilities should sum to ~1.0"
    //     //     );
    //     // }
    // }
}
