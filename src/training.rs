#[cfg(target_os = "linux")]
use burn::backend::cuda::CudaDevice;
use burn::data::dataloader::DataLoaderBuilder;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::module::Module;
use burn::nn::loss::CosineEmbeddingLossConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamConfig, SgdConfig};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
#[cfg(target_os = "macos")]
use burn_candle::MetalDevice;
use burn_train::checkpoint::{ComposedCheckpointingStrategy, KeepLastNCheckpoints};
use burn_train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use std::path::Path;

use crate::config::{ModelConfig, TrainingConfig};
use crate::dataset::{ChessBatcher, OXIDataset};
use crate::legal_move_probability_metric::LegalMoveProbabilityMetric;
use crate::model::OXIModel;
use crate::move_accuracy_metric::MoveAccuracyMetric;
use crate::move_distribution_accuracy_metric::MoveDistributionAccuracyMetric;
use crate::policy_loss_metric::PolicyLossMetric;
use crate::side_info_loss_metric::SideInfoLossMetric;
use crate::uncertainty_metric::UncertaintyMetric;
use crate::value_loss_metric::ValueLossMetric;
use crate::wdl_accuracy_metric::WdlAccuracyMetric;

struct CustomRenderer {}

impl MetricsRenderer for CustomRenderer {
    fn update_train(&mut self, _state: MetricState) {
        dbg!(_state);
    }

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        dbg!(item);
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        dbg!(item);
    }
}

/// Train the Oxi model with dataset
pub fn train<B: AutodiffBackend>(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_path: &Path,
    train_ratio: f32,
    max_samples: Option<usize>,
    devices: Vec<B::Device>,
    // device: B::Device,
) -> anyhow::Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    #[cfg(target_os = "linux")]

    #[cfg(target_os = "macos")]
    let devices: Vec<B::Device> = (0..1).map(|_| B::Device::default()).collect();

    // Create model
    let model: OXIModel<B> = OXIModel::new(&devices[0], &model_config);

    // Load dataset with max_samples limit
    let dataset: OXIDataset = if data_path.is_dir() {
        tracing::info!("Loading data from PGN directory: {:?}", data_path);
        OXIDataset::from_pgn_dir_with_limit(data_path, model_config.clone(), max_samples)?
    } else {
        // Handle both .pgn and .pgn.zst files
        tracing::info!("Loading data from PGN file: {:?}", data_path);
        OXIDataset::from_pgn_with_limit(data_path, model_config.clone(), max_samples)?
    };

    // Split into train and validation
    tracing::info!("Splitting dataset with train ratio: {}", train_ratio);
    let (train_dataset, valid_dataset) = dataset.split(train_ratio);

    // Create batchers
    let batcher_train: ChessBatcher<B> = ChessBatcher::new(devices[0].clone());
    let batcher_valid: ChessBatcher<B::InnerBackend> = ChessBatcher::new(devices[0].clone());

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed())
        .num_workers(training_config.num_workers())
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed())
        .num_workers(training_config.num_workers())
        .build(valid_dataset);

    // Create learner
    let learner = LearnerBuilder::new("checkpoints")
        .metric_train_numeric(MoveAccuracyMetric::new())
        .metric_valid_numeric(MoveAccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PolicyLossMetric::new())
        .metric_valid_numeric(PolicyLossMetric::new())
        .metric_train_numeric(ValueLossMetric::new())
        .metric_valid_numeric(ValueLossMetric::new())
        .metric_train_numeric(SideInfoLossMetric::new())
        .metric_valid_numeric(SideInfoLossMetric::new())
        .metric_train_numeric(MoveAccuracyMetric::new())
        .metric_valid_numeric(MoveAccuracyMetric::new())
        .metric_train_numeric(WdlAccuracyMetric::new())
        .metric_train_numeric(LegalMoveProbabilityMetric::default())
        .metric_valid_numeric(LegalMoveProbabilityMetric::default())
        .metric_train_numeric(burn_train::metric::LearningRateMetric::default())
        .metric_train(UncertaintyMetric::<B>::new())
        .metric_valid(UncertaintyMetric::<B::InnerBackend>::new())
        // .metric_train(burn_train::metric::IterationSpeedMetric::new())
        // .with_file_checkpointer(CompactRecorder::new())
        // .with_checkpointing_strategy(
        //     ComposedCheckpointingStrategy::builder()
        //         .add(KeepLastNCheckpoints::new(2))
        //         .build(),
        // )
        .devices(devices)
        .num_epochs(training_config.num_epochs)
        // .renderer(CustomRenderer {})
        // .summary()
        .build(
            model,
            AdamConfig::new()
                // .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
                .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                    training_config.weight_decay() as f32,
                )))
                .with_grad_clipping(Some(GradientClippingConfig::Norm(2.0)))
                .init(),
            CosineAnnealingLrSchedulerConfig::new(training_config.learning_rate, 1000)
                .with_min_lr(training_config.learning_rate / 10.0)
                .init()
                .unwrap(),
        );

    // Train the model
    let _trained_model = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}

/// Load a trained model from checkpoint
pub fn load_model<B: Backend>(
    path: &Path,
    model_config: &ModelConfig,
    device: &Device<B>,
) -> anyhow::Result<OXIModel<B>> {
    let record = CompactRecorder::new().load(path.to_path_buf().into(), device)?;

    let model = OXIModel::new(device, model_config).load_record(record);

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

    #[test]
    fn test_model_creation() {
        let device = NdArrayDevice::default();
        let config = ModelConfig::default();
        let model: OXIModel<NdArray> = OXIModel::new(&device, &config);

        // Test forward pass with dummy data
        let batch_size = 2;
        let board = Tensor::zeros([batch_size, 16, 8, 8], &device);
        let elo_self = Tensor::zeros([batch_size, 1], &device);
        let elo_oppo = Tensor::zeros([batch_size, 1], &device);

        let (policy, value, side_info) = model.forward(board, elo_self, elo_oppo);

        assert_eq!(policy.dims(), [batch_size, 4096]);
        assert_eq!(value.dims(), [batch_size, 1]);
        assert_eq!(side_info.dims(), [batch_size, 13]);
    }

    #[test]
    fn test_minimal_save_load() {
        type Backend = NdArray;
        let device = NdArrayDevice::default();

        // Create minimal model config
        let model_config = ModelConfig::new(1, 32);

        // Create model
        let model: OXIModel<Backend> = OXIModel::new(&device, &model_config);

        // Create temporary directory for model
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.mpk");

        // Save the model
        save_model(model, &model_path).unwrap();

        // Load the model
        let loaded_model: OXIModel<Backend> =
            load_model(&model_path, &model_config, &device).unwrap();

        // Create inference engine
        let engine = InferenceEngine::new(loaded_model, model_config, device);

        // Test inference
        let test_fen = "5rk1/2p3p1/p2q1r1p/1p1p2p1/3P1nN1/2P1RP2/PP1Q2PP/4R1K1 w - - 0 31";
        let predictions = engine.predict(test_fen, 1600, 1600, 1.0, 10).unwrap();

        // Verify we get predictions
        assert!(
            !predictions.is_empty(),
            "Should get at least one move prediction"
        );
    }

    #[test]
    fn test_save_load_inference() {
        use crate::dataset::{ChessBatcher, OXIDataset};
        use burn::data::dataloader::DataLoaderBuilder;
        use burn::optim::AdamConfig;
        use burn::train::metric::LossMetric;
        use burn::train::LearnerBuilder;
        use burn_autodiff::Autodiff;
        use std::path::PathBuf;

        type Backend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        // Create minimal model config
        let model_config = ModelConfig::new(2, 32);

        let mut training_config = TrainingConfig::default();
        training_config.batch_size = 32;
        training_config.learning_rate = 0.001;

        // Create temporary directory for model
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.mpk");

        // Load real PGN data
        let pgn_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgn");

        let dataset =
            OXIDataset::from_pgn_dir_with_limit(&pgn_dir, model_config.clone(), None).unwrap();
        let dataset_len = dataset.len();

        // Split dataset for train/valid (use same dataset for simplicity)
        let train_dataset = dataset;
        let valid_dataset =
            OXIDataset::from_pgn_dir_with_limit(&pgn_dir, model_config.clone(), None).unwrap();

        // Create model
        let model: OXIModel<Backend> = OXIModel::new(&device, &model_config);

        // Create batchers
        let batcher_train = ChessBatcher::<Backend>::new(device.clone());
        let batcher_valid = ChessBatcher::<NdArray>::new(device.clone());

        // Create data loaders
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(training_config.batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(train_dataset);

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(training_config.batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(valid_dataset);

        // Train the model
        let learner = LearnerBuilder::new(temp_dir.path())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .devices(vec![device.clone()])
            .num_epochs(1) // Run one full epoch through the data
            .summary()
            .build(
                model,
                AdamConfig::new().init(),
                training_config.learning_rate,
            );

        // Limit the dataset size for faster training
        let _num_batches = (dataset_len / training_config.batch_size).min(10); // Limit to 10 batches for testing

        // Train the model
        let trained_model = learner.fit(dataloader_train, dataloader_valid);

        // Save the trained model
        save_model(trained_model, &model_path).unwrap();

        // Load the model for inference
        let loaded_model: OXIModel<NdArray> =
            load_model(&model_path, &model_config, &device).unwrap();

        // Create inference engine
        let engine = InferenceEngine::new(loaded_model, model_config, device);

        // Test inference on the requested position
        let test_fen = "5rk1/2p3p1/p2q1r1p/1p1p2p1/3P1nN1/2P1RP2/PP1Q2PP/4R1K1 w - - 0 31";
        let predictions = engine.predict(test_fen, 1600, 1600, 1.0, 20).unwrap();

        // Verify we get predictions
        assert!(
            !predictions.is_empty(),
            "Should get at least one move prediction"
        );

        // Check if g4f6 is among the predictions
        let _has_g4f6 = predictions.iter().any(|p| p.uci_move == "g4f6");

        // Verify the model can make legal predictions
        for pred in &predictions {
            assert!(pred.probability > 0.0 && pred.probability <= 1.0);
            assert!(pred.win_prob >= 0.0 && pred.win_prob <= 1.0);
            assert!(pred.draw_prob >= 0.0 && pred.draw_prob <= 1.0);
            assert!(pred.loss_prob >= 0.0 && pred.loss_prob <= 1.0);
            let prob_sum = pred.win_prob + pred.draw_prob + pred.loss_prob;
            assert!(
                (prob_sum - 1.0).abs() < 0.01,
                "WDL probabilities should sum to ~1.0"
            );
        }
    }
}
