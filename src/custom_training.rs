use crate::metrics_renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, TrainingProgress,
};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Progress;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::LrScheduler;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, MuonConfig, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn_train::metric::IterationSpeedMetric;
use burn_train::metric::{Adaptor, Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};
use burn_train::Interrupter;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::thread_rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::subscriber::DefaultGuard;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt as tracing_fmt;

use crate::chess_output::ChessOutput;
use crate::config::Config;

use crate::dataset::{ChessBatch, ChessBatcher, ChessExample, ChessItem, OXIDataset};
use crate::debug_prediction_monitor::DebugPredictionMonitor;
use crate::gradient_norm_metric::{
    compute_gradient_norm_with_breakdown, GradientNormBreakdown, GradientNormInput,
    GradientNormMetric,
};
use crate::gradnorm::{GradNormProbeResult, GradNormState, GradNormTask};
use crate::lr_plateau_metric::{LrPlateauInput, LrPlateauMetric};
use crate::model::OXIModel;
use crate::move_accuracy_metric::MoveTopKAccuracyMetric;
use crate::pgn_processor::process_pgn_directory_iter;
use crate::pgn_processor::process_tcec_directory_iter;
use crate::policy_loss_metric::{PolicyLossInput, PolicyLossMetric};
use crate::puzzle_processor::{process_puzzle_file_iter, MixedExampleIterator};
use crate::reduce_on_plateau_scheduler::ReduceOnPlateauScheduler;
use crate::time_usage_loss_metric::{TimeUsageLossInput, TimeUsageLossMetric};
use crate::training_stage_metric::{TrainingStage, TrainingStageInput, TrainingStageMetric};
use crate::tui::OxiTuiRenderer;
use crate::value_loss_metric::{ValueLossInput, ValueLossMetric};
use crate::wdl_accuracy_metric::WdlAccuracyMetric;
use crate::weight_decay::{ValueTowerGradientFilter, WeightDecayGroups};

/// Simple CLI renderer for when TUI is disabled
struct SimpleCliRenderer;

impl SimpleCliRenderer {
    fn new() -> Self {
        Self
    }
}

impl MetricsRendererTraining for SimpleCliRenderer {
    fn update_train(&mut self, _state: MetricState) {
        // Metrics are printed directly in the training loop
    }
    fn update_valid(&mut self, _state: MetricState) {}
    fn render_train(&mut self, _item: TrainingProgress) {}
    fn render_valid(&mut self, _item: TrainingProgress) {}
}

impl MetricsRendererEvaluation for SimpleCliRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
    fn render_test(&mut self, _item: EvaluationProgress) {}
}

impl MetricsRenderer for SimpleCliRenderer {
    fn manual_close(&mut self) {}
}

use std::fs::OpenOptions;
use std::io::Write;

struct MetricFileLogger {
    dir: PathBuf,
}

impl MetricFileLogger {
    fn new(base_dir: Option<&Path>) -> Self {
        let dir = match base_dir {
            Some(base) => base.join("metrics_logs"),
            None => PathBuf::from("metrics_logs"),
        };
        let _ = std::fs::create_dir_all(&dir);
        Self { dir }
    }

    fn log(&self, metric_name: &str, iteration: usize, value: f64) {
        let safe_name = metric_name.replace(['/', '\\', '|', ' '], "_");
        let path = self.dir.join(format!("{}.log", safe_name));
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(file, "{}\t{:.8}", iteration, value);
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum EloBucket {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl EloBucket {
    fn label(self) -> &'static str {
        match self {
            EloBucket::Beginner => "Beginner",
            EloBucket::Intermediate => "Intermediate",
            EloBucket::Advanced => "Advanced",
            EloBucket::Expert => "Expert",
        }
    }
}

const ELO_BUCKETS: [EloBucket; 4] = [
    EloBucket::Beginner,
    EloBucket::Intermediate,
    EloBucket::Advanced,
    EloBucket::Expert,
];

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum GameStageBucket {
    Opening,
    Middlegame,
    Endgame,
}

impl GameStageBucket {
    fn label(self) -> &'static str {
        match self {
            GameStageBucket::Opening => "Opening",
            GameStageBucket::Middlegame => "Middlegame",
            GameStageBucket::Endgame => "Endgame",
        }
    }
}

const GAME_STAGE_BUCKETS: [GameStageBucket; 3] = [
    GameStageBucket::Opening,
    GameStageBucket::Middlegame,
    GameStageBucket::Endgame,
];

#[derive(Default, Copy, Clone)]
struct AccuracyCounter {
    correct: usize,
    total: usize,
}

impl AccuracyCounter {
    fn update(&mut self, correct: bool) {
        self.total += 1;
        if correct {
            self.correct += 1;
        }
    }

    fn accuracy(self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some(self.correct as f64 / self.total as f64)
        }
    }
}

fn categorize_elo(elo: i32) -> Option<EloBucket> {
    if elo <= 0 {
        return None;
    }
    if elo <= 1600 {
        Some(EloBucket::Beginner)
    } else if elo <= 2000 {
        Some(EloBucket::Intermediate)
    } else if elo <= 2300 {
        Some(EloBucket::Advanced)
    } else {
        Some(EloBucket::Expert)
    }
}

fn categorize_stage(move_count: usize) -> GameStageBucket {
    if move_count < 20 {
        GameStageBucket::Opening
    } else if move_count < 50 {
        GameStageBucket::Middlegame
    } else {
        GameStageBucket::Endgame
    }
}

const SCORE_WINDOW: usize = 100;

/// Reference dimension for μP-informed LR scaling.
/// Base LRs (adamw_base_lr, muon_base_lr) are defined at this width.
/// When embed_dim differs, they are automatically adjusted:
///   - AdamW: scales as d_ref / d (decreases with width)
///   - Muon: scales as sqrt(d_ref / d) (decreases slower)
///   - Embedding: width-independent (no scaling)
const LR_REFERENCE_DIM: f64 = 256.0;

const MODEL_DIR_NAME: &str = "model";
const MODEL_FILE_NAME: &str = "model.mpk";
const PARAMS_FILE_NAME: &str = "params.json";
const OPT_MUON_FILE_NAME: &str = "optimizer_muon.mpk";
const OPT_DECAY_NORMAL_FILE_NAME: &str = "optimizer_decay_normal.mpk";
const OPT_DECAY_HIGH_FILE_NAME: &str = "optimizer_decay_high.mpk";
const OPT_NO_DECAY_NORMAL_FILE_NAME: &str = "optimizer_no_decay_normal.mpk";
const OPT_NO_DECAY_HIGH_FILE_NAME: &str = "optimizer_no_decay_high.mpk";
const GRADNORM_STATE_FILE_NAME: &str = "gradnorm_state.bin";

fn save_optimizer_state<B: AutodiffBackend, O>(
    optimizer: &O,
    recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    path: PathBuf,
) -> anyhow::Result<()>
where
    O: Optimizer<OXIModel<B>, B>,
{
    let record = optimizer.to_record();
    recorder
        .record(record, path)
        .map_err(|err| anyhow::anyhow!(err))
}

fn load_optimizer_state<B: AutodiffBackend, O>(
    optimizer: O,
    recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    path: PathBuf,
    device: &B::Device,
) -> anyhow::Result<O>
where
    O: Optimizer<OXIModel<B>, B>,
{
    let record = recorder
        .load(path, device)
        .map_err(|err| anyhow::anyhow!(err))?;
    Ok(optimizer.load_record(record))
}

fn save_training_state<B, OM, O1, O2, O3, O4>(
    model: &OXIModel<B>,
    config: &Config,
    gradnorm_state: &GradNormState,
    optim_muon: &OM,
    optim_decay_normal: &O1,
    optim_decay_high: &O2,
    optim_no_decay_normal: &O3,
    optim_no_decay_high: &O4,
    directory: &Path,
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
    OM: Optimizer<OXIModel<B>, B>,
    O1: Optimizer<OXIModel<B>, B>,
    O2: Optimizer<OXIModel<B>, B>,
    O3: Optimizer<OXIModel<B>, B>,
    O4: Optimizer<OXIModel<B>, B>,
{
    std::fs::create_dir_all(directory)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    model
        .clone()
        .valid()
        .save_file(directory.join(MODEL_FILE_NAME), &recorder)
        .map_err(|err| anyhow::anyhow!(err))?;

    let params_path = directory.join(PARAMS_FILE_NAME);
    let params_json = serde_json::to_string_pretty(config)
        .map_err(|err| anyhow::anyhow!("Failed to serialize config: {}", err))?;
    std::fs::write(&params_path, params_json)?;

    save_optimizer_state::<B, OM>(optim_muon, &recorder, directory.join(OPT_MUON_FILE_NAME))?;
    save_optimizer_state::<B, O1>(
        optim_decay_normal,
        &recorder,
        directory.join(OPT_DECAY_NORMAL_FILE_NAME),
    )?;
    save_optimizer_state::<B, O2>(
        optim_decay_high,
        &recorder,
        directory.join(OPT_DECAY_HIGH_FILE_NAME),
    )?;
    save_optimizer_state::<B, O3>(
        optim_no_decay_normal,
        &recorder,
        directory.join(OPT_NO_DECAY_NORMAL_FILE_NAME),
    )?;
    save_optimizer_state::<B, O4>(
        optim_no_decay_high,
        &recorder,
        directory.join(OPT_NO_DECAY_HIGH_FILE_NAME),
    )?;

    let gradnorm_path = directory.join(GRADNORM_STATE_FILE_NAME);
    let gradnorm_bytes = bincode::serde::encode_to_vec(gradnorm_state, bincode::config::standard())
        .map_err(|err: bincode::error::EncodeError| anyhow::anyhow!(err))?;
    std::fs::write(&gradnorm_path, gradnorm_bytes)?;

    Ok(())
}

#[derive(Clone, Debug)]
struct RollingMetric {
    values: VecDeque<f64>,
    capacity: usize,
}

impl RollingMetric {
    fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    fn average(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let sum: f64 = self.values.iter().sum();
        Some(sum / self.values.len() as f64)
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

fn numeric_entry_value(entry: &NumericEntry) -> Option<f64> {
    let value = match entry {
        NumericEntry::Value(v) => *v,
        NumericEntry::Aggregated {
            aggregated_value, ..
        } => *aggregated_value,
    };
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn numeric_entry_raw_value(entry: &NumericEntry) -> f64 {
    match entry {
        NumericEntry::Value(v) => *v,
        NumericEntry::Aggregated {
            aggregated_value, ..
        } => *aggregated_value,
    }
}

fn split_items_across_devices<T: Clone>(items: &[T], num_devices: usize) -> Vec<Vec<T>> {
    if num_devices == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(num_devices);
    let mut start = 0usize;

    for device_index in 0..num_devices {
        let remaining = items.len().saturating_sub(start);
        let remaining_devices = num_devices - device_index;

        if remaining == 0 {
            result.push(Vec::new());
            continue;
        }

        let chunk_size = (remaining + remaining_devices - 1) / remaining_devices;
        let end = start + chunk_size.min(remaining);
        result.push(items[start..end].to_vec());
        start = end;
    }

    result
}

fn sample_gradnorm_items(items: &[ChessItem], sample_size: usize) -> Vec<ChessItem> {
    if items.len() <= sample_size {
        return items.to_vec();
    }

    let mut rng = thread_rng();
    items
        .iter()
        .choose_multiple(&mut rng, sample_size)
        .into_iter()
        .cloned()
        .collect()
}

#[derive(Clone, Copy)]
struct GradNormWeights {
    policy: f32,
    value: f32,
    time: f32,
    aux: f32,
}

impl GradNormWeights {
    fn from_state(state: &GradNormState) -> Self {
        Self {
            policy: state.weight_for(GradNormTask::Policy),
            value: state.weight_for(GradNormTask::Value),
            time: state.weight_for(GradNormTask::TimeUsage),
            aux: state.weight_for(GradNormTask::Auxiliary),
        }
    }

    /// Create weights for value tower only training: policy=0, time=0, aux=0
    fn for_value_tower_only(state: &GradNormState) -> Self {
        Self {
            policy: 0.0,
            value: state.weight_for(GradNormTask::Value),
            time: 0.0,
            aux: 0.0,
        }
    }
}

fn move_output_to_device<B: Backend>(output: ChessOutput<B>, device: &B::Device) -> ChessOutput<B> {
    ChessOutput {
        loss: output.loss.to_device(device),
        policy_loss: output.policy_loss.to_device(device),
        value_loss: output.value_loss.to_device(device),
        time_usage_loss: output.time_usage_loss.to_device(device),
        aux_loss: output.aux_loss.to_device(device),
        base_policy_loss: output.base_policy_loss.to_device(device),
        base_value_loss: output.base_value_loss.to_device(device),
        base_time_usage_loss: output.base_time_usage_loss.to_device(device),
        base_aux_loss: output.base_aux_loss.to_device(device),
        policy_output: output.policy_output.to_device(device),
        policy_targets: output.policy_targets.to_device(device),
        value_output: output.value_output.to_device(device),
        value_targets: output.value_targets.to_device(device),
        legal_moves_mask: output.legal_moves_mask.to_device(device),
        uncertainties: output.uncertainties,
        raw_policy_loss: output
            .raw_policy_loss
            .map(|tensor| tensor.to_device(device)),
        raw_value_loss: output.raw_value_loss.map(|tensor| tensor.to_device(device)),
        raw_time_usage_loss: output
            .raw_time_usage_loss
            .map(|tensor| tensor.to_device(device)),
        aux_mobility_loss: output.aux_mobility_loss,
        aux_material_loss: output.aux_material_loss,
        aux_mobility_mae: output.aux_mobility_mae,
        aux_material_mae: output.aux_material_mae,
        aux_side_info_loss: output.aux_side_info_loss,
        aux_from_square_loss: output.aux_from_square_loss,
        aux_to_square_loss: output.aux_to_square_loss,
        aux_from_square_accuracy: output.aux_from_square_accuracy,
        aux_to_square_accuracy: output.aux_to_square_accuracy,
    }
    .detach()
}

fn apply_gradnorm_weights_to_output<B: Backend>(
    output: &mut ChessOutput<B>,
    weights: GradNormWeights,
) {
    output.policy_loss = output.base_policy_loss.clone() * weights.policy;
    output.value_loss = output.base_value_loss.clone() * weights.value;
    output.time_usage_loss = output.base_time_usage_loss.clone() * weights.time;
    output.aux_loss = output.base_aux_loss.clone() * weights.aux;
    output.loss = output.policy_loss.clone()
        + output.value_loss.clone()
        + output.time_usage_loss.clone()
        + output.aux_loss.clone();
}

fn combine_outputs<B: Backend>(outputs: &[ChessOutput<B>], device: &B::Device) -> ChessOutput<B> {
    assert!(
        !outputs.is_empty(),
        "Cannot combine outputs from an empty slice"
    );

    let mut total_items = 0usize;
    let mut sum_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_policy_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_value_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_time_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_aux_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_policy_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_value_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_time_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_aux_loss = Tensor::<B, 1>::zeros([1], device);

    let mut sum_aux_mobility_loss = 0.0f32;
    let mut sum_aux_material_loss = 0.0f32;
    let mut sum_aux_mobility_mae = 0.0f32;
    let mut sum_aux_material_mae = 0.0f32;
    let mut sum_aux_side_info_loss = 0.0f32;
    let mut sum_aux_from_square_loss = 0.0f32;
    let mut sum_aux_to_square_loss = 0.0f32;
    let mut sum_aux_from_square_accuracy = 0.0f32;
    let mut sum_aux_to_square_accuracy = 0.0f32;

    let all_raw_policy = outputs.iter().all(|o| o.raw_policy_loss.is_some());
    let all_raw_value = outputs.iter().all(|o| o.raw_value_loss.is_some());
    let all_raw_time = outputs.iter().all(|o| o.raw_time_usage_loss.is_some());

    let mut sum_raw_policy_loss = all_raw_policy.then(|| Tensor::<B, 1>::zeros([1], device));
    let mut sum_raw_value_loss = all_raw_value.then(|| Tensor::<B, 1>::zeros([1], device));
    let mut sum_raw_time_loss = all_raw_time.then(|| Tensor::<B, 1>::zeros([1], device));

    let mut policy_outputs = Vec::with_capacity(outputs.len());
    let mut policy_targets = Vec::with_capacity(outputs.len());
    let mut value_outputs = Vec::with_capacity(outputs.len());
    let mut value_targets = Vec::with_capacity(outputs.len());
    let mut legal_masks = Vec::with_capacity(outputs.len());

    for output in outputs {
        let batch_size = output.policy_output.dims()[0];
        total_items += batch_size;
        let batch_scalar = batch_size as f32;

        sum_loss = sum_loss + output.loss.clone() * batch_scalar;
        sum_policy_loss = sum_policy_loss + output.policy_loss.clone() * batch_scalar;
        sum_value_loss = sum_value_loss + output.value_loss.clone() * batch_scalar;
        sum_time_loss = sum_time_loss + output.time_usage_loss.clone() * batch_scalar;
        sum_aux_loss = sum_aux_loss + output.aux_loss.clone() * batch_scalar;

        sum_base_policy_loss =
            sum_base_policy_loss + output.base_policy_loss.clone() * batch_scalar;
        sum_base_value_loss = sum_base_value_loss + output.base_value_loss.clone() * batch_scalar;
        sum_base_time_loss =
            sum_base_time_loss + output.base_time_usage_loss.clone() * batch_scalar;
        sum_base_aux_loss = sum_base_aux_loss + output.base_aux_loss.clone() * batch_scalar;

        sum_aux_mobility_loss += output.aux_mobility_loss * batch_scalar;
        sum_aux_material_loss += output.aux_material_loss * batch_scalar;
        sum_aux_mobility_mae += output.aux_mobility_mae * batch_scalar;
        sum_aux_material_mae += output.aux_material_mae * batch_scalar;
        sum_aux_side_info_loss += output.aux_side_info_loss * batch_scalar;
        sum_aux_from_square_loss += output.aux_from_square_loss * batch_scalar;
        sum_aux_to_square_loss += output.aux_to_square_loss * batch_scalar;
        sum_aux_from_square_accuracy += output.aux_from_square_accuracy * batch_scalar;
        sum_aux_to_square_accuracy += output.aux_to_square_accuracy * batch_scalar;

        if let Some(sum) = sum_raw_policy_loss.as_mut() {
            if let Some(raw) = output.raw_policy_loss.as_ref() {
                *sum = sum.clone() + raw.clone() * batch_scalar;
            }
        }

        if let Some(sum) = sum_raw_value_loss.as_mut() {
            if let Some(raw) = output.raw_value_loss.as_ref() {
                *sum = sum.clone() + raw.clone() * batch_scalar;
            }
        }

        if let Some(sum) = sum_raw_time_loss.as_mut() {
            if let Some(raw) = output.raw_time_usage_loss.as_ref() {
                *sum = sum.clone() + raw.clone() * batch_scalar;
            }
        }

        policy_outputs.push(output.policy_output.clone());
        policy_targets.push(output.policy_targets.clone());
        value_outputs.push(output.value_output.clone());
        value_targets.push(output.value_targets.clone());
        legal_masks.push(output.legal_moves_mask.clone());
    }

    assert!(total_items > 0, "Combined batch must contain samples");
    let total_scalar = total_items as f32;

    let loss = sum_loss / total_scalar;
    let policy_loss = sum_policy_loss / total_scalar;
    let value_loss = sum_value_loss / total_scalar;
    let time_usage_loss = sum_time_loss / total_scalar;
    let aux_loss = sum_aux_loss / total_scalar;

    let base_policy_loss = sum_base_policy_loss / total_scalar;
    let base_value_loss = sum_base_value_loss / total_scalar;
    let base_time_usage_loss = sum_base_time_loss / total_scalar;
    let base_aux_loss = sum_base_aux_loss / total_scalar;

    let raw_policy_loss = sum_raw_policy_loss.map(|sum| sum / total_scalar);
    let raw_value_loss = sum_raw_value_loss.map(|sum| sum / total_scalar);
    let raw_time_usage_loss = sum_raw_time_loss.map(|sum| sum / total_scalar);

    let policy_output = Tensor::cat(policy_outputs, 0);
    let policy_targets = Tensor::cat(policy_targets, 0);
    let value_output = Tensor::cat(value_outputs, 0);
    let value_targets = Tensor::cat(value_targets, 0);
    let legal_moves_mask = Tensor::cat(legal_masks, 0);

    let uncertainties = outputs.iter().find_map(|output| output.uncertainties);

    ChessOutput {
        loss,
        policy_loss,
        value_loss,
        time_usage_loss,
        aux_loss,
        base_policy_loss,
        base_value_loss,
        base_time_usage_loss,
        base_aux_loss,
        policy_output,
        policy_targets,
        value_output,
        value_targets,
        legal_moves_mask,
        uncertainties,
        raw_policy_loss,
        raw_value_loss,
        raw_time_usage_loss,
        aux_mobility_loss: sum_aux_mobility_loss / total_scalar,
        aux_material_loss: sum_aux_material_loss / total_scalar,
        aux_mobility_mae: sum_aux_mobility_mae / total_scalar,
        aux_material_mae: sum_aux_material_mae / total_scalar,
        aux_side_info_loss: sum_aux_side_info_loss / total_scalar,
        aux_from_square_loss: sum_aux_from_square_loss / total_scalar,
        aux_to_square_loss: sum_aux_to_square_loss / total_scalar,
        aux_from_square_accuracy: sum_aux_from_square_accuracy / total_scalar,
        aux_to_square_accuracy: sum_aux_to_square_accuracy / total_scalar,
    }
    .detach()
}

fn compute_gradnorm_probe<B: AutodiffBackend>(
    model: OXIModel<B>,
    batch: ChessBatch<B>,
    weights: GradNormWeights,
) -> Vec<GradNormProbeResult>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    let mut results = Vec::new();

    let tasks = [
        (GradNormTask::Policy, weights.policy),
        (GradNormTask::Value, weights.value),
        (GradNormTask::TimeUsage, weights.time),
        (GradNormTask::Auxiliary, weights.aux),
    ];

    for (task, weight) in tasks {
        if weight <= 0.0 {
            continue;
        }

        let output = model.forward_classification(batch.clone());
        let base_loss_tensor = match task {
            GradNormTask::Policy => output.base_policy_loss.clone(),
            GradNormTask::Value => output.base_value_loss.clone(),
            GradNormTask::TimeUsage => output.base_time_usage_loss.clone(),
            GradNormTask::Auxiliary => output.base_aux_loss.clone(),
        };
        let base_loss_value = base_loss_tensor.clone().into_scalar().elem::<f32>();
        // Use base (unweighted) loss for gradient norm measurement.
        // GradNorm needs to compare the *intrinsic* gradient magnitudes of each task,
        // not the currently-weighted ones, to avoid a feedback loop.
        let grads = base_loss_tensor.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        let breakdown = compute_gradient_norm_with_breakdown(&grads_params, &model, false);
        results.push(GradNormProbeResult {
            task,
            base_loss: base_loss_value,
            grad_norm: breakdown.total() as f32,
        });
    }

    results
}

struct WorkerRequest {
    items: Vec<ChessItem>,
    weights: GradNormWeights,
}

struct GradNormProbeRequest {
    items: Vec<ChessItem>,
    weights: GradNormWeights,
}

type ModelRecord<B> = <OXIModel<B> as Module<B>>::Record;

enum WorkerResponse<B: AutodiffBackend> {
    Training {
        grads: GradientsParams,
        output: ChessOutput<B>,
    },
    GradNormProbe(Vec<GradNormProbeResult>),
}

enum WorkerCommand<B: AutodiffBackend>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    Run(WorkerRequest),
    GradNormProbe(GradNormProbeRequest),
    UpdateModel(ModelRecord<B>),
    Terminate,
}

struct DeviceWorker<B: AutodiffBackend>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    sender: Sender<WorkerCommand<B>>,
    receiver: Receiver<WorkerResponse<B>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl<B: AutodiffBackend> DeviceWorker<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
    B::Device: Clone + Send + 'static,
{
    fn new(device: B::Device, main_device: B::Device, initial_model: OXIModel<B>) -> Self {
        let (command_tx, command_rx) = mpsc::channel::<WorkerCommand<B>>();
        let (response_tx, response_rx) = mpsc::channel::<WorkerResponse<B>>();

        let handle = thread::spawn(move || {
            let batcher = ChessBatcher::<B>::new(device.clone());
            let mut model = initial_model;

            while let Ok(command) = command_rx.recv() {
                match command {
                    WorkerCommand::Run(request) => {
                        if request.items.is_empty() {
                            continue;
                        }

                        let t_batch_start = Instant::now();
                        let batch = batcher.batch(request.items, &device);
                        let batch_time = t_batch_start.elapsed();

                        let t_transfer_start = Instant::now();
                        let batch = batch.to_device(&device);
                        let transfer_time = t_transfer_start.elapsed();

                        let t_forward_start = Instant::now();
                        let mut output = model.forward_classification(batch);
                        let forward_time = t_forward_start.elapsed();

                        apply_gradnorm_weights_to_output(&mut output, request.weights);

                        let t_backward_start = Instant::now();
                        let loss = output.loss.clone();
                        let grads = GradientsParams::from_grads(loss.backward(), &model);
                        let backward_time = t_backward_start.elapsed();

                        tracing::info!(
                            "perf_timing: batch={:?} transfer={:?} forward={:?} backward={:?}",
                            batch_time,
                            transfer_time,
                            forward_time,
                            backward_time
                        );

                        let output_main = move_output_to_device(output, &main_device);
                        if response_tx
                            .send(WorkerResponse::Training {
                                grads,
                                output: output_main,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    WorkerCommand::GradNormProbe(request) => {
                        if request.items.is_empty() {
                            if response_tx
                                .send(WorkerResponse::GradNormProbe(Vec::new()))
                                .is_err()
                            {
                                break;
                            }
                            continue;
                        }

                        let batch = batcher.batch(request.items, &device).to_device(&device);
                        let probe_model = model.clone();
                        let results = compute_gradnorm_probe(probe_model, batch, request.weights);
                        if response_tx
                            .send(WorkerResponse::GradNormProbe(results))
                            .is_err()
                        {
                            break;
                        }
                    }
                    WorkerCommand::UpdateModel(record) => {
                        model = model.load_record(record);
                    }
                    WorkerCommand::Terminate => break,
                }
            }
        });

        Self {
            sender: command_tx,
            receiver: response_rx,
            handle: Some(handle),
        }
    }

    fn send(&self, command: WorkerCommand<B>) {
        let _ = self.sender.send(command);
    }

    fn recv(&self) -> Option<WorkerResponse<B>> {
        self.receiver.recv().ok()
    }

    fn terminate(&mut self) {
        let _ = self.sender.send(WorkerCommand::Terminate);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct DeviceWorkers<B: AutodiffBackend>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
    B::Device: Clone + Send + 'static,
{
    workers: Vec<DeviceWorker<B>>,
}

impl<B: AutodiffBackend> DeviceWorkers<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
    B::Device: Clone + Send + 'static,
{
    fn new(model: &OXIModel<B>, devices: &[B::Device], main_device: &B::Device) -> Self {
        let workers = devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let worker_model = if index == 0 {
                    model.clone()
                } else {
                    model.clone().fork(device)
                };
                DeviceWorker::new(device.clone(), main_device.clone(), worker_model)
            })
            .collect();

        Self { workers }
    }

    fn get(&self, index: usize) -> &DeviceWorker<B> {
        &self.workers[index]
    }

    fn broadcast_model(&self, model: &OXIModel<B>) {
        for worker in &self.workers {
            let record = model.clone().into_record();
            worker.send(WorkerCommand::UpdateModel(record));
        }
    }

    fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.terminate();
        }
    }
}

struct ShuffleBuffer<I: Iterator<Item = ChessExample>> {
    iter: I,
    buffer: Vec<ChessExample>,
    capacity: usize,
    exhausted: bool,
}

impl<I: Iterator<Item = ChessExample>> ShuffleBuffer<I> {
    fn new(iter: I, capacity: usize) -> Self {
        Self {
            iter,
            buffer: Vec::with_capacity(capacity),
            capacity,
            exhausted: false,
        }
    }

    fn fill_buffer(&mut self) {
        while self.buffer.len() < self.capacity {
            match self.iter.next() {
                Some(example) => self.buffer.push(example),
                None => {
                    self.exhausted = true;
                    break;
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.exhausted
    }

    fn sample_batch(&mut self, batch_size: usize, rng: &mut impl rand::Rng) -> Vec<ChessExample> {
        let pre_fill_len = self.buffer.len();
        self.fill_buffer();
        let post_fill_len = self.buffer.len();

        if post_fill_len != pre_fill_len {
            tracing::debug!(
                "ShuffleBuffer: filled {} -> {} examples (exhausted: {})",
                pre_fill_len,
                post_fill_len,
                self.exhausted
            );
        }

        if self.buffer.is_empty() {
            tracing::warn!(
                "ShuffleBuffer: buffer empty after fill attempt (exhausted: {})",
                self.exhausted
            );
            return Vec::new();
        }

        self.buffer.shuffle(rng);
        let take_count = batch_size.min(self.buffer.len());
        self.buffer.drain(..take_count).collect()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

fn log_gradient_breakdown(
    breakdown: &GradientNormBreakdown,
    config: &Config,
    optimizer_step: usize,
    muon_lr: f64,
    adamw_lr: f64,
    embedding_lr: f64,
) {
    tracing::info!(
        target: "gradient_debug",
        "gradient_debug: step={} total_grad_norm={:.4e}",
        optimizer_step,
        breakdown.total()
    );

    // Log per-layer diagnostics: grad_norm, weight_norm, update_ratio, SNR
    for layer in breakdown
        .per_layer
        .iter()
        .take(config.gradient_layer_limit())
    {
        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: layer={} grad_norm={:.4e} weight_norm={:.4e} update_ratio={:.4e} snr={:.4e} grad_mean={:.4e} grad_std={:.4e} numel={}",
            layer.name,
            layer.norm,
            layer.weight_norm,
            layer.update_ratio,
            layer.grad_snr,
            layer.grad_mean,
            layer.grad_std,
            layer.numel,
        );
    }

    // Log effective update ratios accounting for optimizer type
    // For Muon layers: NS orthogonalizes the gradient, so raw grad norms are meaningless.
    // The actual update: delta = muon_lr * lr_adjust(shape) * NS(momentum(grad))
    // ||NS(.)|| = sqrt(min(A,B)), so ||delta||_F = muon_lr * lr_adjust * sqrt(min(A,B))
    // We precomputed muon_update_scale = sqrt(sum of (sqrt(min(A,B)) * lr_adjust)^2) per layer.
    tracing::info!(
        target: "gradient_debug",
        "gradient_debug: effective_lrs muon={:.4e} adamw={:.4e} embedding={:.4e}",
        muon_lr, adamw_lr, embedding_lr,
    );

    for layer in breakdown
        .per_layer
        .iter()
        .filter(|l| l.muon_numel > 0)
        .take(config.gradient_layer_limit())
    {
        // ||update||_F = muon_lr * muon_update_scale
        let update_frob = muon_lr * layer.muon_update_scale;
        let update_rms = update_frob / (layer.muon_numel as f64).sqrt();
        let muon_update_ratio = if layer.muon_weight_rms > 0.0 {
            update_rms / layer.muon_weight_rms
        } else {
            0.0
        };
        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: muon_update layer={} update_rms={:.4e} weight_rms={:.4e} ratio={:.4e} muon_numel={}",
            layer.name,
            update_rms,
            layer.muon_weight_rms,
            muon_update_ratio,
            layer.muon_numel,
        );
    }

    // Log layer gradient ratio summary (first block vs last block, max/min ratio)
    let block_layers: Vec<&_> = breakdown
        .per_layer
        .iter()
        .filter(|l| l.name.starts_with("blocks."))
        .collect();
    if block_layers.len() >= 2 {
        // Sort by block index for first/last comparison
        let mut sorted_blocks: Vec<&_> = block_layers.clone();
        sorted_blocks.sort_by_key(|l| {
            l.name
                .strip_prefix("blocks.")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        });
        let first_norm = sorted_blocks.first().map(|l| l.norm).unwrap_or(0.0);
        let last_norm = sorted_blocks.last().map(|l| l.norm).unwrap_or(0.0);
        let first_last_ratio = if last_norm > 0.0 {
            first_norm / last_norm
        } else {
            f64::INFINITY
        };

        let max_norm = block_layers.iter().map(|l| l.norm).fold(0.0_f64, f64::max);
        let min_norm = block_layers
            .iter()
            .map(|l| l.norm)
            .fold(f64::INFINITY, f64::min);
        let max_min_ratio = if min_norm > 0.0 {
            max_norm / min_norm
        } else {
            f64::INFINITY
        };

        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: layer_ratio first/last={:.2} max/min={:.2} (first={:.4e} last={:.4e} max={:.4e} min={:.4e})",
            first_last_ratio,
            max_min_ratio,
            first_norm,
            last_norm,
            max_norm,
            min_norm,
        );
    }

    for head in breakdown.per_head.iter().take(config.gradient_head_limit()) {
        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: layer={} projection={} head={} grad_norm={:.4e}",
            head.layer,
            head.component.as_str(),
            head.head_index,
            head.norm
        );
    }
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

fn current_utc_timestamp_strings() -> (String, String) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let total_seconds = now.as_secs() as i64;
    let days = total_seconds.div_euclid(86_400);
    let seconds_of_day = total_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = (seconds_of_day / 3_600) as i32;
    let minute = ((seconds_of_day % 3_600) / 60) as i32;
    let second = (seconds_of_day % 60) as i32;

    let slug = format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}");
    let display = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");
    (slug, display)
}

fn format_metric_line(name: &str, history: &RollingMetric, as_percentage: bool) -> String {
    if history.len() == 0 {
        return format!("- {name}: N/A (no samples)");
    }

    match history.average() {
        Some(value) => {
            let samples = history.len();
            if as_percentage {
                format!(
                    "- {name}: {percent:.2}% (avg of {samples} updates)",
                    percent = value * 100.0,
                    samples = samples
                )
            } else {
                format!(
                    "- {name}: {metric_value:.6} (avg of {samples} updates)",
                    metric_value = value,
                    samples = samples
                )
            }
        }
        None => format!("- {name}: N/A (no samples)"),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_scoresheet(
    config: &Config,
    resume_status: &str,
    iteration: usize,
    items_processed: usize,
    train_size: usize,
    total_batches: usize,
    total_optimizer_steps: usize,
    grad_accumulation_steps: usize,
    training_duration: Duration,
    policy_history: &RollingMetric,
    value_history: &RollingMetric,
    time_history: &RollingMetric,
    top1_history: &RollingMetric,
    top5_history: &RollingMetric,
    wdl_history: &RollingMetric,
    gradient_norm_history: &RollingMetric,
) -> anyhow::Result<String> {
    let (timestamp_slug, timestamp_display) = current_utc_timestamp_strings();
    let filename = format!("sheet-{timestamp_slug}.txt");

    let metrics_section = [
        format_metric_line("Policy Loss", policy_history, false),
        format_metric_line("Value Loss", value_history, false),
        format_metric_line("Time Usage Loss", time_history, false),
        format_metric_line("Move Top-1 Accuracy", top1_history, true),
        format_metric_line("Move Top-5 Accuracy", top5_history, true),
        format_metric_line("WDL Accuracy", wdl_history, false),
        format_metric_line("Gradient Norm", gradient_norm_history, false),
    ]
    .join("\n");

    let data_path_display = config
        .data_path
        .as_ref()
        .map(|p| format!("{}", p.display()))
        .unwrap_or_else(|| "N/A".to_string());

    let batch_size_display = config
        .batch_size
        .map(|v| v.to_string())
        .unwrap_or_else(|| "auto".to_string());

    let max_samples_display = config
        .max_samples
        .map(|v| v.to_string())
        .unwrap_or_else(|| "N/A".to_string());

    let skip_display = config
        .skip
        .map(|v| v.to_string())
        .unwrap_or_else(|| "0".to_string());

    let timeout_display = config
        .timeout
        .map(|v| v.to_string())
        .unwrap_or_else(|| "N/A".to_string());

    let effective_batch_size = grad_accumulation_steps * config.physical_batch_size;
    let base_batch_size = 1024.0;
    let batch_scale = effective_batch_size as f64 / base_batch_size;

    // μP-informed dimension scaling: base LRs are defined at LR_REFERENCE_DIM (256)
    //   - AdamW hidden LR: scales as d_ref/d (decreases with width)
    //   - Muon LR: scales as sqrt(d_ref/d) (decreases slower with width)
    //   - Embedding LR: width-independent (no d-scaling)
    let d = config.embed_dim() as f64;
    let adamw_dim_scale = LR_REFERENCE_DIM / d;
    let muon_dim_scale = (LR_REFERENCE_DIM / d).sqrt();

    let initial_adamw_lr =
        config.adamw_base_lr * adamw_dim_scale * batch_scale * config.lr_multiplier;
    let initial_muon_lr = config.muon_base_lr * muon_dim_scale * batch_scale * config.lr_multiplier;
    let initial_embedding_lr = config.embedding_base_lr * batch_scale * config.lr_multiplier;
    // Use AdamW LR as the scheduler's LR (the "base" for plateau reduction)
    let initial_lr = initial_adamw_lr;
    let muon_to_adamw_lr_ratio = initial_muon_lr / initial_adamw_lr;
    let embedding_to_adamw_lr_ratio = initial_embedding_lr / initial_adamw_lr;
    let warmup_iterations = (config.warmup_multiplier * effective_batch_size as f64) as usize;

    let initial_adamw_lr =
        config.adamw_base_lr * adamw_dim_scale * batch_scale * config.lr_multiplier;
    let initial_muon_lr = config.muon_base_lr * muon_dim_scale * batch_scale * config.lr_multiplier;
    let initial_embedding_lr = config.embedding_base_lr * batch_scale * config.lr_multiplier;
    // Use AdamW LR as the scheduler's LR (the "base" for plateau reduction)
    let initial_lr = initial_adamw_lr;
    let muon_to_adamw_lr_ratio = initial_muon_lr / initial_adamw_lr;
    let embedding_to_adamw_lr_ratio = initial_embedding_lr / initial_adamw_lr;
    let warmup_iterations = (config.warmup_multiplier * effective_batch_size as f64) as usize;

    let train_size_display = if train_size == usize::MAX {
        "streaming".to_string()
    } else {
        train_size.to_string()
    };
    let total_batches_display = if total_batches == usize::MAX {
        "streaming".to_string()
    } else {
        total_batches.to_string()
    };
    let total_optimizer_steps_display = if total_optimizer_steps == usize::MAX {
        "streaming".to_string()
    } else {
        total_optimizer_steps.to_string()
    };

    let content = format!(
        "OXI Training Scoresheet\n\
Timestamp (UTC): {timestamp_display}\n\
Resume Status: {resume_status}\n\
\n\
Training Summary\n\
- Iterations processed: {iteration}\n\
- Items processed: {items_processed}\n\
- Training set size: {train_size_display}\n\
- Total batches (estimated): {total_batches_display}\n\
- Optimizer steps (estimated): {total_optimizer_steps_display}\n\
- Grad accumulation steps: {grad_accumulation_steps}\n\
- Duration (seconds): {duration:.2}\n\
\n\
Model & Data Configuration\n\
- Data path: {data_path_display}\n\
- Max samples: {max_samples}\n\
- Skip: {skip}\n\
- Timeout: {timeout}\n\
- Batch size: {batch_size_display}\n\
- Physical batch size: {physical_batch}\n\
- Num devices: {num_devices}\n\
- Train ratio: {train_ratio}\n\
- Seed: {seed}\n\
- Pretrain samples: {pretrain_samples}\n\
- Checkpoint interval: {checkpoint_interval}\n\
- Resume flag: {resume_flag}\n\
- Embed dim: {embed_dim}\n\
- Num heads: {num_heads}\n\
- Num layers: {num_layers}\n\
- MLP hidden multiplier: 2.5 (SwiGLU)\n\
- Policy loss weight: {policy_loss_weight}\n\
- Value loss weight: {value_loss_weight}\n\
- Value entropy weight: {value_entropy_weight}\n\
- Time usage loss weight: {time_usage_loss_weight}\n\
- Weight decay: {weight_decay}\n\
- Gradient clip: {gradient_clip}\n\
- AdamW base LR: {adamw_base_lr:.6} (at d_ref=256)\n\
- Muon base LR: {muon_base_lr:.6} (at d_ref=256)\n\
- Embedding base LR: {embedding_base_lr:.6} (width-independent)\n\
- Initial AdamW LR: {initial_adamw_lr:.6}\n\
- Initial Muon LR: {initial_muon_lr:.6}\n\
- Initial Embedding LR: {initial_embedding_lr:.6}\n\
- Muon/AdamW LR ratio: {muon_to_adamw_lr_ratio:.2}x\n\
- Embedding/AdamW LR ratio: {embedding_to_adamw_lr_ratio:.2}x\n\
- LR min: {lr_min}\n\
- LR multiplier: {lr_multiplier}\n\
- Warmup iterations: {warmup_iterations}\n\
- Warmup multiplier: {warmup_multiplier}\n\
- LR window size: {lr_window_size}\n\
- LR improvement threshold: {lr_improvement_threshold:.2}%\n\
- LR reduction factor: {lr_reduction_factor}\n\
\n\
Metric Averages (last min({window}, N) updates)\n{metrics_section}\n",
        duration = training_duration.as_secs_f64(),
        max_samples = max_samples_display,
        skip = skip_display,
        timeout = timeout_display,
        physical_batch = config.physical_batch_size,
        num_devices = config.num_devices,
        train_ratio = config.train_ratio,
        seed = config.seed,
        pretrain_samples = config.pretrain_samples,
        checkpoint_interval = config.checkpoint_interval,
        resume_flag = config.resume.unwrap_or(false),
        embed_dim = config.embed_dim(),
        num_heads = config.num_heads(),
        num_layers = config.num_layers(),
        policy_loss_weight = config.policy_loss_weight,
        value_loss_weight = config.value_loss_weight,
        value_entropy_weight = config.value_entropy_weight,
        time_usage_loss_weight = config.time_usage_loss_weight,
        weight_decay = config.weight_decay,
        gradient_clip = config.gradient_clip,
        adamw_base_lr = config.adamw_base_lr,
        muon_base_lr = config.muon_base_lr,
        embedding_base_lr = config.embedding_base_lr,
        initial_adamw_lr = initial_adamw_lr,
        initial_muon_lr = initial_muon_lr,
        initial_embedding_lr = initial_embedding_lr,
        muon_to_adamw_lr_ratio = muon_to_adamw_lr_ratio,
        embedding_to_adamw_lr_ratio = embedding_to_adamw_lr_ratio,
        lr_min = config.lr_min,
        lr_multiplier = config.lr_multiplier,
        warmup_iterations = warmup_iterations,
        warmup_multiplier = config.warmup_multiplier,
        lr_window_size = config.lr_window_size,
        lr_improvement_threshold = config.lr_improvement_threshold * 100.0,
        lr_reduction_factor = config.lr_reduction_factor,
        window = SCORE_WINDOW,
    );

    fs::write(&filename, content)?;
    Ok(filename)
}

pub fn init_train_logging(log_dir: Option<&Path>) -> WorkerGuard {
    let dir = log_dir.unwrap_or_else(|| Path::new("."));
    if let Some(d) = log_dir {
        let _ = std::fs::create_dir_all(d);
    }
    let file_appender = tracing_appender::rolling::never(dir, "train.log");
    let (non_blocking, worker_guard) = tracing_appender::non_blocking(file_appender);
    let subscriber = tracing_fmt()
        .with_ansi(false)
        .with_file(false)
        .with_target(false)
        .without_time()
        .with_writer(non_blocking)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);

    worker_guard
}

fn sync_backend_if_supported<B: AutodiffBackend>(device: &B::Device) {
    let backend_name = B::name(device);
    if backend_name.contains("<metal>") {
        return;
    }
    B::sync(device);
}

fn load_tcec_examples(data_path: &Path, max_count: usize) -> anyhow::Result<Vec<ChessExample>> {
    let tcec_path = data_path.join("tcec");
    if !tcec_path.exists() {
        anyhow::bail!(
            "TCEC games directory not found: {}. Run 'download-tcec' command first.",
            tcec_path.display()
        );
    }

    println!("Loading TCEC examples from {}...", tcec_path.display());

    let examples_iter = process_tcec_directory_iter(&tcec_path)?;
    let examples: Vec<ChessExample> = examples_iter.take(max_count).collect();

    println!("Loaded {} TCEC examples", examples.len());
    Ok(examples)
}

pub fn train_custom<B: AutodiffBackend>(
    config: Config,
    devices: Vec<B::Device>,
) -> anyhow::Result<()>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    println!("Using custom training loop");
    tracing::info!("Using custom training loop");

    if config.forward_timing_enabled() {
        crate::forward_timing::enable_timing();
        crate::forward_timing::set_sample_interval(config.forward_timing_interval());
        println!(
            "Forward pass timing enabled (sample interval: {})",
            config.forward_timing_interval()
        );
        tracing::info!(
            "Forward pass timing enabled (sample interval: {})",
            config.forward_timing_interval()
        );
    }

    // Create model
    let mut model: OXIModel<B> = OXIModel::new(&devices[0], &config);

    let mut resume_status = "Not requested".to_string();
    let mut resume_optimizer_dir: Option<PathBuf> = None;

    if config.resume.unwrap_or(false) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let resume_dir = Path::new(MODEL_DIR_NAME);
        let model_file = resume_dir.join(MODEL_FILE_NAME);

        if resume_dir.is_dir() && model_file.exists() {
            tracing::info!("Resuming training from {}", model_file.display());
            println!("Resuming training from {}", model_file.display());
            model = model
                .load_file(model_file.clone(), &recorder, &devices[0])
                .map_err(|err| anyhow::anyhow!(err.to_string()))?;
            resume_status = format!("Requested; loaded {}", model_file.display());
            resume_optimizer_dir = Some(resume_dir.to_path_buf());
        } else {
            let legacy_path = Path::new("model.mpk");
            if legacy_path.exists() {
                tracing::info!(
                    "Resuming training from legacy checkpoint {}",
                    legacy_path.display()
                );
                println!(
                    "Resuming training from legacy checkpoint {}",
                    legacy_path.display()
                );
                model = model
                    .load_file(legacy_path.to_path_buf(), &recorder, &devices[0])
                    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
                resume_status = format!("Requested; loaded {}", legacy_path.display());
            } else {
                tracing::warn!(
                    "Resume requested but checkpoint {} (or legacy {}) was not found; starting fresh",
                    model_file.display(),
                    legacy_path.display()
                );
                println!(
                    "Resume requested but {} (or legacy {}) was not found; starting fresh",
                    model_file.display(),
                    legacy_path.display()
                );
                resume_status = format!("Requested; missing {}", model_file.display());
            }
        }
    }

    let weight_decay_groups = WeightDecayGroups::new::<B, _>(&model);
    let (
        decay_params,
        no_decay_params,
        normal_lr_params,
        high_lr_params,
        muon_params,
        adamw_params,
    ) = weight_decay_groups.counts();

    // Create value tower gradient filter for potential value tower only training
    let value_tower_filter = ValueTowerGradientFilter::new::<B, _>(&model);
    let value_tower_param_count = value_tower_filter.count();

    // Embedding LR ratio: embedding_lr / adamw_lr (computed from initial LRs)
    // This replaces the old sqrt(embed_dim) multiplier with μP-informed separate embedding LR
    let d = config.embed_dim() as f64;
    let adamw_dim_scale = LR_REFERENCE_DIM / d;
    // lr_multiplier is now the ratio of embedding LR to AdamW LR at this d
    // (used in the optimizer step for the high-LR group)
    let lr_multiplier = (config.embedding_base_lr) / (config.adamw_base_lr * adamw_dim_scale);

    println!("\nParameter grouping summary:");
    println!(
        "  Weight decay: {} with decay, {} without decay",
        decay_params, no_decay_params
    );
    println!(
        "  Learning rate: {} normal LR, {} high LR (embeddings)",
        normal_lr_params, high_lr_params
    );
    println!(
        "  Optimizer: {} Muon (2D+ weights), {} AdamW (biases/embeds/norms/heads)",
        muon_params, adamw_params
    );
    println!(
        "  Value tower params: {} (for value tower only training)",
        value_tower_param_count
    );
    println!(
        "  Embedding LR: {:.6} (base), ratio to AdamW: {:.2}x (μP: d/d_ref={:.2}x)",
        config.embedding_base_lr,
        lr_multiplier,
        d / LR_REFERENCE_DIM,
    );
    println!("  Min LR: {:.6}", config.lr_min);
    println!("  Embedding min LR: {:.6}\n", config.lr_min * lr_multiplier);

    tracing::info!(
        "Parameter grouping: {} decay, {} no_decay; {} normal_lr, {} high_lr; {} muon, {} adamw; embedding_lr_ratio={:.4}",
        decay_params,
        no_decay_params,
        normal_lr_params,
        high_lr_params,
        muon_params,
        adamw_params,
        lr_multiplier
    );
    tracing::info!(
        "Min LR: {:.6}; Embedding min LR: {:.6}",
        config.lr_min,
        config.lr_min * lr_multiplier
    );

    // Set up streaming data loading
    let path = config.data_path.clone().expect("Data path not set");
    let data_path = Path::new(&path);

    // Pretrain phase: load TCEC (computer engine) games if configured (loaded upfront, small amount)
    let mut pretrain_batches: Vec<Vec<ChessExample>> = Vec::new();
    let num_pretrain_batches;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    if config.pretrain_samples > 0 {
        println!(
            "Loading {} TCEC samples for pretraining...",
            config.pretrain_samples
        );

        match load_tcec_examples(data_path, config.pretrain_samples) {
            Ok(mut pretrain_examples) => {
                println!(
                    "Loaded {} TCEC examples for pretraining",
                    pretrain_examples.len()
                );

                pretrain_examples.shuffle(&mut rng);

                let physical_batch_size = config.physical_batch_size;
                let num_pretrain = pretrain_examples.len();
                let average_tcec_ratio = 0.5;
                num_pretrain_batches = ((num_pretrain as f64
                    / (physical_batch_size as f64 * average_tcec_ratio))
                    .ceil() as usize)
                    .max(1);

                println!(
                    "Will create {} pretrain batches (100% -> 0% TCEC)",
                    num_pretrain_batches
                );

                let mut pretrain_iter = pretrain_examples.into_iter().peekable();

                for batch_num in 0..num_pretrain_batches {
                    let progress = batch_num as f64 / (num_pretrain_batches - 1).max(1) as f64;
                    let tcec_percentage = 1.0 - progress;
                    let num_tcec = (physical_batch_size as f64 * tcec_percentage).round() as usize;

                    let mut batch: Vec<ChessExample> =
                        pretrain_iter.by_ref().take(num_tcec).collect();
                    batch.shuffle(&mut rng);
                    pretrain_batches.push(batch);
                }

                let total_pretrain_examples: usize = pretrain_batches.iter().map(|b| b.len()).sum();
                println!(
                    "Created {} pretrain batches with {} total examples",
                    pretrain_batches.len(),
                    total_pretrain_examples
                );
            }
            Err(e) => {
                println!("Warning: Failed to load TCEC games: {}", e);
                println!("Continuing without pretraining phase");
                num_pretrain_batches = 0;
            }
        }
    } else {
        num_pretrain_batches = 0;
    }

    // Set up streaming iterator for human games
    let pgn_iter: Box<dyn Iterator<Item = ChessExample>> = if data_path.is_dir() {
        tracing::info!("Setting up streaming from PGN directory: {:?}", data_path);
        println!(
            "Streaming data from PGN directory: {:?} (shuffle buffer: {})",
            data_path, config.shuffle_buffer_size
        );
        Box::new(process_pgn_directory_iter(data_path)?)
    } else {
        anyhow::bail!(
            "Streaming mode requires a directory path, got file: {:?}",
            data_path
        );
    };

    // Set up puzzle iterator if puzzle sampling is enabled
    // In value tower only mode, skip puzzles entirely
    let puzzle_ratio = if config.skip_policy_loss() {
        println!("Skipping puzzle mixing (value tower only mode)");
        0.0
    } else {
        config.puzzle_sampling_ratio
    };
    let examples_iter: Box<dyn Iterator<Item = ChessExample>> = if puzzle_ratio > 0.0 {
        let puzzle_path = config
            .puzzle_path
            .clone()
            .unwrap_or_else(|| data_path.join("puzzles/lichess_db_puzzle.csv.zst"));

        if puzzle_path.exists() {
            println!(
                "Mixing puzzles from {:?} at {:.0}% ratio",
                puzzle_path,
                puzzle_ratio * 100.0
            );
            tracing::info!(
                "Mixing puzzles from {:?} at {:.1}% ratio",
                puzzle_path,
                puzzle_ratio * 100.0
            );

            let puzzle_iter = process_puzzle_file_iter(&puzzle_path)?;
            let mixed_rng = rand::rngs::StdRng::seed_from_u64(config.seed + 1);
            Box::new(MixedExampleIterator::new(
                pgn_iter,
                puzzle_iter,
                puzzle_ratio,
                mixed_rng,
            ))
        } else {
            println!(
                "Warning: Puzzle file not found at {:?}, proceeding without puzzles",
                puzzle_path
            );
            tracing::warn!("Puzzle file not found at {:?}", puzzle_path);
            pgn_iter
        }
    } else {
        pgn_iter
    };

    let mut shuffle_buffer = ShuffleBuffer::new(examples_iter, config.shuffle_buffer_size);

    println!(
        "Streaming mode enabled with shuffle buffer size: {}",
        config.shuffle_buffer_size
    );
    tracing::info!(
        "Streaming mode enabled with shuffle buffer size: {}",
        config.shuffle_buffer_size
    );

    println!("Pre-filling shuffle buffer...");
    shuffle_buffer.fill_buffer();
    println!(
        "Shuffle buffer initial fill: {} examples (exhausted: {})",
        shuffle_buffer.len(),
        shuffle_buffer.is_exhausted()
    );
    if shuffle_buffer.len() == 0 {
        anyhow::bail!("No examples found in shuffle buffer after initial fill. Check that your data directory contains valid PGN files.");
    }

    // For streaming, we don't know total size upfront
    let train_size: usize = config.max_samples.unwrap_or(usize::MAX);
    println!(
        "Train size: {} (streaming, max_samples limit)",
        if train_size == usize::MAX {
            "unlimited".to_string()
        } else {
            train_size.to_string()
        }
    );

    // Calculate gradient accumulation steps
    let grad_accumulation_steps = if let Some(batch_size) = config.batch_size {
        let mut steps = batch_size / config.physical_batch_size;
        steps = steps.max(1);
        println!("Gradient accumulation steps: {}", steps);
        steps
    } else {
        1
    };

    // Calculate effective batch size and initial learning rate
    // μP-informed dimension scaling: base LRs are defined at LR_REFERENCE_DIM (256)
    let effective_batch_size = grad_accumulation_steps * config.physical_batch_size;
    let base_batch_size = 1024.0;
    let batch_scale = effective_batch_size as f64 / base_batch_size;

    let d = config.embed_dim() as f64;
    let adamw_dim_scale = LR_REFERENCE_DIM / d;
    let muon_dim_scale = (LR_REFERENCE_DIM / d).sqrt();

    let initial_adamw_lr =
        config.adamw_base_lr * adamw_dim_scale * batch_scale * config.lr_multiplier;
    let initial_muon_lr = config.muon_base_lr * muon_dim_scale * batch_scale * config.lr_multiplier;
    let initial_embedding_lr = config.embedding_base_lr * batch_scale * config.lr_multiplier;
    let initial_lr = initial_adamw_lr;
    let muon_to_adamw_lr_ratio = initial_muon_lr / initial_adamw_lr;
    let embedding_to_adamw_lr_ratio = initial_embedding_lr / initial_adamw_lr;
    let warmup_iterations = (config.warmup_multiplier * effective_batch_size as f64) as usize;
    println!(
        "Effective batch size: {}, d_ref: {}, d: {}, adamw_dim_scale: {:.4}, muon_dim_scale: {:.4}",
        effective_batch_size,
        LR_REFERENCE_DIM as usize,
        config.embed_dim(),
        adamw_dim_scale,
        muon_dim_scale
    );
    println!(
        "AdamW LR: {:.6}, Muon LR: {:.6} (ratio: {:.2}x), Embedding LR: {:.6} (ratio: {:.2}x), Warmup: {}",
        initial_adamw_lr, initial_muon_lr, muon_to_adamw_lr_ratio,
        initial_embedding_lr, embedding_to_adamw_lr_ratio, warmup_iterations
    );
    println!(
        "AdamW LR: {:.6}, Muon LR: {:.6} (ratio: {:.2}x), Embedding LR: {:.6} (ratio: {:.2}x), Warmup: {}",
        initial_adamw_lr, initial_muon_lr, muon_to_adamw_lr_ratio,
        initial_embedding_lr, embedding_to_adamw_lr_ratio, warmup_iterations
    );

    // Create optimizers (5 groups: muon, adamw_decay+normal_lr, adamw_decay+high_lr, adamw_no_decay+normal_lr, adamw_no_decay+high_lr)
    let grad_clipping = if config.gradient_clip > 0.0 {
        Some(GradientClippingConfig::Norm(config.gradient_clip as f32))
    } else {
        None
    };

    let cautious = config.cautious_weight_decay.unwrap_or(true);

    // Muon optimizer for 2D+ hidden layer weight matrices
    let muon_weight_decay = if config.weight_decay > 0.0 {
        Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        ))
    } else {
        None
    };
    let muon_lr_adjust = match config.muon_lr_adjust() {
        "match_rms_adamw" => burn::optim::AdjustLrFn::MatchRmsAdamW,
        _ => burn::optim::AdjustLrFn::Original,
    };
    let mut optim_muon = MuonConfig::new()
        .with_weight_decay(muon_weight_decay)
        .with_adjust_lr_fn(muon_lr_adjust)
        .init();
    println!(
        "  Muon optimizer: enabled={}, lr_adjust={}, weight_decay={}",
        config.use_muon(),
        config.muon_lr_adjust(),
        config.weight_decay
    );

    // AdamW optimizers for everything else
    let mut optim_decay_normal = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_epsilon(config.adam_epsilon)
        .with_cautious_weight_decay(cautious)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_decay_high = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_epsilon(config.adam_epsilon)
        .with_cautious_weight_decay(cautious)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_no_decay_normal = AdamWConfig::new()
        .with_weight_decay(0.0)
        .with_epsilon(config.adam_epsilon)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_no_decay_high = AdamWConfig::new()
        .with_weight_decay(0.0)
        .with_epsilon(config.adam_epsilon)
        .with_grad_clipping(grad_clipping)
        .init();

    let mut gradnorm_state = GradNormState::new(&config);

    if let Some(resume_dir) = resume_optimizer_dir.clone() {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let device = &devices[0];

        let muon_path = resume_dir.join(OPT_MUON_FILE_NAME);
        if muon_path.exists() {
            optim_muon = load_optimizer_state(optim_muon, &recorder, muon_path, device)?;
        } else {
            println!(
                "Optimizer state {} not found; continuing with fresh state",
                OPT_MUON_FILE_NAME
            );
        }

        let decay_normal_path = resume_dir.join(OPT_DECAY_NORMAL_FILE_NAME);
        if decay_normal_path.exists() {
            optim_decay_normal =
                load_optimizer_state(optim_decay_normal, &recorder, decay_normal_path, device)?;
        } else {
            println!(
                "Optimizer state {} not found; continuing with fresh state",
                OPT_DECAY_NORMAL_FILE_NAME
            );
        }

        let decay_high_path = resume_dir.join(OPT_DECAY_HIGH_FILE_NAME);
        if decay_high_path.exists() {
            optim_decay_high =
                load_optimizer_state(optim_decay_high, &recorder, decay_high_path, device)?;
        } else {
            println!(
                "Optimizer state {} not found; continuing with fresh state",
                OPT_DECAY_HIGH_FILE_NAME
            );
        }

        let no_decay_normal_path = resume_dir.join(OPT_NO_DECAY_NORMAL_FILE_NAME);
        if no_decay_normal_path.exists() {
            optim_no_decay_normal = load_optimizer_state(
                optim_no_decay_normal,
                &recorder,
                no_decay_normal_path,
                device,
            )?;
        } else {
            println!(
                "Optimizer state {} not found; continuing with fresh state",
                OPT_NO_DECAY_NORMAL_FILE_NAME
            );
        }

        let no_decay_high_path = resume_dir.join(OPT_NO_DECAY_HIGH_FILE_NAME);
        if no_decay_high_path.exists() {
            optim_no_decay_high =
                load_optimizer_state(optim_no_decay_high, &recorder, no_decay_high_path, device)?;
        } else {
            println!(
                "Optimizer state {} not found; continuing with fresh state",
                OPT_NO_DECAY_HIGH_FILE_NAME
            );
        }

        let gradnorm_path = resume_dir.join(GRADNORM_STATE_FILE_NAME);
        if gradnorm_path.exists() {
            match std::fs::read(&gradnorm_path) {
                Ok(bytes) => {
                    match bincode::serde::decode_from_slice::<GradNormState, _>(
                        &bytes,
                        bincode::config::standard(),
                    ) {
                        Ok((mut state, _)) => {
                            println!("Loaded GradNorm state from {}", gradnorm_path.display());
                            tracing::info!(
                                "Loaded GradNorm state from {}",
                                gradnorm_path.display()
                            );
                            if !config.gradnorm_enabled() {
                                state.set_enabled(false);
                                state.reset_weights_to_config(&config);
                            }
                            gradnorm_state = state;
                        }
                        Err(err) => {
                            println!(
                                "Warning: Failed to load GradNorm state from {}: {}",
                                gradnorm_path.display(),
                                err
                            );
                            tracing::warn!(
                                "Failed to load GradNorm state from {}: {}",
                                gradnorm_path.display(),
                                err
                            );
                        }
                    }
                }
                Err(err) => {
                    println!(
                        "Warning: Failed to read GradNorm state {}: {}",
                        gradnorm_path.display(),
                        err
                    );
                    tracing::warn!(
                        "Failed to read GradNorm state {}: {}",
                        gradnorm_path.display(),
                        err
                    );
                }
            }
        } else {
            println!(
                "GradNorm state {} not found; continuing with fresh state",
                GRADNORM_STATE_FILE_NAME
            );
            tracing::info!(
                "GradNorm state {} not found; continuing with fresh state",
                GRADNORM_STATE_FILE_NAME
            );
        }
    }

    // For streaming mode, we don't know total size upfront
    let total_batches = if train_size == usize::MAX {
        usize::MAX
    } else {
        (train_size + config.physical_batch_size - 1) / config.physical_batch_size
    };
    let total_optimizer_steps = if total_batches == usize::MAX {
        usize::MAX
    } else {
        (total_batches + grad_accumulation_steps - 1) / grad_accumulation_steps
    };

    if total_batches == usize::MAX {
        println!("Streaming mode: total batches unknown (will train until data exhausted or min LR reached)");
    } else {
        println!(
            "Total batches: {}, Total optimizer steps: {}",
            total_batches, total_optimizer_steps
        );
    }
    println!(
        "AdamW LR: {:.6}, Muon LR: {:.6}, Min LR: {}",
        initial_adamw_lr, initial_muon_lr, config.lr_min
    );
    println!(
        "LR window: {}, LR improvement threshold: {:.2}%, LR reduction factor: {}",
        config.lr_window_size,
        config.lr_improvement_threshold * 100.0,
        config.lr_reduction_factor
    );

    // ==================== LR RANGE FINDER MODE ====================
    if config.lr_range_finder.unwrap_or(false) {
        let accum_steps = grad_accumulation_steps;
        let eff_batch = accum_steps * config.physical_batch_size;

        // Sweep a single multiplier applied to all configured LRs.
        // The ratios between AdamW/Muon/Embedding are fixed by config + μP d-scaling.
        // We just find the right overall magnitude.
        println!("\n=== LR RANGE FINDER MODE ===");
        println!(
            "Configured LRs: AdamW={:.2e}, Muon={:.2e}, Embedding={:.2e}",
            initial_adamw_lr, initial_muon_lr, initial_embedding_lr
        );
        println!("Sweeping a multiplier from 0.01x to 100x of configured LRs");

        let mult_min = 0.01_f64;
        let mult_max = 100.0_f64;
        let num_steps = 200;
        let log_factor = (mult_max / mult_min).ln() / num_steps as f64;
        println!(
            "Physical batch: {}, Accumulation steps: {}, Effective batch: {}",
            config.physical_batch_size, accum_steps, eff_batch
        );

        let log_dir = config
            .log_dir
            .as_deref()
            .unwrap_or(std::path::Path::new("."));
        std::fs::create_dir_all(log_dir)?;
        let tsv_path = log_dir.join("lr_range_finder.tsv");
        let mut tsv_file =
            std::fs::File::create(&tsv_path).expect("Failed to create lr_range_finder.tsv");
        use std::io::Write;
        writeln!(
            tsv_file,
            "step\tadamw_lr\tmuon_lr\tembedding_lr\tpolicy_loss\ttotal_loss\tgrad_norm"
        )
        .unwrap();

        let dataset_for_processing = OXIDataset::new(Vec::new(), config.clone());
        let main_device = devices[0].clone();
        let device_workers = DeviceWorkers::<B>::new(&model, &devices, &main_device);

        let mut best_loss = f64::MAX;
        let mut diverged = false;

        for step in 0..num_steps {
            if diverged {
                break;
            }

            let mult = mult_min * (log_factor * step as f64).exp();
            let adamw_sweep_lr = initial_adamw_lr * mult;
            let muon_sweep_lr = initial_muon_lr * mult;
            let embedding_sweep_lr = initial_embedding_lr * mult;

            // Accumulate gradients over multiple micro-batches
            let mut grad_accumulator_rf = GradientsAccumulator::new();
            let mut total_policy_loss = 0.0_f64;
            let mut total_total_loss = 0.0_f64;
            let mut micro_batches_done = 0usize;
            let mut data_exhausted = false;

            for _micro in 0..accum_steps {
                let batch_examples =
                    shuffle_buffer.sample_batch(config.physical_batch_size, &mut rng);
                if batch_examples.is_empty() {
                    data_exhausted = true;
                    break;
                }

                let items: Vec<_> = batch_examples
                    .par_iter()
                    .filter_map(|ex| dataset_for_processing.process_example(ex).ok())
                    .collect();

                if items.is_empty() {
                    continue;
                }

                // Dispatch to workers
                let device_splits = split_items_across_devices(&items, devices.len());
                let gradnorm_weights = GradNormWeights::from_state(&gradnorm_state);
                let mut active_workers_rf = Vec::new();

                for (device_index, device_items) in device_splits.into_iter().enumerate() {
                    if device_items.is_empty() {
                        continue;
                    }
                    device_workers
                        .get(device_index)
                        .send(WorkerCommand::Run(WorkerRequest {
                            items: device_items,
                            weights: gradnorm_weights,
                        }));
                    active_workers_rf.push(device_index);
                }

                // Collect grads and outputs for this micro-batch
                let mut outputs: Vec<ChessOutput<B>> = Vec::new();
                for device_index in active_workers_rf {
                    if let Some(response) = device_workers.get(device_index).recv() {
                        match response {
                            WorkerResponse::Training { grads, output } => {
                                let grads_main = grads.to_device(&devices[0], &model);
                                grad_accumulator_rf.accumulate(&model, grads_main);
                                outputs.push(output);
                            }
                            _ => {}
                        }
                    }
                }

                if !outputs.is_empty() {
                    let combined = combine_outputs(&outputs, &devices[0]);
                    total_policy_loss += combined
                        .base_policy_loss
                        .clone()
                        .into_scalar()
                        .elem::<f32>() as f64;
                    total_total_loss += combined.loss.clone().into_scalar().elem::<f32>() as f64;
                    micro_batches_done += 1;
                }
            }

            if data_exhausted && micro_batches_done == 0 {
                println!("LR range finder: data exhausted at step {}", step);
                break;
            }

            if micro_batches_done == 0 {
                continue;
            }

            // Average losses over micro-batches
            let policy_loss = (total_policy_loss / micro_batches_done as f64) as f32;
            let total_loss = (total_total_loss / micro_batches_done as f64) as f32;

            let grads = grad_accumulator_rf.grads();
            let grad_breakdown = compute_gradient_norm_with_breakdown(&grads, &model, false);
            let grad_norm = grad_breakdown.total();

            // Split and step
            let split = weight_decay_groups.split_grads::<B, _>(&model, grads);
            model = optim_muon.step(muon_sweep_lr, model, split.muon);
            model = optim_decay_normal.step(adamw_sweep_lr, model, split.adamw_decay_normal);
            model = optim_decay_high.step(embedding_sweep_lr, model, split.adamw_decay_high);
            model = optim_no_decay_normal.step(adamw_sweep_lr, model, split.adamw_no_decay_normal);
            model = optim_no_decay_high.step(embedding_sweep_lr, model, split.adamw_no_decay_high);
            device_workers.broadcast_model(&model);

            writeln!(
                tsv_file,
                "{}\t{:.8}\t{:.8}\t{:.8}\t{:.6}\t{:.6}\t{:.6}",
                step,
                adamw_sweep_lr,
                muon_sweep_lr,
                embedding_sweep_lr,
                policy_loss,
                total_loss,
                grad_norm
            )
            .unwrap();

            if step % 20 == 0 {
                println!(
                    "  step={:>3} adamw={:.2e} muon={:.2e} embed={:.2e} policy={:.4} total={:.4} grad={:.4} ({}x{})",
                    step, adamw_sweep_lr, muon_sweep_lr, embedding_sweep_lr, policy_loss, total_loss, grad_norm,
                    micro_batches_done, config.physical_batch_size
                );
            }

            // Track divergence
            if (total_loss as f64) < best_loss {
                best_loss = total_loss as f64;
            }
            if total_loss > 4.0 * best_loss as f32
                || total_loss.is_nan()
                || total_loss.is_infinite()
            {
                println!(
                    "  DIVERGED at step={} adamw_lr={:.2e} muon_lr={:.2e} loss={:.4} (best was {:.4})",
                    step, adamw_sweep_lr, muon_sweep_lr, total_loss, best_loss
                );
                diverged = true;
            }
        }

        println!(
            "\nLR range finder complete. Results saved to: {}",
            tsv_path.display()
        );
        println!(
            "Effective batch size: {} ({}x{})",
            eff_batch, accum_steps, config.physical_batch_size
        );
        println!("Best loss seen: {:.4}", best_loss);
        println!("Look for the LR region with steepest loss decrease (just before divergence).");

        // Shut down workers
        device_workers.shutdown();

        return Ok(());
    }
    // ==================== END LR RANGE FINDER ====================

    let mut lr_scheduler = ReduceOnPlateauScheduler::with_cosine(
        initial_lr,
        config.lr_min,
        config.lr_reduction_factor,
        config.lr_window_size,
        config.lr_improvement_threshold,
        warmup_iterations,
        config.cosine_period,
    );

    let mut loss_metric = LossMetric::new();
    let mut policy_loss_metric = PolicyLossMetric::new();
    let mut value_loss_metric = ValueLossMetric::new();
    let mut time_usage_loss_metric = TimeUsageLossMetric::new();
    let mut move_top1_metric = MoveTopKAccuracyMetric::new(1);
    let mut move_top5_metric = MoveTopKAccuracyMetric::new(5);
    let mut wdl_accuracy_metric = WdlAccuracyMetric::new();
    let mut gradient_norm_metric = GradientNormMetric::<B>::new();
    let mut policy_history = RollingMetric::new(SCORE_WINDOW);
    let mut value_history = RollingMetric::new(SCORE_WINDOW);
    let mut time_history = RollingMetric::new(SCORE_WINDOW);
    let mut top1_history = RollingMetric::new(SCORE_WINDOW);
    let mut top5_history = RollingMetric::new(SCORE_WINDOW);
    let mut wdl_history = RollingMetric::new(SCORE_WINDOW);
    let mut gradient_norm_history = RollingMetric::new(SCORE_WINDOW);
    let mut iteration_speed_metric = IterationSpeedMetric::new();
    let mut lr_metric = LrPlateauMetric::new();
    let mut stage_metric = TrainingStageMetric::new();
    let metric_logger = MetricFileLogger::new(config.log_dir.as_deref());

    println!(
        "Starting training for {} batches (checkpoint every {} items)",
        total_batches, config.checkpoint_interval
    );

    let start_time = Instant::now();
    let mut iteration = 0;
    let mut grad_accumulator = GradientsAccumulator::new();
    let mut accumulation_count = 0;
    let mut items_processed = 0;
    let mut current_lr;
    let mut optimizer_step = 0usize;

    // Create checkpoints directory
    std::fs::create_dir_all("checkpoints")?;

    // Process examples into ChessItems and chunk into batches
    // Only need config for process_example, not the examples themselves
    let dataset_for_processing = OXIDataset::new(Vec::new(), config.clone());
    let mut debug_monitor =
        DebugPredictionMonitor::new(&dataset_for_processing, devices[0].clone())?;

    // Create interrupter and renderer
    let interruptor = Interrupter::new();
    let main_device = devices[0].clone();
    let device_workers = DeviceWorkers::<B>::new(&model, &devices, &main_device);

    let mut renderer: Box<dyn MetricsRenderer> = if config.disable_tui.unwrap_or(false) {
        println!("TUI disabled, using CLI renderer");
        Box::new(SimpleCliRenderer::new())
    } else {
        println!("Using TUI renderer");
        Box::new(OxiTuiRenderer::new(
            interruptor.clone(),
            total_batches,
            train_size,
        ))
    };

    let pretrain_batch_count = pretrain_batches.len();
    let mut pretrain_batch_iter = pretrain_batches.into_iter();
    let max_samples = config.max_samples.unwrap_or(usize::MAX);

    // Track if we're in value tower only mode (skip_policy_loss or after main training completes)
    let mut value_tower_only_mode = config.skip_policy_loss();
    if value_tower_only_mode {
        println!("VALUE TOWER ONLY MODE: Enabled via skip_policy_loss flag");
        println!("  - Only value tower parameters will be updated");
        println!("  - Policy loss weight set to 0");
        println!("  - Puzzles disabled");
        println!("  - LR reset to initial value");
        tracing::info!(
            "Value tower only mode enabled: skip_policy_loss={}, value_tower_params={}",
            config.skip_policy_loss(),
            value_tower_param_count
        );
    }

    println!(
        "Starting training loop: {} pretrain batches, shuffle buffer: {} examples",
        pretrain_batch_count,
        shuffle_buffer.len()
    );
    tracing::info!(
        "Starting training loop: {} pretrain batches, shuffle buffer: {} examples",
        pretrain_batch_count,
        shuffle_buffer.len()
    );

    loop {
        let loop_iteration = iteration + 1;
        tracing::debug!(
            "Loop iteration {} start: items_processed={}, max_samples={}, buffer_len={}, buffer_exhausted={}",
            loop_iteration, items_processed, max_samples, shuffle_buffer.len(), shuffle_buffer.is_exhausted()
        );

        if interruptor.should_stop() {
            println!(
                "Training interrupted by user (iteration {})",
                loop_iteration
            );
            tracing::info!(
                "EXIT_CONDITION: user_interrupt at iteration {} (items_processed={}, buffer_len={}, buffer_exhausted={})",
                loop_iteration, items_processed, shuffle_buffer.len(), shuffle_buffer.is_exhausted()
            );
            break;
        }

        if lr_scheduler.should_stop() {
            if value_tower_only_mode {
                // Already in value tower only mode, stop training
                println!(
                    "Training stopped: reached min LR in value tower only mode with <{:.2}% improvement",
                    config.lr_improvement_threshold * 100.0
                );
                tracing::info!(
                    "EXIT_CONDITION: lr_min_reached_value_tower at iteration {} (current_lr={:.2e}, min_lr={:.2e})",
                    loop_iteration, lr_scheduler.get_lr(), config.lr_min
                );
                break;
            } else {
                // Transition from main training to value tower only mode
                tracing::info!("========================================");
                tracing::info!("TRANSITIONING TO VALUE TOWER ONLY MODE");
                tracing::info!("========================================");
                tracing::info!(
                    "Main training reached min LR with <{:.2}% improvement",
                    config.lr_improvement_threshold * 100.0
                );
                tracing::info!("Now training only value tower parameters");
                tracing::info!("  - Resetting LR to initial value");
                tracing::info!("  - Setting policy loss weight to 0");
                tracing::info!("  - Only updating value tower params");
                tracing::info!("========================================");
                tracing::info!(
                    "STAGE_TRANSITION: main_to_value_tower at iteration {} (lr was {:.2e}, resetting to {:.2e})",
                    loop_iteration, lr_scheduler.get_lr(), initial_lr
                );

                value_tower_only_mode = true;
                lr_scheduler.reset_to_initial();

                // Continue training instead of breaking
                continue;
            }
        }

        if items_processed >= max_samples {
            println!(
                "Training stopped: reached max_samples limit ({})",
                max_samples
            );
            tracing::info!(
                "EXIT_CONDITION: max_samples_reached at iteration {} (items_processed={}, max_samples={})",
                loop_iteration, items_processed, max_samples
            );
            break;
        }

        let t_full_iteration_start = Instant::now();

        let (batch_examples, batch_source): (Vec<ChessExample>, &str) =
            if let Some(pretrain_batch) = pretrain_batch_iter.next() {
                // If pretrain batch is empty (e.g., last batch with 0% TCEC), fall through to shuffle_buffer
                if pretrain_batch.is_empty() {
                    (
                        shuffle_buffer.sample_batch(config.physical_batch_size, &mut rng),
                        "shuffle_buffer",
                    )
                } else {
                    (pretrain_batch, "pretrain")
                }
            } else {
                (
                    shuffle_buffer.sample_batch(config.physical_batch_size, &mut rng),
                    "shuffle_buffer",
                )
            };

        if batch_examples.is_empty() {
            println!(
                "Training stopped: data exhausted (source: {}, shuffle_buffer len: {}, exhausted: {})",
                batch_source,
                shuffle_buffer.len(),
                shuffle_buffer.is_exhausted()
            );
            tracing::info!(
                "EXIT_CONDITION: data_exhausted at iteration {} (source={}, buffer_len={}, buffer_exhausted={})",
                loop_iteration, batch_source, shuffle_buffer.len(), shuffle_buffer.is_exhausted()
            );
            break;
        }

        iteration += 1;
        tracing::info!(
            "Starting iteration {} with {} examples from {}",
            iteration,
            batch_examples.len(),
            batch_source
        );

        let t_process_start = Instant::now();
        let items_all: Vec<_> = batch_examples
            .par_iter()
            .filter_map(|example| dataset_for_processing.process_example(example).ok())
            .collect();
        let process_time = t_process_start.elapsed();

        if items_all.is_empty() {
            continue;
        }

        let current_batch_size = items_all.len();
        items_processed += current_batch_size;

        // Split batch across devices for parallel execution
        let device_splits = split_items_across_devices(&items_all, devices.len());
        let gradnorm_weights = if value_tower_only_mode {
            GradNormWeights::for_value_tower_only(&gradnorm_state)
        } else {
            GradNormWeights::from_state(&gradnorm_state)
        };
        let mut active_workers = Vec::new();

        for (device_index, device_items) in device_splits.into_iter().enumerate() {
            if device_items.is_empty() {
                continue;
            }

            let request = WorkerRequest {
                items: device_items,
                weights: gradnorm_weights,
            };

            device_workers
                .get(device_index)
                .send(WorkerCommand::Run(request));
            active_workers.push(device_index);
        }

        let mut device_outputs: Vec<ChessOutput<B>> = Vec::new();

        let t_worker_wait_start = Instant::now();
        for worker_index in active_workers {
            if let Some(response) = device_workers.get(worker_index).recv() {
                match response {
                    WorkerResponse::Training { grads, output } => {
                        let grads_main = grads.to_device(&devices[0], &model);
                        grad_accumulator.accumulate(&model, grads_main);
                        device_outputs.push(output);
                    }
                    WorkerResponse::GradNormProbe(_) => {
                        // Probe responses should not be received during the training phase.
                        continue;
                    }
                }
            }
        }
        let worker_wait_time = t_worker_wait_start.elapsed();

        if device_outputs.is_empty() {
            continue;
        }
        accumulation_count += 1;

        // Combine outputs back on the main device for logging/metrics
        let t_combine_start = Instant::now();
        let output = combine_outputs(&device_outputs, &devices[0]);
        drop(device_outputs);
        let combine_time = t_combine_start.elapsed();

        let t_sync_start = Instant::now();
        sync_backend_if_supported::<B>(&devices[0]);
        #[cfg(not(feature = "backend-cuda"))]
        B::memory_cleanup(&devices[0]);
        let sync_time = t_sync_start.elapsed();

        tracing::info!(
            "perf_main_loop: iter={} process_examples={:?} worker_wait={:?} combine={:?} sync={:?}",
            iteration,
            process_time,
            worker_wait_time,
            combine_time,
            sync_time
        );

        let t_post_main_loop = Instant::now();
        gradnorm_state.record_batch_losses(iteration, &output);

        // Record batch in ReduceOnPlateau scheduler:
        // - Main training: use raw policy loss (so GradNorm weighting on other heads cannot prematurely trigger LR reductions)
        // - Value tower only: use value loss (since policy loss is 0)
        let plateau_loss = if value_tower_only_mode {
            let value_loss_tensor = output.base_value_loss.clone();
            let loss = value_loss_tensor.into_scalar().elem::<f32>() as f64;
            if iteration % 100 == 0 {
                tracing::info!(
                    "Value tower plateau tracking: iteration={}, value_loss={:.6}",
                    iteration,
                    loss
                );
            }
            loss
        } else {
            let policy_loss_tensor = output
                .raw_policy_loss
                .clone()
                .unwrap_or_else(|| output.base_policy_loss.clone());
            policy_loss_tensor.into_scalar().elem::<f32>() as f64
        };
        metric_logger.log("plateau_loss", iteration, plateau_loss);
        let _measurement_recorded = lr_scheduler.record_batch(plateau_loss);

        // Get current learning rate from scheduler
        current_lr = lr_scheduler.get_lr();

        // Update metrics metadata
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed,
                items_total: train_size,
            },
            epoch: 1,
            epoch_total: 1,
            iteration,
            lr: Some(current_lr),
        };

        // Update model when accumulation is complete
        let should_update =
            accumulation_count >= grad_accumulation_steps || iteration == total_batches;
        let should_compute_full_metrics = config
            .full_metrics_interval()
            .map_or(false, |interval| iteration % interval == 0);
        let t_optimizer_start = Instant::now();
        let mut grad_norm_compute_time = std::time::Duration::ZERO;
        let mut split_grads_time = std::time::Duration::ZERO;
        let mut optim_step_time = std::time::Duration::ZERO;
        if should_update {
            let grads = grad_accumulator.grads();
            let next_step = optimizer_step + 1;

            let t_split = Instant::now();
            // In value tower only mode, filter gradients to only value tower params
            let grads_to_split = if value_tower_only_mode {
                let filtered = value_tower_filter.filter_grads::<B, _>(&model, grads);
                tracing::debug!(
                    "Value tower only: filtered grads count = {}",
                    filtered.len()
                );
                filtered
            } else {
                grads
            };
            split_grads_time = t_split.elapsed();

            // Compute gradient norm on the gradients we're actually using
            let t_grad_norm = Instant::now();
            let need_breakdown = config.log_gradient_breakdown() && should_compute_full_metrics;
            let gradient_breakdown =
                compute_gradient_norm_with_breakdown(&grads_to_split, &model, need_breakdown);
            grad_norm_compute_time = t_grad_norm.elapsed();
            let gradient_norm_value = gradient_breakdown.total();

            if value_tower_only_mode && next_step % 100 == 0 {
                tracing::info!(
                    "Value tower only step {}: grad_norm={:.6e}, grads_count={}",
                    next_step,
                    gradient_norm_value,
                    grads_to_split.len()
                );
            }

            if need_breakdown {
                let bd_adamw_lr = current_lr;
                let bd_muon_lr = current_lr * muon_to_adamw_lr_ratio;
                let bd_embedding_lr = current_lr * lr_multiplier;
                log_gradient_breakdown(
                    &gradient_breakdown,
                    &config,
                    next_step,
                    bd_muon_lr,
                    bd_adamw_lr,
                    bd_embedding_lr,
                );
            }

            // Compute and log L2 penalty from weight decay at full metrics interval
            if should_compute_full_metrics {
                let l2_penalty =
                    weight_decay_groups.compute_l2_penalty::<B, _>(&model, config.weight_decay);
                tracing::info!(
                    target: "weight_decay",
                    "step={} l2_penalty={:.6} weight_decay={:.6}",
                    next_step,
                    l2_penalty,
                    config.weight_decay
                );
            }

            let split = weight_decay_groups.split_grads::<B, _>(&model, grads_to_split);

            // Log per-optimizer-group gradient counts every 50 iterations
            if next_step % 50 == 0 {
                tracing::info!(
                    "optimizer_groups: step={} muon_grads={} adamw_decay_normal={} adamw_decay_high={} adamw_no_decay_normal={} adamw_no_decay_high={}",
                    next_step,
                    split.muon.len(),
                    split.adamw_decay_normal.len(),
                    split.adamw_decay_high.len(),
                    split.adamw_no_decay_normal.len(),
                    split.adamw_no_decay_high.len()
                );
            }

            let grad_norm_input = GradientNormInput::new(gradient_norm_value);
            let grad_norm_entry = gradient_norm_metric.update(&grad_norm_input, &metadata);
            let grad_norm_numeric = Numeric::value(&gradient_norm_metric);
            if let Some(value) = numeric_entry_value(&grad_norm_numeric) {
                gradient_norm_history.push(value);
            }
            renderer.update_train(MetricState::Numeric {
                name: gradient_norm_metric.name().to_string(),
                entry: grad_norm_entry,
                value: grad_norm_numeric,
            });

            // Skip GradNorm probing in value tower only mode (only one loss is active)
            if !value_tower_only_mode && gradnorm_state.should_update_weights(next_step) {
                let probe_items = sample_gradnorm_items(&items_all, config.gradnorm_probe_size());
                if !probe_items.is_empty() {
                    let probe_device_index = if devices.len() > 1 { 1 } else { 0 };
                    let probe_request = GradNormProbeRequest {
                        items: probe_items,
                        weights: GradNormWeights::from_state(&gradnorm_state),
                    };
                    device_workers
                        .get(probe_device_index)
                        .send(WorkerCommand::GradNormProbe(probe_request));

                    if let Some(response) = device_workers.get(probe_device_index).recv() {
                        if let WorkerResponse::GradNormProbe(results) = response {
                            let _ = gradnorm_state.apply_probe_results(next_step, &results);
                        }
                    }
                }
            }

            // Apply different learning rates per optimizer group
            let adamw_lr = current_lr;
            let muon_lr = current_lr * muon_to_adamw_lr_ratio;
            let high_lr = current_lr * lr_multiplier;

            // Log learning rates periodically
            if next_step % 100 == 0 {
                tracing::info!(
                    "step={} adamw_lr={:.6} muon_lr={:.6} high_lr={:.6} (embed_mult={:.4})",
                    next_step,
                    adamw_lr,
                    muon_lr,
                    high_lr,
                    lr_multiplier
                );
            }

            let t_optim_step = Instant::now();
            model = optim_muon.step(muon_lr, model, split.muon);
            model = optim_decay_normal.step(adamw_lr, model, split.adamw_decay_normal);
            model = optim_decay_high.step(high_lr, model, split.adamw_decay_high);
            model = optim_no_decay_normal.step(adamw_lr, model, split.adamw_no_decay_normal);
            model = optim_no_decay_high.step(high_lr, model, split.adamw_no_decay_high);
            optim_step_time = t_optim_step.elapsed();

            device_workers.broadcast_model(&model);

            accumulation_count = 0;
            optimizer_step += 1;
        }
        let optimizer_time = t_optimizer_start.elapsed();

        tracing::info!(
            "perf_optimizer_breakdown: iter={} grad_norm_compute={:?} split_grads={:?} optim_step={:?} total={:?}",
            iteration,
            grad_norm_compute_time,
            split_grads_time,
            optim_step_time,
            optimizer_time
        );

        let t_metrics_start = Instant::now();
        // Update each metric and send to renderer
        let gradnorm_snapshot = gradnorm_state.status_snapshot();

        let t_loss = Instant::now();
        let loss_entry = loss_metric.update(&output.adapt(), &metadata);
        let loss_value = Numeric::value(&loss_metric);
        if let Some(loss) = numeric_entry_value(&loss_value) {
            metric_logger.log("total_loss", iteration, loss);
        }
        renderer.update_train(MetricState::Numeric {
            name: loss_metric.name().to_string(),
            entry: loss_entry,
            value: loss_value.clone(),
        });
        let loss_metric_time = t_loss.elapsed();

        if should_compute_full_metrics {
            let raw_loss = numeric_entry_raw_value(&loss_value);
            if !raw_loss.is_finite() {
                println!(
                    "Training stopped: NaN/Inf detected in loss at iteration {} (loss={})",
                    iteration, raw_loss
                );
                tracing::error!(
                    "EXIT_CONDITION: nan_detected at iteration {} (loss={})",
                    iteration,
                    raw_loss
                );
                break;
            }
        }

        let t_policy = Instant::now();
        let mut policy_input: PolicyLossInput<B> = output.adapt();
        if let Some(status) = gradnorm_snapshot
            .iter()
            .find(|status| status.task == GradNormTask::Policy)
        {
            policy_input = policy_input.with_grad_info(status.weight, status.last_grad_norm);
        }
        let policy_entry = policy_loss_metric.update(&policy_input, &metadata);
        let policy_value = Numeric::value(&policy_loss_metric);
        if let Some(value) = numeric_entry_value(&policy_value) {
            policy_history.push(value);
            metric_logger.log("policy_loss", iteration, value);
        }
        renderer.update_train(MetricState::Numeric {
            name: policy_loss_metric.name().to_string(),
            entry: policy_entry,
            value: policy_value,
        });
        let policy_metric_time = t_policy.elapsed();

        let t_value = Instant::now();
        let mut value_input: ValueLossInput<B> = output.adapt();
        if let Some(status) = gradnorm_snapshot
            .iter()
            .find(|status| status.task == GradNormTask::Value)
        {
            value_input = value_input.with_grad_info(status.weight, status.last_grad_norm);
        }
        let value_entry = value_loss_metric.update(&value_input, &metadata);
        let value_value = Numeric::value(&value_loss_metric);
        if let Some(value) = numeric_entry_value(&value_value) {
            value_history.push(value);
            metric_logger.log("value_loss", iteration, value);
        }
        renderer.update_train(MetricState::Numeric {
            name: value_loss_metric.name().to_string(),
            entry: value_entry,
            value: value_value,
        });
        let value_metric_time = t_value.elapsed();

        let t_time_usage = Instant::now();
        let mut time_input: TimeUsageLossInput<B> = output.adapt();
        if let Some(status) = gradnorm_snapshot
            .iter()
            .find(|status| status.task == GradNormTask::TimeUsage)
        {
            time_input = time_input.with_grad_info(status.weight, status.last_grad_norm);
        }
        let _time_entry = time_usage_loss_metric.update(&time_input, &metadata);
        let time_value = Numeric::value(&time_usage_loss_metric);
        if let Some(value) = numeric_entry_value(&time_value) {
            time_history.push(value);
            metric_logger.log("time_usage_loss", iteration, value);
        }
        let time_usage_metric_time = t_time_usage.elapsed();

        // Aux head metrics (mobility + material)
        if output.aux_mobility_loss > 0.0 || output.aux_material_loss > 0.0 {
            let mob_loss = output.aux_mobility_loss as f64;
            let mat_loss = output.aux_material_loss as f64;
            renderer.update_train(MetricState::Numeric {
                name: "Aux Loss|Mobility".to_string(),
                entry: SerializedEntry::new(
                    format!("Mobility MSE: {mob_loss:.6}"),
                    format!("{mob_loss:.6}"),
                ),
                value: NumericEntry::Value(mob_loss),
            });
            renderer.update_train(MetricState::Numeric {
                name: "Aux Loss|Material".to_string(),
                entry: SerializedEntry::new(
                    format!("Material MSE: {mat_loss:.6}"),
                    format!("{mat_loss:.6}"),
                ),
                value: NumericEntry::Value(mat_loss),
            });
            metric_logger.log("aux_mobility_loss", iteration, mob_loss);
            metric_logger.log("aux_material_loss", iteration, mat_loss);

            let mob_mae = output.aux_mobility_mae as f64;
            let mat_mae = output.aux_material_mae as f64;
            renderer.update_train(MetricState::Numeric {
                name: "Aux Head MAE|Mobility".to_string(),
                entry: SerializedEntry::new(
                    format!("Mobility MAE: {mob_mae:.4}"),
                    format!("{mob_mae:.4}"),
                ),
                value: NumericEntry::Value(mob_mae),
            });
            renderer.update_train(MetricState::Numeric {
                name: "Aux Head MAE|Material".to_string(),
                entry: SerializedEntry::new(
                    format!("Material MAE: {mat_mae:.4}"),
                    format!("{mat_mae:.4}"),
                ),
                value: NumericEntry::Value(mat_mae),
            });
            metric_logger.log("aux_mobility_mae", iteration, mob_mae);
            metric_logger.log("aux_material_mae", iteration, mat_mae);
        }

        // Maia 2-style auxiliary metrics (side info, from/to square)
        if output.aux_side_info_loss > 0.0
            || output.aux_from_square_loss > 0.0
            || output.aux_to_square_loss > 0.0
        {
            let si_loss = output.aux_side_info_loss as f64;
            let from_loss = output.aux_from_square_loss as f64;
            let to_loss = output.aux_to_square_loss as f64;
            renderer.update_train(MetricState::Numeric {
                name: "Aux Loss|Side Info".to_string(),
                entry: SerializedEntry::new(
                    format!("Side Info BCE: {si_loss:.6}"),
                    format!("{si_loss:.6}"),
                ),
                value: NumericEntry::Value(si_loss),
            });
            renderer.update_train(MetricState::Numeric {
                name: "Aux Loss|From Sq".to_string(),
                entry: SerializedEntry::new(
                    format!("From Sq BCE: {from_loss:.6}"),
                    format!("{from_loss:.6}"),
                ),
                value: NumericEntry::Value(from_loss),
            });
            renderer.update_train(MetricState::Numeric {
                name: "Aux Loss|To Sq".to_string(),
                entry: SerializedEntry::new(
                    format!("To Sq BCE: {to_loss:.6}"),
                    format!("{to_loss:.6}"),
                ),
                value: NumericEntry::Value(to_loss),
            });
            metric_logger.log("aux_side_info_loss", iteration, si_loss);
            metric_logger.log("aux_from_square_loss", iteration, from_loss);
            metric_logger.log("aux_to_square_loss", iteration, to_loss);

            let from_acc = output.aux_from_square_accuracy as f64;
            let to_acc = output.aux_to_square_accuracy as f64;
            renderer.update_train(MetricState::Numeric {
                name: "Aux Accuracy|From Sq".to_string(),
                entry: SerializedEntry::new(
                    format!("From Sq Acc: {from_acc:.4}"),
                    format!("{from_acc:.4}"),
                ),
                value: NumericEntry::Value(from_acc),
            });
            renderer.update_train(MetricState::Numeric {
                name: "Aux Accuracy|To Sq".to_string(),
                entry: SerializedEntry::new(
                    format!("To Sq Acc: {to_acc:.4}"),
                    format!("{to_acc:.4}"),
                ),
                value: NumericEntry::Value(to_acc),
            });
            metric_logger.log("aux_from_square_accuracy", iteration, from_acc);
            metric_logger.log("aux_to_square_accuracy", iteration, to_acc);
        }

        let t_top1 = Instant::now();
        let _move_top1_entry = move_top1_metric.update(&output.adapt(), &metadata);
        let top1_metric_time = t_top1.elapsed();

        let t_elo_breakdown = Instant::now();
        if let (Ok(predicted_indices), Ok(target_indices)) = (
            output
                .policy_output
                .clone()
                .argmax(1)
                .squeeze_dim::<1>(1)
                .into_data()
                .convert::<i32>()
                .to_vec::<i32>(),
            output
                .policy_targets
                .clone()
                .into_data()
                .convert::<i32>()
                .to_vec::<i32>(),
        ) {
            if predicted_indices.len() == target_indices.len()
                && predicted_indices.len() == items_all.len()
            {
                let mut elo_counters: HashMap<EloBucket, AccuracyCounter> = HashMap::new();
                let mut stage_counters: HashMap<GameStageBucket, AccuracyCounter> = HashMap::new();
                let mut puzzle_counter = AccuracyCounter::default();
                let mut human_games_counter = AccuracyCounter::default();

                for (idx, item) in items_all.iter().enumerate() {
                    let correct = predicted_indices[idx] == target_indices[idx];
                    if item.is_puzzle {
                        puzzle_counter.update(correct);
                    } else {
                        human_games_counter.update(correct);
                        if let Some(bucket) = categorize_elo(item.elo_self) {
                            elo_counters.entry(bucket).or_default().update(correct);
                        }
                    }
                    let stage_bucket = categorize_stage(item.global_features.move_count);
                    stage_counters
                        .entry(stage_bucket)
                        .or_default()
                        .update(correct);
                }

                if let Some(accuracy) = human_games_counter.accuracy() {
                    let metric_name = "Move Top-1 Accuracy".to_string();
                    let entry = SerializedEntry::new(
                        format!(
                            "Top-1: {:.1}% ({}/{})",
                            accuracy * 100.0,
                            human_games_counter.correct,
                            human_games_counter.total
                        ),
                        format!("{accuracy:.4}"),
                    );
                    renderer.update_train(MetricState::Numeric {
                        name: metric_name.clone(),
                        entry,
                        value: NumericEntry::Value(accuracy),
                    });
                    top1_history.push(accuracy);
                    metric_logger.log("top1_accuracy", iteration, accuracy);
                }

                if let Some(accuracy) = puzzle_counter.accuracy() {
                    let metric_name = "Puzzle Solve Rate".to_string();
                    let entry = SerializedEntry::new(
                        format!(
                            "Puzzle: {:.1}% ({}/{})",
                            accuracy * 100.0,
                            puzzle_counter.correct,
                            puzzle_counter.total
                        ),
                        format!("{accuracy:.4}"),
                    );
                    renderer.update_train(MetricState::Numeric {
                        name: metric_name,
                        entry,
                        value: NumericEntry::Value(accuracy),
                    });
                    metric_logger.log("puzzle_solve_rate", iteration, accuracy);
                }

                for bucket in ELO_BUCKETS {
                    if let Some(counter) = elo_counters.get(&bucket) {
                        if let Some(accuracy) = counter.accuracy() {
                            let metric_name =
                                format!("Move Top-1 Accuracy by Elo|{}", bucket.label());
                            let entry = SerializedEntry::new(
                                format!(
                                    "{}: {:.1}% ({}/{})",
                                    bucket.label(),
                                    accuracy * 100.0,
                                    counter.correct,
                                    counter.total
                                ),
                                format!("{accuracy:.4}"),
                            );
                            renderer.update_train(MetricState::Numeric {
                                name: metric_name,
                                entry,
                                value: NumericEntry::Value(accuracy),
                            });
                        }
                    }
                }

                for bucket in GAME_STAGE_BUCKETS {
                    if let Some(counter) = stage_counters.get(&bucket) {
                        if let Some(accuracy) = counter.accuracy() {
                            let metric_name =
                                format!("Move Top-1 Accuracy by Game Stage|{}", bucket.label());
                            let entry = SerializedEntry::new(
                                format!(
                                    "{}: {:.1}% ({}/{})",
                                    bucket.label(),
                                    accuracy * 100.0,
                                    counter.correct,
                                    counter.total
                                ),
                                format!("{accuracy:.4}"),
                            );
                            renderer.update_train(MetricState::Numeric {
                                name: metric_name,
                                entry,
                                value: NumericEntry::Value(accuracy),
                            });
                        }
                    }
                }
            }
        }
        let elo_breakdown_time = t_elo_breakdown.elapsed();

        let t_top5 = Instant::now();
        if should_compute_full_metrics {
            let move_top5_entry = move_top5_metric.update(&output.adapt(), &metadata);
            let move_top5_value = Numeric::value(&move_top5_metric);
            if let Some(value) = numeric_entry_value(&move_top5_value) {
                top5_history.push(value);
            }
            renderer.update_train(MetricState::Numeric {
                name: move_top5_metric.name().to_string(),
                entry: move_top5_entry,
                value: move_top5_value,
            });
        }
        let top5_metric_time = t_top5.elapsed();

        let t_wdl = Instant::now();
        let wdl_acc_entry = wdl_accuracy_metric.update(&output.adapt(), &metadata);
        let wdl_acc_value = Numeric::value(&wdl_accuracy_metric);
        if let Some(value) = numeric_entry_value(&wdl_acc_value) {
            wdl_history.push(value);
        }
        renderer.update_train(MetricState::Numeric {
            name: wdl_accuracy_metric.name().to_string(),
            entry: wdl_acc_entry,
            value: wdl_acc_value,
        });
        let wdl_metric_time = t_wdl.elapsed();

        let t_misc = Instant::now();
        let iteration_speed_entry = iteration_speed_metric.update(&output.adapt(), &metadata);
        renderer.update_train(MetricState::Generic {
            name: iteration_speed_metric.name().to_string(),
            entry: iteration_speed_entry,
        });

        let lr_plateau_input = LrPlateauInput::new(
            current_lr,
            lr_scheduler.relative_improvement(),
            lr_scheduler.window_fill_ratio(),
            lr_scheduler.num_reductions(),
            config.lr_improvement_threshold,
            lr_scheduler.is_warming_up(),
            lr_scheduler.warmup_progress(),
        );
        let lr_entry = lr_metric.update(&lr_plateau_input, &metadata);
        let lr_value = Numeric::value(&lr_metric);
        renderer.update_train(MetricState::Numeric {
            name: lr_metric.name().to_string(),
            entry: lr_entry,
            value: lr_value,
        });

        // Determine if we're in the pretrain phase (using mixed batches with easy examples)
        let is_in_pretrain_phase = iteration <= num_pretrain_batches && !value_tower_only_mode;

        let stage_input = TrainingStageInput {
            stage: if value_tower_only_mode {
                TrainingStage::ValueTowerOnly
            } else if is_in_pretrain_phase {
                let progress =
                    ((iteration - 1) as f64 / (num_pretrain_batches - 1).max(1) as f64).min(1.0);
                let tcec_percentage = 1.0 - progress;

                TrainingStage::Pretrain {
                    iteration,
                    total: num_pretrain_batches,
                    tcec_percentage,
                }
            } else {
                TrainingStage::MainTraining
            },
        };
        let stage_entry = stage_metric.update(&stage_input, &metadata);
        renderer.update_train(MetricState::Generic {
            name: stage_metric.name().to_string(),
            entry: stage_entry,
        });
        let misc_metric_time = t_misc.elapsed();

        let metrics_time = t_metrics_start.elapsed();

        tracing::info!(
            "perf_metrics_breakdown: iter={} loss={:?} policy={:?} value={:?} time_usage={:?} top1={:?} elo_breakdown={:?} top5={:?} wdl={:?} misc={:?} total={:?}",
            iteration,
            loss_metric_time,
            policy_metric_time,
            value_metric_time,
            time_usage_metric_time,
            top1_metric_time,
            elo_breakdown_time,
            top5_metric_time,
            wdl_metric_time,
            misc_metric_time,
            metrics_time
        );

        let t_render_start = Instant::now();
        // Render progress
        let progress = TrainingProgress {
            progress: Progress {
                items_processed,
                items_total: train_size,
            },
            epoch: 1,
            epoch_total: 1,
            iteration,
        };
        let t_render_train = Instant::now();
        renderer.render_train(progress);
        let render_train_time = t_render_train.elapsed();

        let t_debug_monitor = Instant::now();
        if should_compute_full_metrics {
            if let Some(monitor) = debug_monitor.as_mut() {
                monitor.evaluate(iteration, &model, renderer.as_mut())?;
            }
        }
        let debug_monitor_time = t_debug_monitor.elapsed();
        let render_time = t_render_start.elapsed();

        tracing::info!(
            "perf_render_breakdown: iter={} render_train={:?} debug_monitor={:?} total={:?}",
            iteration,
            render_train_time,
            debug_monitor_time,
            render_time
        );

        // Checkpoint at specified intervals based on items processed
        // Check if we've crossed a checkpoint boundary
        let prev_checkpoint_num = (iteration - 1) / config.checkpoint_interval;
        let curr_checkpoint_num = iteration / config.checkpoint_interval;
        if curr_checkpoint_num > prev_checkpoint_num {
            tracing::info!(
                "Saving checkpoint at iteration {} to {}",
                iteration,
                MODEL_DIR_NAME
            );

            save_training_state(
                &model,
                &config,
                &gradnorm_state,
                &optim_muon,
                &optim_decay_normal,
                &optim_decay_high,
                &optim_no_decay_normal,
                &optim_no_decay_high,
                Path::new(MODEL_DIR_NAME),
            )?;
        }

        let post_main_loop_time = t_post_main_loop.elapsed();
        let full_iteration_time = t_full_iteration_start.elapsed();
        tracing::info!(
            "perf_full_iteration: iter={} total={:?} post_main_loop={:?} optimizer={:?} metrics={:?} render={:?}",
            iteration,
            full_iteration_time,
            post_main_loop_time,
            optimizer_time,
            metrics_time,
            render_time
        );
    }

    tracing::info!(
        "EXIT_CONDITION: loop_ended at iteration {} (items_processed={}, max_samples={})",
        iteration,
        items_processed,
        max_samples
    );

    // Shut down device workers before finalizing
    device_workers.shutdown();

    // Final save
    let training_duration = start_time.elapsed();
    println!(
        "Training completed in {:.2} seconds ({} iterations)",
        training_duration.as_secs_f64(),
        iteration
    );

    let final_top1 = top1_history.values.back().copied().unwrap_or(0.0);
    let max_top1 = top1_history.values.iter().copied().fold(0.0_f64, f64::max);
    let final_loss = policy_history.values.back().copied().unwrap_or(f64::MAX);
    let min_loss = policy_history
        .values
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);
    tracing::info!(
        "sweep_final: iterations={} final_top1={:.6} max_top1={:.6} final_loss={:.6} min_loss={:.6}",
        iteration, final_top1, max_top1, final_loss, min_loss
    );

    let scoresheet_path = write_scoresheet(
        &config,
        &resume_status,
        iteration,
        items_processed,
        train_size,
        total_batches,
        total_optimizer_steps,
        grad_accumulation_steps,
        training_duration,
        &policy_history,
        &value_history,
        &time_history,
        &top1_history,
        &top5_history,
        &wdl_history,
        &gradient_norm_history,
    )?;
    println!("Saved training scoresheet to {}", scoresheet_path);
    tracing::info!("Saved training scoresheet to {}", scoresheet_path);

    save_training_state(
        &model,
        &config,
        &gradnorm_state,
        &optim_muon,
        &optim_decay_normal,
        &optim_decay_high,
        &optim_no_decay_normal,
        &optim_no_decay_high,
        Path::new(MODEL_DIR_NAME),
    )?;

    Ok(())
}