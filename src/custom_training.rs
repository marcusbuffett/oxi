use crate::metrics_renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, TrainingProgress,
};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Progress;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::LrScheduler;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn_train::metric::IterationSpeedMetric;
use burn_train::metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric, NumericEntry};
use burn_train::Interrupter;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::thread_rng;
use rand::SeedableRng;
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
    compute_gradient_norm, GradientNormBreakdown, GradientNormInput, GradientNormMetric,
};
use crate::gradnorm::{GradNormProbeResult, GradNormState, GradNormTask};
use crate::lr_plateau_metric::{LrPlateauInput, LrPlateauMetric};
use crate::model::OXIModel;
use crate::move_accuracy_metric::MoveTopKAccuracyMetric;
use crate::pgn_processor::process_tcec_directory_iter;
use crate::policy_loss_metric::{PolicyLossInput, PolicyLossMetric};
use crate::reduce_on_plateau_scheduler::ReduceOnPlateauScheduler;
use crate::time_usage_loss_metric::{TimeUsageLossInput, TimeUsageLossMetric};
use crate::training_stage_metric::{TrainingStage, TrainingStageInput, TrainingStageMetric};
use crate::tui::OxiTuiRenderer;
use crate::value_loss_metric::{ValueLossInput, ValueLossMetric};
use crate::wdl_accuracy_metric::WdlAccuracyMetric;
use crate::weight_decay::WeightDecayGroups;

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
    if move_count < 40 {
        GameStageBucket::Opening
    } else if move_count < 100 {
        GameStageBucket::Middlegame
    } else {
        GameStageBucket::Endgame
    }
}

const SCORE_WINDOW: usize = 100;

const MODEL_DIR_NAME: &str = "model";
const MODEL_FILE_NAME: &str = "model.mpk";
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

fn save_training_state<B, O1, O2, O3, O4>(
    model: &OXIModel<B>,
    gradnorm_state: &GradNormState,
    optim_decay_normal: &O1,
    optim_decay_high: &O2,
    optim_no_decay_normal: &O3,
    optim_no_decay_high: &O4,
    directory: &Path,
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
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
        NumericEntry::Aggregated { current, .. } => *current,
    };
    if value.is_finite() {
        Some(value)
    } else {
        None
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
}

impl GradNormWeights {
    fn from_state(state: &GradNormState) -> Self {
        Self {
            policy: state.weight_for(GradNormTask::Policy),
            value: state.weight_for(GradNormTask::Value),
            time: state.weight_for(GradNormTask::TimeUsage),
        }
    }
}

fn move_output_to_device<B: Backend>(output: ChessOutput<B>, device: &B::Device) -> ChessOutput<B> {
    ChessOutput {
        loss: output.loss.to_device(device),
        policy_loss: output.policy_loss.to_device(device),
        value_loss: output.value_loss.to_device(device),
        time_usage_loss: output.time_usage_loss.to_device(device),
        base_policy_loss: output.base_policy_loss.to_device(device),
        base_value_loss: output.base_value_loss.to_device(device),
        base_time_usage_loss: output.base_time_usage_loss.to_device(device),
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
    output.loss =
        output.policy_loss.clone() + output.value_loss.clone() + output.time_usage_loss.clone();
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
    let mut sum_base_policy_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_value_loss = Tensor::<B, 1>::zeros([1], device);
    let mut sum_base_time_loss = Tensor::<B, 1>::zeros([1], device);

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

        sum_base_policy_loss =
            sum_base_policy_loss + output.base_policy_loss.clone() * batch_scalar;
        sum_base_value_loss = sum_base_value_loss + output.base_value_loss.clone() * batch_scalar;
        sum_base_time_loss =
            sum_base_time_loss + output.base_time_usage_loss.clone() * batch_scalar;

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

    let base_policy_loss = sum_base_policy_loss / total_scalar;
    let base_value_loss = sum_base_value_loss / total_scalar;
    let base_time_usage_loss = sum_base_time_loss / total_scalar;

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
        base_policy_loss,
        base_value_loss,
        base_time_usage_loss,
        policy_output,
        policy_targets,
        value_output,
        value_targets,
        legal_moves_mask,
        uncertainties,
        raw_policy_loss,
        raw_value_loss,
        raw_time_usage_loss,
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
        };
        let base_loss_value = base_loss_tensor.clone().into_scalar().elem::<f32>();
        let weighted_loss = base_loss_tensor * weight;
        let grads = weighted_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        let breakdown: GradientNormBreakdown = compute_gradient_norm(&grads_params, &model);
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

                        let batch = batcher.batch(request.items, &device).to_device(&device);
                        let mut output = model.forward_classification(batch);

                        apply_gradnorm_weights_to_output(&mut output, request.weights);

                        let loss = output.loss.clone();
                        let grads = GradientsParams::from_grads(loss.backward(), &model);

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

fn log_gradient_breakdown(
    breakdown: &GradientNormBreakdown,
    config: &Config,
    optimizer_step: usize,
) {
    tracing::info!(
        target: "gradient_debug",
        "gradient_debug: step={} total_grad_norm={:.6}",
        optimizer_step,
        breakdown.total()
    );

    for layer in breakdown
        .per_layer
        .iter()
        .take(config.gradient_layer_limit())
    {
        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: layer={} grad_norm={:.6}",
            layer.name,
            layer.norm
        );
    }

    for head in breakdown.per_head.iter().take(config.gradient_head_limit()) {
        tracing::info!(
            target: "gradient_debug",
            "gradient_debug: layer={} projection={} head={} grad_norm={:.6}",
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
    let initial_lr = 0.001 * (effective_batch_size as f64 / 16000.0).sqrt();

    let content = format!(
        "OXI Training Scoresheet\n\
Timestamp (UTC): {timestamp_display}\n\
Resume Status: {resume_status}\n\
\n\
Training Summary\n\
- Iterations processed: {iteration}\n\
- Items processed: {items_processed}\n\
- Training set size: {train_size}\n\
- Total batches (estimated): {total_batches}\n\
- Optimizer steps (estimated): {total_optimizer_steps}\n\
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
- MLP ratio: {mlp_ratio}\n\
- Policy loss weight: {policy_loss_weight}\n\
- Value loss weight: {value_loss_weight}\n\
- Value entropy weight: {value_entropy_weight}\n\
- Time usage loss weight: {time_usage_loss_weight}\n\
- Weight decay: {weight_decay}\n\
- Gradient clip: {gradient_clip}\n\
- Initial LR: {initial_lr:.6}\n\
- LR min: {lr_min}\n\
- Measurement batch size: {measurement_batch_size}\n\
- LR patience: {lr_patience}\n\
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
        mlp_ratio = config.mlp_ratio(),
        policy_loss_weight = config.policy_loss_weight,
        value_loss_weight = config.value_loss_weight,
        value_entropy_weight = config.value_entropy_weight,
        time_usage_loss_weight = config.time_usage_loss_weight,
        weight_decay = config.weight_decay,
        gradient_clip = config.gradient_clip,
        initial_lr = initial_lr,
        lr_min = config.lr_min,
        measurement_batch_size = config.measurement_batch_size,
        lr_patience = config.lr_patience,
        lr_reduction_factor = config.lr_reduction_factor,
        window = SCORE_WINDOW,
    );

    fs::write(&filename, content)?;
    Ok(filename)
}

pub fn init_train_logging() -> WorkerGuard {
    let file_appender = tracing_appender::rolling::never(".", "train.log");
    let (non_blocking, worker_guard) = tracing_appender::non_blocking(file_appender);
    let subscriber = tracing_fmt()
        .with_ansi(false)
        .with_file(false)
        .with_target(false)
        .without_time()
        .with_writer(non_blocking)
        .finish();

    // Try to set the global default, but don't panic if it's already set
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
    let (decay_params, no_decay_params, normal_lr_params, high_lr_params) =
        weight_decay_groups.counts();

    // Calculate LR multiplier for embeddings and scale parameters
    let lr_multiplier = (config.embed_dim() as f64).sqrt();

    println!("\nParameter grouping summary:");
    println!(
        "  Weight decay: {} with decay, {} without decay",
        decay_params, no_decay_params
    );
    println!(
        "  Learning rate: {} normal LR, {} high LR (embeddings + scale params)",
        normal_lr_params, high_lr_params
    );
    println!(
        "  LR multiplier for high LR params: {:.4}x (sqrt({}) = {:.4})",
        lr_multiplier,
        config.embed_dim(),
        lr_multiplier
    );
    println!("  Min LR: {:.6}", config.lr_min);
    println!("  High LR min: {:.6}\n", config.lr_min * lr_multiplier);

    tracing::info!(
        "Parameter grouping: {} decay, {} no_decay; {} normal_lr, {} high_lr; lr_multiplier={:.4}",
        decay_params,
        no_decay_params,
        normal_lr_params,
        high_lr_params,
        lr_multiplier
    );
    tracing::info!(
        "Min LR: {:.6}; High LR min: {:.6}",
        config.lr_min,
        config.lr_min * lr_multiplier
    );

    // Load dataset
    let path = config.data_path.clone().expect("Data path not set");
    let data_path = Path::new(&path);
    let dataset: OXIDataset = if data_path.is_dir() {
        tracing::info!("Loading data from PGN directory: {:?}", data_path);
        OXIDataset::from_pgn_dir_with_limit(data_path, config.clone(), config.max_samples)?
    } else {
        tracing::info!("Loading data from PGN file: {:?}", data_path);
        OXIDataset::from_pgn_with_limit(data_path, config.clone(), config.max_samples)?
    };

    tracing::info!("Training with {} samples", dataset.examples.len());
    println!(
        "{}",
        crate::model_prediction_logger::format_elo_histogram(&dataset.examples)
    );
    println!(
        "{}",
        crate::model_prediction_logger::format_ply_histogram(&dataset.examples)
    );

    // Pretrain phase: load TCEC (computer engine) games if configured
    let mut pretrain_examples = Vec::new();
    if config.pretrain_samples > 0 {
        println!(
            "Loading {} TCEC samples for pretraining...",
            config.pretrain_samples
        );

        match load_tcec_examples(data_path, config.pretrain_samples) {
            Ok(examples) => {
                pretrain_examples = examples;
                println!(
                    "Loaded {} TCEC examples for pretraining",
                    pretrain_examples.len()
                );
            }
            Err(e) => {
                println!("Warning: Failed to load TCEC games: {}", e);
                println!("Continuing without pretraining phase");
            }
        }
    }

    if (config.train_ratio - 1.0).abs() > f32::EPSILON {
        tracing::info!(
            "Ignoring train_ratio ({}) – using entire dataset for training",
            config.train_ratio
        );
    }

    let mut train_examples = dataset.examples;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Shuffle both pretrain and train examples first
    pretrain_examples.shuffle(&mut rng);
    train_examples.shuffle(&mut rng);
    println!(
        "Shuffled {} pretrain examples and {} training examples",
        pretrain_examples.len(),
        train_examples.len()
    );

    // Create mixed batches: taper from 100% TCEC to 0% TCEC, then continue with human games
    let mut batches: Vec<Vec<ChessExample>> = Vec::new();
    let num_pretrain_batches;

    if !pretrain_examples.is_empty() {
        println!(
            "Creating mixed batches with {} TCEC samples (100% -> 0% TCEC)",
            pretrain_examples.len()
        );

        // Linear taper from 100% to 0% means average is 50% TCEC per batch
        // num_batches = num_pretrain / (batch_size * 0.5)
        let physical_batch_size = config.physical_batch_size;
        let num_pretrain = pretrain_examples.len();
        let average_tcec_ratio = 0.5;
        num_pretrain_batches = ((num_pretrain as f64
            / (physical_batch_size as f64 * average_tcec_ratio))
            .ceil() as usize)
            .max(1);

        println!(
            "Will create {} pretrain batches using {} TCEC samples (avg {:.0}% TCEC per batch)",
            num_pretrain_batches,
            num_pretrain,
            average_tcec_ratio * 100.0
        );

        let mut pretrain_iter = pretrain_examples.into_iter();
        let mut human_iter = train_examples.into_iter();
        let mut tcec_count = 0;
        let mut human_count = 0;

        for batch_num in 0..num_pretrain_batches {
            let progress = batch_num as f64 / (num_pretrain_batches - 1).max(1) as f64;
            let tcec_percentage = 1.0 - progress;

            let num_tcec = (physical_batch_size as f64 * tcec_percentage).round() as usize;
            let num_human = physical_batch_size - num_tcec;

            let mut batch = Vec::with_capacity(physical_batch_size);

            for _ in 0..num_tcec {
                if let Some(example) = pretrain_iter.next() {
                    batch.push(example);
                    tcec_count += 1;
                } else if let Some(example) = human_iter.next() {
                    batch.push(example);
                    human_count += 1;
                }
            }

            let human_chunk: Vec<_> = human_iter.by_ref().take(num_human).collect();
            human_count += human_chunk.len();
            batch.extend(human_chunk);

            batch.shuffle(&mut rng);
            batches.push(batch);
        }

        loop {
            let batch_chunk: Vec<_> = human_iter.by_ref().take(physical_batch_size).collect();
            if batch_chunk.is_empty() {
                break;
            }
            human_count += batch_chunk.len();
            let mut batch = batch_chunk;
            batch.shuffle(&mut rng);
            batches.push(batch);
        }

        println!(
            "Created {} batches using {} TCEC and {} human samples",
            batches.len(),
            tcec_count,
            human_count
        );
    } else {
        num_pretrain_batches = 0;

        // No pretrain examples, use iterator and extend for efficiency
        let mut train_iter = train_examples.into_iter();
        loop {
            let batch_chunk: Vec<_> = train_iter
                .by_ref()
                .take(config.physical_batch_size)
                .collect();
            if batch_chunk.is_empty() {
                break;
            }
            let mut batch = batch_chunk;
            batch.shuffle(&mut rng);
            batches.push(batch);
        }
    }

    // Calculate train size from batches without flattening
    let train_size: usize = batches.iter().map(|b| b.len()).sum();
    println!("Train size: {}", train_size);

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
    // LR scales with sqrt of batch size: batch_size=16000 -> lr=3e-2
    let effective_batch_size = grad_accumulation_steps * config.physical_batch_size;
    let initial_lr = 0.001 * (effective_batch_size as f64 / 16000.0).sqrt() * config.lr_multiplier;
    println!(
        "Effective batch size: {}, Initial LR: {:.6} (multiplier: {})",
        effective_batch_size, initial_lr, config.lr_multiplier
    );

    // Create optimizers (4 groups: decay+normal_lr, decay+high_lr, no_decay+normal_lr, no_decay+high_lr)
    let grad_clipping = if config.gradient_clip > 0.0 {
        Some(GradientClippingConfig::Norm(config.gradient_clip as f32))
    } else {
        None
    };

    let mut optim_decay_normal = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_decay_high = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_no_decay_normal = AdamWConfig::new()
        .with_weight_decay(0.0)
        .with_grad_clipping(grad_clipping.clone())
        .init();

    let mut optim_no_decay_high = AdamWConfig::new()
        .with_weight_decay(0.0)
        .with_grad_clipping(grad_clipping)
        .init();

    let mut gradnorm_state = GradNormState::new(&config);

    if let Some(resume_dir) = resume_optimizer_dir.clone() {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let device = &devices[0];

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
                        Ok((state, _)) => {
                            println!("Loaded GradNorm state from {}", gradnorm_path.display());
                            tracing::info!(
                                "Loaded GradNorm state from {}",
                                gradnorm_path.display()
                            );
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

    // Calculate total training steps (one epoch only)
    // This is the number of physical batches (include final partial batch)
    let total_batches = (train_size + config.physical_batch_size - 1) / config.physical_batch_size;
    // Number of optimizer steps (accounting for gradient accumulation)
    let total_optimizer_steps =
        (total_batches + grad_accumulation_steps - 1) / grad_accumulation_steps;

    println!(
        "Total batches: {}, Total optimizer steps: {}",
        total_batches, total_optimizer_steps
    );
    println!("Initial LR: {:.6}, Min LR: {}", initial_lr, config.lr_min);
    println!(
        "Measurement batch size: {}, LR patience: {}, LR reduction factor: {}",
        config.measurement_batch_size, config.lr_patience, config.lr_reduction_factor
    );

    // Create ReduceOnPlateau learning rate scheduler
    let mut lr_scheduler = ReduceOnPlateauScheduler::new(
        initial_lr,
        config.lr_min,
        config.lr_reduction_factor,
        config.lr_patience,
        config.measurement_batch_size,
    );

    // Adjust measurement batch size to be a multiple of physical batch size
    lr_scheduler.adjust_measurement_batch_size(config.physical_batch_size);
    println!(
        "Adjusted measurement batch size to: {} (multiple of physical batch size {})",
        lr_scheduler.measurement_batch_size(),
        config.physical_batch_size
    );

    // Initialize metrics
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

    // Single epoch training loop - iterate through pre-created batches
    for batch_examples in &batches {
        // Check if user requested stop
        if interruptor.should_stop() {
            println!("Training interrupted by user");
            tracing::info!("Training interrupted by user");
            break;
        }

        // Check if scheduler indicates training should stop (at min LR with no improvement)
        if lr_scheduler.should_stop() {
            println!(
                "Training stopped: reached min LR with no improvement for {} measurement batches",
                config.lr_patience
            );
            tracing::info!(
                "Training stopped: reached min LR with no improvement for {} measurement batches",
                config.lr_patience
            );
            break;
        }

        iteration += 1;

        // Convert ChessExamples to ChessItems
        let items_all: Vec<_> = batch_examples
            .iter()
            .filter_map(|example| dataset_for_processing.process_example(example).ok())
            .collect();

        if items_all.is_empty() {
            break;
        }

        let current_batch_size = items_all.len();
        items_processed += current_batch_size;

        // Split batch across devices for parallel execution
        let device_splits = split_items_across_devices(&items_all, devices.len());
        let gradnorm_weights = GradNormWeights::from_state(&gradnorm_state);
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

        if device_outputs.is_empty() {
            continue;
        }
        accumulation_count += 1;

        // Combine outputs back on the main device for logging/metrics
        let output = combine_outputs(&device_outputs, &devices[0]);
        drop(device_outputs);

        sync_backend_if_supported::<B>(&devices[0]);
        B::memory_cleanup(&devices[0]);
        gradnorm_state.record_batch_losses(iteration, &output);

        // Record batch in ReduceOnPlateau scheduler using raw policy loss so GradNorm weighting
        // on other heads cannot prematurely trigger LR reductions.
        let policy_loss_tensor = output
            .raw_policy_loss
            .clone()
            .unwrap_or_else(|| output.base_policy_loss.clone());
        let batch_policy_loss = policy_loss_tensor.into_scalar().elem::<f32>() as f64;
        let _measurement_recorded =
            lr_scheduler.record_batch(current_batch_size, batch_policy_loss);

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
        if should_update {
            let grads = grad_accumulator.grads();

            // Update gradient norm metric before optimizer step.
            let gradient_breakdown = compute_gradient_norm(&grads, &model);
            let gradient_norm_value = gradient_breakdown.total();
            let next_step = optimizer_step + 1;

            if config.log_gradient_breakdown() {
                if next_step % config.gradient_breakdown_interval() == 0 {
                    log_gradient_breakdown(&gradient_breakdown, &config, next_step);
                }
            }

            // Compute and log L2 penalty from weight decay at specified interval
            if next_step % config.l2_penalty_log_interval() == 0 {
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

            let (grads_decay_normal, grads_decay_high, grads_no_decay_normal, grads_no_decay_high) =
                weight_decay_groups.split_grads::<B, _>(&model, grads);

            let grad_norm_input = GradientNormInput::new(gradient_norm_value);
            let grad_norm_entry = gradient_norm_metric.update(&grad_norm_input, &metadata);
            let grad_norm_numeric = Numeric::value(&gradient_norm_metric);
            if let Some(value) = numeric_entry_value(&grad_norm_numeric) {
                gradient_norm_history.push(value);
            }
            renderer.update_train(MetricState::Numeric(grad_norm_entry, grad_norm_numeric));

            if gradnorm_state.should_update_weights(next_step) {
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

            // Apply different learning rates: normal LR for normal params, lr_multiplier * LR for high LR params
            let high_lr = current_lr * lr_multiplier;

            // Log learning rates periodically
            if next_step % 100 == 0 {
                tracing::info!(
                    "step={} base_lr={:.6} high_lr={:.6} (multiplier={:.4})",
                    next_step,
                    current_lr,
                    high_lr,
                    lr_multiplier
                );
            }

            model = optim_decay_normal.step(current_lr, model, grads_decay_normal);
            model = optim_decay_high.step(high_lr, model, grads_decay_high);
            model = optim_no_decay_normal.step(current_lr, model, grads_no_decay_normal);
            model = optim_no_decay_high.step(high_lr, model, grads_no_decay_high);

            device_workers.broadcast_model(&model);

            accumulation_count = 0;
            optimizer_step += 1;
        }

        // Update each metric and send to renderer
        let gradnorm_snapshot = gradnorm_state.status_snapshot();

        let loss_entry = loss_metric.update(&output.adapt(), &metadata);
        let loss_value = Numeric::value(&loss_metric);
        renderer.update_train(MetricState::Numeric(loss_entry, loss_value));

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
        }
        renderer.update_train(MetricState::Numeric(policy_entry, policy_value));

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
        }
        renderer.update_train(MetricState::Numeric(value_entry, value_value));

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
        }

        let move_top1_entry = move_top1_metric.update(&output.adapt(), &metadata);
        let move_top1_value = Numeric::value(&move_top1_metric);
        if let Some(value) = numeric_entry_value(&move_top1_value) {
            top1_history.push(value);
        }
        renderer.update_train(MetricState::Numeric(move_top1_entry, move_top1_value));

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

                for (idx, item) in items_all.iter().enumerate() {
                    let correct = predicted_indices[idx] == target_indices[idx];
                    if let Some(bucket) = categorize_elo(item.elo_self) {
                        elo_counters.entry(bucket).or_default().update(correct);
                    }
                    let stage_bucket = categorize_stage(item.global_features.move_count);
                    stage_counters
                        .entry(stage_bucket)
                        .or_default()
                        .update(correct);
                }

                for bucket in ELO_BUCKETS {
                    if let Some(counter) = elo_counters.get(&bucket) {
                        if let Some(accuracy) = counter.accuracy() {
                            let metric_name =
                                format!("Move Top-1 Accuracy by Elo|{}", bucket.label());
                            let entry = MetricEntry::new(
                                metric_name.clone().into(),
                                format!(
                                    "{}: {:.1}% ({}/{})",
                                    bucket.label(),
                                    accuracy * 100.0,
                                    counter.correct,
                                    counter.total
                                ),
                                format!("{accuracy:.4}"),
                            );
                            renderer.update_train(MetricState::Numeric(
                                entry,
                                NumericEntry::Value(accuracy),
                            ));
                        }
                    }
                }

                for bucket in GAME_STAGE_BUCKETS {
                    if let Some(counter) = stage_counters.get(&bucket) {
                        if let Some(accuracy) = counter.accuracy() {
                            let metric_name =
                                format!("Move Top-1 Accuracy by Game Stage|{}", bucket.label());
                            let entry = MetricEntry::new(
                                metric_name.clone().into(),
                                format!(
                                    "{}: {:.1}% ({}/{})",
                                    bucket.label(),
                                    accuracy * 100.0,
                                    counter.correct,
                                    counter.total
                                ),
                                format!("{accuracy:.4}"),
                            );
                            renderer.update_train(MetricState::Numeric(
                                entry,
                                NumericEntry::Value(accuracy),
                            ));
                        }
                    }
                }
            }
        }

        let move_top5_entry = move_top5_metric.update(&output.adapt(), &metadata);
        let move_top5_value = Numeric::value(&move_top5_metric);
        if let Some(value) = numeric_entry_value(&move_top5_value) {
            top5_history.push(value);
        }
        renderer.update_train(MetricState::Numeric(move_top5_entry, move_top5_value));

        let wdl_acc_entry = wdl_accuracy_metric.update(&output.adapt(), &metadata);
        let wdl_acc_value = Numeric::value(&wdl_accuracy_metric);
        if let Some(value) = numeric_entry_value(&wdl_acc_value) {
            wdl_history.push(value);
        }
        renderer.update_train(MetricState::Numeric(wdl_acc_entry, wdl_acc_value));

        let iteration_speed_entry = iteration_speed_metric.update(&output.adapt(), &metadata);
        renderer.update_train(MetricState::Generic(iteration_speed_entry));

        let lr_plateau_input = LrPlateauInput::new(
            current_lr,
            lr_scheduler.best_loss(),
            lr_scheduler.batches_without_improvement(),
            config.lr_patience,
        );
        let lr_entry = lr_metric.update(&lr_plateau_input, &metadata);
        let lr_value = Numeric::value(&lr_metric);
        renderer.update_train(MetricState::Numeric(lr_entry, lr_value));

        // Determine if we're in the pretrain phase (using mixed batches with easy examples)
        let is_in_pretrain_phase = iteration <= num_pretrain_batches;

        let stage_input = TrainingStageInput {
            stage: if is_in_pretrain_phase {
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
        renderer.update_train(MetricState::Generic(stage_entry));

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
        renderer.render_train(progress);

        if let Some(monitor) = debug_monitor.as_mut() {
            monitor.evaluate(iteration, &model, renderer.as_mut())?;
        }

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
                &gradnorm_state,
                &optim_decay_normal,
                &optim_decay_high,
                &optim_no_decay_normal,
                &optim_no_decay_high,
                Path::new(MODEL_DIR_NAME),
            )?;
        }
    }

    // Shut down device workers before finalizing
    device_workers.shutdown();

    // Final save
    let training_duration = start_time.elapsed();
    println!(
        "Training completed in {:.2} seconds ({} iterations)",
        training_duration.as_secs_f64(),
        iteration
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
        &gradnorm_state,
        &optim_decay_normal,
        &optim_decay_high,
        &optim_no_decay_normal,
        &optim_no_decay_high,
        Path::new(MODEL_DIR_NAME),
    )?;

    Ok(())
}
