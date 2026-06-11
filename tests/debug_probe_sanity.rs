//! Manual sanity check for the debug prediction probe: loads a real
//! checkpoint, runs the DebugPredictionMonitor (the TUI probe pipeline), and
//! prints the top moves it would display. Guards against the model-frame
//! mirroring bug where black-to-move probe positions were fed unmirrored and
//! produced nonsense predictions.
//!
//! Run with a checkpoint at checkpoints/<name>/ (model.mpk + params.json):
//!   OXI_PROBE_MODEL_DIR=checkpoints/policy_ce_fixed_stable_step44k_20260611 \
//!   cargo test --release --features "train backend-tch" --test debug_probe_sanity -- --ignored --nocapture
#![cfg(all(feature = "train", feature = "backend-tch"))]

use burn::backend::Autodiff;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

use oxi::config::Config;
use oxi::dataset::OXIDataset;
use oxi::debug_prediction_monitor::DebugPredictionMonitor;
use oxi::metrics_renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, TrainingProgress,
};
use oxi::model::OXIModel;

type B = Autodiff<burn::backend::LibTorch<f32>>;

#[derive(Default)]
struct CapturingRenderer {
    predictions: Vec<(String, Vec<(String, f64)>)>,
}

impl MetricsRendererTraining for CapturingRenderer {
    fn update_train(&mut self, state: MetricState) {
        if let MetricState::Predictions(metric) = state {
            self.predictions.push((
                metric.name.clone(),
                metric
                    .predictions
                    .iter()
                    .map(|p| (p.label.clone(), p.probability))
                    .collect(),
            ));
        }
    }
    fn update_valid(&mut self, _state: MetricState) {}
    fn render_train(&mut self, _item: TrainingProgress) {}
    fn render_valid(&mut self, _item: TrainingProgress) {}
}

impl MetricsRendererEvaluation for CapturingRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
    fn render_test(&mut self, _item: EvaluationProgress) {}
}

impl MetricsRenderer for CapturingRenderer {
    fn manual_close(&mut self) {}
}

#[test]
#[ignore = "needs a real checkpoint; run manually"]
fn debug_probe_predictions_are_sane() {
    let model_dir = std::env::var("OXI_PROBE_MODEL_DIR")
        .unwrap_or_else(|_| "checkpoints/policy_ce_fixed_stable_step44k_20260611".to_string());
    let params = std::fs::read_to_string(format!("{model_dir}/params.json"))
        .expect("params.json missing — set OXI_PROBE_MODEL_DIR");
    let config: Config = serde_json::from_str(&params).expect("parse params.json");
    let _ = oxi::config::set_global_config(config.clone());

    let device = if cfg!(target_os = "macos") {
        burn_tch::LibTorchDevice::Mps
    } else {
        burn_tch::LibTorchDevice::Cpu
    };

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = OXIModel::<B>::new(&device, &config)
        .load_file(format!("{model_dir}/model.mpk"), &recorder, &device)
        .expect("load checkpoint");

    let dataset = OXIDataset::new(Vec::new(), config);
    let mut monitor = DebugPredictionMonitor::<B>::new(&dataset, device)
        .expect("build monitor")
        .expect("monitor should have positions");

    let mut renderer = CapturingRenderer::default();
    monitor
        .evaluate(0, &model, &mut renderer)
        .expect("evaluate probe");

    assert!(!renderer.predictions.is_empty(), "no predictions captured");
    for (name, moves) in &renderer.predictions {
        println!("{name}:");
        for (san, prob) in moves {
            println!("  {san:8} {:.1}%", prob * 100.0);
        }
    }
}
