use burn::data::dataloader::Progress;
use burn_train::metric::{NumericEntry, SerializedEntry};

/// Alias representing the name of an evaluation run.
pub type EvaluationName = String;

/// Progress information for training or validation splits.
#[derive(Clone, Debug)]
pub struct TrainingProgress {
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
}

/// Progress information for evaluation splits.
#[derive(Clone, Debug)]
pub struct EvaluationProgress {
    pub progress: Progress,
    pub iteration: usize,
}

/// Represents a single prediction entry for debug monitoring.
#[derive(Clone, Debug)]
pub struct PredictionEntry {
    pub label: String,
    /// Probability in the range [0.0, 1.0].
    pub probability: f64,
}

/// Specialized metric payload carrying a list of predictions.
#[derive(Clone, Debug)]
pub struct PredictionMetric {
    pub name: String,
    pub formatted: String,
    pub predictions: Vec<PredictionEntry>,
}

/// State update communicated to metric renderers.
#[allow(clippy::large_enum_variant)]
pub enum MetricState {
    Generic {
        name: String,
        entry: SerializedEntry,
    },
    Numeric {
        name: String,
        entry: SerializedEntry,
        value: NumericEntry,
    },
    Predictions(PredictionMetric),
}

pub trait MetricsRendererTraining {
    fn update_train(&mut self, state: MetricState);
    fn update_valid(&mut self, state: MetricState);
    fn render_train(&mut self, item: TrainingProgress);
    fn render_valid(&mut self, item: TrainingProgress);

    fn on_train_end(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

pub trait MetricsRendererEvaluation {
    fn update_test(&mut self, name: EvaluationName, state: MetricState);
    fn render_test(&mut self, item: EvaluationProgress);
}

pub trait MetricsRenderer: MetricsRendererTraining + MetricsRendererEvaluation {
    fn manual_close(&mut self);
}
