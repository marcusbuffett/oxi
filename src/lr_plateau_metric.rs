use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric, NumericEntry};

/// Input type for learning rate metric with plateau information
#[derive(Clone)]
pub struct LrPlateauInput {
    /// Current learning rate
    pub lr: f64,
    /// Best (lowest) loss observed so far
    pub best_loss: Option<f64>,
    /// Number of measurement batches without improvement
    pub batches_without_improvement: usize,
    /// Patience threshold (batches before reducing LR)
    pub patience: usize,
}

impl LrPlateauInput {
    pub fn new(
        lr: f64,
        best_loss: Option<f64>,
        batches_without_improvement: usize,
        patience: usize,
    ) -> Self {
        Self {
            lr,
            best_loss,
            batches_without_improvement,
            patience,
        }
    }
}

/// Metric for tracking learning rate with ReduceOnPlateau information
#[derive(Default, Clone)]
pub struct LrPlateauMetric {
    current_lr: f64,
    best_loss: Option<f64>,
    batches_without_improvement: usize,
    patience: usize,
}

impl LrPlateauMetric {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for LrPlateauMetric {
    type Input = LrPlateauInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.current_lr = input.lr;
        self.best_loss = input.best_loss;
        self.batches_without_improvement = input.batches_without_improvement;
        self.patience = input.patience;

        let best_loss_display = self
            .best_loss
            .map(|loss| format!("{loss:.6}"))
            .unwrap_or_else(|| "N/A".to_string());

        let progress_pct = if self.patience > 0 {
            (self.batches_without_improvement as f64 / self.patience as f64 * 100.0).min(100.0)
        } else {
            0.0
        };

        let formatted = format!(
            "LR: {lr:.6}, Best Loss: {best}, Plateau: {batches}/{patience} ({progress:.0}%)",
            lr = self.current_lr,
            best = best_loss_display,
            batches = self.batches_without_improvement,
            patience = self.patience,
            progress = progress_pct
        );

        MetricEntry::new(
            "Learning Rate".to_string().into(),
            formatted.clone(),
            formatted,
        )
    }

    fn clear(&mut self) {
        self.current_lr = 0.0;
        self.best_loss = None;
        self.batches_without_improvement = 0;
        self.patience = 0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Learning Rate".to_string().into()
    }
}

impl Numeric for LrPlateauMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_lr)
    }
}
