use burn::train::metric::{Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};

#[derive(Clone)]
pub struct LrPlateauInput {
    pub lr: f64,
    pub relative_improvement: Option<f64>,
    pub window_fill_ratio: f64,
    pub num_reductions: usize,
    pub improvement_threshold: f64,
    pub is_warming_up: bool,
    pub warmup_progress: f64,
}

impl LrPlateauInput {
    pub fn new(
        lr: f64,
        relative_improvement: Option<f64>,
        window_fill_ratio: f64,
        num_reductions: usize,
        improvement_threshold: f64,
        is_warming_up: bool,
        warmup_progress: f64,
    ) -> Self {
        Self {
            lr,
            relative_improvement,
            window_fill_ratio,
            num_reductions,
            improvement_threshold,
            is_warming_up,
            warmup_progress,
        }
    }
}

#[derive(Default, Clone)]
pub struct LrPlateauMetric {
    current_lr: f64,
    relative_improvement: Option<f64>,
    window_fill_ratio: f64,
    num_reductions: usize,
    improvement_threshold: f64,
    is_warming_up: bool,
    warmup_progress: f64,
}

impl LrPlateauMetric {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for LrPlateauMetric {
    type Input = LrPlateauInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.current_lr = input.lr;
        self.relative_improvement = input.relative_improvement;
        self.window_fill_ratio = input.window_fill_ratio;
        self.num_reductions = input.num_reductions;
        self.improvement_threshold = input.improvement_threshold;
        self.is_warming_up = input.is_warming_up;
        self.warmup_progress = input.warmup_progress;

        let lr_display = if self.current_lr < 0.01 {
            format!("{:.2e}", self.current_lr)
        } else {
            format!("{:.6}", self.current_lr)
        };

        let warmup_display = if self.is_warming_up {
            format!(" | warmup: {:.0}%", self.warmup_progress * 100.0)
        } else {
            String::new()
        };

        let improvement_display = self
            .relative_improvement
            .map(|r| format!("{:.2}%", r * 100.0))
            .unwrap_or_else(|| "...".to_string());

        let threshold = self.improvement_threshold * 100.0;
        let fill_pct = self.window_fill_ratio * 100.0;

        let formatted = format!(
            "LR: {}{} | improvement: {} (thresh: {:.2}%) | fill: {:.0}% | reductions: {}",
            lr_display,
            warmup_display,
            improvement_display,
            threshold,
            fill_pct,
            self.num_reductions
        );

        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.current_lr = 0.0;
        self.relative_improvement = None;
        self.window_fill_ratio = 0.0;
        self.num_reductions = 0;
        self.improvement_threshold = 0.005;
        self.is_warming_up = false;
        self.warmup_progress = 0.0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Learning Rate".to_string().into()
    }
}

impl Numeric for LrPlateauMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_lr)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_lr)
    }
}
