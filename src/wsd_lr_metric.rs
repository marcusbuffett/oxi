use burn::train::metric::{Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};

use crate::wsd_scheduler::WsdPhase;

#[derive(Clone)]
pub struct WsdLrInput {
    pub lr: f64,
    pub phase: WsdPhase,
}

#[derive(Clone)]
pub struct WsdLrMetric {
    current_lr: f64,
    phase: WsdPhase,
}

impl Default for WsdLrMetric {
    fn default() -> Self {
        Self {
            current_lr: 0.0,
            phase: WsdPhase::Stable,
        }
    }
}

impl WsdLrMetric {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for WsdLrMetric {
    type Input = WsdLrInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.current_lr = input.lr;
        self.phase = input.phase;

        let lr_display = if self.current_lr < 0.01 {
            format!("{:.2e}", self.current_lr)
        } else {
            format!("{:.6}", self.current_lr)
        };

        let phase_display = match self.phase {
            WsdPhase::Warmup { progress } => format!("warmup: {:.0}%", progress * 100.0),
            WsdPhase::Stable => "stable".to_string(),
            WsdPhase::Decay { progress } => format!("decay: {:.0}%", progress * 100.0),
        };

        let formatted = format!("LR: {} | {}", lr_display, phase_display);
        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Learning Rate".to_string().into()
    }
}

impl Numeric for WsdLrMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_lr)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_lr)
    }
}
