use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric, NumericEntry};

use crate::gradnorm::{GradNormTask, GradNormTaskStatus};

#[derive(Clone, Debug)]
pub struct GradNormStatusInput {
    pub tasks: Vec<GradNormTaskStatus>,
}

impl GradNormStatusInput {
    pub fn new(tasks: Vec<GradNormTaskStatus>) -> Self {
        Self { tasks }
    }
}

#[derive(Default, Clone)]
pub struct GradNormStatusMetric {
    last_tasks: Vec<GradNormTaskStatus>,
}

impl GradNormStatusMetric {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for GradNormStatusMetric {
    type Input = GradNormStatusInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.last_tasks = input.tasks.clone();
        let formatted = if self.last_tasks.is_empty() {
            "GradNorm disabled".to_string()
        } else {
            self.last_tasks
                .iter()
                .map(format_status)
                .collect::<Vec<_>>()
                .join(" | ")
        };

        MetricEntry::new(
            "GradNorm Status".to_string().into(),
            formatted.clone(),
            formatted,
        )
    }

    fn clear(&mut self) {
        self.last_tasks.clear();
    }

    fn name(&self) -> std::sync::Arc<String> {
        "GradNorm Status".to_string().into()
    }
}

impl Numeric for GradNormStatusMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(0.0)
    }
}

fn format_status(status: &GradNormTaskStatus) -> String {
    let name = match status.task {
        GradNormTask::Policy => "policy",
        GradNormTask::Value => "value",
        GradNormTask::TimeUsage => "time",
    };

    if !status.enabled {
        return format!(
            "{name}: disabled (weight={weight:.4})",
            weight = status.weight
        );
    }

    let loss_display = status
        .last_loss
        .map(|loss| {
            if loss < 0.01 {
                format!("{loss:.2e}")
            } else {
                format!("{loss:.6}")
            }
        })
        .unwrap_or_else(|| "N/A".to_string());
    let weighted_display = status
        .last_weighted_loss
        .map(|loss| {
            if loss < 0.01 {
                format!("{loss:.2e}")
            } else {
                format!("{loss:.6}")
            }
        })
        .unwrap_or_else(|| "N/A".to_string());
    let grad_display = status
        .last_grad_norm
        .map(|grad| {
            if grad < 0.01 {
                format!("{grad:.2e}")
            } else {
                format!("{grad:.4}")
            }
        })
        .unwrap_or_else(|| "N/A".to_string());

    format!(
        "{name}: loss={loss}, weight={weight:.4}, priority={priority:.2}, weighted={weighted}, grad={grad}",
        loss = loss_display,
        weight = status.weight,
        priority = status.priority,
        weighted = weighted_display,
        grad = grad_display
    )
}
