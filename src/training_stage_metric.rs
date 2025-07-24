use burn_train::metric::{Metric, MetricEntry, MetricMetadata};

/// Metric to display the current training stage (pretrain vs main training)
#[derive(Default, Clone)]
pub struct TrainingStageMetric {
    stage: String,
}

impl TrainingStageMetric {
    pub fn new() -> Self {
        Self {
            stage: "Initializing".to_string(),
        }
    }

    pub fn set_stage(&mut self, stage: String) {
        self.stage = stage;
    }

    pub fn stage(&self) -> String {
        self.stage.clone()
    }
}

/// Training stage enum
#[derive(Debug, Clone, Copy)]
pub enum TrainingStage {
    Pretrain {
        iteration: usize,
        total: usize,
        easy_percentage: f64,
    },
    MainTraining,
}

/// Input type for the training stage metric
pub struct TrainingStageInput {
    pub stage: TrainingStage,
}

impl Metric for TrainingStageMetric {
    type Input = TrainingStageInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.stage = match input.stage {
            TrainingStage::Pretrain {
                iteration,
                total,
                easy_percentage,
            } => {
                format!(
                    "Pretrain ({}/{} - {:.0}% easy)",
                    iteration,
                    total,
                    easy_percentage * 100.0
                )
            }
            TrainingStage::MainTraining => "Main Training".to_string(),
        };

        MetricEntry::new(
            "Stage".to_string().into(),
            self.stage.clone(),
            self.stage.clone(),
        )
    }

    fn clear(&mut self) {
        self.stage = "Initializing".to_string();
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Stage".to_string().into()
    }
}
