use burn_train::metric::{Metric, MetricMetadata, SerializedEntry};

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingStage {
    Pretrain {
        iteration: usize,
        total: usize,
        tcec_percentage: f64,
    },
    MainTraining,
}

/// Input type for the training stage metric
pub struct TrainingStageInput {
    pub stage: TrainingStage,
}

impl Metric for TrainingStageMetric {
    type Input = TrainingStageInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.stage = match input.stage {
            TrainingStage::Pretrain {
                iteration,
                total,
                tcec_percentage,
            } => {
                format!(
                    "Pretrain ({}/{} - {:.0}% TCEC)",
                    iteration,
                    total,
                    tcec_percentage * 100.0
                )
            }
            TrainingStage::MainTraining => "Main Training".to_string(),
        };

        SerializedEntry::new(self.stage.clone(), self.stage.clone())
    }

    fn clear(&mut self) {
        self.stage = "Initializing".to_string();
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Stage".to_string().into()
    }
}
