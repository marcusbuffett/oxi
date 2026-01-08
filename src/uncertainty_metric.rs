use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricMetadata, SerializedEntry};

/// Input type for uncertainty metric
pub struct UncertaintyInput {
    /// Current uncertainty values (sigma) for each loss component
    pub policy_sigma: f32,
    pub value_sigma: f32,
    pub time_usage_sigma: f32,
}

impl UncertaintyInput {
    pub fn new(policy_sigma: f32, value_sigma: f32, time_usage_sigma: f32) -> Self {
        Self {
            policy_sigma,
            value_sigma,
            time_usage_sigma,
        }
    }
}

/// Metric for tracking uncertainty values (sigma) for each loss component
#[derive(Default, Clone)]
pub struct UncertaintyMetric<B: Backend> {
    policy_sigma: f32,
    value_sigma: f32,
    time_usage_sigma: f32,
    _backend: PhantomData<B>,
}

impl<B: Backend> UncertaintyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for UncertaintyMetric<B> {
    type Input = UncertaintyInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.policy_sigma = input.policy_sigma;
        self.value_sigma = input.value_sigma;
        self.time_usage_sigma = input.time_usage_sigma;

        // Format the display to show all three uncertainties
        let formatted = format!(
            "Policy: {:.7}, Value: {:.7}, TimeUsage: {:.7}",
            self.policy_sigma, self.value_sigma, self.time_usage_sigma
        );

        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.policy_sigma = 1.0;
        self.value_sigma = 1.0;
        self.time_usage_sigma = 1.0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Uncertainties".to_string().into()
    }
}
