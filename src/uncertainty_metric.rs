use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata};

/// Input type for uncertainty metric
pub struct UncertaintyInput {
    /// Current uncertainty values (sigma) for each loss component
    pub policy_sigma: f32,
    pub value_sigma: f32,
    pub side_info_sigma: f32,
}

impl UncertaintyInput {
    pub fn new(policy_sigma: f32, value_sigma: f32, side_info_sigma: f32) -> Self {
        Self {
            policy_sigma,
            value_sigma,
            side_info_sigma,
        }
    }
}

/// Metric for tracking uncertainty values (sigma) for each loss component
#[derive(Default)]
pub struct UncertaintyMetric<B: Backend> {
    policy_sigma: f32,
    value_sigma: f32,
    side_info_sigma: f32,
    _backend: PhantomData<B>,
}

impl<B: Backend> UncertaintyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for UncertaintyMetric<B> {
    type Input = UncertaintyInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.policy_sigma = input.policy_sigma;
        self.value_sigma = input.value_sigma;
        self.side_info_sigma = input.side_info_sigma;
        
        // Format the display to show all three uncertainties
        let formatted = format!(
            "Policy: {:.3}, Value: {:.3}, SideInfo: {:.3}",
            self.policy_sigma, self.value_sigma, self.side_info_sigma
        );
        
        MetricEntry::new(
            "Uncertainties (σ)".to_string(),
            formatted.clone(),
            formatted,
        )
    }

    fn clear(&mut self) {
        self.policy_sigma = 1.0;
        self.value_sigma = 1.0;
        self.side_info_sigma = 1.0;
    }

    fn name(&self) -> String {
        "Uncertainties".to_string()
    }
}