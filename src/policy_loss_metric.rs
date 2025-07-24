use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric, NumericEntry};

/// Input type for policy loss metric
pub struct PolicyLossInput<B: Backend> {
    /// Policy loss value (after uncertainty weighting)
    pub loss: Tensor<B, 1>,
    /// Raw policy loss (before uncertainty weighting)
    pub raw_loss: Option<Tensor<B, 1>>,
    /// Current GradNorm weight applied to this loss
    pub grad_weight: Option<f32>,
    /// Last measured gradient norm for this head
    pub grad_norm: Option<f32>,
}

impl<B: Backend> PolicyLossInput<B> {
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self {
            loss,
            raw_loss: None,
            grad_weight: None,
            grad_norm: None,
        }
    }

    pub fn with_raw_loss(mut self, raw_loss: Tensor<B, 1>) -> Self {
        self.raw_loss = Some(raw_loss);
        self
    }

    pub fn with_grad_info(mut self, weight: f32, grad_norm: Option<f32>) -> Self {
        self.grad_weight = Some(weight);
        self.grad_norm = grad_norm;
        self
    }
}

/// Metric for tracking policy loss
#[derive(Default, Clone)]
pub struct PolicyLossMetric<B: Backend> {
    current_value: f64,
    current_raw_value: Option<f64>,
    current_weight: Option<f64>,
    current_grad_norm: Option<f64>,
    _backend: PhantomData<B>,
}

impl<B: Backend> PolicyLossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for PolicyLossMetric<B> {
    type Input = PolicyLossInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let loss_value = input.loss.clone().into_scalar().elem::<f32>() as f64;

        self.current_value = loss_value;

        let raw_value = input
            .raw_loss
            .as_ref()
            .map(|raw| raw.clone().into_scalar().elem::<f32>() as f64);
        self.current_raw_value = raw_value;

        self.current_weight = input.grad_weight.map(|w| w as f64);
        self.current_grad_norm = input.grad_norm.map(|g| g as f64);

        let base_value = raw_value.unwrap_or(loss_value);
        let weight_display = self
            .current_weight
            .map(|w| format!("{w:.4}"))
            .unwrap_or_else(|| "N/A".to_string());
        let grad_display = self
            .current_grad_norm
            .map(|g| format!("{g:.4}"))
            .unwrap_or_else(|| "N/A".to_string());

        let formatted = format!(
            "Loss: {base:.6}, Weight: {weight}, Weighted: {weighted:.6}, Grad: {grad}",
            base = base_value,
            weight = weight_display,
            weighted = loss_value,
            grad = grad_display
        );

        MetricEntry::new(
            "Policy Loss".to_string().into(),
            formatted.clone(),
            formatted,
        )
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
        self.current_raw_value = None;
        self.current_weight = None;
        self.current_grad_norm = None;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Policy Loss".to_string().into()
    }
}

impl<B: Backend> Numeric for PolicyLossMetric<B> {
    fn value(&self) -> NumericEntry {
        // Return raw loss if available, otherwise return weighted loss
        // Cap at 10 for display purposes
        let value = self.current_raw_value.unwrap_or(self.current_value);
        NumericEntry::Value(value.min(10.0))
    }
}
