use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// Input type for value loss metric
pub struct ValueLossInput<B: Backend> {
    /// Value loss value (after uncertainty weighting)
    pub loss: Tensor<B, 1>,
    /// Raw value loss (before uncertainty weighting)
    pub raw_loss: Option<Tensor<B, 1>>,
}

impl<B: Backend> ValueLossInput<B> {
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self {
            loss,
            raw_loss: None,
        }
    }

    pub fn with_raw_loss(mut self, raw_loss: Tensor<B, 1>) -> Self {
        self.raw_loss = Some(raw_loss);
        self
    }
}

/// Metric for tracking value loss
#[derive(Default)]
pub struct ValueLossMetric<B: Backend> {
    current_value: f64,
    current_raw_value: Option<f64>,
    _backend: PhantomData<B>,
}

impl<B: Backend> ValueLossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for ValueLossMetric<B> {
    type Input = ValueLossInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let loss_value = input.loss.clone().into_scalar().elem::<f32>() as f64;

        self.current_value = loss_value;

        let raw_value = input
            .raw_loss
            .as_ref()
            .map(|raw| raw.clone().into_scalar().elem::<f32>() as f64);
        self.current_raw_value = raw_value;

        let formatted = if let Some(raw) = raw_value {
            format!("Raw: {:.6}, Weighted: {:.6}", raw, loss_value)
        } else {
            format!("{:.6}", loss_value)
        };

        MetricEntry::new("Value Loss".to_string(), formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
        self.current_raw_value = None;
    }

    fn name(&self) -> String {
        "Value Loss".to_string()
    }
}

impl<B: Backend> Numeric for ValueLossMetric<B> {
    fn value(&self) -> f64 {
        // Return raw loss if available, otherwise return weighted loss
        self.current_raw_value.unwrap_or(self.current_value).min(2.0)
    }
}

