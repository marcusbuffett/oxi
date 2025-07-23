use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// Input type for side info loss metric
pub struct SideInfoLossInput<B: Backend> {
    /// Side info loss value (after uncertainty weighting)
    pub loss: Tensor<B, 1>,
    /// Raw side info loss (before uncertainty weighting)
    pub raw_loss: Option<Tensor<B, 1>>,
}

impl<B: Backend> SideInfoLossInput<B> {
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

/// Metric for tracking side info loss
#[derive(Default)]
pub struct SideInfoLossMetric<B: Backend> {
    current_value: f64,
    current_raw_value: Option<f64>,
    _backend: PhantomData<B>,
}

impl<B: Backend> SideInfoLossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for SideInfoLossMetric<B> {
    type Input = SideInfoLossInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        // return MetricEntry::new(
        //     "SideInfo Loss".to_string(),
        //     "N/A".to_string(),
        //     "N/A".to_string(),
        // );
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

        MetricEntry::new("SideInfo Loss".to_string(), formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
        self.current_raw_value = None;
    }

    fn name(&self) -> String {
        "SideInfo Loss".to_string()
    }
}

impl<B: Backend> Numeric for SideInfoLossMetric<B> {
    fn value(&self) -> f64 {
        // Return raw loss if available, otherwise return weighted loss
        self.current_raw_value.unwrap_or(self.current_value)
    }
}
