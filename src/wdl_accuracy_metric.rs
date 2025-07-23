use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::tensor::ElementConversion;

/// The WDL (Win/Draw/Loss) accuracy metric.
/// This metric tracks the accuracy of win/draw/loss predictions from the value head.
#[derive(Default)]
pub struct WdlAccuracyMetric<B: Backend> {
    /// Current batch's accuracy percentage
    current_value: f64,
    _b: PhantomData<B>,
}

/// The [WDL accuracy metric](WdlAccuracyMetric) input type.
pub struct WdlAccuracyInput<B: Backend> {
    /// Value head outputs: [batch_size, 3] where columns are [win, draw, loss] probabilities/logits
    pub outputs: Tensor<B, 2>,
    /// Target values: [batch_size, 3] as one-hot encoded [win, draw, loss]
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> WdlAccuracyInput<B> {
    /// Creates a new WdlAccuracyInput
    pub fn new(outputs: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        Self { outputs, targets }
    }
}

impl<B: Backend> WdlAccuracyMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for WdlAccuracyMetric<B> {
    type Input = WdlAccuracyInput<B>;

    fn update(&mut self, input: &WdlAccuracyInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let outputs = &input.outputs;
        let targets = &input.targets;

        let [batch_size, num_classes] = outputs.dims();
        
        // Verify we have 3 classes (win, draw, loss)
        debug_assert_eq!(num_classes, 3, "WDL head should output 3 classes");

        // Get predicted classes (argmax along class dimension)
        let predicted_classes = outputs.clone().argmax(1);
        
        // Get target classes (argmax of one-hot encoded targets)
        let target_classes = targets.clone().argmax(1);

        // Compare predictions with targets
        let correct_predictions = predicted_classes
            .equal(target_classes)
            .int()
            .sum()
            .into_scalar()
            .elem::<i32>() as usize;

        // Calculate accuracy percentage for current batch
        let accuracy = if batch_size > 0 {
            100.0 * (correct_predictions as f64) / (batch_size as f64)
        } else {
            0.0
        };

        self.current_value = accuracy;

        MetricEntry::new(
            "WDL Accuracy".to_string(),
            format!("{:.2}%", accuracy),
            format!("{:.2}", accuracy),
        )
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
    }

    fn name(&self) -> String {
        "WDL Accuracy".to_string()
    }
}

impl<B: Backend> Numeric for WdlAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.current_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::Progress;

    #[test]
    fn test_wdl_accuracy_perfect() {
        let device = NdArrayDevice::default();
        let mut metric = WdlAccuracyMetric::<NdArray>::new();

        // Create test data: 3 examples with perfect predictions
        // Outputs as logits (before softmax)
        let outputs = Tensor::from_data(
            [
                [5.0, 0.0, 0.0],  // Strong win prediction
                [0.0, 5.0, 0.0],  // Strong draw prediction
                [0.0, 0.0, 5.0],  // Strong loss prediction
            ],
            &device,
        );

        // Targets as one-hot encoded
        let targets = Tensor::from_data(
            [
                [1.0, 0.0, 0.0],  // Win
                [0.0, 1.0, 0.0],  // Draw
                [0.0, 0.0, 1.0],  // Loss
            ],
            &device,
        );

        let input = WdlAccuracyInput::new(outputs, targets);
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 0,
            epoch_total: 1,
            iteration: 0,
            lr: None,
        };
        let entry = metric.update(&input, &metadata);

        assert_eq!(metric.value(), 100.0);
        assert!(entry.formatted.contains("100.00%"));
    }

    #[test]
    fn test_wdl_accuracy_partial() {
        let device = NdArrayDevice::default();
        let mut metric = WdlAccuracyMetric::<NdArray>::new();

        // Create test data: 4 examples, 2 correct, 2 incorrect
        let outputs = Tensor::from_data(
            [
                [5.0, 0.0, 0.0],  // Predict win (correct)
                [5.0, 0.0, 0.0],  // Predict win (incorrect, should be draw)
                [0.0, 0.0, 5.0],  // Predict loss (correct)
                [0.0, 5.0, 0.0],  // Predict draw (incorrect, should be win)
            ],
            &device,
        );

        let targets = Tensor::from_data(
            [
                [1.0, 0.0, 0.0],  // Win
                [0.0, 1.0, 0.0],  // Draw
                [0.0, 0.0, 1.0],  // Loss
                [1.0, 0.0, 0.0],  // Win
            ],
            &device,
        );

        let input = WdlAccuracyInput::new(outputs, targets);
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 0,
            epoch_total: 1,
            iteration: 0,
            lr: None,
        };
        metric.update(&input, &metadata);

        assert_eq!(metric.value(), 50.0);
    }

    #[test]
    fn test_wdl_accuracy_batch_update() {
        let device = NdArrayDevice::default();
        let mut metric = WdlAccuracyMetric::<NdArray>::new();

        // First batch: 2/3 correct
        let outputs1 = Tensor::from_data(
            [
                [5.0, 0.0, 0.0],  // Correct
                [5.0, 0.0, 0.0],  // Correct
                [0.0, 5.0, 0.0],  // Incorrect
            ],
            &device,
        );

        let targets1 = Tensor::from_data(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            &device,
        );

        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 0,
            epoch_total: 1,
            iteration: 0,
            lr: None,
        };
        metric.update(&WdlAccuracyInput::new(outputs1, targets1), &metadata);

        // First batch: 2/3 correct = 66.67%
        assert!((metric.value() - 66.67).abs() < 0.01);

        // Second batch: 1/2 correct
        let outputs2 = Tensor::from_data(
            [
                [0.0, 0.0, 5.0],  // Correct
                [0.0, 5.0, 0.0],  // Incorrect
            ],
            &device,
        );

        let targets2 = Tensor::from_data(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            &device,
        );

        metric.update(&WdlAccuracyInput::new(outputs2, targets2), &metadata);

        // Second batch: 1/2 correct = 50% (not accumulated)
        assert_eq!(metric.value(), 50.0);
        
        // Test clear
        metric.clear();
        assert_eq!(metric.value(), 0.0);
    }
}