use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// Input type for move accuracy metric
#[derive(Clone)]
pub struct MoveAccuracyInput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    /// Policy targets (ground truth move indices) [batch_size]
    pub policy_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> MoveAccuracyInput<B> {
    pub fn new(policy_logits: Tensor<B, 2>, policy_targets: Tensor<B, 1, Int>) -> Self {
        Self {
            policy_logits,
            policy_targets,
        }
    }
}

/// Metric for tracking the average probability assigned to the correct move
#[derive(Default)]
pub struct MoveAccuracyMetric<B: Backend> {
    current: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> MoveAccuracyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for MoveAccuracyMetric<B> {
    type Input = MoveAccuracyInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = input.policy_logits.shape().dims[0];

        // Convert logits to probabilities
        let probs = softmax(input.policy_logits.clone(), 1);

        // Sum of probabilities assigned to correct moves
        let mut batch_prob_sum = 0.0;

        // Process each example in the batch
        for i in 0..batch_size {
            // Get target index for this example
            let target = input
                .policy_targets
                .clone()
                .slice([i..i + 1])
                .into_scalar()
                .elem::<i32>() as usize;

            // Get probability assigned to the target move
            let target_prob = probs
                .clone()
                .slice([i..i + 1, target..target + 1])
                .into_scalar()
                .elem::<f32>() as f64;

            batch_prob_sum += target_prob;
        }

        // Calculate average probability for this batch
        let batch_avg = if batch_size > 0 {
            batch_prob_sum / batch_size as f64
        } else {
            0.0
        };

        // Update current value
        self.current = batch_avg;

        MetricEntry::new(
            "Move Accuracy".to_string(),
            format!("Avg Prob: {batch_avg:.4}"),
            format!("{batch_avg:.4}"),
        )
    }

    fn clear(&mut self) {
        self.current = 0.0;
    }

    fn name(&self) -> String {
        "Move Accuracy".to_string()
    }
}

impl<B: Backend> Numeric for MoveAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.current
    }
}

/// Metric for tracking top-1 accuracy (percentage of correct top predictions)
#[derive(Default)]
pub struct MoveTop1AccuracyMetric<B: Backend> {
    current: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> MoveTop1AccuracyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for MoveTop1AccuracyMetric<B> {
    type Input = MoveAccuracyInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = input.policy_logits.shape().dims[0];

        // Get predicted moves (argmax of logits)
        let predicted_moves = input.policy_logits.clone().argmax(1);

        // Count correct predictions
        let mut correct_count = 0;

        // Process each example in the batch
        for i in 0..batch_size {
            // Get target index for this example
            let target = input
                .policy_targets
                .clone()
                .slice([i..i + 1])
                .into_scalar()
                .elem::<i32>();

            // Get predicted index for this example
            let predicted = predicted_moves
                .clone()
                .slice([i..i + 1])
                .into_scalar()
                .elem::<i32>();

            if target == predicted {
                correct_count += 1;
            }
        }

        // Calculate top-1 accuracy for this batch
        let batch_accuracy = if batch_size > 0 {
            correct_count as f64 / batch_size as f64
        } else {
            0.0
        };

        // Update current value
        self.current = batch_accuracy;

        MetricEntry::new(
            "Move Top-1 Accuracy".to_string(),
            format!("Top-1: {:.1}%", batch_accuracy * 100.0),
            format!("{batch_accuracy:.4}"),
        )
    }

    fn clear(&mut self) {
        self.current = 0.0;
    }

    fn name(&self) -> String {
        "Move Top-1 Accuracy".to_string()
    }
}

impl<B: Backend> Numeric for MoveTop1AccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.current
    }
}
