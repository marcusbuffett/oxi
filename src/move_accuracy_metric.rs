use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// Input type for move accuracy metric
#[derive(Clone)]
pub struct MoveAccuracyInput<B: Backend> {
    /// Policy logits [batch_size, num_moves]
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
            format!("Avg Prob: {:.4}", batch_avg),
            format!("{:.4}", batch_avg),
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

