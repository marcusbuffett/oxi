use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric, NumericEntry};

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
#[derive(Default, Clone)]
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
            "Move Accuracy".to_string().into(),
            format!("Avg Prob: {batch_avg:.4}"),
            format!("{batch_avg:.4}"),
        )
    }

    fn clear(&mut self) {
        self.current = 0.0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Move Accuracy".to_string().into()
    }
}

impl<B: Backend> Numeric for MoveAccuracyMetric<B> {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}

/// Metric for tracking top-k accuracy (percentage of targets contained in the model's top-k predictions)
#[derive(Clone)]
pub struct MoveTopKAccuracyMetric<B: Backend> {
    current: f64,
    k: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> MoveTopKAccuracyMetric<B> {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Top-k accuracy metric requires k > 0");
        Self {
            current: 0.0,
            k,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MoveTopKAccuracyMetric<B> {
    type Input = MoveAccuracyInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = input.policy_logits.shape().dims[0];
        let num_moves = input.policy_logits.shape().dims[1];
        let effective_k = self.k.min(num_moves);

        let mut correct_count = 0;

        if batch_size > 0 && effective_k > 0 {
            let (_, top_indices) = input
                .policy_logits
                .clone()
                .topk_with_indices(effective_k, 1);

            for i in 0..batch_size {
                let target = input
                    .policy_targets
                    .clone()
                    .slice([i..i + 1])
                    .into_scalar()
                    .elem::<i32>();

                let candidate_indices =
                    top_indices.clone().slice([i..i + 1]).reshape([effective_k]);

                for j in 0..effective_k {
                    let candidate = candidate_indices
                        .clone()
                        .slice([j..j + 1])
                        .into_scalar()
                        .elem::<i32>();
                    if candidate == target {
                        correct_count += 1;
                        break;
                    }
                }
            }
        }

        let batch_accuracy = if batch_size > 0 {
            correct_count as f64 / batch_size as f64
        } else {
            0.0
        };

        self.current = batch_accuracy;

        let label = format!("Move Top-{} Accuracy", self.k);
        MetricEntry::new(
            label.clone().into(),
            format!("Top-{}: {:.1}%", self.k, batch_accuracy * 100.0),
            format!("{batch_accuracy:.4}"),
        )
    }

    fn clear(&mut self) {
        self.current = 0.0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        format!("Move Top-{} Accuracy", self.k).to_string().into()
    }
}

impl<B: Backend> Numeric for MoveTopKAccuracyMetric<B> {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}
