use std::marker::PhantomData;

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::train::metric::{Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};

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

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let batch_size = input.policy_logits.shape().dims::<2>()[0];

        // Convert logits to probabilities
        let probs = softmax(input.policy_logits.clone(), 1);

        // Vectorized: gather probabilities at target indices
        // targets: [batch_size] -> reshape to [batch_size, 1] for gather
        let targets_2d = input.policy_targets.clone().reshape([batch_size, 1]);

        // Gather the probability at each target index: [batch_size, 1]
        let target_probs = probs.gather(1, targets_2d);

        // Sum all target probabilities and get mean (single GPU sync)
        let batch_avg = target_probs.mean().into_scalar().elem::<f32>() as f64;

        // Update current value
        self.current = batch_avg;

        SerializedEntry::new(
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

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}

/// Combined metric for computing both top-1 and top-k accuracy efficiently.
/// Uses argmax for top-1 (fast) and only computes topk once for top-k.
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

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let batch_size = input.policy_logits.shape().dims::<2>()[0];

        let batch_accuracy = if batch_size > 0 {
            if self.k == 1 {
                let predictions = input.policy_logits.clone().argmax(1);
                let targets = input.policy_targets.clone().reshape([batch_size, 1]);
                let correct = predictions.equal(targets);
                let correct_count = correct.int().sum().into_scalar().elem::<i32>() as f64;
                correct_count / batch_size as f64
            } else {
                // Fast top-k: count moves with higher logits than target.
                // If fewer than k moves beat the target, target is in top-k.
                // O(batch * moves) comparisons vs O(batch * moves * log(moves)) for topk sort.
                let targets_2d = input.policy_targets.clone().reshape([batch_size, 1]);

                let target_logits = input.policy_logits.clone().gather(1, targets_2d);

                let higher_than_target = input.policy_logits.clone().greater(target_logits);

                let num_higher = higher_than_target.int().sum_dim(1);

                let in_top_k = num_higher.lower_elem(self.k as i32);

                let correct_count = in_top_k.int().sum().into_scalar().elem::<i32>() as f64;
                correct_count / batch_size as f64
            }
        } else {
            0.0
        };

        self.current = batch_accuracy;

        SerializedEntry::new(
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

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}
