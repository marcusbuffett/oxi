use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// A metric that measures the sum probability assigned to legal moves
/// after applying softmax normalization. This should be close to 1.0 if the
/// model is correctly outputting valid probability distributions.
#[derive(Default)]
pub struct LegalMoveProbabilityMetric<B: Backend> {
    /// Current batch's average legal move probability
    current_value: f64,
    _backend: PhantomData<B>,
}

/// Input for the legal move probability metric
pub struct LegalMoveProbabilityInput<B: Backend> {
    /// Policy logits (after masking illegal moves with large negative values)
    pub policy_logits: Tensor<B, 2>,
    /// Legal moves mask (1.0 for legal moves, 0.0 for illegal)
    pub legal_moves_mask: Tensor<B, 2>,
}

impl<B: Backend> LegalMoveProbabilityMetric<B> {
    /// Creates a new metric instance
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for LegalMoveProbabilityMetric<B> {
    type Input = LegalMoveProbabilityInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        // Apply softmax to get probabilities from logits
        let policy_probs = burn::tensor::activation::softmax(input.policy_logits.clone(), 1);

        // Multiply probabilities by legal moves mask to get only legal move probabilities
        let legal_probs = policy_probs * input.legal_moves_mask.clone();

        // Sum legal probabilities for each position
        let legal_prob_sums = legal_probs.sum_dim(1);

        // Get the average across the batch
        let batch_size = legal_prob_sums.shape().dims[0];
        let total_legal_prob = legal_prob_sums.sum().to_data().as_slice::<f32>().unwrap()[0] as f64;
        let avg_legal_prob = total_legal_prob / batch_size as f64;

        self.current_value = avg_legal_prob;

        MetricEntry::new(
            "Legal Move Probability".to_string(),
            format!("{avg_legal_prob:.4}"),
            format!("{avg_legal_prob:.4}"),
        )
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
    }

    fn name(&self) -> String {
        "Legal Move Probability".to_string()
    }
}

impl<B: Backend> Numeric for LegalMoveProbabilityMetric<B> {
    fn value(&self) -> f64 {
        self.current_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::Progress;
    use burn_ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_legal_move_probability_perfect() {
        let device = NdArrayDevice::default();
        let mut metric = LegalMoveProbabilityMetric::<NdArray>::new();

        // Create test data: 2 positions with different numbers of legal moves
        // Position 1: 3 legal moves (indices 0, 1, 2)
        // Position 2: 2 legal moves (indices 1, 3)
        let batch_size = 2;
        let num_moves = 5;

        // Policy logits (after masking)
        let policy_logits = Tensor::from_data(
            [
                [0.0, 0.0, 0.0, -1e9, -1e9],  // 3 legal moves
                [-1e9, 0.0, -1e9, 0.0, -1e9], // 2 legal moves
            ],
            &device,
        );

        // Legal moves mask
        let legal_moves_mask = Tensor::from_data(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0], // 3 legal moves
                [0.0, 1.0, 0.0, 1.0, 0.0], // 2 legal moves
            ],
            &device,
        );

        let input = LegalMoveProbabilityInput {
            policy_logits,
            legal_moves_mask,
        };

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

        // After softmax, all probability mass should be on legal moves
        // So the sum should be very close to 1.0 for each position
        let value = metric.value();
        assert!(
            value > 0.99 && value <= 1.0,
            "Expected value close to 1.0, got {}",
            value
        );
        assert!(entry.formatted.contains("1.00") || entry.formatted.contains("0.99"));
    }

    #[test]
    fn test_legal_move_probability_batch() {
        let device = NdArrayDevice::default();
        let mut metric = LegalMoveProbabilityMetric::<NdArray>::new();

        // Test batch with mixed legal move counts
        let policy_logits = Tensor::from_data(
            [
                [-1e9, 0.0, 0.0, 0.0],   // 3 legal moves
                [0.0, -1e9, -1e9, -1e9], // 1 legal move
                [0.0, 0.0, -1e9, -1e9],  // 2 legal moves
            ],
            &device,
        );

        let legal_moves_mask = Tensor::from_data(
            [
                [0.0, 1.0, 1.0, 1.0], // 3 legal moves
                [1.0, 0.0, 0.0, 0.0], // 1 legal move
                [1.0, 1.0, 0.0, 0.0], // 2 legal moves
            ],
            &device,
        );

        let input = LegalMoveProbabilityInput {
            policy_logits,
            legal_moves_mask,
        };

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

        // Each position should have sum close to 1.0, so average should be close to 1.0
        let value = metric.value();
        assert!(
            value > 0.99 && value <= 1.0,
            "Expected value close to 1.0, got {}",
            value
        );

        // Clear and update with a new batch
        metric.clear();

        let policy_logits2 = Tensor::from_data(
            [
                [0.0, 0.0, -1e9], // 2 legal moves
            ],
            &device,
        );

        let legal_moves_mask2 = Tensor::from_data(
            [
                [1.0, 1.0, 0.0], // 2 legal moves
            ],
            &device,
        );

        let input2 = LegalMoveProbabilityInput {
            policy_logits: policy_logits2,
            legal_moves_mask: legal_moves_mask2,
        };

        metric.update(&input2, &metadata);

        // Should only reflect the current batch
        let value2 = metric.value();
        assert!(
            value2 > 0.99 && value2 <= 1.0,
            "Expected value close to 1.0, got {}",
            value2
        );
    }
}
