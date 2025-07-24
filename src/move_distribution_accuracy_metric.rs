use core::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};

/// Input type for move distribution accuracy metric
pub struct MoveDistributionAccuracyInput<B: Backend> {
    /// Policy logits [batch_size, num_moves]
    pub policy_logits: Tensor<B, 2>,
    /// Target move distributions [batch_size, num_moves]
    pub target_distributions: Tensor<B, 2>,
}

impl<B: Backend> MoveDistributionAccuracyInput<B> {
    pub fn new(policy_logits: Tensor<B, 2>, target_distributions: Tensor<B, 2>) -> Self {
        Self {
            policy_logits,
            target_distributions,
        }
    }
}

/// Metric for tracking distribution overlap/similarity
#[derive(Default)]
pub struct MoveDistributionAccuracyMetric<B: Backend> {
    current_overlap: f64,
    current_kl_div: f64,
    current_js_div: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> MoveDistributionAccuracyMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for MoveDistributionAccuracyMetric<B> {
    type Input = MoveDistributionAccuracyInput<B>;

    fn update(&mut self, _input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        MetricEntry::new(
            "TODO: implement move distribution accuracy metric".to_string(),
            "TODO".to_string(),
            "TODO".to_string(),
        )
        // let batch_size = input.policy_logits.shape().dims[0];
        //
        // // Convert logits to probabilities
        // let pred_probs = softmax(input.policy_logits.clone(), 1);
        //
        // // Add small epsilon to avoid log(0)
        // let epsilon = 1e-10;
        // let pred_probs = pred_probs.add_scalar(epsilon);
        // let target_probs = input.target_distributions.clone().add_scalar(epsilon);
        //
        // // Normalize target distributions (in case they don't sum to 1)
        // let target_sums = target_probs.clone().sum_dim(1);
        // let target_probs = target_probs / target_sums.unsqueeze();
        //
        // // Compute overlap (sum of minimum probabilities)
        // let overlap = pred_probs.clone().min_pair(target_probs.clone()).sum_dim(1);
        // let batch_overlap = overlap.mean().into_scalar().elem::<f32>() as f64;
        //
        // // Compute KL divergence: sum(target * log(target/pred))
        // let log_pred = pred_probs.clone().log();
        // let log_target = target_probs.clone().log();
        // let kl_div = (target_probs.clone() * (log_target.clone() - log_pred.clone()))
        //     .sum_dim(1)
        //     .mean()
        //     .into_scalar()
        //     .elem::<f32>() as f64;
        //
        // // Compute JS divergence (symmetric version of KL)
        // let m = (pred_probs.clone() + target_probs.clone()) / 2.0;
        // let log_m = m.clone().log();
        // let js_div_part1 = (pred_probs * (log_pred - log_m.clone())).sum_dim(1).mean();
        // let js_div_part2 = (target_probs * (log_target - log_m)).sum_dim(1).mean();
        // let js_div_sum = js_div_part1.mul_scalar(0.5) + js_div_part2.mul_scalar(0.5);
        // let js_div = js_div_sum.into_scalar().elem::<f32>() as f64;
        //
        // self.current_overlap = batch_overlap;
        // self.current_kl_div = kl_div;
        // self.current_js_div = js_div;
        //
        // MetricEntry::new(
        //     "Move Distribution Accuracy".to_string(),
        //     format!(
        //         "Overlap: {:.2}%, KL: {:.4}, JS: {:.4}",
        //         batch_overlap * 100.0,
        //         kl_div,
        //         js_div
        //     ),
        //     format!("{:.2}", batch_overlap * 100.0),
        // )
    }

    fn clear(&mut self) {
        self.current_overlap = 0.0;
        self.current_kl_div = 0.0;
        self.current_js_div = 0.0;
    }

    fn name(&self) -> String {
        "Move Distribution Accuracy".to_string()
    }
}

impl<B: Backend> Numeric for MoveDistributionAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.current_overlap * 100.0
    }
}
