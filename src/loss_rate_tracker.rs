use burn::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// Tracks loss improvement rates for rate-based balancing
#[derive(Debug)]
pub struct LossRateTracker {
    /// Window size for computing loss rates (number of recent measurements)
    window_size: usize,
    /// Historical loss values for each component
    policy_losses: VecDeque<f32>,
    value_losses: VecDeque<f32>,
    time_usage_losses: VecDeque<f32>,
    /// Current loss rates (improvement per step) for each component
    policy_rate: f32,
    value_rate: f32,
    time_usage_rate: f32,
}

impl LossRateTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            policy_losses: VecDeque::with_capacity(window_size + 1),
            value_losses: VecDeque::with_capacity(window_size + 1),
            time_usage_losses: VecDeque::with_capacity(window_size + 1),
            policy_rate: 0.0,
            value_rate: 0.0,
            time_usage_rate: 0.0,
        }
    }

    /// Add new loss measurements and update rates
    pub fn update(&mut self,
                  policy_loss: f32,
                  value_loss: f32,
                  time_usage_loss: f32) {

        // Add new measurements
        self.policy_losses.push_back(policy_loss);
        self.value_losses.push_back(value_loss);
        self.time_usage_losses.push_back(time_usage_loss);

        // Maintain window size
        if self.policy_losses.len() > self.window_size + 1 {
            self.policy_losses.pop_front();
            self.value_losses.pop_front();
            self.time_usage_losses.pop_front();
        }

        // Update rates if we have enough data
        if self.policy_losses.len() >= 2 {
            self.policy_rate = self.compute_rate(&self.policy_losses);
            self.value_rate = self.compute_rate(&self.value_losses);
            self.time_usage_rate = self.compute_rate(&self.time_usage_losses);
        }
    }

    /// Compute improvement rate using linear regression over the window
    fn compute_rate(&self, losses: &VecDeque<f32>) -> f32 {
        if losses.len() < 2 {
            return 0.0;
        }

        let n = losses.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &loss) in losses.iter().enumerate() {
            let x = i as f32;
            let y = loss;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Return negative slope as improvement rate (more negative = faster improvement)
        -slope
    }

    /// Get current loss improvement rates (higher = faster improvement)
    pub fn get_rates(&self) -> (f32, f32, f32) {
        (self.policy_rate, self.value_rate, self.time_usage_rate)
    }

    /// Get normalized weights based on improvement rates (for DWA-style balancing)
    pub fn get_normalized_weights(&self, temperature: f32) -> (f32, f32, f32) {
        let (p_rate, v_rate, t_rate) = self.get_rates();

        // Avoid division by zero and handle edge cases
        let rates = [p_rate.max(1e-8), v_rate.max(1e-8), t_rate.max(1e-8)];

        // Apply temperature scaling and softmax normalization
        let max_rate = rates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_rates: Vec<f32> = rates.iter().map(|&r| ((r - max_rate) / temperature).exp()).collect();

        let sum_exp = exp_rates.iter().sum::<f32>();
        let normalized = exp_rates.iter().map(|&exp| exp / sum_exp).collect::<Vec<_>>();

        (normalized[0], normalized[1], normalized[2])
    }

    /// Check if we have enough data for reliable rate computation
    pub fn is_ready(&self) -> bool {
        self.policy_losses.len() >= self.window_size
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.policy_losses.clear();
        self.value_losses.clear();
        self.time_usage_losses.clear();
        self.policy_rate = 0.0;
        self.value_rate = 0.0;
        self.time_usage_rate = 0.0;
    }
}

// Global shared tracker for DWA (Dynamic Weight Averaging)
static GLOBAL_LOSS_TRACKER: OnceLock<Arc<Mutex<LossRateTracker>>> = OnceLock::new();

/// Set the global loss tracker (call once during initialization)
pub fn set_global_loss_tracker(tracker: Arc<Mutex<LossRateTracker>>) {
    let _ = GLOBAL_LOSS_TRACKER.set(tracker);
}

/// Get the global loss tracker if initialized
pub fn get_global_loss_tracker() -> Option<&'static Arc<Mutex<LossRateTracker>>> {
    GLOBAL_LOSS_TRACKER.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_rate_tracker() {
        let mut tracker = LossRateTracker::new(3);

        // Add some test data with decreasing losses (improving)
        tracker.update(1.0, 1.0, 1.0);
        tracker.update(0.9, 0.8, 0.85);
        tracker.update(0.8, 0.6, 0.7);

        assert!(tracker.is_ready());

        let (p_rate, v_rate, t_rate) = tracker.get_rates();

        // All rates should be positive (indicating improvement)
        assert!(p_rate > 0.0, "Policy should show improvement");
        assert!(v_rate > 0.0, "Value should show improvement");
        assert!(t_rate > 0.0, "Time usage should show improvement");

        // Value loss improved the most, so should have highest rate
        assert!(v_rate > p_rate, "Value should improve faster than policy");
        assert!(v_rate > t_rate, "Value should improve faster than time usage");
    }

    #[test]
    fn test_normalized_weights() {
        let mut tracker = LossRateTracker::new(2);

        // Simulate different improvement rates
        tracker.update(1.0, 1.0, 1.0);
        tracker.update(0.9, 0.7, 0.8); // Value improves fastest, policy slowest

        let (p_w, v_w, t_w) = tracker.get_normalized_weights(1.0);

        // Weights should sum to 1
        let sum = p_w + v_w + t_w;
        assert!((sum - 1.0).abs() < 0.001, "Weights should sum to 1, got {}", sum);

        // Value should get highest weight (fastest improvement)
        // Policy should get lowest weight (slowest improvement)
        assert!(v_w > p_w, "Value should get higher weight than policy");
        assert!(v_w > t_w, "Value should get higher weight than time usage");
    }
}
