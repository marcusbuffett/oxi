use burn::lr_scheduler::LrScheduler;
use burn::tensor::backend::Backend;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct PlateauDetector {
    window: VecDeque<f64>,
    window_size: usize,
    improvement_threshold: f64,
}

impl PlateauDetector {
    pub fn new(window_size: usize, improvement_threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            improvement_threshold,
        }
    }

    pub fn record(&mut self, loss: f64) {
        if !loss.is_finite() {
            return;
        }

        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(loss);
    }

    pub fn is_plateau(&self) -> bool {
        if self.window.len() < self.window_size {
            return false;
        }

        let (old_avg, new_avg) = match self.half_window_averages() {
            Some(avgs) => avgs,
            None => return false,
        };

        if old_avg <= 0.0 {
            return false;
        }

        let relative_improvement = (old_avg - new_avg) / old_avg;
        relative_improvement < self.improvement_threshold
    }

    fn half_window_averages(&self) -> Option<(f64, f64)> {
        if self.window.len() < 4 {
            return None;
        }

        let mid = self.window.len() / 2;
        let first_half: Vec<f64> = self.window.iter().take(mid).copied().collect();
        let second_half: Vec<f64> = self.window.iter().skip(mid).copied().collect();

        if first_half.is_empty() || second_half.is_empty() {
            return None;
        }

        let old_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let new_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

        Some((old_avg, new_avg))
    }

    pub fn relative_improvement(&self) -> Option<f64> {
        let (old_avg, new_avg) = self.half_window_averages()?;

        if old_avg <= 0.0 {
            return None;
        }

        Some((old_avg - new_avg) / old_avg)
    }

    pub fn oldest_loss(&self) -> Option<f64> {
        self.window.front().copied()
    }

    pub fn newest_loss(&self) -> Option<f64> {
        self.window.back().copied()
    }

    pub fn fill_ratio(&self) -> f64 {
        self.window.len() as f64 / self.window_size as f64
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn current_window_len(&self) -> usize {
        self.window.len()
    }

    pub fn improvement_threshold(&self) -> f64 {
        self.improvement_threshold
    }

    pub fn reset(&mut self) {
        self.window.clear();
    }

    pub fn window_values(&self) -> Vec<f64> {
        self.window.iter().copied().collect()
    }
}

#[derive(Clone, Debug)]
pub struct ReduceOnPlateauScheduler {
    initial_lr: f64,
    current_lr: f64,
    min_lr: f64,
    reduction_factor: f64,
    detector: PlateauDetector,
    iteration: usize,
    num_reductions: usize,
    warmup_iterations: usize,
    cosine_period: usize,
}

impl ReduceOnPlateauScheduler {
    pub fn new(
        initial_lr: f64,
        min_lr: f64,
        reduction_factor: f64,
        window_size: usize,
        improvement_threshold: f64,
        warmup_iterations: usize,
    ) -> Self {
        Self::with_cosine(initial_lr, min_lr, reduction_factor, window_size, improvement_threshold, warmup_iterations, 0)
    }

    pub fn with_cosine(
        initial_lr: f64,
        min_lr: f64,
        reduction_factor: f64,
        window_size: usize,
        improvement_threshold: f64,
        warmup_iterations: usize,
        cosine_period: usize,
    ) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            min_lr,
            reduction_factor,
            detector: PlateauDetector::new(window_size, improvement_threshold),
            iteration: 0,
            num_reductions: 0,
            warmup_iterations,
            cosine_period,
        }
    }

    pub fn record_batch(&mut self, loss: f64) -> bool {
        self.iteration += 1;
        self.record_measurement(loss)
    }

    fn record_measurement(&mut self, loss: f64) -> bool {
        if !loss.is_finite() {
            return false;
        }

        if self.is_warming_up() {
            return false;
        }

        self.detector.record(loss);

        if self.detector.is_plateau() {
            let new_lr = (self.current_lr * self.reduction_factor).max(self.min_lr);
            if new_lr < self.current_lr {
                let window_values = self.detector.window_values();
                let old_loss = self.detector.oldest_loss().unwrap_or(f64::NAN);
                let new_loss = self.detector.newest_loss().unwrap_or(f64::NAN);
                let rel_improvement = self.detector.relative_improvement().unwrap_or(f64::NAN);
                let threshold = self.detector.improvement_threshold();

                tracing::warn!(
                    target: "plateau_detection",
                    "PLATEAU DETECTED at iteration {}: LR {} -> {} | loss: {:.6} -> {:.6} | improvement: {:.4}% < threshold: {:.4}% | window_size: {} | reduction #{}",
                    self.iteration,
                    self.current_lr,
                    new_lr,
                    old_loss,
                    new_loss,
                    rel_improvement * 100.0,
                    threshold * 100.0,
                    window_values.len(),
                    self.num_reductions + 1
                );

                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("plateau_detection.log")
                {
                    let _ = writeln!(
                        file,
                        "--- Plateau Detection at iteration {} ---",
                        self.iteration
                    );
                    let _ = writeln!(file, "LR: {} -> {}", self.current_lr, new_lr);
                    let _ = writeln!(file, "Loss: {:.6} -> {:.6}", old_loss, new_loss);
                    let _ = writeln!(
                        file,
                        "Relative improvement: {:.4}% (threshold: {:.4}%)",
                        rel_improvement * 100.0,
                        threshold * 100.0
                    );
                    let _ = writeln!(file, "Window values ({}):", window_values.len());
                    for (i, v) in window_values.iter().enumerate() {
                        let _ = writeln!(file, "  [{:4}] {:.6}", i, v);
                    }
                    let _ = writeln!(file, "");
                }

                self.current_lr = new_lr;
                self.num_reductions += 1;
                self.detector.reset();
                return true;
            }
        }
        false
    }

    pub fn get_lr(&self) -> f64 {
        if self.warmup_iterations > 0 && self.iteration < self.warmup_iterations {
            let warmup_progress = self.iteration as f64 / self.warmup_iterations as f64;
            self.current_lr * warmup_progress
        } else if self.cosine_period > 0 {
            // Cosine decay from current_lr to min_lr over cosine_period steps post-warmup
            let steps_since_warmup = self.iteration.saturating_sub(self.warmup_iterations);
            let progress = (steps_since_warmup as f64 / self.cosine_period as f64).min(1.0);
            let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.current_lr - self.min_lr) * cosine_factor
        } else {
            self.current_lr
        }
    }

    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    pub fn is_warming_up(&self) -> bool {
        self.warmup_iterations > 0 && self.iteration < self.warmup_iterations
    }

    pub fn warmup_progress(&self) -> f64 {
        if self.warmup_iterations == 0 {
            1.0
        } else {
            (self.iteration as f64 / self.warmup_iterations as f64).min(1.0)
        }
    }

    pub fn relative_improvement(&self) -> Option<f64> {
        self.detector.relative_improvement()
    }

    pub fn window_fill_ratio(&self) -> f64 {
        self.detector.fill_ratio()
    }

    pub fn num_reductions(&self) -> usize {
        self.num_reductions
    }

    pub fn improvement_threshold(&self) -> f64 {
        self.detector.improvement_threshold()
    }

    pub fn should_stop(&self) -> bool {
        if self.is_warming_up() {
            return false;
        }
        (self.current_lr - self.min_lr).abs() < 1e-10 && self.detector.is_plateau()
    }

    /// Reset the scheduler to initial state (for transitioning to a new training phase)
    pub fn reset_to_initial(&mut self) {
        self.current_lr = self.initial_lr;
        self.iteration = 0;
        self.num_reductions = 0;
        self.detector.reset();
        tracing::info!(
            "LR scheduler reset to initial: lr={:.6}, warmup_iterations={}",
            self.initial_lr,
            self.warmup_iterations
        );
    }
}

impl LrScheduler for ReduceOnPlateauScheduler {
    type Record<B: Backend> = (
        f64,      // initial_lr
        f64,      // current_lr
        f64,      // min_lr
        f64,      // reduction_factor
        usize,    // window_size
        f64,      // improvement_threshold
        usize,    // iteration
        usize,    // num_reductions
        (usize, usize), // (warmup_iterations, cosine_period)
        Vec<f64>, // detector window
    );

    fn step(&mut self) -> f64 {
        self.get_lr()
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        (
            self.initial_lr,
            self.current_lr,
            self.min_lr,
            self.reduction_factor,
            self.detector.window_size,
            self.detector.improvement_threshold,
            self.iteration,
            self.num_reductions,
            (self.warmup_iterations, self.cosine_period),
            self.detector.window.iter().copied().collect(),
        )
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.initial_lr = record.0;
        self.current_lr = record.1;
        self.min_lr = record.2;
        self.reduction_factor = record.3;
        self.detector = PlateauDetector::new(record.4, record.5);
        for loss in record.9 {
            self.detector.record(loss);
        }
        self.iteration = record.6;
        self.num_reductions = record.7;
        self.warmup_iterations = record.8.0;
        self.cosine_period = record.8.1;
        self
    }
}