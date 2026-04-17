use burn::lr_scheduler::LrScheduler;
use burn::tensor::backend::Backend;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct PlateauDetector {
    window: VecDeque<f64>,
    window_size: usize,
    /// t-statistic threshold: plateau is declared when t < this value
    /// (i.e., we can't confidently say loss is still decreasing)
    t_threshold: f64,
}

impl PlateauDetector {
    pub fn new(window_size: usize, t_threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            t_threshold,
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

    /// Welch's t-test comparing first half vs second half of the window.
    /// Returns the t-statistic (positive means loss decreased from first to second half).
    fn welch_t_statistic(&self) -> Option<f64> {
        if self.window.len() < self.window_size {
            return None;
        }

        let mid = self.window.len() / 2;
        let n1 = mid as f64;
        let n2 = (self.window.len() - mid) as f64;

        if n1 < 2.0 || n2 < 2.0 {
            return None;
        }

        let (sum1, sum_sq1) = self
            .window
            .iter()
            .take(mid)
            .fold((0.0, 0.0), |(s, sq), &v| (s + v, sq + v * v));
        let (sum2, sum_sq2) = self
            .window
            .iter()
            .skip(mid)
            .fold((0.0, 0.0), |(s, sq), &v| (s + v, sq + v * v));

        let mean1 = sum1 / n1;
        let mean2 = sum2 / n2;

        // Sample variance: Var = (sum_sq - n*mean^2) / (n - 1)
        let var1 = (sum_sq1 - n1 * mean1 * mean1) / (n1 - 1.0);
        let var2 = (sum_sq2 - n2 * mean2 * mean2) / (n2 - 1.0);

        if var1 <= 0.0 && var2 <= 0.0 {
            return None;
        }

        let se = (var1 / n1 + var2 / n2).sqrt();
        if se < 1e-15 {
            return None;
        }

        // Positive t means first half had higher loss (= loss is decreasing)
        Some((mean1 - mean2) / se)
    }

    pub fn is_plateau(&self) -> bool {
        match self.welch_t_statistic() {
            Some(t) => t < self.t_threshold,
            None => false,
        }
    }

    /// Returns the Welch t-statistic if the window is full.
    pub fn t_statistic(&self) -> Option<f64> {
        self.welch_t_statistic()
    }

    /// Returns the relative improvement (fraction) between first and second half means.
    /// Kept for display/logging purposes.
    pub fn relative_improvement(&self) -> Option<f64> {
        if self.window.len() < 4 {
            return None;
        }
        let mid = self.window.len() / 2;
        let first_half_mean: f64 = self.window.iter().take(mid).sum::<f64>() / mid as f64;
        let second_half_mean: f64 =
            self.window.iter().skip(mid).sum::<f64>() / (self.window.len() - mid) as f64;

        if first_half_mean <= 0.0 {
            return None;
        }
        Some((first_half_mean - second_half_mean) / first_half_mean)
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

    pub fn t_threshold(&self) -> f64 {
        self.t_threshold
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
}

impl ReduceOnPlateauScheduler {
    pub fn new(
        initial_lr: f64,
        min_lr: f64,
        reduction_factor: f64,
        window_size: usize,
        t_threshold: f64,
        warmup_iterations: usize,
    ) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            min_lr,
            reduction_factor,
            detector: PlateauDetector::new(window_size, t_threshold),
            iteration: 0,
            num_reductions: 0,
            warmup_iterations,
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
                let t_stat = self.detector.t_statistic().unwrap_or(f64::NAN);
                let rel_improvement = self.detector.relative_improvement().unwrap_or(f64::NAN);
                let threshold = self.detector.t_threshold();

                tracing::warn!(
                    target: "plateau_detection",
                    "PLATEAU DETECTED at iteration {}: LR {} -> {} | loss: {:.6} -> {:.6} | t-stat: {:.3} < threshold: {:.3} | rel_improvement: {:.4}% | window_size: {} | reduction #{}",
                    self.iteration,
                    self.current_lr,
                    new_lr,
                    old_loss,
                    new_loss,
                    t_stat,
                    threshold,
                    rel_improvement * 100.0,
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
                        "Welch t-stat: {:.3} (threshold: {:.3})",
                        t_stat, threshold
                    );
                    let _ = writeln!(
                        file,
                        "Relative improvement: {:.4}%",
                        rel_improvement * 100.0
                    );
                    let _ = writeln!(file, "Window values ({}):", window_values.len());
                    for (i, v) in window_values.iter().enumerate() {
                        let _ = writeln!(file, "  [{:4}] {:.6}", i, v);
                    }
                    let _ = writeln!(file);
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

    pub fn t_statistic(&self) -> Option<f64> {
        self.detector.t_statistic()
    }

    pub fn window_fill_ratio(&self) -> f64 {
        self.detector.fill_ratio()
    }

    pub fn num_reductions(&self) -> usize {
        self.num_reductions
    }

    pub fn t_threshold(&self) -> f64 {
        self.detector.t_threshold()
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
        f64,      // t_threshold
        usize,    // iteration
        usize,    // num_reductions
        usize,    // warmup_iterations
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
            self.detector.t_threshold,
            self.iteration,
            self.num_reductions,
            self.warmup_iterations,
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
        self.warmup_iterations = record.8;
        self
    }
}
