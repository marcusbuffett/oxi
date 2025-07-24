use burn::lr_scheduler::LrScheduler;
use burn::tensor::backend::Backend;
use std::f64::consts::PI;

/// Cosine annealing learning rate scheduler with linear warmup
#[derive(Clone, Debug)]
pub struct CosineAnnealingWithWarmup {
    warmup_steps: f64,
    total_steps: f64,
    max_lr: f64,
    min_lr: f64,
    step: f64,
}

impl CosineAnnealingWithWarmup {
    /// Create a new cosine annealing scheduler with linear warmup
    ///
    /// # Arguments
    /// * `warmup_steps` - Number of steps for linear warmup
    /// * `total_steps` - Total number of training steps (for cosine decay)
    /// * `max_lr` - Maximum learning rate (reached after warmup)
    /// * `min_lr` - Minimum learning rate (at end of training)
    pub fn new(warmup_steps: usize, total_steps: usize, max_lr: f64, min_lr: f64) -> Self {
        Self {
            warmup_steps: warmup_steps as f64,
            total_steps: total_steps as f64,
            max_lr,
            min_lr,
            step: 0.0,
        }
    }
}

impl LrScheduler for CosineAnnealingWithWarmup {
    type Record<B: Backend> = (usize, usize, f64, f64, usize);

    fn step(&mut self) -> f64 {
        self.step += 1.0;

        if self.step < self.warmup_steps {
            // Linear warmup from 0 to max_lr
            self.max_lr * (self.step / self.warmup_steps)
        } else {
            // Cosine annealing from max_lr to min_lr
            let progress =
                (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps).max(1.0);
            let progress = progress.min(1.0);

            let cosine_decay = 0.5 * (1.0 + (progress * PI).cos());
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        }
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        (
            self.warmup_steps as usize,
            self.total_steps as usize,
            self.max_lr,
            self.min_lr,
            self.step as usize,
        )
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.warmup_steps = record.0 as f64;
        self.total_steps = record.1 as f64;
        self.max_lr = record.2;
        self.min_lr = record.3;
        self.step = record.4 as f64;
        self
    }
}
