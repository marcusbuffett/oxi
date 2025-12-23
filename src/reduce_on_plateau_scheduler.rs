use burn::lr_scheduler::LrScheduler;
use burn::tensor::backend::Backend;

/// ReduceOnPlateau learning rate scheduler
/// Reduces learning rate when loss plateaus for a specified number of measurement batches
#[derive(Clone, Debug)]
pub struct ReduceOnPlateauScheduler {
    /// Current learning rate
    current_lr: f64,
    /// Minimum learning rate (never go below this)
    min_lr: f64,
    /// Factor to reduce LR by when plateau detected (e.g., 0.5 means halve the LR)
    reduction_factor: f64,
    /// Number of measurement batches without improvement before reducing LR
    patience: usize,
    /// Number of samples to accumulate before recording a measurement
    measurement_batch_size: usize,
    /// Number of samples accumulated since last measurement
    samples_accumulated: usize,
    /// Best (lowest) loss observed so far
    best_loss: Option<f64>,
    /// Number of measurement batches since best loss was updated
    batches_without_improvement: usize,
    /// Total number of samples processed
    total_samples_processed: usize,
}

impl ReduceOnPlateauScheduler {
    /// Create a new ReduceOnPlateau scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `min_lr` - Minimum learning rate (floor)
    /// * `reduction_factor` - Factor to multiply LR by when reducing (e.g., 0.5)
    /// * `patience` - Number of measurement batches without improvement before reducing LR
    /// * `measurement_batch_size` - Number of samples per measurement batch
    pub fn new(
        initial_lr: f64,
        min_lr: f64,
        reduction_factor: f64,
        patience: usize,
        measurement_batch_size: usize,
    ) -> Self {
        Self {
            current_lr: initial_lr,
            min_lr,
            reduction_factor,
            patience,
            measurement_batch_size,
            samples_accumulated: 0,
            best_loss: None,
            batches_without_improvement: 0,
            total_samples_processed: 0,
        }
    }

    /// Adjust measurement batch size to be a multiple of physical batch size
    pub fn adjust_measurement_batch_size(&mut self, physical_batch_size: usize) {
        // Round up to nearest multiple of physical_batch_size
        let multiple =
            (self.measurement_batch_size + physical_batch_size - 1) / physical_batch_size;
        self.measurement_batch_size = multiple * physical_batch_size;
    }

    /// Get the adjusted measurement batch size
    pub fn measurement_batch_size(&self) -> usize {
        self.measurement_batch_size
    }

    /// Record samples processed in this batch and optionally a loss measurement
    /// Returns true if a measurement was recorded (measurement batch completed)
    pub fn record_batch(&mut self, batch_size: usize, loss: f64) -> bool {
        self.samples_accumulated += batch_size;
        self.total_samples_processed += batch_size;

        // Check if we've reached a measurement boundary
        if self.samples_accumulated >= self.measurement_batch_size {
            self.samples_accumulated = 0;
            self.record_measurement(loss);
            true
        } else {
            false
        }
    }

    /// Record a loss measurement and potentially reduce learning rate
    fn record_measurement(&mut self, loss: f64) {
        if !loss.is_finite() {
            return;
        }

        match self.best_loss {
            None => {
                // First measurement
                self.best_loss = Some(loss);
                self.batches_without_improvement = 0;
            }
            Some(best) => {
                if loss < best {
                    // New best loss! Reset patience counter and update best
                    self.best_loss = Some(loss);
                    self.batches_without_improvement = 0;
                } else {
                    // No improvement
                    self.batches_without_improvement += 1;

                    // Check if we should reduce LR
                    if self.batches_without_improvement >= self.patience {
                        let new_lr = (self.current_lr * self.reduction_factor).max(self.min_lr);
                        if new_lr < self.current_lr {
                            self.current_lr = new_lr;
                            // Reset best loss when LR is reduced
                            self.best_loss = Some(loss);
                            self.batches_without_improvement = 0;
                        }
                    }
                }
            }
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Get current best loss
    pub fn best_loss(&self) -> Option<f64> {
        self.best_loss
    }

    /// Get batches without improvement
    pub fn batches_without_improvement(&self) -> usize {
        self.batches_without_improvement
    }

    /// Check if training should stop (at min LR with no improvement for patience batches)
    pub fn should_stop(&self) -> bool {
        // Stop if we're at minimum LR and haven't improved for patience batches
        (self.current_lr - self.min_lr).abs() < 1e-10
            && self.batches_without_improvement >= self.patience
    }
}

impl LrScheduler for ReduceOnPlateauScheduler {
    type Record<B: Backend> = (
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
        Option<f64>,
        usize,
        usize,
    );

    fn step(&mut self) -> f64 {
        // Unlike the cosine scheduler, we don't step automatically
        // The LR only changes when record_batch detects a plateau
        self.current_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        (
            self.current_lr,
            self.min_lr,
            self.reduction_factor,
            self.patience,
            self.measurement_batch_size,
            self.samples_accumulated,
            self.best_loss,
            self.batches_without_improvement,
            self.total_samples_processed,
        )
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.current_lr = record.0;
        self.min_lr = record.1;
        self.reduction_factor = record.2;
        self.patience = record.3;
        self.measurement_batch_size = record.4;
        self.samples_accumulated = record.5;
        self.best_loss = record.6;
        self.batches_without_improvement = record.7;
        self.total_samples_processed = record.8;
        self
    }
}
