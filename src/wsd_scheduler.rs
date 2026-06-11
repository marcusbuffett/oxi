use std::time::Instant;

/// Warmup-Stable-Decay (WSD) learning rate scheduler.
///
/// Three phases:
///   1. Warmup: LR ramps linearly from 0 to base_lr over `warmup_fraction`
///      of the run budget.
///   2. Stable: LR held constant at base_lr.
///   3. Decay: LR ramps linearly from base_lr down to `final_lr` over the
///      final `decay_fraction` of the run's budget.
///
/// The budget comes from `--max-samples` (sample-based progress), `--timeout`
/// (wall-clock progress), or both (whichever is further along governs, since
/// the run exits at whichever limit is hit first). With `--lr-hold` there is
/// no budget and the stable phase lasts until the run is stopped; a decay
/// phase can then be run later by resuming the checkpoint with a budget.
#[derive(Clone, Debug)]
pub struct WsdScheduler {
    base_lr: f64,
    final_lr: f64,
    warmup_fraction: f64,
    decay_fraction: f64,
    /// Sample budget (from --max-samples), if any.
    budget_samples: Option<usize>,
    /// Wall-clock budget in seconds (from --timeout), if any.
    budget_secs: Option<f64>,
    start: Instant,
    items_processed: usize,
}

/// Which phase the scheduler is currently in, with progress through that
/// phase in [0, 1] for display purposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WsdPhase {
    Warmup { progress: f64 },
    Stable,
    Decay { progress: f64 },
}

impl WsdScheduler {
    pub fn new(
        base_lr: f64,
        final_lr: f64,
        warmup_fraction: f64,
        decay_fraction: f64,
        budget_samples: Option<usize>,
        budget_secs: Option<u64>,
    ) -> Self {
        Self {
            base_lr,
            final_lr: final_lr.min(base_lr),
            warmup_fraction: warmup_fraction.clamp(0.0, 1.0),
            decay_fraction: decay_fraction.clamp(0.0, 1.0),
            budget_samples,
            budget_secs: budget_secs.map(|s| s as f64),
            start: Instant::now(),
            items_processed: 0,
        }
    }

    /// Record progress. Call once per training iteration with the cumulative
    /// number of samples processed.
    pub fn update(&mut self, items_processed: usize) {
        self.items_processed = items_processed;
    }

    /// Fraction of the run budget consumed, in [0, 1]. When both sample and
    /// wall-clock budgets exist, the further-along one governs (the run exits
    /// at whichever limit is hit first). None when there is no budget.
    fn budget_progress(&self) -> Option<f64> {
        self.budget_progress_at(self.start.elapsed().as_secs_f64())
    }

    fn budget_progress_at(&self, elapsed_secs: f64) -> Option<f64> {
        let sample_progress = self
            .budget_samples
            .map(|total| self.items_processed as f64 / total.max(1) as f64);
        let time_progress = self.budget_secs.map(|total| elapsed_secs / total.max(1e-9));
        match (sample_progress, time_progress) {
            (Some(s), Some(t)) => Some(s.max(t)),
            (Some(s), None) => Some(s),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        }
    }

    pub fn phase(&self) -> WsdPhase {
        self.phase_at(self.start.elapsed().as_secs_f64())
    }

    fn phase_at(&self, elapsed_secs: f64) -> WsdPhase {
        if self.warmup_fraction > 0.0 {
            if let Some(progress) = self.budget_progress_at(elapsed_secs) {
                if progress < self.warmup_fraction {
                    return WsdPhase::Warmup {
                        progress: (progress / self.warmup_fraction).min(1.0),
                    };
                }
            }
        }
        let Some(progress) = self.budget_progress_at(elapsed_secs) else {
            return WsdPhase::Stable;
        };
        let decay_start = 1.0 - self.decay_fraction;
        if progress < decay_start || self.decay_fraction <= 0.0 {
            WsdPhase::Stable
        } else {
            WsdPhase::Decay {
                progress: ((progress - decay_start) / self.decay_fraction).min(1.0),
            }
        }
    }

    pub fn get_lr(&self) -> f64 {
        self.lr_at(self.start.elapsed().as_secs_f64())
    }

    fn lr_at(&self, elapsed_secs: f64) -> f64 {
        match self.phase_at(elapsed_secs) {
            WsdPhase::Warmup { progress } => self.base_lr * progress,
            WsdPhase::Stable => self.base_lr,
            WsdPhase::Decay { progress } => {
                self.base_lr + (self.final_lr - self.base_lr) * progress
            }
        }
    }

    pub fn is_warming_up(&self) -> bool {
        matches!(self.phase(), WsdPhase::Warmup { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_budget_scheduler() -> WsdScheduler {
        // base_lr 1.0, final_lr 0.0, warmup over first 10%, decay over final
        // 20% of a 10_000-sample budget (decay starts at 8000 samples).
        WsdScheduler::new(1.0, 0.0, 0.1, 0.2, Some(10_000), None)
    }

    #[test]
    fn warmup_ramps_linearly() {
        let mut s = sample_budget_scheduler();
        s.update(0);
        assert_eq!(s.lr_at(0.0), 0.0);
        s.update(500);
        assert!((s.lr_at(0.0) - 0.5).abs() < 1e-9);
        s.update(1000);
        assert_eq!(s.lr_at(0.0), 1.0);
    }

    #[test]
    fn stable_phase_holds_base_lr() {
        let mut s = sample_budget_scheduler();
        for items in [1000, 4000, 7999] {
            s.update(items);
            assert_eq!(s.lr_at(0.0), 1.0, "items={items}");
            assert_eq!(s.phase_at(0.0), WsdPhase::Stable, "items={items}");
        }
    }

    #[test]
    fn decay_ramps_to_final_lr() {
        let mut s = sample_budget_scheduler();
        s.update(9000);
        assert!((s.lr_at(0.0) - 0.5).abs() < 1e-9);
        s.update(10_000);
        assert!(s.lr_at(0.0).abs() < 1e-9);
        s.update(20_000); // past budget: clamped at final_lr
        assert!(s.lr_at(0.0).abs() < 1e-9);
    }

    #[test]
    fn wallclock_budget_decays_by_time() {
        let mut s = WsdScheduler::new(1.0, 0.1, 0.0, 0.5, None, Some(100));
        s.update(1_000_000);
        assert_eq!(s.lr_at(0.0), 1.0);
        assert_eq!(s.lr_at(49.0), 1.0);
        assert!((s.lr_at(75.0) - 0.55).abs() < 1e-9);
        assert!((s.lr_at(100.0) - 0.1).abs() < 1e-9);
        assert!((s.lr_at(500.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn both_budgets_use_whichever_is_further_along() {
        // 50% decay fraction; sample budget 10k, time budget 100s.
        let mut s = WsdScheduler::new(1.0, 0.0, 0.0, 0.5, Some(10_000), Some(100));
        // Samples ahead of time: 90% through samples, 10% through time.
        s.update(9000);
        assert!((s.lr_at(10.0) - 0.2).abs() < 1e-9);
        // Time ahead of samples: 10% through samples, 90% through time.
        s.update(1000);
        assert!((s.lr_at(90.0) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn no_budget_holds_constant_forever() {
        let mut s = WsdScheduler::new(1.0, 0.0, 0.02, 0.15, None, None);
        s.update(1_000_000_000);
        assert_eq!(s.lr_at(1e9), 1.0);
        assert_eq!(s.phase_at(1e9), WsdPhase::Stable);
    }

    #[test]
    fn warmup_fraction_uses_timeout_budget_progress() {
        let s = WsdScheduler::new(1.0, 0.0, 0.02, 0.15, None, Some(100));
        assert_eq!(s.phase_at(0.0), WsdPhase::Warmup { progress: 0.0 });
        assert_eq!(s.phase_at(1.0), WsdPhase::Warmup { progress: 0.5 });
        assert_eq!(s.phase_at(2.0), WsdPhase::Stable);
    }
}
