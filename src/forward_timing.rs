//! Forward pass timing instrumentation for identifying GPU bottlenecks.
//!
//! Use `TimingScope` to measure individual operations and `ForwardPassTimer` to aggregate.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Global flag to enable/disable timing (avoids overhead when not profiling)
static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Counter for sampling (only time every Nth forward pass)
static FORWARD_PASS_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Sample interval (time every Nth forward pass to reduce overhead)
static TIMING_SAMPLE_INTERVAL: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// Thread-local storage for timing data
    static TIMING_DATA: RefCell<TimingData> = RefCell::new(TimingData::new());
}

/// Enable forward pass timing
pub fn enable_timing() {
    TIMING_ENABLED.store(true, Ordering::SeqCst);
}

/// Disable forward pass timing
pub fn disable_timing() {
    TIMING_ENABLED.store(false, Ordering::SeqCst);
}

/// Set sampling interval (time every Nth forward pass)
pub fn set_sample_interval(interval: u64) {
    TIMING_SAMPLE_INTERVAL.store(interval.max(1), Ordering::SeqCst);
}

/// Check if timing is currently active for this forward pass
pub fn is_timing_active() -> bool {
    if !TIMING_ENABLED.load(Ordering::Relaxed) {
        return false;
    }
    let counter = FORWARD_PASS_COUNTER.load(Ordering::Relaxed);
    let interval = TIMING_SAMPLE_INTERVAL.load(Ordering::Relaxed);
    counter % interval == 0
}

/// Start a new forward pass timing session
pub fn start_forward_pass() {
    FORWARD_PASS_COUNTER.fetch_add(1, Ordering::Relaxed);
    if is_timing_active() {
        TIMING_DATA.with(|data| {
            data.borrow_mut().clear();
        });
    }
}

/// Record timing for a named operation
pub fn record_timing(name: &str, duration: Duration) {
    if is_timing_active() {
        TIMING_DATA.with(|data| {
            data.borrow_mut().record(name, duration);
        });
    }
}

/// Get all accumulated timing data and log it
pub fn finish_and_log_forward_pass() {
    if is_timing_active() {
        TIMING_DATA.with(|data| {
            let timing = data.borrow();
            if !timing.timings.is_empty() {
                let total: Duration = timing.timings.values().sum();
                let mut sorted: Vec<_> = timing.timings.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort by duration descending

                let mut log_msg = format!(
                    "forward_timing: total={:.3}ms | ",
                    total.as_secs_f64() * 1000.0
                );

                for (name, dur) in sorted.iter().take(20) {
                    let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
                    log_msg.push_str(&format!(
                        "{}={:.3}ms ({:.1}%) | ",
                        name,
                        dur.as_secs_f64() * 1000.0,
                        pct
                    ));
                }

                tracing::info!("{}", log_msg);
            }
        });
    }
}

/// Get timing summary as a formatted string
pub fn get_timing_summary() -> Option<String> {
    if !is_timing_active() {
        return None;
    }
    TIMING_DATA.with(|data| {
        let timing = data.borrow();
        if timing.timings.is_empty() {
            return None;
        }

        let total: Duration = timing.timings.values().sum();
        let mut sorted: Vec<_> = timing.timings.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        let mut lines = vec![format!(
            "Forward Pass Timing (total: {:.3}ms)",
            total.as_secs_f64() * 1000.0
        )];

        for (name, dur) in sorted {
            let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
            lines.push(format!(
                "  {:40} {:8.3}ms ({:5.1}%)",
                name,
                dur.as_secs_f64() * 1000.0,
                pct
            ));
        }

        Some(lines.join("\n"))
    })
}

/// Internal timing data storage
struct TimingData {
    timings: HashMap<String, Duration>,
}

impl TimingData {
    fn new() -> Self {
        Self {
            timings: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.timings.clear();
    }

    fn record(&mut self, name: &str, duration: Duration) {
        *self.timings.entry(name.to_string()).or_default() += duration;
    }
}

/// RAII timing scope - measures time from creation to drop
pub struct TimingScope {
    name: &'static str,
    start: Instant,
    active: bool,
    sync_fn: Option<Box<dyn FnMut() + Send>>,
}

impl TimingScope {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let active = is_timing_active();
        Self {
            name,
            start: Instant::now(),
            active,
            sync_fn: None,
        }
    }

    #[inline]
    pub fn new_with_sync<B: burn::tensor::backend::Backend>(
        name: &'static str,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let active = is_timing_active();
        if active {
            let _ = B::sync(device);
        }
        let device_clone = device.clone();
        Self {
            name,
            start: Instant::now(),
            active,
            sync_fn: if active {
                Some(Box::new(move || {
                    let _ = B::sync(&device_clone);
                }))
            } else {
                None
            },
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        if self.active {
            if let Some(mut sync_fn) = self.sync_fn.take() {
                sync_fn();
            }
            record_timing(self.name, self.start.elapsed());
        }
    }
}

/// Macro for convenient timing scope creation
#[macro_export]
macro_rules! time_scope {
    ($name:expr) => {
        let _timing_scope = $crate::forward_timing::TimingScope::new($name);
    };
    ($name:expr, $device:expr) => {
        let _timing_scope = $crate::forward_timing::TimingScope::new_with_sync::<B>($name, $device);
    };
}

/// Macro for timing a block of code
#[macro_export]
macro_rules! timed {
    ($name:expr, $block:expr) => {{
        let _scope = $crate::forward_timing::TimingScope::new($name);
        $block
    }};
}
