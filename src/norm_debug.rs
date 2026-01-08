use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::prelude::*;
use tracing::info;

use crate::config::get_global_config;

static SNAPSHOT_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DEBUG_ACTIVE: Cell<bool> = Cell::new(false);
    static CURRENT_LAYER: Cell<Option<usize>> = Cell::new(None);
    static CURRENT_STREAM: RefCell<Option<String>> = RefCell::new(None);
}

#[derive(Debug, Clone)]
struct TensorStats {
    l2: f32,
    rms: f32,
    mean: f32,
    mean_abs: f32,
    l1: f32,
    entropy: f32,
    max_abs: f32,
    numel: usize,
    shape: Vec<usize>,
}

/// Guard that enables or disables norm logging for the lifetime of the scope.
pub struct NormDebugScope {
    active: bool,
    previous: bool,
    snapshot_idx: usize,
}

impl NormDebugScope {
    /// Start a new norm-debug snapshot. Logging only occurs if enabled in config and the
    /// sampling interval matches the current forward pass.
    pub fn start(context: &str) -> Self {
        let config = get_global_config();
        if !config.log_tensor_norms() {
            return Self {
                active: false,
                previous: false,
                snapshot_idx: 0,
            };
        }

        let count = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        let active = config
            .full_metrics_interval()
            .map_or(false, |interval| count % interval == 0);

        let previous = DEBUG_ACTIVE.with(|flag| {
            let prev = flag.get();
            flag.set(active);
            prev
        });

        if active {
            info!(
                target: "norm_debug",
                "norm_debug: ===== snapshot {} ({}) =====",
                count,
                context
            );
        }

        Self {
            active,
            previous,
            snapshot_idx: count,
        }
    }

    #[allow(dead_code)]
    pub fn active(&self) -> bool {
        self.active
    }

    #[allow(dead_code)]
    pub fn snapshot_index(&self) -> usize {
        self.snapshot_idx
    }
}

impl Drop for NormDebugScope {
    fn drop(&mut self) {
        DEBUG_ACTIVE.with(|flag| flag.set(self.previous));
    }
}

/// Guard that records the current transformer layer index for downstream logging.
pub struct LayerScope {
    previous: Option<usize>,
    current: usize,
    active: bool,
}

impl LayerScope {
    pub fn enter(layer_idx: usize) -> Self {
        let active = is_active();
        let previous = CURRENT_LAYER.with(|cell| {
            let prev = cell.get();
            cell.set(Some(layer_idx));
            prev
        });

        if active {
            info!(
                target: "norm_debug",
                "norm_debug: -- layer {} begin --",
                layer_idx
            );
        }

        Self {
            previous,
            current: layer_idx,
            active,
        }
    }
}

impl Drop for LayerScope {
    fn drop(&mut self) {
        CURRENT_LAYER.with(|cell| cell.set(self.previous));
        if self.active {
            info!(
                target: "norm_debug",
                "norm_debug: -- layer {} end --",
                self.current
            );
        }
    }
}

/// Guard that tracks the current logical stream (encoder/policy/value/etc).
pub struct StreamScope {
    previous: Option<String>,
}

impl StreamScope {
    pub fn enter<S: Into<String>>(stream: S) -> Self {
        let active = is_active();
        let stream_name = stream.into();
        let previous = CURRENT_STREAM.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let prev = borrow.clone();
            *borrow = Some(stream_name.clone());
            prev
        });

        if active {
            info!(
                target: "norm_debug",
                "norm_debug: stream -> {}",
                stream_name
            );
        }

        Self { previous }
    }
}

impl Drop for StreamScope {
    fn drop(&mut self) {
        CURRENT_STREAM.with(|cell| {
            let mut borrow = cell.borrow_mut();
            *borrow = self.previous.clone();
        });
    }
}

fn is_active() -> bool {
    DEBUG_ACTIVE.with(|flag| flag.get())
}

fn collect_stats<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Option<TensorStats> {
    let dims = tensor.dims();
    if dims.iter().any(|&d| d == 0) {
        return None;
    }
    let numel = dims.iter().product::<usize>();
    if numel == 0 {
        return None;
    }

    let l2_sq = tensor
        .clone()
        .powi_scalar(2)
        .sum()
        .into_scalar()
        .elem::<f32>();
    let l2 = l2_sq.sqrt();
    let rms = if numel > 0 {
        (l2_sq / numel as f32).sqrt()
    } else {
        0.0
    };
    let mean = tensor.clone().mean().into_scalar().elem::<f32>();
    let abs_tensor = tensor.clone().abs();
    let mean_abs = abs_tensor.clone().mean().into_scalar().elem::<f32>();
    let l1 = mean_abs * numel as f32;
    let max_abs = abs_tensor.clone().max().into_scalar().elem::<f32>();
    let abs_sum = abs_tensor.clone().sum().into_scalar().elem::<f32>();
    let entropy = if abs_sum <= 0.0 || !abs_sum.is_finite() {
        0.0
    } else {
        let probs = abs_tensor.div_scalar(abs_sum);
        let probs_safe = probs.clone().clamp_min(1e-12);
        let log_probs = probs_safe.log();
        let entropy_term = probs * log_probs;
        -entropy_term.sum().into_scalar().elem::<f32>()
    };

    Some(TensorStats {
        l2,
        rms,
        mean,
        mean_abs,
        l1,
        entropy,
        max_abs,
        numel,
        shape: dims.to_vec(),
    })
}

fn preview_values<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    limit: usize,
) -> Option<Vec<f32>> {
    let dims = tensor.dims();
    let numel = dims.iter().product::<usize>();
    if numel == 0 || numel > limit {
        return None;
    }
    let data = tensor.clone().into_data().convert::<f32>();
    data.to_vec::<f32>().ok()
}

fn current_stream() -> Option<String> {
    CURRENT_STREAM.with(|cell| cell.borrow().clone())
}

fn current_layer() -> Option<usize> {
    CURRENT_LAYER.with(|cell| cell.get())
}

/// Log tensor statistics if norm debugging is active.
pub fn log_tensor_stats<B: Backend, const D: usize>(label: &str, tensor: &Tensor<B, D>) {
    if !is_active() {
        return;
    }

    let Some(stats) = collect_stats(tensor) else {
        return;
    };

    let stream = current_stream().unwrap_or_else(|| "main".to_string());
    let layer = current_layer()
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "-".to_string());

    info!(
        target: "norm_debug",
        "norm_debug: [{}] layer={} label={} | l2={:.4} rms={:.4} mean={:.4} mean_abs={:.4} l1={:.4} entropy={:.4} max_abs={:.4} numel={} shape={:?}",
        stream,
        layer,
        label,
        stats.l2,
        stats.rms,
        stats.mean,
        stats.mean_abs,
        stats.l1,
        stats.entropy,
        stats.max_abs,
        stats.numel,
        stats.shape,
    );

    let config = get_global_config();
    if let Some(values) = preview_values(tensor, config.norm_preview_limit()) {
        info!(
            target: "norm_debug",
            "norm_debug: [{}] layer={} label={} | values={:?}",
            stream,
            layer,
            label,
            values
        );
    }
}

/// Log an arbitrary message under the norm-debug context.
pub fn log_message(message: &str) {
    if !is_active() {
        return;
    }
    let stream = current_stream().unwrap_or_else(|| "main".to_string());
    let layer = current_layer()
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "-".to_string());

    info!(
        target: "norm_debug",
        "norm_debug: [{}] layer={} | {}",
        stream,
        layer,
        message
    );
}

/// Log a single-query 8x8 attention heatmap using Unicode shaded blocks for quick inspection.
pub fn log_attention_heatmap<B: Backend>(
    label: &str,
    tensor: &Tensor<B, 3>,
    batch_index: usize,
    query_index: Option<usize>,
) {
    if !is_active() {
        return;
    }

    let [batch, rows, cols] = tensor.dims();
    if batch == 0 || rows == 0 || cols == 0 || rows != cols {
        return;
    }

    let batch_idx = batch_index.min(batch - 1);
    let selected: Tensor<B, 2> = tensor
        .clone()
        .slice([batch_idx..batch_idx + 1, 0..rows, 0..cols])
        .squeeze_dim(0);

    let data = selected.into_data().convert::<f32>();
    let Ok(values) = data.to_vec::<f32>() else {
        return;
    };

    let heat_chars: [char; 8] = [' ', '·', '▪', '▢', '▤', '▦', '▩', '█'];

    let chosen_query = query_index.unwrap_or_else(|| {
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        for row in 0..rows {
            let row_slice = &values[row * cols..(row + 1) * cols];
            let row_peak = row_slice.iter().cloned().fold(0.0_f32, f32::max);
            if row_peak > best_val {
                best_val = row_peak;
                best_idx = row;
            }
        }
        best_idx
    });
    let query_idx = chosen_query.min(rows - 1);
    let row_values = &values[query_idx * cols..(query_idx + 1) * cols];

    let mut row_max = row_values
        .iter()
        .cloned()
        .fold(0.0_f32, |acc, v| acc.max(v));
    if !row_max.is_finite() || row_max <= 0.0 {
        row_max = 1.0;
    }

    let mut lines = Vec::with_capacity(8);
    for board_row in 0..8 {
        let mut line = String::with_capacity(8);
        for board_col in 0..8 {
            let idx = board_row * 8 + board_col;
            let val = row_values.get(idx).cloned().unwrap_or(0.0);
            let mut norm = (val / row_max).sqrt();
            if !norm.is_finite() || norm < 0.0 {
                norm = 0.0;
            }
            if norm > 1.0 {
                norm = 1.0;
            }
            let level = (norm * (heat_chars.len() - 1) as f32).round() as usize;
            let char_idx = level.min(heat_chars.len() - 1);
            line.push(heat_chars[char_idx]);
        }
        lines.push(line);
    }

    let heatmap = lines.join("\n");

    let stream = current_stream().unwrap_or_else(|| "main".to_string());
    let layer = current_layer()
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "-".to_string());
    let board_row = query_idx / 8;
    let board_col = query_idx % 8;

    info!(
        target: "norm_debug",
        "norm_debug: [{}] layer={} label={} batch={} query_idx={} board_row={} board_col={} heatmap:\n{}",
        stream,
        layer,
        label,
        batch_idx,
        query_idx,
        board_row,
        board_col,
        heatmap
    );
}
