pub mod calibration;
pub mod config;
pub mod constants;
pub mod distribution_utils;
pub mod encoding;
pub mod eval_dataset;
pub mod factorized_policy;
pub mod inference;
pub mod model;
pub mod move_encoding;
pub mod moves;
pub mod relative_position_transformer;
pub mod smolgen;
pub mod spatial_conv;
pub mod stockfish;
pub mod value_tower;

#[cfg(not(feature = "train"))]
pub mod train_stubs {
    use burn::prelude::*;

    pub struct TimingScope;
    impl TimingScope {
        pub fn new_with_sync<B: Backend>(_name: &str, _device: &B::Device) -> Self {
            Self
        }
    }

    pub struct NormDebugScope;
    impl NormDebugScope {
        pub fn start(_name: &str) -> Self {
            Self
        }
    }

    pub struct StreamScope;
    impl StreamScope {
        pub fn enter(_name: &str) -> Self {
            Self
        }
    }

    pub struct LayerScope;
    impl LayerScope {
        pub fn enter(_idx: usize) -> Self {
            Self
        }
    }

    #[inline(always)]
    pub fn start_forward_pass() {}

    #[inline(always)]
    pub fn finish_and_log_forward_pass() {}

    #[inline(always)]
    pub fn log_tensor_stats<B: Backend, const D: usize>(_name: &str, _tensor: &Tensor<B, D>) {}

    #[inline(always)]
    pub fn log_model_predictions<B: Backend, T>(
        _policy_logits: &Tensor<B, 2>,
        _value_logits: &Tensor<B, 2>,
        _time_usage_logits: &Tensor<B, 2>,
        _batch: &T,
    ) {
    }
}

#[cfg(feature = "train")]
pub mod chess_output;
#[cfg(feature = "train")]
pub mod custom_training;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod debug_prediction_monitor;
#[cfg(feature = "train")]
pub mod forward_timing;
#[cfg(feature = "train")]
pub mod gradient_norm_metric;
#[cfg(feature = "train")]
pub mod gradnorm;
#[cfg(feature = "train")]
pub mod gradnorm_status_metric;
#[cfg(feature = "train")]
pub mod legal_move_probability_metric;
#[cfg(feature = "train")]
pub mod lr_plateau_metric;
#[cfg(feature = "train")]
pub mod metrics_renderer;
#[cfg(feature = "train")]
pub mod model_prediction_logger;
#[cfg(feature = "train")]
pub mod move_accuracy_metric;
#[cfg(feature = "train")]
pub mod norm_debug;
#[cfg(feature = "train")]
pub mod opening_family_metric;
#[cfg(feature = "train")]
pub mod pgn_processor;
#[cfg(feature = "train")]
pub mod policy_loss_metric;
#[cfg(feature = "train")]
pub mod puzzle_processor;
#[cfg(feature = "train")]
pub mod reduce_on_plateau_scheduler;
#[cfg(feature = "train")]
pub mod time_usage_loss_metric;
#[cfg(feature = "train")]
pub mod training_stage_metric;
#[cfg(feature = "train")]
pub mod training_stream;
#[cfg(feature = "train")]
pub mod tui;
#[cfg(feature = "train")]
pub mod uncertainty_metric;
#[cfg(feature = "train")]
pub mod value_loss_metric;
#[cfg(feature = "train")]
pub mod wdl_accuracy_metric;
#[cfg(feature = "train")]
pub mod weight_decay;

#[cfg(test)]
pub mod test_backend;
#[cfg(test)]
pub mod test_tensor_consistency;
