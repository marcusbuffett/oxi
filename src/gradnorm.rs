use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::chess_output::ChessOutput;
use crate::config::Config;

const EPS: f32 = 1e-8;
/// Absolute floor on any gradnorm-managed weight. Prevents weights from
/// collapsing to zero (and being numerically unrecoverable) without tying the
/// floor to the initial config value.
const WEIGHT_FLOOR: f32 = 1e-6;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum GradNormTask {
    Policy = 0,
    Value = 1,
    TimeUsage = 2,
    Auxiliary = 3,
    Calibration = 4,
    PolicyRegret = 5,
}

impl GradNormTask {
    pub fn iter() -> impl Iterator<Item = GradNormTask> {
        [
            GradNormTask::Policy,
            GradNormTask::Value,
            GradNormTask::TimeUsage,
            GradNormTask::Auxiliary,
            GradNormTask::Calibration,
            GradNormTask::PolicyRegret,
        ]
        .into_iter()
    }

    pub fn name(&self) -> &'static str {
        match self {
            GradNormTask::Policy => "policy",
            GradNormTask::Value => "value",
            GradNormTask::TimeUsage => "time_usage",
            GradNormTask::Auxiliary => "auxiliary",
            GradNormTask::Calibration => "calibration",
            GradNormTask::PolicyRegret => "policy_regret",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradNormTaskStatus {
    pub task: GradNormTask,
    pub weight: f32,
    pub enabled: bool,
    pub priority: f32,
    pub last_loss: Option<f32>,
    pub last_weighted_loss: Option<f32>,
    pub last_loss_step: Option<usize>,
    pub last_grad_norm: Option<f32>,
    pub last_grad_step: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskState {
    task: GradNormTask,
    weight: f32,
    initial_weight: f32,
    enabled: bool,
    priority: f32,
    initial_loss: Option<f32>,
    last_loss: Option<f32>,
    last_weighted_loss: Option<f32>,
    last_loss_step: Option<usize>,
    last_grad_norm: Option<f32>,
    last_grad_step: Option<usize>,
}

impl TaskState {
    fn new(task: GradNormTask, initial_weight: f32, priority: f32) -> Self {
        let enabled = initial_weight > 0.0;
        Self {
            task,
            weight: initial_weight.max(0.0),
            initial_weight: initial_weight.max(0.0),
            enabled,
            priority: priority.max(EPS),
            initial_loss: None,
            last_loss: None,
            last_weighted_loss: None,
            last_loss_step: None,
            last_grad_norm: None,
            last_grad_step: None,
        }
    }

    fn status(&self) -> GradNormTaskStatus {
        GradNormTaskStatus {
            task: self.task,
            weight: self.weight,
            enabled: self.enabled,
            priority: self.priority,
            last_loss: self.last_loss,
            last_weighted_loss: self.last_weighted_loss,
            last_loss_step: self.last_loss_step,
            last_grad_norm: self.last_grad_norm,
            last_grad_step: self.last_grad_step,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GradNormProbeResult {
    pub task: GradNormTask,
    pub base_loss: f32,
    pub grad_norm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradNormState {
    tasks: Vec<TaskState>,
    alpha: f32,
    lr: f32,
    interval: usize,
    enabled: bool,
    reference_total_weight: f32,
}

impl GradNormState {
    pub fn new(config: &Config) -> Self {
        let tasks = GradNormTask::iter()
            .map(|task| {
                let (weight, priority) = match task {
                    GradNormTask::Policy => {
                        (config.policy_loss_weight, config.gradnorm_policy_priority())
                    }
                    GradNormTask::Value => {
                        (config.value_loss_weight, config.gradnorm_value_priority())
                    }
                    GradNormTask::TimeUsage => (
                        config.time_usage_loss_weight,
                        config.gradnorm_time_priority(),
                    ),
                    GradNormTask::Auxiliary => {
                        (config.aux_loss_weight, config.gradnorm_aux_priority())
                    }
                    GradNormTask::Calibration => (
                        config.calibration_loss_weight(),
                        config.gradnorm_calibration_priority(),
                    ),
                    GradNormTask::PolicyRegret => (
                        config.policy_regret_loss_weight(),
                        config.gradnorm_policy_regret_priority(),
                    ),
                };
                TaskState::new(task, weight, priority)
            })
            .collect::<Vec<_>>();

        let reference_total_weight: f32 = tasks
            .iter()
            .filter(|task| task.enabled)
            .map(|task| task.weight)
            .sum();

        let enabled = config.gradnorm_enabled()
            && tasks.iter().filter(|task| task.enabled).count() > 1
            && reference_total_weight > 0.0;

        Self {
            tasks,
            alpha: config.gradnorm_alpha(),
            lr: config.gradnorm_learning_rate(),
            interval: config.gradnorm_interval(),
            enabled,
            reference_total_weight: reference_total_weight.max(0.0),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn reset_weights_to_config(&mut self, config: &Config) {
        for task_state in &mut self.tasks {
            let weight = match task_state.task {
                GradNormTask::Policy => config.policy_loss_weight,
                GradNormTask::Value => config.value_loss_weight,
                GradNormTask::TimeUsage => config.time_usage_loss_weight,
                GradNormTask::Auxiliary => config.aux_loss_weight,
                GradNormTask::Calibration => config.calibration_loss_weight(),
                GradNormTask::PolicyRegret => config.policy_regret_loss_weight(),
            };
            task_state.weight = weight;
            task_state.enabled = weight > 0.0;
        }
    }

    pub fn weight_for(&self, task: GradNormTask) -> f32 {
        self.tasks[task as usize].weight
    }

    pub fn status_snapshot(&self) -> Vec<GradNormTaskStatus> {
        self.tasks.iter().map(TaskState::status).collect()
    }

    pub fn should_update_weights(&self, next_optimizer_step: usize) -> bool {
        if !self.enabled {
            return false;
        }

        if self.tasks.iter().filter(|task| task.enabled).count() <= 1 {
            return false;
        }

        let needs_bootstrap = self
            .tasks
            .iter()
            .any(|task| task.enabled && task.last_grad_step.is_none());

        if needs_bootstrap {
            return true;
        }

        // Early adjustment: also fire at step 10 so loss weights stabilize quickly
        const EARLY_ADJUSTMENT_STEP: usize = 10;
        if next_optimizer_step == EARLY_ADJUSTMENT_STEP {
            return true;
        }

        next_optimizer_step % self.interval == 0
    }

    pub fn record_batch_losses<B: Backend>(&mut self, batch_step: usize, output: &ChessOutput<B>) {
        for task_state in &mut self.tasks {
            let (base_loss_tensor, enabled_by_output) = match task_state.task {
                GradNormTask::Policy => (&output.base_policy_loss, true),
                GradNormTask::Value => (&output.base_value_loss, true),
                GradNormTask::TimeUsage => (&output.base_time_usage_loss, true),
                GradNormTask::Auxiliary => (&output.base_aux_loss, true),
                GradNormTask::Calibration => (
                    &output.base_calibration_loss,
                    output.calibration_labeled_fraction > 0.0,
                ),
                GradNormTask::PolicyRegret => (
                    &output.base_policy_regret_loss,
                    output.calibration_labeled_fraction > 0.0,
                ),
            };

            let base_loss = if enabled_by_output {
                Some(base_loss_tensor.clone().into_scalar().elem::<f32>())
            } else {
                None
            };

            if let Some(loss) = base_loss {
                if task_state.initial_loss.is_none() && loss > 0.0 {
                    task_state.initial_loss = Some(loss);
                }
                task_state.last_loss = Some(loss);
                task_state.last_weighted_loss = Some(loss * task_state.weight);
                task_state.last_loss_step = Some(batch_step);
            } else {
                task_state.last_loss = None;
                task_state.last_weighted_loss = None;
                task_state.last_loss_step = Some(batch_step);
            }
        }
    }

    pub fn apply_weights_to_output<B: Backend>(&self, output: &mut ChessOutput<B>) {
        let policy_weight = self.weight_for(GradNormTask::Policy);
        let value_weight = self.weight_for(GradNormTask::Value);
        let time_weight = self.weight_for(GradNormTask::TimeUsage);
        let aux_weight = self.weight_for(GradNormTask::Auxiliary);
        let calibration_weight = self.weight_for(GradNormTask::Calibration);
        let policy_regret_weight = self.weight_for(GradNormTask::PolicyRegret);

        output.policy_loss = output.base_policy_loss.clone() * policy_weight;
        output.value_loss = output.base_value_loss.clone() * value_weight;
        output.time_usage_loss = output.base_time_usage_loss.clone() * time_weight;
        output.aux_loss = output.base_aux_loss.clone() * aux_weight;
        output.calibration_loss = output.base_calibration_loss.clone() * calibration_weight;
        output.policy_regret_loss =
            output.base_policy_regret_loss.clone() * policy_regret_weight;
        output.loss = output.policy_loss.clone()
            + output.value_loss.clone()
            + output.time_usage_loss.clone()
            + output.aux_loss.clone()
            + output.calibration_loss.clone()
            + output.policy_regret_loss.clone();
    }

    pub fn apply_probe_results(
        &mut self,
        next_optimizer_step: usize,
        results: &[GradNormProbeResult],
    ) -> Option<Vec<GradNormTaskStatus>> {
        if !self.enabled {
            return None;
        }

        let active_indices: Vec<usize> = self
            .tasks
            .iter()
            .enumerate()
            .filter(|(_, task)| task.enabled)
            .map(|(idx, _)| idx)
            .collect();

        if active_indices.len() <= 1 {
            return None;
        }

        let mut gradient_norms = Vec::with_capacity(active_indices.len());
        let mut base_losses = Vec::with_capacity(active_indices.len());

        for &task_index in &active_indices {
            let task_kind = self.tasks[task_index].task;
            if let Some(result) = results.iter().find(|result| result.task == task_kind) {
                let base_loss_value = result.base_loss;
                base_losses.push(base_loss_value);

                if self.tasks[task_index].initial_loss.is_none() && base_loss_value > 0.0 {
                    self.tasks[task_index].initial_loss = Some(base_loss_value);
                }

                gradient_norms.push(result.grad_norm);
            } else {
                return None;
            }
        }

        let g_avg = gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32;

        let mut updated_weights: Vec<(usize, f32)> = Vec::with_capacity(active_indices.len());
        for ((&task_index, grad_norm), base_loss) in active_indices
            .iter()
            .zip(gradient_norms.into_iter())
            .zip(base_losses.into_iter())
        {
            let task_state = &mut self.tasks[task_index];
            let initial_loss = task_state
                .initial_loss
                .unwrap_or(base_loss.max(EPS))
                .max(EPS);
            let loss_ratio = (base_loss.max(EPS) / initial_loss).powf(self.alpha);
            let target_grad = g_avg * loss_ratio * task_state.priority;
            let current_grad = grad_norm.max(EPS);
            let ratio = (target_grad + EPS) / current_grad;
            let adjusted = task_state.weight * ratio.powf(self.lr);
            tracing::info!(
                "gradnorm_update: task={} weight={:.6} grad={:.2e} target_grad={:.2e} ratio={:.4} adjusted={:.6} loss_ratio={:.4}",
                task_state.task.name(), task_state.weight, current_grad, target_grad, ratio, adjusted, loss_ratio
            );
            task_state.last_grad_norm = Some(current_grad);
            task_state.last_grad_step = Some(next_optimizer_step);
            updated_weights.push((task_index, adjusted.max(EPS)));
        }

        for (index, new_weight) in updated_weights {
            self.tasks[index].weight = new_weight;
        }

        self.renormalize_weights();

        // Clamp weights AFTER renormalization. The ceiling is tied to the initial
        // weight (prevents any task from dominating during noisy bootstrap probes
        // by more than 10x its intended scale). The floor is an absolute value —
        // operator intent no longer constrains how low gradnorm can drive a task,
        // which keeps behavior predictable when initial weights are set on
        // different scales (e.g. raw-cp hinge vs O(1) CE losses).
        for task in &mut self.tasks {
            if task.enabled {
                let initial = task.initial_weight.max(EPS);
                task.weight = task.weight.clamp(WEIGHT_FLOOR, initial * 10.0);
            }
        }

        let final_weights: Vec<String> = self
            .tasks
            .iter()
            .filter(|t| t.enabled)
            .map(|t| format!("{}={:.6}", t.task.name(), t.weight))
            .collect();
        tracing::info!("gradnorm_post_renorm: {}", final_weights.join(" "));

        Some(self.status_snapshot())
    }

    fn renormalize_weights(&mut self) {
        if self.reference_total_weight <= 0.0 {
            return;
        }

        let sum_active: f32 = self
            .tasks
            .iter()
            .filter(|task| task.enabled)
            .map(|task| task.weight)
            .sum();

        if sum_active <= 0.0 {
            return;
        }

        let scale = self.reference_total_weight / sum_active;
        for task in &mut self.tasks {
            if task.enabled {
                task.weight = (task.weight * scale).max(EPS);
            }
        }

        self.refresh_weighted_losses();
    }

    fn refresh_weighted_losses(&mut self) {
        for task in &mut self.tasks {
            if task.enabled {
                if let Some(loss) = task.last_loss {
                    task.last_weighted_loss = Some(loss * task.weight);
                }
            }
        }
    }
}
