use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use burn::module::{list_param_ids, AutodiffModule, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use tracing::info;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightDecayGroup {
    Decay,
    NoDecay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LearningRateGroup {
    Normal,
    HighLR, // For embeddings and scale params that should get sqrt(embed_dim) multiplier
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParameterGroup {
    decay: WeightDecayGroup,
    lr: LearningRateGroup,
}

/// Represents a partition of model parameters into weight-decayed and non-decayed groups,
/// as well as different learning rate groups for embeddings.
pub struct WeightDecayGroups {
    assignments: HashMap<ParamId, ParameterGroup>,
}

impl WeightDecayGroups {
    /// Build the weight decay grouping for a model once at optimizer initialization time.
    pub fn new<B, M>(model: &M) -> Self
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        let mut classifier = WeightDecayClassifier::<B>::new();
        model.visit(&mut classifier);

        #[cfg(debug_assertions)]
        {
            // Ensure every parameter received a classification.
            let expected = list_param_ids(&model.valid());
            debug_assert_eq!(
                expected.len(),
                classifier.assignments.len(),
                "Weight decay classifier did not cover all parameters"
            );
        }

        let groups = Self {
            assignments: classifier.assignments,
        };

        let (decay_count, no_decay_count, normal_lr_count, high_lr_count) = groups.counts();
        info!(
            "[WeightDecayGroups::new] Total assignments: {} decay, {} no_decay, {} normal_lr, {} high_lr",
            decay_count, no_decay_count, normal_lr_count, high_lr_count
        );

        // Log detailed parameter grouping
        info!("[WeightDecayGroups::new] Parameter group assignments:");

        let mut high_lr_params: Vec<_> = classifier
            .param_paths
            .iter()
            .filter(|(_, group)| group.lr == LearningRateGroup::HighLR)
            .collect();
        high_lr_params.sort_by(|a, b| a.0.cmp(&b.0));

        if !high_lr_params.is_empty() {
            info!(
                "[WeightDecayGroups::new] High LR parameters ({} total):",
                high_lr_params.len()
            );
            for (path, group) in high_lr_params.iter() {
                info!(
                    "[WeightDecayGroups::new]   {} (decay={:?})",
                    path, group.decay
                );
            }
        }

        // Show all normal LR parameters with normalized paths (e.g., blocks.* instead of blocks.0, blocks.1, etc.)
        let mut normal_lr_params: Vec<_> = classifier
            .param_paths
            .iter()
            .filter(|(_, group)| group.lr == LearningRateGroup::Normal)
            .map(|(path, group)| (normalize_path(path), *group))
            .collect();

        // Deduplicate by normalized path
        let mut seen = HashSet::new();
        normal_lr_params.retain(|(path, _)| seen.insert(path.clone()));
        normal_lr_params.sort_by(|a, b| a.0.cmp(&b.0));

        if !normal_lr_params.is_empty() {
            info!(
                "[WeightDecayGroups::new] Normal LR parameters ({} unique patterns of {} total):",
                normal_lr_params.len(),
                normal_lr_count
            );
            for (path, group) in normal_lr_params.iter() {
                info!(
                    "[WeightDecayGroups::new]   {} (decay={:?})",
                    path, group.decay
                );
            }
        }

        groups
    }

    /// Compute the L2 penalty (regularization loss) for parameters in the decay group.
    /// Returns: 0.5 * weight_decay * sum(||param||^2) for all decay-group parameters.
    pub fn compute_l2_penalty<B, M>(&self, model: &M, weight_decay: f64) -> f64
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        let mut visitor = L2PenaltyVisitor::<B>::new(&self.assignments);
        model.visit(&mut visitor);
        let total_squared_norm = visitor.total_squared_norm;
        0.5 * weight_decay * total_squared_norm
    }

    /// Split gradients into 4 sets: (decay+normal_lr, decay+high_lr, no_decay+normal_lr, no_decay+high_lr)
    pub fn split_grads<B, M>(
        &self,
        model: &M,
        grads: GradientsParams,
    ) -> (
        GradientsParams,
        GradientsParams,
        GradientsParams,
        GradientsParams,
    )
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        info!(
            "[split_grads] Starting gradient split with {} total grads",
            grads.len()
        );
        let mut splitter = GradientsSplitVisitor::<B>::new(&self.assignments, grads);
        model.visit(&mut splitter);
        let (decay_normal, decay_high, no_decay_normal, no_decay_high) = splitter.finish();
        info!(
            "[split_grads] Result: {} decay+normal, {} decay+high, {} no_decay+normal, {} no_decay+high",
            decay_normal.len(), decay_high.len(), no_decay_normal.len(), no_decay_high.len()
        );
        (decay_normal, decay_high, no_decay_normal, no_decay_high)
    }

    /// Return (decay_count, no_decay_count, normal_lr_count, high_lr_count) for logging/debugging.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut decay = 0;
        let mut no_decay = 0;
        let mut normal_lr = 0;
        let mut high_lr = 0;

        for group in self.assignments.values() {
            match group.decay {
                WeightDecayGroup::Decay => decay += 1,
                WeightDecayGroup::NoDecay => no_decay += 1,
            }
            match group.lr {
                LearningRateGroup::Normal => normal_lr += 1,
                LearningRateGroup::HighLR => high_lr += 1,
            }
        }

        (decay, no_decay, normal_lr, high_lr)
    }
}

/// Classifies parameters by walking the module once and recording assignments.
struct WeightDecayClassifier<B: AutodiffBackend> {
    assignments: HashMap<ParamId, ParameterGroup>,
    path_stack: Vec<String>,
    backend: PhantomData<B>,
    // Track parameter paths for logging
    param_paths: Vec<(String, ParameterGroup)>,
}

impl<B: AutodiffBackend> WeightDecayClassifier<B> {
    fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            path_stack: Vec::new(),
            backend: PhantomData,
            param_paths: Vec::new(),
        }
    }

    fn classify_path(&self, path: &[String]) -> ParameterGroup {
        let path_str = path.join(".");

        // First determine weight decay group
        let decay = if path_contains(path, "bias") {
            WeightDecayGroup::NoDecay
        } else if path_contains(path, "gamma") || path_contains(path, "beta") {
            WeightDecayGroup::NoDecay
        } else if path_contains(path, "token_embed") {
            WeightDecayGroup::NoDecay
        } else if path_contains(path, "global_embed") {
            WeightDecayGroup::NoDecay
        } else if path_contains(path, "gate_logits") {
            WeightDecayGroup::NoDecay
        } else if path_contains(path, "policy_uncertainty")
            || path_contains(path, "value_uncertainty")
            || path_contains(path, "side_info_uncertainty")
            || path_contains(path, "time_usage_uncertainty")
        {
            WeightDecayGroup::NoDecay
        } else {
            WeightDecayGroup::Decay
        };

        let lr = if (path_contains(path, "token_embed") && path_contains(path, "weight"))
            || (path_contains(path, "global_embed") && path_contains(path, "weight"))
        {
            LearningRateGroup::HighLR
        } else {
            LearningRateGroup::Normal
        };

        ParameterGroup { decay, lr }
    }
}

impl<B: AutodiffBackend> ModuleVisitor<B> for WeightDecayClassifier<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path_stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let id = param.id;
        if self.assignments.contains_key(&id) {
            return;
        }
        let assignment = self.classify_path(&self.path_stack);
        let path_str = self.path_stack.join(".");
        self.param_paths.push((path_str, assignment));
        self.assignments.insert(id, assignment);
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        _tensor: &Tensor<B, D>,
    ) {
        let assignment = self.classify_path(path);
        let path_str = path.join(".");
        self.param_paths.push((path_str, assignment));
        self.assignments.insert(id, assignment);
    }
}

/// Utility visitor that partitions gradients into 4 groups based on decay and LR.
struct GradientsSplitVisitor<'a, B: AutodiffBackend> {
    assignments: &'a HashMap<ParamId, ParameterGroup>,
    remaining: GradientsParams,
    grads_decay_normal: GradientsParams,
    grads_decay_high: GradientsParams,
    grads_no_decay_normal: GradientsParams,
    grads_no_decay_high: GradientsParams,
    backend: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> GradientsSplitVisitor<'a, B> {
    fn new(assignments: &'a HashMap<ParamId, ParameterGroup>, grads: GradientsParams) -> Self {
        Self {
            assignments,
            remaining: grads,
            grads_decay_normal: GradientsParams::new(),
            grads_decay_high: GradientsParams::new(),
            grads_no_decay_normal: GradientsParams::new(),
            grads_no_decay_high: GradientsParams::new(),
            backend: PhantomData,
        }
    }

    fn finish(
        self,
    ) -> (
        GradientsParams,
        GradientsParams,
        GradientsParams,
        GradientsParams,
    ) {
        debug_assert!(
            self.remaining.is_empty(),
            "Unassigned gradients remain after gradient split ({} tensors)",
            self.remaining.len()
        );
        (
            self.grads_decay_normal,
            self.grads_decay_high,
            self.grads_no_decay_normal,
            self.grads_no_decay_high,
        )
    }
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientsSplitVisitor<'a, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let id = param.id;
        let Some(grad) = self.remaining.remove::<B::InnerBackend, D>(id) else {
            return;
        };

        let group = self
            .assignments
            .get(&id)
            .copied()
            .unwrap_or(ParameterGroup {
                decay: WeightDecayGroup::Decay,
                lr: LearningRateGroup::Normal,
            });

        match (group.decay, group.lr) {
            (WeightDecayGroup::Decay, LearningRateGroup::Normal) => {
                self.grads_decay_normal
                    .register::<B::InnerBackend, D>(id, grad);
            }
            (WeightDecayGroup::Decay, LearningRateGroup::HighLR) => {
                self.grads_decay_high
                    .register::<B::InnerBackend, D>(id, grad);
            }
            (WeightDecayGroup::NoDecay, LearningRateGroup::Normal) => {
                self.grads_no_decay_normal
                    .register::<B::InnerBackend, D>(id, grad);
            }
            (WeightDecayGroup::NoDecay, LearningRateGroup::HighLR) => {
                self.grads_no_decay_high
                    .register::<B::InnerBackend, D>(id, grad);
            }
        }
    }
}

/// Visitor that computes the total squared L2 norm of parameters in the decay group.
struct L2PenaltyVisitor<'a, B: AutodiffBackend> {
    assignments: &'a HashMap<ParamId, ParameterGroup>,
    total_squared_norm: f64,
    backend: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> L2PenaltyVisitor<'a, B> {
    fn new(assignments: &'a HashMap<ParamId, ParameterGroup>) -> Self {
        Self {
            assignments,
            total_squared_norm: 0.0,
            backend: PhantomData,
        }
    }
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for L2PenaltyVisitor<'a, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let id = param.id;
        let tensor = param.val();
        let group = self
            .assignments
            .get(&id)
            .copied()
            .unwrap_or(ParameterGroup {
                decay: WeightDecayGroup::Decay,
                lr: LearningRateGroup::Normal,
            });

        // Only accumulate norm for parameters in the decay group
        if group.decay == WeightDecayGroup::Decay {
            let norm_sq = tensor
                .clone()
                .powi_scalar(2)
                .sum()
                .to_data()
                .to_vec::<f32>()
                .unwrap()[0] as f64;
            self.total_squared_norm += norm_sq;
        }
    }
}

fn path_contains(path: &[String], needle: &str) -> bool {
    path.iter()
        .flat_map(|segment| segment.split('.'))
        .any(|part| part == needle)
}

fn normalize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('.').collect();
    parts
        .iter()
        .map(|&part| {
            if part.chars().all(|c| c.is_ascii_digit()) {
                "*"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}
