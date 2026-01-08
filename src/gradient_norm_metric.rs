use core::marker::PhantomData;
use std::cmp::Ordering;
use std::collections::HashMap;

use burn::module::{AutodiffModule, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};

use crate::config::{get_global_config, Config};

/// Aggregated gradient norm information collected during a backward pass.
#[derive(Debug, Clone)]
pub struct GradientNormBreakdown {
    /// Overall L2 norm across all parameters.
    pub total: f64,
    /// L2 norms aggregated by logical layer/module name.
    pub per_layer: Vec<LayerGradientNorm>,
    /// L2 norms aggregated per attention head (Q/K/V projections).
    pub per_head: Vec<HeadGradientNorm>,
}

impl GradientNormBreakdown {
    pub fn total(&self) -> f64 {
        self.total
    }
}

/// Gradient norm associated with a single layer or module.
#[derive(Debug, Clone)]
pub struct LayerGradientNorm {
    pub name: String,
    pub norm: f64,
}

/// Which projection a head gradient corresponds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadComponent {
    Query,
    Key,
    Value,
}

impl HeadComponent {
    pub fn as_str(&self) -> &'static str {
        match self {
            HeadComponent::Query => "query",
            HeadComponent::Key => "key",
            HeadComponent::Value => "value",
        }
    }
}

/// Gradient norm information for an individual attention head.
#[derive(Debug, Clone)]
pub struct HeadGradientNorm {
    pub layer: String,
    pub component: HeadComponent,
    pub head_index: usize,
    pub norm: f64,
}

/// Compute gradient norm statistics (total norm, per-layer, per-head).
/// When `need_breakdown` is false, skips per-layer and per-head tracking for faster execution.
pub fn compute_gradient_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    grads: &GradientsParams,
    model: &M,
) -> GradientNormBreakdown {
    compute_gradient_norm_with_breakdown(grads, model, true)
}

/// Compute gradient norm with optional breakdown.
/// When `need_breakdown` is false, only computes total norm (skips expensive head slicing).
pub fn compute_gradient_norm_with_breakdown<B: AutodiffBackend, M: AutodiffModule<B>>(
    grads: &GradientsParams,
    model: &M,
    need_breakdown: bool,
) -> GradientNormBreakdown {
    let mut visitor = GradientNormVisitor::new(grads, need_breakdown);
    model.visit(&mut visitor);
    visitor.into_breakdown()
}

struct GradientNormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    total_norm_squared: f64,
    per_layer_norms: HashMap<String, f64>,
    per_head_norms: HashMap<HeadKey, f64>,
    config: &'static Config,
    path_stack: Vec<String>,
    need_breakdown: bool,
    _backend: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> GradientNormVisitor<'a, B> {
    fn new(grads: &'a GradientsParams, need_breakdown: bool) -> Self {
        Self {
            grads,
            total_norm_squared: 0.0,
            per_layer_norms: HashMap::new(),
            per_head_norms: HashMap::new(),
            config: get_global_config(),
            path_stack: Vec::new(),
            need_breakdown,
            _backend: PhantomData,
        }
    }

    fn into_breakdown(self) -> GradientNormBreakdown {
        let mut per_layer: Vec<LayerGradientNorm> = self
            .per_layer_norms
            .into_iter()
            .map(|(name, norm_sq)| LayerGradientNorm {
                name,
                norm: norm_sq.sqrt(),
            })
            .collect();
        per_layer.sort_by(|a, b| b.norm.partial_cmp(&a.norm).unwrap_or(Ordering::Equal));

        let mut per_head: Vec<HeadGradientNorm> = self
            .per_head_norms
            .into_iter()
            .map(|(key, norm_sq)| HeadGradientNorm {
                layer: key.layer,
                component: key.component,
                head_index: key.head_index,
                norm: norm_sq.sqrt(),
            })
            .collect();
        per_head.sort_by(|a, b| b.norm.partial_cmp(&a.norm).unwrap_or(Ordering::Equal));

        GradientNormBreakdown {
            total: self.total_norm_squared.sqrt(),
            per_layer,
            per_head,
        }
    }

    fn handle_float<const D: usize>(&mut self, id: ParamId, path: &[String]) {
        if let Some(grad_tensor) = self.grads.get::<B::InnerBackend, D>(id) {
            let norm_sq = grad_tensor
                .clone()
                .powi_scalar(2)
                .sum()
                .into_scalar()
                .elem::<f32>() as f64;
            self.total_norm_squared += norm_sq;

            if self.need_breakdown {
                let layer_key = self.layer_key(path);
                *self.per_layer_norms.entry(layer_key.clone()).or_default() += norm_sq;
                self.accumulate_head_norm::<D>(path, &layer_key, &grad_tensor);
            }
        }
    }

    fn layer_key(&self, path: &[String]) -> String {
        if let Some(idx) = path.iter().position(|p| p == "blocks") {
            if let Some(layer_idx) = path.get(idx + 1) {
                if !layer_idx.is_empty() {
                    return format!("blocks.{}", layer_idx);
                }
            }
        }
        path.iter()
            .find(|segment| !segment.is_empty())
            .cloned()
            .unwrap_or_else(|| "root".to_string())
    }

    fn is_fused_qkv(&self, path: &[String]) -> bool {
        path.iter().any(|p| p == "qkv_proj")
    }

    fn head_component(&self, path: &[String]) -> Option<HeadComponent> {
        if path.iter().any(|p| p == "q_proj") {
            Some(HeadComponent::Query)
        } else if path.iter().any(|p| p == "k_proj") {
            Some(HeadComponent::Key)
        } else if path.iter().any(|p| p == "v_proj") {
            Some(HeadComponent::Value)
        } else {
            None
        }
    }

    fn accumulate_head_norm<const D: usize>(
        &mut self,
        path: &[String],
        layer_key: &str,
        grad_tensor: &Tensor<B::InnerBackend, D>,
    ) {
        if D != 1 && D != 2 {
            return;
        }

        if self.is_fused_qkv(path) {
            self.accumulate_fused_qkv_head_norm::<D>(layer_key, grad_tensor);
            return;
        }

        let Some(component) = self.head_component(path) else {
            return;
        };

        let head_dim = self.config.head_dim();
        let head_count = self.config.num_heads();

        let axis = if D == 1 { 0 } else { 1 };
        let axis_size = grad_tensor.dims()[axis];

        for head_index in 0..head_count {
            let start = head_index * head_dim;
            if start >= axis_size {
                break;
            }
            let length = head_dim.min(axis_size - start);
            let slice = grad_tensor.clone().narrow(axis, start, length);
            let norm_sq = slice.powi_scalar(2).sum().into_scalar().elem::<f32>() as f64;

            let key = HeadKey {
                layer: layer_key.to_string(),
                component,
                head_index,
            };
            *self.per_head_norms.entry(key).or_default() += norm_sq;
        }
    }

    fn accumulate_fused_qkv_head_norm<const D: usize>(
        &mut self,
        layer_key: &str,
        grad_tensor: &Tensor<B::InnerBackend, D>,
    ) {
        let embed_dim = self.config.embed_dim();
        let head_dim = self.config.head_dim();
        let head_count = self.config.num_heads();

        let axis = if D == 1 { 0 } else { 1 };

        let components = [
            (HeadComponent::Query, 0),
            (HeadComponent::Key, embed_dim),
            (HeadComponent::Value, 2 * embed_dim),
        ];

        for (component, component_offset) in components {
            for head_index in 0..head_count {
                let start = component_offset + head_index * head_dim;
                let slice = grad_tensor.clone().narrow(axis, start, head_dim);
                let norm_sq = slice.powi_scalar(2).sum().into_scalar().elem::<f32>() as f64;

                let key = HeadKey {
                    layer: layer_key.to_string(),
                    component,
                    head_index,
                };
                *self.per_head_norms.entry(key).or_default() += norm_sq;
            }
        }
    }
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientNormVisitor<'a, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path_stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        if !self.path_stack.is_empty() {
            self.path_stack.pop();
        }
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.path_stack.is_empty() {
            return;
        }

        let path = self.path_stack.clone();
        self.handle_float::<D>(param.id, &path);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HeadKey {
    layer: String,
    component: HeadComponent,
    head_index: usize,
}

/// Input type for gradient norm metric
#[derive(Clone, Copy)]
pub struct GradientNormInput {
    /// The computed gradient norm value
    pub norm: f64,
}

impl GradientNormInput {
    pub fn new(norm: f64) -> Self {
        Self { norm }
    }
}

/// Metric for tracking the L2 norm of all gradients
#[derive(Default, Clone)]
pub struct GradientNormMetric<B: Backend> {
    current_value: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> GradientNormMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for GradientNormMetric<B> {
    type Input = GradientNormInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.current_value = input.norm;

        let formatted = if input.norm < 0.01 {
            format!("{:.2e}", input.norm)
        } else {
            format!("{:.6}", input.norm)
        };

        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.current_value = 0.0;
    }

    fn name(&self) -> std::sync::Arc<String> {
        "Gradient Norm".to_string().into()
    }
}

impl<B: Backend> Numeric for GradientNormMetric<B> {
    fn value(&self) -> NumericEntry {
        // Cap at 10 for display purposes
        NumericEntry::Value(self.current_value.min(10.0))
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_value.min(10.0))
    }
}
