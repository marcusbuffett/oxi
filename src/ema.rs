//! Exponential moving average of model weights, used as training
//! instrumentation: at constant LR the live weights diffuse in a noise cloud
//! around the local basin, so live metrics flatline while the basin center is
//! still drifting. The EMA model sits near the center of that cloud —
//! evaluating it on a fixed set shows the true slow progress, and when the
//! EMA curve goes flat the run has genuinely plateaued.
//!
//! Cost: one flattened copy of every float param (~4 bytes/param) kept on
//! device, plus one fused multiply-add per param per update — negligible next
//! to a training step.

use std::collections::HashMap;

use burn::module::{AutodiffModule, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

pub struct EmaWeights<B: Backend> {
    beta: f64,
    store: HashMap<ParamId, Tensor<B, 1>>,
    updates: usize,
}

impl<B: Backend> EmaWeights<B> {
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            store: HashMap::new(),
            updates: 0,
        }
    }

    /// Fold the current weights of an autodiff model into the average.
    /// The first call initializes the average to the current weights.
    pub fn update<AB, M>(&mut self, model: &M)
    where
        AB: AutodiffBackend<InnerBackend = B>,
        M: AutodiffModule<AB>,
    {
        let mut visitor = EmaUpdateVisitor::<AB> {
            store: &mut self.store,
            beta: self.beta,
            first: self.updates == 0,
        };
        model.visit(&mut visitor);
        self.updates += 1;
    }

    /// Replace every float param of `model` (an inner-backend module) with its
    /// EMA value. Params never seen by `update` are left untouched.
    pub fn apply<M: Module<B>>(&self, model: M) -> M {
        let mut mapper = EmaApplyMapper { store: &self.store };
        model.map(&mut mapper)
    }

    pub fn updates(&self) -> usize {
        self.updates
    }
}

struct EmaUpdateVisitor<'a, AB: AutodiffBackend> {
    store: &'a mut HashMap<ParamId, Tensor<AB::InnerBackend, 1>>,
    beta: f64,
    first: bool,
}

impl<'a, AB: AutodiffBackend> ModuleVisitor<AB> for EmaUpdateVisitor<'a, AB> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<AB, D>>) {
        let flat: Tensor<AB::InnerBackend, 1> = param.val().inner().flatten(0, D - 1);
        if self.first {
            self.store.insert(param.id, flat);
        } else if let Some(ema) = self.store.get_mut(&param.id) {
            *ema = ema.clone() * self.beta + flat * (1.0 - self.beta);
        } else {
            self.store.insert(param.id, flat);
        }
    }
}

struct EmaApplyMapper<'a, B: Backend> {
    store: &'a HashMap<ParamId, Tensor<B, 1>>,
}

impl<'a, B: Backend> ModuleMapper<B> for EmaApplyMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let out = match self.store.get(&id) {
            Some(flat) => flat.clone().reshape(tensor.dims()),
            None => tensor,
        };
        Param::from_mapped_value(id, out, mapper)
    }
}
