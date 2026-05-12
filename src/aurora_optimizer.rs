use burn::module::AutodiffModule;
use burn::optim::{
    adaptor::OptimizerAdaptor,
    decay::WeightDecayConfig,
    momentum::{Momentum, MomentumConfig, MomentumState},
    AdjustLrFn, LearningRate, SimpleOptimizer,
};
use burn::record::Record;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ops::Device, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuonUpdateKind {
    Muon,
    Aurora,
}

impl MuonUpdateKind {
    pub fn parse(value: &str) -> Self {
        match value {
            "muon" => Self::Muon,
            "aurora" => Self::Aurora,
            _ => {
                tracing::warn!(
                    "Unknown muon_optimizer={value:?}; using aurora. Valid values: aurora, muon"
                );
                Self::Aurora
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct AuroraConfig {
    weight_decay: Option<WeightDecayConfig>,
    momentum: MomentumConfig,
    ns_coefficients: (f32, f32, f32),
    epsilon: f32,
    ns_steps: usize,
    adjust_lr_fn: AdjustLrFn,
    update_kind: MuonUpdateKind,
    pp_iterations: usize,
    pp_beta: f32,
}

impl AuroraConfig {
    pub fn new() -> Self {
        Self {
            weight_decay: None,
            momentum: MomentumConfig {
                momentum: 0.95,
                dampening: 0.0,
                nesterov: true,
            },
            ns_coefficients: (3.4445, -4.775, 2.0315),
            epsilon: 1e-7,
            ns_steps: 5,
            adjust_lr_fn: AdjustLrFn::Original,
            update_kind: MuonUpdateKind::Aurora,
            pp_iterations: 2,
            pp_beta: 0.5,
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: Option<WeightDecayConfig>) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_adjust_lr_fn(mut self, adjust_lr_fn: AdjustLrFn) -> Self {
        self.adjust_lr_fn = adjust_lr_fn;
        self
    }

    pub fn with_update_kind(mut self, update_kind: MuonUpdateKind) -> Self {
        self.update_kind = update_kind;
        self
    }

    pub fn with_pp_iterations(mut self, pp_iterations: usize) -> Self {
        self.pp_iterations = pp_iterations.max(1);
        self
    }

    pub fn with_pp_beta(mut self, pp_beta: f32) -> Self {
        self.pp_beta = pp_beta.max(f32::EPSILON);
        self
    }

    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Aurora<B::InnerBackend>, M, B> {
        let momentum = Momentum::new(&self.momentum);
        let weight_decay_penalty = self.weight_decay.as_ref().map(|wd| wd.penalty);

        let optim = Aurora {
            momentum,
            ns_params: NewtonSchulzParams::new(self.ns_coefficients, self.ns_steps),
            weight_decay_penalty,
            epsilon: self.epsilon,
            adjust_lr_fn: self.adjust_lr_fn,
            update_kind: self.update_kind,
            pp_iterations: self.pp_iterations,
            pp_beta: self.pp_beta,
        };

        OptimizerAdaptor::from(optim)
    }
}

#[derive(Clone, Copy)]
struct NewtonSchulzParams {
    a: f32,
    b: f32,
    c: f32,
    steps: usize,
}

impl NewtonSchulzParams {
    fn new(coefficients: (f32, f32, f32), steps: usize) -> Self {
        Self {
            a: coefficients.0,
            b: coefficients.1,
            c: coefficients.2,
            steps,
        }
    }
}

#[derive(Clone)]
pub struct Aurora<B: Backend> {
    momentum: Momentum<B>,
    ns_params: NewtonSchulzParams,
    weight_decay_penalty: Option<f32>,
    epsilon: f32,
    adjust_lr_fn: AdjustLrFn,
    update_kind: MuonUpdateKind,
    pp_iterations: usize,
    pp_beta: f32,
}

impl<B: Backend> Aurora<B> {
    fn adjust_lr(&self, lr: LearningRate, shape: &[usize]) -> LearningRate {
        if shape.len() < 2 {
            return lr;
        }

        let a = shape[0] as f64;
        let b = shape[1] as f64;
        let ratio = match self.adjust_lr_fn {
            AdjustLrFn::Original => (a / b).max(1.0).sqrt(),
            AdjustLrFn::MatchRmsAdamW => 0.2 * a.max(b).sqrt(),
        };

        lr * ratio
    }

    fn zeropower_via_newtonschulz<const D: usize>(&self, g: Tensor<B, D>) -> Tensor<B, D> {
        let shape = g.shape();
        let dim_m2 = shape[D - 2];
        let dim_m1 = shape[D - 1];

        let (mut x, needs_transpose) = if dim_m2 > dim_m1 {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .clamp_min(self.epsilon)
            .unsqueeze();

        x = x.div(norm);

        let NewtonSchulzParams { a, b, c, steps } = self.ns_params;

        for _ in 0..steps {
            let x_t = x.clone().swap_dims(D - 2, D - 1);
            let a_matrix = x.clone().matmul(x_t);
            let a_squared = a_matrix.clone().matmul(a_matrix.clone());
            let b_matrix = a_matrix.mul_scalar(b).add(a_squared.mul_scalar(c));
            x = x.clone().mul_scalar(a).add(b_matrix.matmul(x.clone()));
        }

        if needs_transpose {
            x = x.swap_dims(D - 2, D - 1);
        }

        x
    }

    fn aurora_update<const D: usize>(&self, grad: Tensor<B, D>) -> Tensor<B, D> {
        let shape = grad.shape();
        let rows = shape[D - 2];
        let cols = shape[D - 1];

        if rows == cols {
            return self.zeropower_via_newtonschulz(grad);
        }

        let (working, transposed, tall_rows, tall_cols) = if rows < cols {
            (grad.swap_dims(D - 2, D - 1), true, cols, rows)
        } else {
            (grad, false, rows, cols)
        };

        let target_row_sq = tall_cols as f64 / tall_rows as f64;
        let row_norm = working
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(self.epsilon);
        let mut row_scale = row_norm.recip();
        let mut update = self.zeropower_via_newtonschulz(row_scale.clone() * working.clone());

        for _ in 1..self.pp_iterations {
            let row_sq = update
                .clone()
                .powf_scalar(2.0)
                .sum_dim(D - 1)
                .clamp_min(self.epsilon * self.epsilon);
            let correction = (row_sq.recip() * target_row_sq).powf_scalar(self.pp_beta);
            row_scale = row_scale * correction;
            update = self.zeropower_via_newtonschulz(row_scale.clone() * working.clone());
        }

        if transposed {
            update.swap_dims(D - 2, D - 1)
        } else {
            update
        }
    }
}

#[derive(Record, Clone)]
pub struct AuroraState<B: Backend, const D: usize> {
    pub momentum: MomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for Aurora<B> {
    type State<const D: usize> = AuroraState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(
            D == 2,
            "Muon/Aurora iteration requires 2D tensors, got {}D",
            D
        );

        let state_momentum = state.map(|s| s.momentum);
        let (grad, new_momentum_state) = self.momentum.transform(grad, state_momentum);

        let update = match self.update_kind {
            MuonUpdateKind::Muon => self.zeropower_via_newtonschulz(grad),
            MuonUpdateKind::Aurora => self.aurora_update(grad),
        };

        let adjusted_lr = self.adjust_lr(lr, &tensor.shape());
        let tensor = if let Some(penalty) = self.weight_decay_penalty {
            tensor.mul_scalar(1.0 - lr * penalty as f64)
        } else {
            tensor
        };

        let delta = update.mul_scalar(adjusted_lr);
        let new_state = AuroraState {
            momentum: new_momentum_state,
        };

        (tensor - delta, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}
