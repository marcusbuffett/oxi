use crate::legal_move_probability_metric::LegalMoveProbabilityInput;
use crate::move_accuracy_metric::MoveAccuracyInput;
use crate::move_distribution_accuracy_metric::MoveDistributionAccuracyInput;
use crate::policy_loss_metric::PolicyLossInput;
use crate::side_info_loss_metric::SideInfoLossInput;
use crate::time_usage_loss_metric::TimeUsageLossInput;
use crate::uncertainty_metric::UncertaintyInput;
use crate::value_loss_metric::ValueLossInput;
use crate::wdl_accuracy_metric::WdlAccuracyInput;
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::train::metric::{Adaptor, LossInput};
use burn_ndarray::NdArray;
use burn_train::metric::ItemLazy;

/// Custom classification output for chess that includes separate policy and value outputs
#[derive(Debug, Clone)]
pub struct ChessOutput<B: Backend> {
    /// The combined loss
    pub loss: Tensor<B, 1>,
    /// The policy loss component (after uncertainty weighting)
    pub policy_loss: Tensor<B, 1>,
    /// The raw policy loss (before uncertainty weighting, but after config weighting)
    pub raw_policy_loss: Option<Tensor<B, 1>>,
    /// The value loss component (after uncertainty weighting)
    pub value_loss: Tensor<B, 1>,
    /// The raw value loss (before uncertainty weighting, but after config weighting)
    pub raw_value_loss: Option<Tensor<B, 1>>,
    /// The side info loss component (after uncertainty weighting)
    pub side_info_loss: Tensor<B, 1>,
    /// The raw side info loss (before uncertainty weighting, but after config weighting)
    pub raw_side_info_loss: Option<Tensor<B, 1>>,
    /// The time usage loss component (after uncertainty weighting)
    pub time_usage_loss: Tensor<B, 1>,
    /// The raw time usage loss (before uncertainty weighting, but after config weighting)
    pub raw_time_usage_loss: Option<Tensor<B, 1>>,
    /// The policy output (masked move logits)
    pub policy_output: Tensor<B, 2>,
    /// The policy targets (for move accuracy)
    pub policy_targets: Tensor<B, 1, Int>,
    /// The value output (WDL logits)
    pub value_output: Tensor<B, 2>,
    /// The value targets (one-hot WDL)
    pub value_targets: Tensor<B, 2>,
    /// The legal moves mask (1.0 for legal moves, 0.0 for illegal)
    pub legal_moves_mask: Tensor<B, 2>,
    /// The target move distributions (for distribution accuracy metric)
    pub target_distributions: Option<Tensor<B, 2>>,
    /// Uncertainty values (sigma) for each loss component
    pub uncertainties: Option<(f32, f32, f32, f32)>,
}

impl<B: Backend> ChessOutput<B> {
    /// Creates a new ChessOutput
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        value_loss: Tensor<B, 1>,
        side_info_loss: Tensor<B, 1>,
        time_usage_loss: Tensor<B, 1>,
        policy_output: Tensor<B, 2>,
        policy_targets: Tensor<B, 1, Int>,
        value_output: Tensor<B, 2>,
        value_targets: Tensor<B, 2>,
        legal_moves_mask: Tensor<B, 2>,
    ) -> Self {
        Self {
            loss,
            policy_loss,
            value_loss,
            side_info_loss,
            time_usage_loss,
            policy_output,
            policy_targets,
            value_output,
            value_targets,
            legal_moves_mask,
            target_distributions: None,
            uncertainties: None,
            raw_policy_loss: None,
            raw_value_loss: None,
            raw_side_info_loss: None,
            raw_time_usage_loss: None,
        }
    }

    /// Creates a new ChessOutput with target distributions
    pub fn with_distributions(mut self, target_distributions: Tensor<B, 2>) -> Self {
        self.target_distributions = Some(target_distributions);
        self
    }

    /// Sets the uncertainty values
    pub fn with_uncertainties(mut self, uncertainties: (f32, f32, f32, f32)) -> Self {
        self.uncertainties = Some(uncertainties);
        self
    }

    /// Sets the raw value loss (before uncertainty weighting)
    pub fn with_raw_value_loss(mut self, raw_value_loss: Tensor<B, 1>) -> Self {
        self.raw_value_loss = Some(raw_value_loss);
        self
    }

    /// Sets all raw losses at once
    pub fn with_raw_losses(
        mut self,
        raw_policy_loss: Tensor<B, 1>,
        raw_value_loss: Tensor<B, 1>,
        raw_side_info_loss: Tensor<B, 1>,
        raw_time_usage_loss: Tensor<B, 1>,
    ) -> Self {
        self.raw_policy_loss = Some(raw_policy_loss);
        self.raw_value_loss = Some(raw_value_loss);
        self.raw_side_info_loss = Some(raw_side_info_loss);
        self.raw_time_usage_loss = Some(raw_time_usage_loss);
        self
    }
}

// Implement Adaptor for MoveAccuracyMetric
impl<B: Backend> Adaptor<MoveAccuracyInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> MoveAccuracyInput<B> {
        MoveAccuracyInput::new(self.policy_output.clone(), self.policy_targets.clone())
    }
}

// Implement Adaptor for MoveDistributionAccuracyMetric
impl<B: Backend> Adaptor<MoveDistributionAccuracyInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> MoveDistributionAccuracyInput<B> {
        MoveDistributionAccuracyInput::new(
            self.policy_output.clone(),
            self.target_distributions.clone().unwrap_or_else(|| {
                // If no distributions provided, create one-hot from targets
                let batch_size = self.policy_targets.shape().dims[0];
                let num_moves = self.policy_output.shape().dims[1];
                let device = self.policy_targets.device();

                // Create one-hot encoding
                let mut one_hot = Tensor::zeros([batch_size, num_moves], &device);
                for i in 0..batch_size {
                    let target_idx = self
                        .policy_targets
                        .clone()
                        .slice([i..i + 1])
                        .into_scalar()
                        .elem::<i32>() as usize;
                    one_hot = one_hot.slice_assign(
                        [i..i + 1, target_idx..target_idx + 1],
                        Tensor::ones([1, 1], &device),
                    );
                }
                one_hot
            }),
        )
    }
}

// Implement Adaptor for PolicyLossMetric
impl<B: Backend> Adaptor<PolicyLossInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> PolicyLossInput<B> {
        let mut input = PolicyLossInput::new(self.policy_loss.clone());
        if let Some(raw_loss) = &self.raw_policy_loss {
            input = input.with_raw_loss(raw_loss.clone());
        }
        input
    }
}

// Implement Adaptor for ValueLossMetric
impl<B: Backend> Adaptor<ValueLossInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> ValueLossInput<B> {
        let mut input = ValueLossInput::new(self.value_loss.clone());
        if let Some(raw_loss) = &self.raw_value_loss {
            input = input.with_raw_loss(raw_loss.clone());
        }
        input
    }
}

// Implement Adaptor for SideInfoLossMetric
impl<B: Backend> Adaptor<SideInfoLossInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> SideInfoLossInput<B> {
        let mut input = SideInfoLossInput::new(self.side_info_loss.clone());
        if let Some(raw_loss) = &self.raw_side_info_loss {
            input = input.with_raw_loss(raw_loss.clone());
        }
        input
    }
}

// Implement Adaptor for TimeUsageLossMetric
impl<B: Backend> Adaptor<TimeUsageLossInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> TimeUsageLossInput<B> {
        let mut input = TimeUsageLossInput::new(self.time_usage_loss.clone());
        if let Some(raw_loss) = &self.raw_time_usage_loss {
            input = input.with_raw_loss(raw_loss.clone());
        }
        input
    }
}

// Implement Adaptor for LossMetric
impl<B: Backend> Adaptor<LossInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

// Implement Adaptor for our custom WdlAccuracyMetric
impl<B: Backend> Adaptor<WdlAccuracyInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> WdlAccuracyInput<B> {
        WdlAccuracyInput::new(self.value_output.clone(), self.value_targets.clone())
    }
}

// Implement Adaptor for LegalMoveProbabilityMetric
impl<B: Backend> Adaptor<LegalMoveProbabilityInput<B>> for ChessOutput<B> {
    fn adapt(&self) -> LegalMoveProbabilityInput<B> {
        LegalMoveProbabilityInput {
            policy_logits: self.policy_output.clone(),
            legal_moves_mask: self.legal_moves_mask.clone(),
        }
    }
}

// Implement Adaptor for UncertaintyMetric
impl<B: Backend> Adaptor<UncertaintyInput> for ChessOutput<B> {
    fn adapt(&self) -> UncertaintyInput {
        let (policy_sigma, value_sigma, side_info_sigma, _time_usage_sigma) =
            self.uncertainties.unwrap_or((1.0, 1.0, 1.0, 1.0));
        UncertaintyInput::new(policy_sigma, value_sigma, side_info_sigma)
    }
}

impl<B: Backend> ItemLazy for ChessOutput<B> {
    type ItemSync = ChessOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let target_distributions = self.target_distributions;
        let uncertainties = self.uncertainties;
        let raw_policy_loss = self.raw_policy_loss;
        let raw_value_loss = self.raw_value_loss;
        let raw_side_info_loss = self.raw_side_info_loss;
        let raw_time_usage_loss = self.raw_time_usage_loss;

        let [loss, policy_loss, value_loss, side_info_loss, time_usage_loss, policy_output, policy_targets, value_output, value_targets, legal_moves_mask] =
            Transaction::default()
                .register(self.loss)
                .register(self.policy_loss)
                .register(self.value_loss)
                .register(self.side_info_loss)
                .register(self.time_usage_loss)
                .register(self.policy_output)
                .register(self.policy_targets)
                .register(self.value_output)
                .register(self.value_targets)
                .register(self.legal_moves_mask)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");

        let device = &Default::default();

        let synced_distributions = target_distributions.map(|t| {
            let [dist_data] = Transaction::default()
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            Tensor::from_data(dist_data, device)
        });

        let synced_raw_policy_loss = raw_policy_loss.map(|t| {
            let [raw_data] = Transaction::default()
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            Tensor::from_data(raw_data, device)
        });

        let synced_raw_value_loss = raw_value_loss.map(|t| {
            let [raw_data] = Transaction::default()
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            Tensor::from_data(raw_data, device)
        });

        let synced_raw_side_info_loss = raw_side_info_loss.map(|t| {
            let [raw_data] = Transaction::default()
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            Tensor::from_data(raw_data, device)
        });

        let synced_raw_time_usage_loss = raw_time_usage_loss.map(|t| {
            let [raw_data] = Transaction::default()
                .register(t)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");
            Tensor::from_data(raw_data, device)
        });

        ChessOutput {
            loss: Tensor::from_data(loss, device),
            policy_loss: Tensor::from_data(policy_loss, device),
            value_loss: Tensor::from_data(value_loss, device),
            time_usage_loss: Tensor::from_data(time_usage_loss, device),
            side_info_loss: Tensor::from_data(side_info_loss, device),
            policy_output: Tensor::from_data(policy_output, device),
            policy_targets: Tensor::from_data(policy_targets, device),
            value_output: Tensor::from_data(value_output, device),
            value_targets: Tensor::from_data(value_targets, device),
            legal_moves_mask: Tensor::from_data(legal_moves_mask, device),
            target_distributions: synced_distributions,
            uncertainties,
            raw_policy_loss: synced_raw_policy_loss,
            raw_value_loss: synced_raw_value_loss,
            raw_side_info_loss: synced_raw_side_info_loss,
            raw_time_usage_loss: synced_raw_time_usage_loss,
        }
    }
}
