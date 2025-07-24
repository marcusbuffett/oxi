use burn::tensor::{backend::Backend, Tensor};

/// Approximates log(Γ(x)) using Stirling's approximation with correction terms
///
/// Uses the formula:
/// log(Γ(x)) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2π) + 1/(12x) - 1/(360x³) + ...
///
/// This approximation is accurate for x > 1. For x < 1, we use the recurrence relation:
/// Γ(x) = Γ(x+1) / x, so log(Γ(x)) = log(Γ(x+1)) - log(x)
pub fn log_gamma_approx<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let device = x.device();

    // Constants
    let half: Tensor<B, 1> = Tensor::from_floats([0.5], &device);
    let one: Tensor<B, 1> = Tensor::from_floats([1.0], &device);
    let two_pi: Tensor<B, 1> = Tensor::from_floats([2.0 * std::f32::consts::PI], &device);
    let twelve: Tensor<B, 1> = Tensor::from_floats([12.0], &device);
    let three_sixty: Tensor<B, 1> = Tensor::from_floats([360.0], &device);

    // For x >= 1, use Stirling's approximation directly
    // For x < 1, use recurrence relation: log(Γ(x)) = log(Γ(x+1)) - log(x)
    let x_for_stirling = x.clone().clamp_min(1.0);

    // Stirling's approximation for x >= 1
    let log_x_stirling = x_for_stirling.clone().log();
    let stirling_main = (x_for_stirling.clone() - half.clone()) * log_x_stirling.clone()
        - x_for_stirling.clone()
        + half.clone() * two_pi.log();

    // Correction terms for better accuracy
    let correction1 = one.clone() / (twelve.clone() * x_for_stirling.clone());
    let x_cubed = x_for_stirling.clone().powf_scalar(3.0);
    let correction2 = one.clone() / (three_sixty.clone() * x_cubed);

    let stirling_result = stirling_main + correction1 - correction2;

    // Apply recurrence relation correction for x < 1
    // If x < 1, we computed log(Γ(x_clamped)) where x_clamped = max(x, 1)
    // But we want log(Γ(x)) = log(Γ(x+1)) - log(x) when x < 1
    // Since x_clamped = 1 when x < 1, we have log(Γ(1)) = 0
    // So log(Γ(x)) = 0 - log(x) = -log(x) when x < 1
    let needs_recurrence = x.clone().lower_elem(1.0);
    let recurrence_correction = x.clone().log().neg();

    // Use stirling result for x >= 1, recurrence result for x < 1
    

    needs_recurrence.clone().float() * recurrence_correction
        + (one.clone() - needs_recurrence.float()) * stirling_result
}

/// Computes the log probability density function of a Gamma distribution
///
/// For Gamma(α, θ) where α is shape and θ is scale:
/// log_pdf(x; α, θ) = -log(Γ(α)) - α*log(θ) + (α-1)*log(x) - x/θ
pub fn gamma_log_pdf<B: Backend>(
    x: Tensor<B, 1>,
    alpha: Tensor<B, 1>,
    theta: Tensor<B, 1>,
) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let device = x.device();
    let one: Tensor<B, 1> = Tensor::from_floats([1.0], &device);

    // Ensure all values are positive for numerical stability
    let x_safe = x.clamp_min(1e-8);
    let alpha_safe = alpha.clamp_min(1e-8);
    let theta_safe = theta.clamp_min(1e-8);

    // Compute log-gamma of alpha
    let log_gamma_alpha = log_gamma_approx(alpha_safe.clone());

    // Compute other terms
    let log_theta = theta_safe.clone().log();
    let log_x = x_safe.clone().log();

    // Combine terms: -log(Γ(α)) - α*log(θ) + (α-1)*log(x) - x/θ
    

    log_gamma_alpha.neg() - alpha_safe.clone() * log_theta
        + (alpha_safe - one) * log_x
        - x_safe / theta_safe
}

pub fn gamma_mean<B: Backend>(alpha: Tensor<B, 2>, theta: Tensor<B, 2>) -> Tensor<B, 2>
where
    B::FloatElem: From<f32>,
{
    alpha * theta
}

pub fn mixture_mean<B: Backend>(
    weights: Tensor<B, 2>,
    alphas: Tensor<B, 2>,
    thetas: Tensor<B, 2>,
) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let m = gamma_mean(alphas, thetas);
    (weights * m).sum_dim(1).squeeze(1)
}

/// Computes the log-likelihood of a mixture of two Gamma distributions
///
/// Uses the log-sum-exp trick for numerical stability:
/// log(w1*pdf1 + w2*pdf2) = max + log(exp(log(w1*pdf1) - max) + exp(log(w2*pdf2) - max))
pub fn gamma_mixture_log_likelihood<B: Backend>(
    x: Tensor<B, 1>,
    weights: Tensor<B, 2>, // [batch_size, 2]
    alphas: Tensor<B, 2>,  // [batch_size, 2]
    thetas: Tensor<B, 2>,  // [batch_size, 2]
) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let batch_size = weights.dims()[0];

    // Extract components
    let w1 = weights
        .clone()
        .slice([0..batch_size, 0..1])
        .flatten::<1>(0, 1);
    let w2 = weights
        .clone()
        .slice([0..batch_size, 1..2])
        .flatten::<1>(0, 1);
    let alpha1 = alphas
        .clone()
        .slice([0..batch_size, 0..1])
        .flatten::<1>(0, 1);
    let alpha2 = alphas
        .clone()
        .slice([0..batch_size, 1..2])
        .flatten::<1>(0, 1);
    let theta1 = thetas
        .clone()
        .slice([0..batch_size, 0..1])
        .flatten::<1>(0, 1);
    let theta2 = thetas
        .clone()
        .slice([0..batch_size, 1..2])
        .flatten::<1>(0, 1);

    // Ensure weights are positive and normalized
    let w1_safe = w1.clamp_min(1e-8);
    let w2_safe = w2.clamp_min(1e-8);
    let weight_sum = w1_safe.clone() + w2_safe.clone();
    let w1_norm = w1_safe / weight_sum.clone();
    let w2_norm = w2_safe / weight_sum;

    // Compute log PDFs for both components
    let log_pdf1 = gamma_log_pdf(x.clone(), alpha1, theta1);
    let log_pdf2 = gamma_log_pdf(x.clone(), alpha2, theta2);

    // Compute log(weight * pdf) for each component
    let log_w1_pdf1 = w1_norm.log() + log_pdf1;
    let log_w2_pdf2 = w2_norm.log() + log_pdf2;

    // Log-sum-exp trick for numerical stability
    // Use element-wise max between the two tensors
    let max_val = log_w1_pdf1.clone().max_pair(log_w2_pdf2.clone());
    let exp_sum = (log_w1_pdf1 - max_val.clone()).exp() + (log_w2_pdf2 - max_val.clone()).exp();
    

    max_val + exp_sum.log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_log_gamma_approx() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let result = log_gamma_approx(x);

        // Correct values:
        // log(Γ(1)) = log(0!) = log(1) = 0
        // log(Γ(2)) = log(1!) = log(1) = 0
        // log(Γ(3)) = log(2!) = log(2) ≈ 0.693
        // log(Γ(4)) = log(3!) = log(6) ≈ 1.792
        let expected_approx = [0.0, 0.0, 0.693, 1.792];
        let result_data_holder = result.to_data();
        let result_data = result_data_holder.as_slice::<f32>().unwrap();

        for (i, &expected) in expected_approx.iter().enumerate() {
            assert!(
                (result_data[i] - expected).abs() < 0.1,
                "log_gamma({}) = {}, expected ~{}",
                i + 1,
                result_data[i],
                expected
            );
        }
    }

    #[test]
    fn test_gamma_log_pdf() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0, 2.0], &device);
        let theta = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 1.0], &device);

        let result = gamma_log_pdf(x, alpha, theta);
        let result_data_holder = result.to_data();
        let result_data = result_data_holder.as_slice::<f32>().unwrap();

        // For Gamma(2, 1), the PDF is x * exp(-x)
        // So log_pdf(x) = log(x) + log(exp(-x)) = log(x) - x
        // But we also need to subtract log(Γ(α)) = log(Γ(2)) = log(1) = 0
        // So log_pdf(x) = -log(Γ(2)) - 2*log(1) + (2-1)*log(x) - x/1 = 0 - 0 + log(x) - x = log(x) - x
        let expected = [1.0_f32.ln() - 1.0, 2.0_f32.ln() - 2.0, 3.0_f32.ln() - 3.0];

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (result_data[i] - exp).abs() < 0.1,
                "gamma_log_pdf mismatch at {}: got {}, expected {}",
                i,
                result_data[i],
                exp
            );
        }
    }
}
