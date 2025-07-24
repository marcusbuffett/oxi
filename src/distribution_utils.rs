use burn::tensor::{backend::Backend, Tensor};

/// Approximates log(Γ(x)) using a Stirling-style series with a recurrence fallback for x < 1.
pub fn log_gamma_approx<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let device = x.device();

    let half: Tensor<B, 1> = Tensor::from_floats([0.5], &device);
    let one: Tensor<B, 1> = Tensor::from_floats([1.0], &device);
    let two_pi: Tensor<B, 1> = Tensor::from_floats([2.0 * std::f32::consts::PI], &device);
    let twelve: Tensor<B, 1> = Tensor::from_floats([12.0], &device);
    let three_sixty: Tensor<B, 1> = Tensor::from_floats([360.0], &device);

    let x_for_stirling = x.clone().clamp_min(1.0);

    let log_x_stirling = x_for_stirling.clone().log();
    let stirling_main = (x_for_stirling.clone() - half.clone()) * log_x_stirling.clone()
        - x_for_stirling.clone()
        + half.clone() * two_pi.log();

    let correction1 = one.clone() / (twelve.clone() * x_for_stirling.clone());
    let x_cubed = x_for_stirling.clone().powf_scalar(3.0);
    let correction2 = one.clone() / (three_sixty.clone() * x_cubed);

    let stirling_result = stirling_main + correction1 - correction2;

    let needs_recurrence = x.clone().lower_elem(1.0);
    let recurrence_correction = x.clone().log().neg();

    needs_recurrence.clone().float() * recurrence_correction
        + (one.clone() - needs_recurrence.float()) * stirling_result
}

fn log_beta_fn<B: Backend>(alpha: Tensor<B, 1>, beta: Tensor<B, 1>) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    log_gamma_approx(alpha.clone()) + log_gamma_approx(beta.clone())
        - log_gamma_approx(alpha + beta)
}

/// Computes log pdf of Beta(α, β) over x ∈ (0,1).
pub fn beta_log_pdf<B: Backend>(
    x: Tensor<B, 1>,
    alpha: Tensor<B, 1>,
    beta: Tensor<B, 1>,
) -> Tensor<B, 1>
where
    B::FloatElem: From<f32>,
{
    let device = x.device();
    let one: Tensor<B, 1> = Tensor::from_floats([1.0], &device);
    let eps = 1e-6f32;

    let x_safe = x.clamp(eps, 1.0 - eps);
    let alpha_safe = alpha.clamp_min(eps);
    let beta_safe = beta.clamp_min(eps);

    let log_term_alpha = (alpha_safe.clone() - one.clone()) * x_safe.clone().log();
    let log_term_beta = (beta_safe.clone() - one.clone()) * (one.clone() - x_safe.clone()).log();
    let log_norm = log_beta_fn(alpha_safe, beta_safe);

    log_term_alpha + log_term_beta - log_norm
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
    fn test_beta_log_pdf_matches_known_values() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_floats([0.2, 0.5, 0.8], &device);
        let alpha = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0, 2.0], &device);
        let beta = Tensor::<TestBackend, 1>::from_floats([3.0, 3.0, 3.0], &device);

        let result = beta_log_pdf(x, alpha, beta);
        let result_data_holder = result.to_data();
        let result_data = result_data_holder.as_slice::<f32>().unwrap();

        // Beta(2,3) pdf = 12 * x^(1) * (1-x)^(2)
        let expected = [
            (12.0 * (0.2f32.powf(1.0)) * (1.0 - 0.2f32).powf(2.0)).ln(),
            (12.0 * (0.5f32.powf(1.0)) * (1.0 - 0.5f32).powf(2.0)).ln(),
            (12.0 * (0.8f32.powf(1.0)) * (1.0 - 0.8f32).powf(2.0)).ln(),
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (result_data[i] - exp).abs() < 1e-3,
                "beta_log_pdf mismatch at {}: got {}, expected {}",
                i,
                result_data[i],
                exp
            );
        }
    }
}
