use burn::prelude::*;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use oxi::model::OXIModel;
use oxi::config::ModelConfig;
use oxi::dataset::ChessBatch;
use oxi::encoding::get_elo_bin;

type Backend = Autodiff<NdArray>;

#[test]
fn test_uncertainty_weighting() {
    let device = Default::default();
    let config = ModelConfig::default();
    let model: OXIModel<Backend> = OXIModel::new(&device, &config);
    
    // Check initial uncertainty values (should be 1.0 since log(1) = 0)
    let (sigma_policy, sigma_value, sigma_side_info) = model.get_uncertainties();
    assert!((sigma_policy - 1.0).abs() < 1e-6, "Initial policy sigma should be 1.0, got {}", sigma_policy);
    assert!((sigma_value - 1.0).abs() < 1e-6, "Initial value sigma should be 1.0, got {}", sigma_value);
    assert!((sigma_side_info - 1.0).abs() < 1e-6, "Initial side info sigma should be 1.0, got {}", sigma_side_info);
    
    // Create a dummy batch to test forward pass
    let batch_size = 2;
    let board_input = Tensor::zeros([batch_size, 112, 8, 8], &device);
    
    // Get Elo bins
    let elo_bins = config.elo_bins();
    let elo_bin_1500 = get_elo_bin(1500, &elo_bins) as i32;
    let elo_bin_1600 = get_elo_bin(1600, &elo_bins) as i32;
    
    let elo_self: Tensor<Backend, 2, Int> = Tensor::from_data([[elo_bin_1500], [elo_bin_1600]], &device);
    let elo_oppo: Tensor<Backend, 2, Int> = Tensor::from_data([[elo_bin_1500], [elo_bin_1600]], &device);
    let move_distributions = Tensor::zeros([batch_size, 1968], &device);
    let values = Tensor::zeros([batch_size, 3], &device);
    let side_info: Tensor<Backend, 2, Int> = Tensor::zeros([batch_size, 141], &device);
    let legal_moves = Tensor::ones([batch_size, 1968], &device);
    let fens = vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(); batch_size];
    
    let batch = ChessBatch {
        board_input,
        elo_self,
        elo_oppo,
        move_distributions,
        values,
        side_info,
        legal_moves,
        fens,
    };
    
    // Run forward pass
    let output = model.forward_classification(batch);
    
    // Check that uncertainties are included in output
    assert!(output.uncertainties.is_some(), "Output should include uncertainties");
    let (out_sigma_policy, out_sigma_value, out_sigma_side_info) = output.uncertainties.unwrap();
    assert!((out_sigma_policy - 1.0).abs() < 1e-6, "Output policy sigma should be 1.0");
    assert!((out_sigma_value - 1.0).abs() < 1e-6, "Output value sigma should be 1.0");
    assert!((out_sigma_side_info - 1.0).abs() < 1e-6, "Output side info sigma should be 1.0");
    
    println!("Uncertainty-based loss weighting test passed!");
}