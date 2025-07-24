use burn::prelude::*;
use burn_ndarray::NdArray;
use oxi::config::ModelConfig;
use oxi::model::OXIModel;

type TestBackend = NdArray<f32>;

// #[test]
// fn test_model_with_projections() {
//     let device = Default::default();
//     let config = ModelConfig::default();
//
//     // Create model
//     let model = OXIModel::<TestBackend>::new(&device, &config);
//
//     // Create dummy inputs
//     let batch_size = 2;
//     let board = Tensor::<TestBackend, 4>::zeros([batch_size, 16, 8, 8], &device);
//     let white_elo = Tensor::<TestBackend, 2, Int>::zeros([batch_size, 1], &device);
//     let black_elo = Tensor::<TestBackend, 2, Int>::zeros([batch_size, 1], &device);
//
//     // Forward pass
//     let (policy, value, side_info) = model.forward(board, white_elo, black_elo);
//
//     // Check output shapes
//     assert_eq!(policy.dims(), [batch_size, 4096]); // 64x64 from-to squares
//     assert_eq!(value.dims(), [batch_size, 3]);
//     assert_eq!(side_info.dims(), [batch_size, 141]);
//
//     println!("Model forward pass with projections successful!");
// }
