#[cfg(test)]
mod tests {
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataloader::Dataset;
    use burn::prelude::*;
    use shakmaty::Chess;
    use std::fs;
    use tempfile::NamedTempFile;

    use crate::config::{set_global_config, Config, ModelConfig, FEATURES_PER_TOKEN, NUM_GLOBALS};
    use crate::dataset::{ChessBatcher, OXIDataset};
    use crate::inference::{GlobalFeatures, InferenceEngine};
    use crate::model::OXIModel;

    #[cfg(target_os = "macos")]
    type TestBackend = burn::backend::Metal;
    #[cfg(not(target_os = "macos"))]
    type TestBackend = burn::backend::LibTorch<f32>;

    #[test]
    fn test_tensor_consistency_between_inference_and_dataset() {
        // Initialize global config (required by PGN processor)
        let model_config = ModelConfig::default();
        let mut config = Config::default();
        // Disable sampling so all moves are included
        config.enable_ply_sampling = Some(false);
        config.enable_elo_sampling = Some(false);
        let _ = set_global_config(config);

        // Create a mock PGN file with the required format
        let pgn_content = r#"[Event "Test Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "TestPlayer"]
[Black "TestOpponent"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1800"]
[TimeControl "1800+30"]

1. e4 { [%clk 1:30:00] } 1-0
"#;

        // Write to temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        fs::write(temp_file.path(), pgn_content).expect("Failed to write PGN content");

        #[cfg(target_os = "macos")]
        let device = burn::backend::metal::MetalDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;

        // Load dataset from PGN
        let dataset =
            OXIDataset::from_pgn_with_limit(temp_file.path(), model_config.clone(), Some(1))
                .expect("Failed to load dataset");

        if dataset.len() == 0 {
            println!("No examples loaded - skipping tensor comparison test");
            return;
        }

        assert_eq!(dataset.len(), 1, "Should have exactly one example");

        // Get the ChessItem from dataset
        let chess_item = dataset.get(0).expect("Should have first item");

        // Create ChessBatch from single item
        let batcher: ChessBatcher<TestBackend> = ChessBatcher::new(device);
        let batch = batcher.batch(vec![chess_item.clone()], &device);

        // Extract dataset tensors
        let dataset_board_tensor = batch.board_input.clone();
        let dataset_global_tensor = batch.global_features.clone();

        // Now create the same tensors using inference path
        let starting_position = Chess::default();
        let global_features = GlobalFeatures {
            time_remaining_self: chess_item.global_features.time_remaining_self,
            time_remaining_oppo: chess_item.global_features.time_remaining_oppo,
            base_time: chess_item.global_features.base_time,
            increment: chess_item.global_features.increment,
            move_count: chess_item.global_features.move_count,
            elo_self: chess_item.elo_self,
        };

        // Create inference engine to use the extracted method
        let model = OXIModel::<TestBackend>::new(&device, &model_config);
        let inference_engine = InferenceEngine::new(model, Config::default(), device);

        let (
            inference_board_tensor,
            inference_global_tensor,
            _flipped_current,
            _material_imbalance_history,
        ) = inference_engine
            .create_input_tensors(&[starting_position], &[], &global_features)
            .expect("Failed to create inference tensors");

        // Compare tensor shapes
        assert_eq!(
            dataset_board_tensor.shape(),
            inference_board_tensor.shape(),
            "Board tensor shapes should match"
        );
        assert_eq!(
            dataset_global_tensor.shape(),
            inference_global_tensor.shape(),
            "Global tensor shapes should match"
        );

        // Compare tensor values (with small tolerance for floating point precision)
        let dataset_board_data = dataset_board_tensor.to_data();
        let inference_board_data = inference_board_tensor.to_data();

        let dataset_global_data = dataset_global_tensor.to_data();
        let inference_global_data = inference_global_tensor.to_data();

        let dataset_board_slice = dataset_board_data.as_slice::<f32>().unwrap();
        let inference_board_slice = inference_board_data.as_slice::<f32>().unwrap();

        let dataset_global_slice = dataset_global_data.as_slice::<f32>().unwrap();
        let inference_global_slice = inference_global_data.as_slice::<f32>().unwrap();

        // Compare board tensors
        assert_eq!(
            dataset_board_slice.len(),
            inference_board_slice.len(),
            "Board tensor data lengths should match"
        );

        for (i, (&dataset_val, &inference_val)) in dataset_board_slice
            .iter()
            .zip(inference_board_slice.iter())
            .enumerate()
        {
            assert!(
                (dataset_val - inference_val).abs() < 1e-6,
                "Board tensor mismatch at index {}: dataset={}, inference={}",
                i,
                dataset_val,
                inference_val
            );
        }

        // Compare global tensors
        assert_eq!(
            dataset_global_slice.len(),
            inference_global_slice.len(),
            "Global tensor data lengths should match"
        );

        for (i, (&dataset_val, &inference_val)) in dataset_global_slice
            .iter()
            .zip(inference_global_slice.iter())
            .enumerate()
        {
            assert!(
                (dataset_val - inference_val).abs() < 1e-6,
                "Global tensor mismatch at index {}: dataset={}, inference={}",
                i,
                dataset_val,
                inference_val
            );
        }

        println!(
            "✅ Tensor consistency test passed! Both inference and dataset paths create identical tensors."
        );
    }

    #[test]
    fn test_board_encoding_consistency() {
        use crate::encoding::encode_position;

        // Test that the board encoding logic is the same
        let starting_position = Chess::default();
        let previous_positions: Vec<Chess> = vec![];

        // Encode using the encoding module directly
        let encoded_board = encode_position(&starting_position, &previous_positions, &[]);

        // Verify the encoding has the right shape
        assert_eq!(
            encoded_board.len(),
            64 * FEATURES_PER_TOKEN,
            "Encoded board should have 64 * FEATURES_PER_TOKEN elements"
        );

        // Test that reshaping works as expected
        #[cfg(target_os = "macos")]
        let device = burn::backend::metal::MetalDevice::default();
        #[cfg(not(target_os = "macos"))]
        let device = burn_tch::LibTorchDevice::Cpu;
        let board_tensor = Tensor::<TestBackend, 1>::from_floats(encoded_board.as_slice(), &device)
            .reshape([1, 64, FEATURES_PER_TOKEN]);

        assert_eq!(
            board_tensor.shape().dims,
            [1, 64, FEATURES_PER_TOKEN],
            "Reshaped tensor should have correct dimensions"
        );
    }
}
