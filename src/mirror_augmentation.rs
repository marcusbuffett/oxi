/// Horizontal board mirroring (file reflection) for data augmentation.
///
/// Chess positions are symmetric with respect to horizontal reflection (swapping
/// files a↔h). This module provides batch-level mirroring applied during
/// training to effectively double data diversity at negligible compute cost.
use burn::prelude::*;
use burn::tensor::backend::Backend;
use once_cell::sync::Lazy;

use crate::config::LEGAL_MOVES;

/// Mirror a square index: rank stays, file reverses (a↔h, b↔g, c↔f, d↔e)
const fn mirror_sq(i: usize) -> usize {
    let rank = i / 8;
    let file = i % 8;
    rank * 8 + (7 - file)
}

/// Precomputed square mirror permutation [64]
static MIRROR_SQ: Lazy<[i32; 64]> = Lazy::new(|| {
    let mut arr = [0i32; 64];
    for i in 0..64 {
        arr[i] = mirror_sq(i) as i32;
    }
    arr
});

/// Precomputed move mirror permutation [4864]
static MIRROR_MOVE: Lazy<[i32; LEGAL_MOVES]> = Lazy::new(|| {
    let mut arr = [0i32; LEGAL_MOVES];
    for m in 0..LEGAL_MOVES {
        let from_sq = m / 76;
        let target_idx = m % 76;
        let new_from = mirror_sq(from_sq);
        let new_target = if target_idx < 64 {
            // Regular move: mirror target square
            mirror_sq(target_idx)
        } else if target_idx < 68 {
            // Promotion dir0 (capture left) → dir2 (capture right)
            target_idx + 8
        } else if target_idx < 72 {
            // Promotion dir1 (straight) stays
            target_idx
        } else {
            // Promotion dir2 (capture right) → dir0 (capture left)
            target_idx - 8
        };
        arr[m] = (new_from * 76 + new_target) as i32;
    }
    arr
});

/// Feature-level permutation for file one-hot reversal [65]
/// Most features map to themselves; only file one-hot (indices 51-58) is reversed.
static FEATURE_PERM: Lazy<[i32; 65]> = Lazy::new(|| {
    let mut arr = [0i32; 65];
    for i in 0..65 {
        arr[i] = i as i32;
    }
    // Reverse file one-hot: 51↔58, 52↔57, 53↔56, 54↔55
    arr[51] = 58;
    arr[52] = 57;
    arr[53] = 56;
    arr[54] = 55;
    arr[55] = 54;
    arr[56] = 53;
    arr[57] = 52;
    arr[58] = 51;
    arr
});

/// Feature index for dark-square flag within per-token features.
/// Positional group starts at offset 12 (piece) + 22 (tactical) = 34,
/// dark_square is the 9th positional feature (index 8 within group).
const DARK_SQUARE_FEATURE_IDX: usize = 42;

/// Apply horizontal file-mirroring to an entire training batch in-place.
///
/// Mirrored fields:
/// - board_input [B,64,65]: square permutation + file one-hot reversal + dark-square flip
/// - legal_moves [B,4864]: move index permutation
/// - move_distributions [B,4864]: move index permutation
/// - calibration_move_cp_losses [B,4864]: move index permutation
/// - side_info [B,141]: from-square (13..77) and to-square (77..141) permutation
///
/// Unchanged: values, global_features, value_weights, material_imbalance,
/// calibration_mask, calibration_target_cp_loss, calibration_target_bins,
/// time_usages, fens, items.
pub fn mirror_batch<B: Backend>(batch: crate::dataset::ChessBatch<B>) -> crate::dataset::ChessBatch<B>
where
    B::IntElem: From<i32>,
{
    let device = batch.board_input.device();
    let batch_size = batch.board_input.dims()[0];

    // Precompute index tensors on the batch device
    let sq_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::from(MIRROR_SQ.as_slice()),
        &device,
    );
    let move_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::from(MIRROR_MOVE.as_slice()),
        &device,
    );
    let feat_idx = Tensor::<B, 1, Int>::from_data(
        TensorData::from(FEATURE_PERM.as_slice()),
        &device,
    );

    // 1. Mirror board squares (permute dim 1)
    let mut board = batch.board_input.select(1, sq_idx.clone());

    // 2. Reverse file one-hot features (permute dim 2)
    board = board.select(2, feat_idx);

    // 3. Flip dark-square flag (index 42): binary 0↔1
    let dark = board.clone().slice([0..batch_size, 0..64, DARK_SQUARE_FEATURE_IDX..DARK_SQUARE_FEATURE_IDX + 1]);
    let dark_flipped = dark.neg().add_scalar(1.0f32);
    board = board.slice_assign(
        [0..batch_size, 0..64, DARK_SQUARE_FEATURE_IDX..DARK_SQUARE_FEATURE_IDX + 1],
        dark_flipped,
    );

    // 4. Mirror move-indexed tensors (permute dim 1)
    let legal_moves = batch.legal_moves.select(1, move_idx.clone());
    let move_distributions = batch.move_distributions.select(1, move_idx.clone());
    let calibration_move_cp_losses = batch.calibration_move_cp_losses.select(1, move_idx);

    // 5. Mirror from/to square one-hots in side_info [B, 141]
    let side_info = {
        let si = batch.side_info;
        // First 13 values: piece/check info — unchanged
        let prefix = si.clone().slice([0..batch_size, 0..13]);
        // From-square one-hot: indices 13..77 (64 values)
        let from_sq = si.clone().slice([0..batch_size, 13..77]);
        let from_sq_mirrored = from_sq.select(1, sq_idx.clone());
        // To-square one-hot: indices 77..141 (64 values)
        let to_sq = si.slice([0..batch_size, 77..141]);
        let to_sq_mirrored = to_sq.select(1, sq_idx);
        // Reassemble
        Tensor::cat(vec![prefix, from_sq_mirrored, to_sq_mirrored], 1)
    };

    crate::dataset::ChessBatch {
        board_input: board,
        move_distributions,
        legal_moves,
        side_info,
        calibration_move_cp_losses,
        // Everything else unchanged
        time_usages: batch.time_usages,
        values: batch.values,
        fens: batch.fens,
        global_features: batch.global_features,
        value_weights: batch.value_weights,
        material_imbalance: batch.material_imbalance,
        calibration_mask: batch.calibration_mask,
        calibration_target_cp_loss: batch.calibration_target_cp_loss,
        calibration_target_bins: batch.calibration_target_bins,
        items: batch.items,
    }
}
