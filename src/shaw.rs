use burn::module::{Module, Param};
use burn::nn::{self, Embedding, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, Device, Int, Tensor};

use crate::config::get_global_config;
use crate::norm_debug::{log_attention_heatmap, log_tensor_stats};

/// Shaw et al. (2018) relative position representations for 8x8 chess boards (seq_len=64).
///
/// Implements the key-only asymmetric variant: attention scores use q_i · (k_j + a^K_ij) and
/// values add a relative term via sum_j α_ij · a^V_ij. Learnable vectors a_ij^K, a_ij^V ∈ R^{head_dim}
/// are shared across heads and broadcast across the head dimension. They are indexed by the 2D relative
/// offset (dr, dc) between squares i and j. We use dr, dc in [-7, 7] mapped to a bucket in [0, 224]
/// via (dr + 7) * 15 + (dc + 7).
#[derive(Module, Debug)]
pub struct ShawRelativePositionAttention<B: Backend> {
    // Separate projections so K/V can have fewer heads (grouped-query attention)
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    // Head-shared relative embeddings (dimension = head_dim), broadcast across heads
    // Key-only variant uses a^K for attention and a^V for value context
    a_qk: Embedding<B>,
    a_v: Embedding<B>,
    // Learnable scale factors for relative position contributions
    rel_scale_qk: Param<Tensor<B, 1>>,
    rel_scale_v: Param<Tensor<B, 1>>,
}

impl<B: Backend> ShawRelativePositionAttention<B> {
    pub fn new(device: &Device<B>) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();
        let head_dim = config.head_dim();
        let kv_dim = config.kv_dim();

        // Separate projections to support grouped-query attention (fewer KV heads than Q heads)
        let q_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let k_proj = LinearConfig::new(embed_dim, kv_dim).init(device);
        let v_proj = LinearConfig::new(embed_dim, kv_dim).init(device);
        let o_proj = LinearConfig::new(embed_dim, embed_dim).init(device);

        // 2D relative offsets: (dr, dc) with dr, dc in [-7, 7] → 15 * 15 buckets
        let num_rel = 15 * 15;
        // To get RMS of rel_scores to be > attn_scores_base
        // Use head_dim embeddings and broadcast across heads during the forward pass
        let a_qk = Embedding {
            weight: nn::Initializer::Normal {
                std: 1.0,
                mean: 0.0,
            }
            .init_with([num_rel, head_dim], Some(num_rel), Some(head_dim), device),
        };
        let a_v = Embedding {
            weight: nn::Initializer::Normal {
                std: 1.0,
                mean: 0.0,
            }
            .init_with([num_rel, head_dim], Some(num_rel), Some(head_dim), device),
        };

        // Initialize learnable scale parameters to 0.0 (log space, exp(0) = 1.0)
        let rel_scale_qk = Param::from_tensor(Tensor::zeros([1], device));
        let rel_scale_v = Param::from_tensor(Tensor::zeros([1], device));

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            a_qk,
            a_v,
            rel_scale_qk,
            rel_scale_v,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let config = get_global_config();
        let [batch_size, seq_len, embed_dim] = x.dims();
        debug_assert_eq!(seq_len, 64, "Shaw attention expects seq_len=64 (8x8 board)");
        debug_assert_eq!(embed_dim, config.embed_dim());
        debug_assert!(batch_size > 0, "Batch size must be positive");
        debug_assert!(embed_dim > 0, "Embed dim must be positive");
        log_tensor_stats("attention.input", &x);

        let num_heads = config.num_heads();
        let head_dim = config.head_dim();
        debug_assert_eq!(
            embed_dim,
            num_heads * head_dim,
            "embed_dim must equal num_heads * head_dim"
        );

        // Separate projections for Q, K, V (supports grouped-query attention)
        let q_proj = self.q_proj.forward(x.clone());
        log_tensor_stats("attention.q_linear", &q_proj);
        let k_proj = self.k_proj.forward(x.clone());
        log_tensor_stats("attention.k_linear", &k_proj);
        let v_proj = self.v_proj.forward(x);
        log_tensor_stats("attention.v_linear", &v_proj);

        // debug_assert projection shapes
        debug_assert_eq!(
            q_proj.dims(),
            [batch_size, seq_len, embed_dim],
            "Q projection shape mismatch"
        );
        debug_assert_eq!(
            k_proj.dims(),
            [batch_size, seq_len, embed_dim],
            "K projection shape mismatch"
        );
        debug_assert_eq!(
            v_proj.dims(),
            [batch_size, seq_len, embed_dim],
            "V projection shape mismatch"
        );

        // Reshape to [B, H, S, D]
        let num_kv_heads = config.num_kv_heads();
        let group_size = config.gqa_group_size();
        let q = q_proj
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .permute([0, 2, 1, 3]);
        log_tensor_stats("attention.q_heads", &q);
        let mut k = k_proj
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .permute([0, 2, 1, 3]);
        log_tensor_stats("attention.k_heads", &k);
        let mut v = v_proj
            .reshape([batch_size, seq_len, num_kv_heads, head_dim])
            .permute([0, 2, 1, 3]);
        log_tensor_stats("attention.v_heads", &v);

        if group_size > 1 {
            let mut k_expanded = Vec::with_capacity(group_size);
            let mut v_expanded = Vec::with_capacity(group_size);
            for _ in 0..group_size {
                k_expanded.push(k.clone());
                v_expanded.push(v.clone());
            }
            k = Tensor::cat(k_expanded, 1);
            v = Tensor::cat(v_expanded, 1);
            log_tensor_stats("attention.k_expanded", &k);
            log_tensor_stats("attention.v_expanded", &v);
        }

        // debug_assert multi-head shapes post expansion
        debug_assert_eq!(
            q.dims(),
            [batch_size, num_heads, seq_len, head_dim],
            "Q multi-head reshape failed"
        );
        debug_assert_eq!(
            k.dims(),
            [batch_size, num_heads, seq_len, head_dim],
            "K multi-head reshape failed"
        );
        debug_assert_eq!(
            v.dims(),
            [batch_size, num_heads, seq_len, head_dim],
            "V multi-head reshape failed"
        );

        // Scale queries once so both absolute and relative terms are scaled
        let scale = (head_dim as f32).sqrt();
        let q_scaled = q.clone().div_scalar(scale);
        log_tensor_stats("attention.q_scaled", &q_scaled);

        // Base dot-product attention using scaled queries
        let mut attn_scores = q_scaled.clone().matmul(k.clone().swap_dims(2, 3)); // [B, H, S, S]
        debug_assert_eq!(
            attn_scores.dims(),
            [batch_size, num_heads, seq_len, seq_len],
            "Base attention scores shape mismatch"
        );
        log_tensor_stats("attention.attn_scores_base", &attn_scores);

        // Relative indices [S, S] mapping (i,j) -> bucket in [0, 224]
        // Build relative index every forward (no caching)
        let rel_index = build_rel_index::<B>(seq_len, &q.device()); // [S, S] (Int)
                                                                    // NOTE: Hoisting this tensor outside the forward pass is non-trivial because it
                                                                    // must live on the backend-specific device (including autodiff variants) and would
                                                                    // require plumbing lifetimes through Module init; keep the per-forward rebuild for now.
        debug_assert_eq!(
            rel_index.dims(),
            [seq_len, seq_len],
            "Relative index shape mismatch"
        );

        // Relative embeddings per (i,j): [S,S,head_dim]
        let a_qk_embedded = self.a_qk.forward(rel_index.clone()); // [S,S,D]
        let a_v_embedded = self.a_v.forward(rel_index); // [S,S,D]
        debug_assert_eq!(a_qk_embedded.dims(), [seq_len, seq_len, head_dim]);
        debug_assert_eq!(a_v_embedded.dims(), [seq_len, seq_len, head_dim]);

        // Vectorized computation of q · a^K_ij using batched matmul (use scaled q)
        // q_batched: [B,H,S,1,D]
        let q_batched = q_scaled
            .clone()
            .reshape([batch_size, num_heads, seq_len, 1, head_dim]);
        // a_k_batched: [1,1,S,D,S] from [S,S,D]
        let a_k_batched = a_qk_embedded
            .clone()
            .swap_dims(1, 2) // [S,D,S]
            .reshape([1, 1, seq_len, head_dim, seq_len]);
        // rel_scores: [B,H,S,1,S] -> reshape to [B,H,S,S]
        // Use exp(log_scale) to get the actual scale value
        let scale_qk = self.rel_scale_qk.val().exp();
        log_tensor_stats("scale_qk", &scale_qk);
        let rel_scores = q_batched
            .matmul(a_k_batched)
            .reshape([batch_size, num_heads, seq_len, seq_len])
            .mul(scale_qk.unsqueeze());
        debug_assert_eq!(rel_scores.dims(), [batch_size, num_heads, seq_len, seq_len]);
        log_tensor_stats("attention.rel_scores", &rel_scores);
        attn_scores = attn_scores + rel_scores; // [B,H,S,S] (already scaled)
        log_tensor_stats("attention.attn_scores_combined", &attn_scores);

        // Attention weights
        let attn_weights = softmax(attn_scores, 3); // [B,H,S,S]
        log_tensor_stats("attention.attn_weights", &attn_weights);
        let attn_mean = attn_weights.clone().mean_dim(1).squeeze_dim(1); // [B,S,S]
        log_attention_heatmap("attention.attn_heatmap_mean", &attn_mean, 0, None);

        // Standard context from values
        let context_v = attn_weights.clone().matmul(v.clone()); // [B,H,S,D]
        debug_assert_eq!(context_v.dims(), [batch_size, num_heads, seq_len, head_dim]);
        log_tensor_stats("attention.context_v", &context_v);

        // Relative value context: vectorized sum_j α_ij · a^V_ij via batched matmul
        // attn_batched: [B,H,S,1,S]
        let attn_batched = attn_weights
            .clone()
            .reshape([batch_size, num_heads, seq_len, 1, seq_len]);
        // a_v_batched: [1,1,S,S,D]
        let a_v_batched = a_v_embedded
            .clone()
            .reshape([1, 1, seq_len, seq_len, head_dim]);
        // context_rel_v: [B,H,S,1,D] -> [B,H,S,D]
        // Use exp(log_scale) to get the actual scale value
        let scale_v = self.rel_scale_v.val().exp();
        log_tensor_stats("scale_v", &scale_v);
        let context_rel_v = attn_batched
            .matmul(a_v_batched)
            .reshape([batch_size, num_heads, seq_len, head_dim])
            .mul(scale_v.unsqueeze());
        debug_assert_eq!(
            context_rel_v.dims(),
            [batch_size, num_heads, seq_len, head_dim]
        );
        log_tensor_stats("attention.context_rel_v", &context_rel_v);

        let context = context_v + context_rel_v; // [B,H,S,D]
        log_tensor_stats("attention.context", &context);

        // Merge heads and project out
        let context_merged = context
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, embed_dim]);
        log_tensor_stats("attention.context_merged", &context_merged);

        let output = self.o_proj.forward(context_merged);
        log_tensor_stats("attention.output", &output);
        return output;
    }

    // Removed row-wise builder in favor of fully vectorized computation in forward
}

fn build_rel_index<B: Backend>(seq_len: usize, device: &Device<B>) -> Tensor<B, 2, Int> {
    debug_assert_eq!(seq_len, 64, "This implementation assumes seq_len=64 (8x8)");
    let mut indices: Vec<i32> = Vec::with_capacity(seq_len * seq_len);
    #[cfg(debug_assertions)]
    let mut unique_buckets = std::collections::HashSet::new();

    for i in 0..seq_len {
        let ri = (i / 8) as i32;
        let ci = (i % 8) as i32;
        for j in 0..seq_len {
            let rj = (j / 8) as i32;
            let cj = (j % 8) as i32;
            let dr = (rj - ri).clamp(-7, 7);
            let dc = (cj - ci).clamp(-7, 7);
            let bucket = (dr + 7) * 15 + (dc + 7);
            debug_assert!(
                bucket >= 0 && bucket < 225,
                "Bucket index out of range: {}",
                bucket
            );
            indices.push(bucket);
            #[cfg(debug_assertions)]
            unique_buckets.insert(bucket);
        }
    }

    // Verify we have the expected number of unique relative position buckets
    #[cfg(debug_assertions)]
    debug_assert_eq!(
        unique_buckets.len(),
        15 * 15,
        "Expected 225 unique relative position buckets, got {}",
        unique_buckets.len()
    );

    Tensor::<B, 1, Int>::from_ints(indices.as_slice(), device).reshape([seq_len, seq_len])
}
