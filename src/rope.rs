use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::s;
use burn::tensor::{backend::Backend, Device, Int, Tensor};
use rand::rng;
use rand::Rng as _;

use crate::config::get_global_config;
// Debug helper removed

/// 2D RoPE multi-head attention for 8x8 chess boards.
///
/// Splits the head channel into two halves and applies rotary embeddings
/// using the row index for the first half and the column index for the second half.
#[derive(Module, Debug)]
pub struct RopeRelativePositionAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rel_bias: Option<Param<Tensor<B, 2>>>,
    rel_bias_scale: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> RopeRelativePositionAttention<B> {
    pub fn new(device: &Device<B>, use_rel_bias: bool) -> Self {
        let config = get_global_config();
        let embed_dim = config.embed_dim();
        let num_heads = config.num_heads();
        let head_dim = config.head_dim();
        let kv_dim = embed_dim;

        let q_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let k_proj = LinearConfig::new(embed_dim, kv_dim).init(device);
        let v_proj = LinearConfig::new(embed_dim, kv_dim).init(device);
        let o_proj = LinearConfig::new(embed_dim, embed_dim).init(device);

        let (rel_bias, rel_bias_scale) = if use_rel_bias && !config.disable_shaw_pr {
            let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
            let rel_bias = Param::from_tensor(Tensor::<B, 2>::random(
                burn::tensor::Shape::new([64, 64]),
                distribution,
                device,
            ));
            let rel_bias_scale = Param::from_tensor(Tensor::from_floats([-3.0f32], device));
            (Some(rel_bias), Some(rel_bias_scale))
        } else {
            (None, None)
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rel_bias,
            rel_bias_scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let config = get_global_config();
        let [batch_size, seq_len, embed_dim] = x.dims();
        assert_eq!(
            seq_len, 64,
            "2D RoPE attention expects seq_len=64 (8x8 board)"
        );
        assert_eq!(embed_dim, config.embed_dim());

        let num_heads = config.num_heads();
        let head_dim = config.head_dim();
        assert!(
            head_dim % 4 == 0,
            "head_dim must be divisible by 4 for 2D RoPE"
        );

        // Projections
        let q = self.q_proj.forward(x.clone()); // [B, S, E]
        let k = self.k_proj.forward(x.clone()); // [B, S, E] now
        let v = self.v_proj.forward(x); // [B, S, E]

        // Reshape to [B, H, S, D]
        let q = q
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([batch_size, seq_len, num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        // Apply 2D RoPE to Q and K (unless disabled)
        let (q_rot, k_rot) = if config.disable_rope() {
            (q.clone(), k.clone())
        } else {
            (apply_rope_2d::<B>(q.clone()), apply_rope_2d::<B>(k.clone()))
        };

        // Scaled dot-product attention without relative bias
        let scale = (head_dim as f32).sqrt();
        let mut attn_scores = q_rot.matmul(k_rot.swap_dims(2, 3)).div_scalar(scale); // [B, H, S, S]

        if let Some(rel_bias) = &self.rel_bias {
            if let Some(rel_bias_scale) = &self.rel_bias_scale {
                let bias = rel_bias.val().reshape([1, 1, seq_len, seq_len]);
                {
                    let mut rng = rng();
                    if rng.random::<f32>() < 0.1 {
                        tracing::info!("Relative bias scale: {}", rel_bias_scale.val());
                    }
                }
                let gate = sigmoid(rel_bias_scale.val()).reshape([1, 1, 1, 1]);
                attn_scores = attn_scores + bias * gate;
            }
        }
        let attn_weights = softmax(attn_scores, 3);
        let context = attn_weights.matmul(v); // [B, H, S, D]

        // Merge heads
        let context = context
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, embed_dim]);

        // Output projection
        self.o_proj.forward(context)
    }
}

fn apply_rope_2d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // x: [B, H, S=64, D]
    let [_, _, seq_len, d] = x.dims();
    assert_eq!(seq_len, 64, "apply_rope_2d expects seq_len=64");
    assert!(d % 2 == 0, "Last dim must be even");

    let half = d / 2;
    assert!(
        half % 2 == 0,
        "Half dim must be even to form (even, odd) pairs"
    );

    let device = x.device();
    let x_row = x.clone().slice(s![.., .., .., 0..half]);
    let x_col = x.slice(s![.., .., .., half..d]);

    // Positions 0..63 (compute using float to avoid CUDA floor overload on int)
    let pos_f = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).float(); // [S]
    let eight = Tensor::<B, 1>::from_floats([8.0f32], &device).reshape([1]);
    let rows = pos_f.clone().div(eight.clone()).floor().int();
    let cols = pos_f.sub(rows.clone().float().mul(eight)).int();

    let x_row = apply_rotary_on_half::<B>(x_row, rows);
    let x_col = apply_rotary_on_half::<B>(x_col, cols);

    Tensor::cat(vec![x_row, x_col], 3)
}

fn apply_rotary_on_half<B: Backend>(
    x_half: Tensor<B, 4>,
    pos_index: Tensor<B, 1, Int>,
) -> Tensor<B, 4> {
    // x_half: [B, H, S, Dh], Dh even
    let [batch, heads, seq_len, dh] = x_half.dims();
    assert!(dh % 2 == 0, "Rotary half requires even dimension");
    let d_pair = dh / 2;

    // Build inverse frequencies as on RoPE: 10000^{-2k/Dh}
    let device = x_half.device();
    let inv_freq: Vec<f32> = (0..d_pair)
        .map(|k| 1.0f32 / 10000f32.powf(2.0 * (k as f32) / (dh as f32)))
        .collect();
    let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), &device).reshape([d_pair]); // [d_pair]

    // Build cos/sin with broadcastable shapes
    let pos_f = pos_index.float().reshape([1, 1, seq_len, 1]); // [1,1,S,1]
    let inv_f = inv_freq.reshape([1, 1, 1, d_pair]); // [1,1,1,d_pair]
    let angles = pos_f.mul(inv_f); // [1,1,S,d_pair]
    let cos = angles.clone().cos().reshape([1, 1, seq_len, d_pair, 1]); // [1,1,S,d_pair,1]
    let sin = angles.sin().reshape([1, 1, seq_len, d_pair, 1]); // [1,1,S,d_pair,1]

    // Reshape x_half -> [B, H, S, d_pair, 2]
    let x5 = x_half.reshape([batch, heads, seq_len, d_pair, 2]);
    let idx_even = Tensor::<B, 1, Int>::from_ints([0], &device);
    let idx_odd = Tensor::<B, 1, Int>::from_ints([1], &device);
    let x_even = x5.clone().select(4, idx_even); // [B,H,S,d_pair,1]
    let x_odd = x5.select(4, idx_odd); // [B,H,S,d_pair,1]

    // Apply rotation with broadcasting over the trailing singleton dim
    let x_even_new = x_even
        .clone()
        .mul(cos.clone())
        .sub(x_odd.clone().mul(sin.clone())); // [B,H,S,d_pair,1]
    let x_odd_new = x_odd.mul(cos).add(x_even.mul(sin)); // [B,H,S,d_pair,1]

    // Concatenate back on the last dimension and reshape to [B,H,S,Dh]
    let paired = Tensor::cat(vec![x_even_new, x_odd_new], 4); // [B,H,S,d_pair,2]
    let x_rot = paired.reshape([batch, heads, seq_len, dh]);
    x_rot
}
