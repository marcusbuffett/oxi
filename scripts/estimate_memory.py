#!/usr/bin/env python3
"""
Compare memory usage: OLD architecture vs NEW architecture.
More realistic estimation - don't double count.
"""


def format_bytes(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b} B"


def estimate_old_architecture(
    batch_size=1200,
    embed_dim=320,
    num_layers=24,
    num_heads=8,
    mlp_ratio=4.0,
    seq_len=64,
    num_globals=7,
    dtype_bytes=4,
):
    """OLD architecture: GELU MLP, simple LayerNorm, Linear policy head"""
    hidden_dim = int(embed_dim * mlp_ratio)

    print("=" * 60)
    print("OLD ARCHITECTURE (what worked with batch 1200)")
    print("=" * 60)
    print(f"  embed_dim: {embed_dim}, num_layers: {num_layers}")
    print(f"  mlp_ratio: {mlp_ratio}, hidden_dim: {hidden_dim}")
    print(f"  batch_size: {batch_size}")

    # Parameters per block (OLD)
    block_params = (
        4 * embed_dim * embed_dim  # attn qkvo
        + 2 * embed_dim * 2  # 2 LayerNorms (gamma + beta each)
        + embed_dim * hidden_dim  # fc1
        + hidden_dim * embed_dim  # fc2
    )

    total_blocks = num_layers + 3
    total_params = total_blocks * block_params
    param_memory = total_params * dtype_bytes

    # Activations per block (OLD - GELU MLP)
    # Only count what needs to be stored for backward
    block_acts = (
        batch_size * seq_len * embed_dim  # block input (for residual gradient)
        + 3 * batch_size * seq_len * embed_dim  # Q, K, V (for attn backward)
        + batch_size
        * num_heads
        * seq_len
        * seq_len  # attention weights (softmax output)
        + batch_size * seq_len * hidden_dim  # fc1 output (for gelu backward)
        + batch_size * seq_len * embed_dim  # post-attn (for mlp residual)
    )

    total_acts = total_blocks * block_acts
    act_memory = total_acts * dtype_bytes

    # Policy head (OLD): simple Linear(embed_dim, 76) - negligible

    # Optimizer (AdamW: m + v per param)
    optimizer_memory = 2 * param_memory

    # Total = params + activations + optimizer
    # (gradients are computed on-the-fly, not all stored simultaneously)
    total = param_memory + act_memory + optimizer_memory

    print(f"\n  Parameters: {total_params:,} ({format_bytes(param_memory)})")
    print(
        f"  Per-block params: {block_params:,} ({format_bytes(block_params * dtype_bytes)})"
    )
    print(
        f"  Per-block activations: {block_acts:,} ({format_bytes(block_acts * dtype_bytes)})"
    )
    print(f"  Total activations: {format_bytes(act_memory)}")
    print(f"  Optimizer state: {format_bytes(optimizer_memory)}")
    print(f"  TOTAL: {format_bytes(total)}")

    return total, block_acts * dtype_bytes, act_memory


def estimate_new_architecture(
    batch_size=64,
    embed_dim=384,
    num_layers=24,
    num_heads=8,
    mlp_ratio=2.5,
    seq_len=64,
    num_globals=7,
    policy_rank=64,
    dtype_bytes=4,
):
    """NEW architecture: SwiGLU MLP, FiLM LayerNorm, Factorized policy head"""
    hidden_dim = int(embed_dim * mlp_ratio)

    print("\n" + "=" * 60)
    print("NEW ARCHITECTURE (current changes)")
    print("=" * 60)
    print(f"  embed_dim: {embed_dim}, num_layers: {num_layers}")
    print(f"  mlp_ratio: {mlp_ratio}, hidden_dim: {hidden_dim}")
    print(f"  batch_size: {batch_size}")

    # Parameters per block (NEW - with FiLM)
    block_params = (
        4 * embed_dim * embed_dim  # attn qkvo
        +
        # FiLM LayerNorms: 2 per block, each has LN(2*D) + gamma_proj(G->D) + beta_proj(G->D)
        2 * (embed_dim * 2 + 2 * num_globals * embed_dim)
        + embed_dim * (2 * hidden_dim)  # fused gate+up
        + hidden_dim * embed_dim  # down_proj
    )

    total_blocks = num_layers + 3
    total_params = total_blocks * block_params
    param_memory = total_params * dtype_bytes

    # Activations per block (NEW - SwiGLU + FiLM)
    # SwiGLU needs to store: fused output (for backward through gate/up split)
    # and the activated = silu(gate) * up (for down_proj backward)
    block_acts = (
        batch_size * seq_len * embed_dim  # block input
        + 3 * batch_size * seq_len * embed_dim  # Q, K, V
        + batch_size * num_heads * seq_len * seq_len  # attention weights
        + batch_size * seq_len * embed_dim  # post-attn (for mlp residual)
        +
        # SwiGLU: need fused output for gate/up gradient
        batch_size * seq_len * (2 * hidden_dim)  # fused = [gate, up]
        +
        # Need gate input for silu backward, and silu(gate) for multiply backward
        batch_size * seq_len * hidden_dim  # gate (or silu(gate))
        + batch_size * seq_len * hidden_dim  # up (for multiply backward)
    )

    total_acts = total_blocks * block_acts
    act_memory = total_acts * dtype_bytes

    # Factorized policy head activations
    policy_acts = (
        batch_size * seq_len * policy_rank  # source proj
        + batch_size * seq_len * policy_rank  # target proj
        + batch_size * seq_len * seq_len  # base logits (64x64 matmul result)
    )
    act_memory += policy_acts * dtype_bytes

    optimizer_memory = 2 * param_memory

    total = param_memory + act_memory + optimizer_memory

    print(f"\n  Parameters: {total_params:,} ({format_bytes(param_memory)})")
    print(
        f"  Per-block params: {block_params:,} ({format_bytes(block_params * dtype_bytes)})"
    )
    print(
        f"  Per-block activations: {block_acts:,} ({format_bytes(block_acts * dtype_bytes)})"
    )
    print(f"  Total activations: {format_bytes(act_memory)}")
    print(f"  Optimizer state: {format_bytes(optimizer_memory)}")
    print(f"  TOTAL: {format_bytes(total)}")

    return total, block_acts * dtype_bytes, act_memory


if __name__ == "__main__":
    old_total, old_per_block, old_acts = estimate_old_architecture(
        batch_size=1200, embed_dim=320, num_layers=24, mlp_ratio=4.0
    )

    new_total, new_per_block, new_acts = estimate_new_architecture(
        batch_size=64, embed_dim=384, num_layers=24, mlp_ratio=2.5
    )

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\nOLD (batch=1200, dim=320, mlp=4.0): {format_bytes(old_total)}")
    print(f"  Activations: {format_bytes(old_acts)}")

    print(f"\nNEW (batch=64, dim=384, mlp=2.5): {format_bytes(new_total)}")
    print(f"  Activations: {format_bytes(new_acts)}")

    # Normalize by batch size
    print("\n" + "=" * 60)
    print("PER-SAMPLE COMPARISON")
    print("=" * 60)

    old_per_sample = old_acts / 1200
    new_per_sample = new_acts / 64

    print(f"\nOLD per sample activations: {format_bytes(old_per_sample)}")
    print(f"NEW per sample activations: {format_bytes(new_per_sample)}")
    print(f"Ratio (NEW/OLD): {new_per_sample / old_per_sample:.2f}x")

    # What batch size would NEW support with same memory as OLD?
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    # If old used X GB for batch 1200, new can use same for batch Y
    # new_acts_per_sample * Y = old_acts
    max_new_batch = int(old_acts / new_per_sample)
    print(f"\nIf OLD could do batch=1200, NEW could do batch~{max_new_batch}")

    # Let's verify
    print("\nVerifying with NEW at equivalent batch...")
    estimate_new_architecture(
        batch_size=max_new_batch, embed_dim=384, num_layers=24, mlp_ratio=2.5
    )

    # Key insight: what's actually different?
    print("\n" + "=" * 60)
    print("WHAT CHANGED?")
    print("=" * 60)

    old_hidden = int(320 * 4.0)  # 1280
    new_hidden = int(384 * 2.5)  # 960

    # Old MLP stores: fc1_out (hidden_dim)
    # New SwiGLU stores: fused (2*hidden_dim) + gate (hidden_dim) + up (hidden_dim)

    old_mlp_acts = old_hidden  # just fc1 output for gelu backward
    new_mlp_acts = 2 * new_hidden + new_hidden + new_hidden  # fused + gate + up

    print(f"\nMLP activations per token:")
    print(f"  OLD: {old_mlp_acts} (fc1 output)")
    print(f"  NEW: {new_mlp_acts} (fused + gate + up)")
    print(f"  Ratio: {new_mlp_acts / old_mlp_acts:.2f}x")

    # This is the problem! SwiGLU stores 4x as much per token
    print(
        f"\n>>> SwiGLU stores {new_mlp_acts / old_mlp_acts:.1f}x more MLP activations!"
    )
