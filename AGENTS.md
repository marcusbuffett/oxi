# Agent Instructions for Oxi

## Backend: always tch (LibTorch)

**Use the `tch` (LibTorch) backend for everything — it is the only backend that
works reliably.** It is now the default feature (`backend-tch`), and it covers
every device we run on:
- **CPU / Apple MPS** locally (macOS uses MPS automatically).
- **CUDA** on Linux — install a CUDA-enabled libtorch (the PyTorch CUDA wheels
  from `setup_ubuntu.sh` provide it) and tch uses the GPU automatically. No
  separate CUDA toolkit / `backend-cuda` build is needed.

Do **not** use `backend-cuda`, `backend-wgpu`, `backend-metal`, `backend-candle`,
or `backend-ndarray`. Re-tested 2026-08-08 on an H100: `backend-cuda` and
`backend-candle-cuda` now *build* (burn tracks main), but burn-cuda crashes at
runtime in CubeCL's matmul autotuner on our shapes, and candle-cuda trains
with broken numerics (loss pinned at 10.0, top1 ~random). tch remains the
backend.
Plain `cargo build` / `cargo run` now pulls in `backend-tch`; for training add
`--features train` (so: `cargo run --release --features train -- train ...`).

**Always train with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`.** PyTorch defaults
matmul TF32 to OFF, which leaves H100/A100 tensor cores idle for FP32 matmuls;
the override is a measured ~1.3x end-to-end iteration speedup on the 768/8
model (710ms -> ~530ms per batch-512 iteration) with standard TF32 training
numerics. tch 0.22 exposes no API for this — the env var is the mechanism.

## Local Compilation Check

Before deploying, verify the code compiles with the training features:

```bash
cargo check --features train
```

This catches errors that won't show up otherwise (the default build is
inference-only — no `train`/autodiff).

## Remote Access (GH200 GPU Server)

### SSH Connection
```bash
# Basic SSH
ssh ubuntu@$(mise exec -- printenv REMOTE_IP)

# Run command on remote
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cd /home/ubuntu/oxi && <command>"'

# Examples:
# Watch training logs
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cd /home/ubuntu/oxi && tail -f train.log"'

# Check GPU status
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'nvidia-smi'

# Grep specific logs
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cd /home/ubuntu/oxi && grep perf_full_iteration train.log | tail -20"'
```

### Deploy Code
```bash
./deploy.sh
```

### Common Log Patterns
```bash
# Performance timing
grep perf_full_iteration train.log | tail -20
grep perf_optimizer_breakdown train.log | tail -20
grep perf_timing train.log | tail -20

# Gradient breakdown (requires --log-gradient-breakdown flag)
grep "gradient_debug: step=" train.log           # Total gradient norm per step
grep "gradient_debug: layer=" train.log          # Per-layer breakdown

# Tensor norms (requires --log-tensor-norms flag)
grep "norm_debug:.*snapshot" train.log           # Snapshot markers
grep "norm_debug:.*embed.gates" train.log        # Embedding gate values

# L2 penalty (always logged at full_metrics_interval)
grep "l2_penalty" train.log

# GradNorm status (requires --enable-gradnorm flag)
grep "GradNorm" train.log
grep "loss_weights" train.log

# Training metrics
grep "perf_metrics_breakdown" train.log | tail -20
```

### Training Flags for Diagnostics
```bash
# Gradient breakdown and tensor norms are DISABLED by default
# When enabled, they log every --full-metrics-interval iterations

# Enable gradient breakdown logging
--log-gradient-breakdown

# Enable tensor norm logging
--log-tensor-norms

# Enable GradNorm adaptive loss weighting
--enable-gradnorm

# Control logging frequency
--full-metrics-interval=N    # Controls debug_monitor, gradient breakdown, tensor norms (0 = never)
--gradnorm-interval=N        # GradNorm probe frequency (default: 20)
```

### Run Training on Remote
```bash
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cd /home/ubuntu/oxi && cargo run --release --features train -- train \
  --physical-batch-size=2048 \
  --num-layers=8 \
  --embed-dim=256 \
  --disable-tui"'
```

## Performance Investigation

See `docs/perf-investigation.md` for detailed timing breakdown and optimization history.
