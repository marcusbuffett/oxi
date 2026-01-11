# Agent Instructions for Oxi

## Local Compilation Check

Before deploying, verify the code compiles with the training features:

```bash
cargo check --features "train,backend-tch"
```

This catches errors that won't show up with default features (inference-only mode).

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
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cd /home/ubuntu/oxi && CUDA_PATH=/usr cargo run --release --no-default-features --features backend-cuda -- train \
  --physical-batch-size=2048 \
  --num-layers=8 \
  --embed-dim=256 \
  --disable-tui"'
```

## Performance Investigation

See `docs/perf-investigation.md` for detailed timing breakdown and optimization history.
