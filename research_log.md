# OXI Research Log

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 0 | Baseline (no changes) | 0.590766 (top1=0.3817, wdl=0.5819, aux=0.0760) | — | baseline |
| 0 | Fix from/to square aux loss: BCE→cross-entropy | 0.634516 (top1=0.3770, wdl=0.5918, aux=0.3013) | +0.0438 (top1=-0.0047, wdl=+0.0099, aux=+0.2253) | ✅ |
| 1 | Enhanced aux heads: hidden layer + policy tokens | 0.659628 (top1=0.3910, wdl=0.5879, aux=0.3632) | +0.0251 (top1=+0.0140, wdl=-0.0039, aux=+0.0619) | ✅ |
| 2 | Deeper policy tower + FiLM + wider FFN | 0.654562 (top1=0.3839, wdl=0.6079, aux=0.3402) | -0.0051 (top1=-0.0071, wdl=+0.0200, aux=-0.0230) | ❌ |
| 3 | Trunk aux supervision + policy FiLM conditioning | 0.672744 (top1=0.3870, wdl=0.5971, aux=0.4334) | +0.0131 (top1=-0.0040, wdl=+0.0092, aux=+0.0702) | ✅ |
| 4 | Config validation (no change) | 0.672132 (top1=0.3889, wdl=0.5910, aux=0.4313) | -0.0006 (top1=+0.0019, wdl=-0.0061, aux=-0.0021) | ❌ |
| 5 | Deeper policy tower + value FiLM + value weight boost | 0.664727 (top1=0.3796, wdl=0.5995, aux=0.4264) | ❌ |
| 7 | Residual MLP embedding + value FiLM | 0.182084 (top1=0.0459, wdl=0.4000, aux=0.0142) | -0.4907 (top1=+0.0459, wdl=+0.4000, aux=+0.0142) | ❌ |
| 12 | LR plateau annealing + value FiLM conditioning | 0.700292 (top1=0.4071, wdl=0.5800, aux=0.4992) | +0.0275 (top1=+0.4071, wdl=+0.5800, aux=+0.4992) | ✅ |
| 13 | Policy from/to square bias injection | 0.685887 (top1=0.3982, wdl=0.5670, aux=0.4935) | -0.0144 (top1=-0.0089, wdl=-0.0130, aux=-0.0057) | ❌ |
| 14 | Stochastic depth + full-rank policy head | 0.699780 (top1=0.4055, wdl=0.5840, aux=0.4983) | -0.0005 (top1=-0.0017, wdl=+0.0040, aux=-0.0009) | ❌ |
| 15 | Higher LR + faster annealing + lower weight decay | 0.697790 (top1=0.4026, wdl=0.5885, aux=0.4953) | -0.0025 (top1=-0.0046, wdl=+0.0085, aux=-0.0038) | ❌ |
