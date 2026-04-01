# OXI Research Log

| Iter | Description | Score | Kept |
|------|-------------|-------|------|
| 0 | Baseline (no changes) | 0.590766 (top1=0.3817, wdl=0.5819, aux=0.0760) | baseline |
| 0 | Fix from/to square aux loss: BCE→cross-entropy | 0.634516 (top1=0.3770, wdl=0.5918, aux=0.3013) | ✅ |
| 1 | Enhanced aux heads: hidden layer + policy tokens | 0.659628 (top1=0.3910, wdl=0.5879, aux=0.3632) | ✅ |
| 2 | Deeper policy tower + FiLM + wider FFN | 0.654562 (top1=0.3839, wdl=0.6079, aux=0.3402) | ❌ |
| 3 | Trunk aux supervision + policy FiLM conditioning | 0.672744 (top1=0.3870, wdl=0.5971, aux=0.4334) | ✅ |
| 4 | Config validation (no change) | 0.672132 (top1=0.3889, wdl=0.5910, aux=0.4313) | ❌ |
