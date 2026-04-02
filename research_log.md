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
| 16 | Attention soft-capping + mid-trunk deep supervision | 0.678972 (top1=0.3903, wdl=0.5719, aux=0.4901) | -0.0213 (top1=-0.0168, wdl=-0.0081, aux=-0.0091) | ❌ |

---

**Architecture reset**: embed_dim 256→192, added learnable square embeddings, full-rank policy factorization (rank=embed_dim), 2 derived globals (material balance, game phase), removed mid-trunk deep supervision. Scores below are not directly comparable to above.

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 0 | New baseline (192 embed, square embeds, full-rank policy, derived globals) | 0.728148 (top1=0.4211, wdl=0.6166, aux=0.5075) | — | baseline |
| 1 | 3x3 spatial conv stem (1 layer) | 0.698223 (top1=0.4009, wdl=0.6067, aux=0.4755) | -0.0299 (top1=-0.0202, wdl=-0.0099, aux=-0.0321) | ❌ |
| 2 | Reduce attention heads 8→6 for larger head_dim | 0.737124 (top1=0.4326, wdl=0.6050, aux=0.5143) | +0.0090 (top1=+0.0115, wdl=-0.0116, aux=+0.0068) | ❌ |
| 3 | 6 attention heads + QK-Norm for stable training | 0.757333 (top1=0.4496, wdl=0.6036, aux=0.5327) | +0.0292 (top1=+0.0285, wdl=-0.0129, aux=+0.0252) | ✅ |
| 4 | Wider MLP (8/3x) + nonlinear token embedding | 0.741547 (top1=0.4299, wdl=0.6233, aux=0.5194) | -0.0158 (top1=-0.0197, wdl=+0.0196, aux=-0.0133) | ❌ |

---

**Metric reset**: WDL metric changed from argmax accuracy (0-100%) to mean probability of correct class (0-1). WDL weight in composite changed from 1/3 to 0.5. Composite formula is now: `1.0*top1 + 0.5*wdl + 0.2*aux`. Scores below are not directly comparable to above.

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 0 | New baseline (metric reset: WDL mean-prob, weight 0.5) | 0.822116 (top1=0.4468, wdl=0.5391, aux=0.5285) | — | baseline |
| 1 | GradNorm value priority + aggressive LR annealing | 0.844156 (top1=0.4411, wdl=0.5967, aux=0.5237) | +0.0220 (top1=-0.0057, wdl=+0.0576, aux=-0.0048) | ✅ |
| 2 | Smoother LR decay + from/to loss upweighting | 0.851148 (top1=0.4423, wdl=0.6067, aux=0.5278) | +0.0070 (top1=+0.0012, wdl=+0.0100, aux=+0.0041) | ❌ |
| 3 | Fix value head gradient starvation via loss weight | 0.864733 (top1=0.4468, wdl=0.6237, aux=0.5306) | +0.0206 (top1=+0.0057, wdl=+0.0270, aux=+0.0069) | ✅ |
| 4 | Apply unused value ply-ramp weights to value loss | 0.846371 (top1=0.4457, wdl=0.5898, aux=0.5286) | -0.0184 (top1=-0.0010, wdl=-0.0338, aux=-0.0020) | ❌ |
| 5 | Head residual scaling fix + higher LR | 0.852104 (top1=0.4421, wdl=0.6111, aux=0.5226) | -0.0126 (top1=-0.0047, wdl=-0.0126, aux=-0.0080) | ❌ |
| 6 | Wider aux heads + deeper value MLP + lower decay | 0.856595 (top1=0.4440, wdl=0.6135, aux=0.5293) | -0.0081 (top1=-0.0028, wdl=-0.0102, aux=-0.0013) | ❌ |
| 7 | OK, I've analyzed the codebase thoroughly. Let me now finalize my approach. | 0.863447 (top1=0.4498, wdl=0.6151, aux=0.5302) | -0.0013 (top1=+0.0031, wdl=-0.0085, aux=-0.0005) | ❌ |
| 8 | Gentler LR decay + head-specific residual init | 0.879279 (top1=0.4558, wdl=0.6310, aux=0.5398) | +0.0145 (top1=+0.0091, wdl=+0.0073, aux=+0.0092) | ✅ |
