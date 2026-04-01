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

---

**Config change (2026-03-31):** Updated research model to 384 embed / 8 layers (from 256 embed / 12 layers). Added --gradnorm-interval=100, --checkpoint-interval=0. Baseline will be re-trained.

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 6 | [ERROR] ERROR: subagent cancelled | 0.000000 | — | ⚠️ |
| 6 | Nonlinear embedding + square positions + QK-Norm | 0.230754 (top1=0.0582, wdl=0.4961, aux=0.0361) | -0.4420 (top1=+0.0582, wdl=+0.4961, aux=+0.0361) | ❌ |
| 7 | [COMPILE FAIL] (no final message) | 0.000000 | — | ❌ |
| 7 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 8 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 9 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 10 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 11 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 12 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 13 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 14 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 15 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 16 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 17 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 18 | [ERROR] ERROR: openrouter API error (status=400 Bad Reques | 0.000000 | — | ⚠️ |
| 7 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 8 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 9 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 10 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 11 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 12 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 13 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 14 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 15 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 16 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 17 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 18 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 19 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 20 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 21 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 22 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 23 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 24 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 25 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 26 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 27 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 28 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 29 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 30 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 31 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 32 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 33 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 34 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 35 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 36 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 37 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 38 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 39 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 40 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 41 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 42 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 43 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 44 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 45 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 46 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 47 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 48 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 49 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 50 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 51 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 52 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 53 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 54 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 55 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
| 56 | [ERROR] ERROR: unknown tool id: "subagent.run" | 0.000000 | — | ⚠️ |
