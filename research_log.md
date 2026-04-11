# OXI Research Log

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 0 | Value tower mean-pool skip connection | 1.046771 (top1=0.4493, wdl=0.6844, aux=0.5108, cal=0.3828) | +0.0463 (top1=+0.0185, wdl=+0.0134, aux=+0.0127, cal=+0.0464) | ✅ |
| 1 | Lower LR plateau threshold 0.8% to 0.5% | 1.082571 (top1=0.4608, wdl=0.7096, aux=0.5240, cal=0.4055) | +0.0358 (top1=+0.0115, wdl=+0.0252, aux=+0.0131, cal=+0.0227) | ✅ |
| 2 | Fix GradNorm bootstrap stuck on missing calibration probes | 1.059239 (top1=0.4560, wdl=0.6911, aux=0.5163, cal=0.3861) | -0.0233 (top1=-0.0048, wdl=-0.0185, aux=-0.0077, cal=-0.0194) | ❌ |
| 3 | Raise GradNorm weight floor to prevent aux/cal starvation | 1.075259 (top1=0.4523, wdl=0.6925, aux=0.5299, cal=0.4268) | -0.0073 (top1=-0.0085, wdl=-0.0171, aux=+0.0059, cal=+0.0213) | ❌ |
| 4 | Disable broken GradNorm to eliminate probe overhead | 1.055493 (top1=0.4420, wdl=0.6305, aux=0.5318, cal=0.4797) | -0.0271 (top1=-0.0188, wdl=-0.0791, aux=+0.0079, cal=+0.0742) | ❌ |
| 5 | Lower LR plateau threshold from 0.5% to 0.3% | 1.076957 (top1=0.4582, wdl=0.7004, aux=0.5176, cal=0.4126) | -0.0056 (top1=-0.0026, wdl=-0.0092, aux=-0.0064, cal=+0.0071) | ❌ |
| 6 | Increase policy label smoothing 0.005→0.1 | 1.041291 (top1=0.4486, wdl=0.6923, aux=0.5171, cal=0.3579) | -0.0413 (top1=-0.0122, wdl=-0.0173, aux=-0.0069, cal=-0.0476) | ❌ |
| 7 | Boost policy-implied CPL loss 50x | 0.919296 (top1=0.2812, wdl=0.6009, aux=0.4199, cal=0.6342) | -0.1633 (top1=-0.1796, wdl=-0.1087, aux=-0.1041, cal=+0.2287) | ❌ |
| 8 | FiLM-conditioned policy head | 1.113876 (top1=0.4551, wdl=0.7048, aux=0.5175, cal=0.5072) | +0.0313 (top1=-0.0057, wdl=-0.0048, aux=-0.0065, cal=+0.1017) | ✅ |
| 9 | Elo-Conditional Temperature Scaling | 1.119919 (top1=0.4598, wdl=0.6921, aux=0.5269, cal=0.5218) | +0.0060 (top1=+0.0047, wdl=-0.0128, aux=+0.0094, cal=+0.0147) | ❌ |
| 10 | Metric-Aligned Calibration Loss | 1.166766 (top1=0.4725, wdl=0.7070, aux=0.5394, cal=0.5821) | +0.0529 (top1=+0.0174, wdl=+0.0022, aux=+0.0219, cal=+0.0749) | ✅ |
| 11 | Non-Linear Factorized Policy Head | 1.148648 (top1=0.4587, wdl=0.6865, aux=0.5323, cal=0.6006) | -0.0181 (top1=-0.0139, wdl=-0.0205, aux=-0.0071, cal=+0.0186) | ❌ |
| 12 | Mid-trunk deep supervision for from/to square heads | 1.170902 (top1=0.4784, wdl=0.6954, aux=0.5444, cal=0.5897) | +0.0041 (top1=+0.0059, wdl=-0.0117, aux=+0.0050, cal=+0.0076) | ❌ |
| 13 | Residual spatial conv for local board context | 1.151362 (top1=0.4633, wdl=0.7144, aux=0.5286, cal=0.5630) | -0.0154 (top1=-0.0093, wdl=+0.0073, aux=-0.0108, cal=-0.0191) | ❌ |
| 14 | Wider shallower architecture 224d×10L×7H | 1.155768 (top1=0.4614, wdl=0.7126, aux=0.5279, cal=0.5813) | -0.0110 (top1=-0.0112, wdl=+0.0056, aux=-0.0115, cal=-0.0008) | ❌ |
| 15 | Stochastic depth regularization for trunk transformer | 1.157168 (top1=0.4737, wdl=0.6922, aux=0.5332, cal=0.5767) | -0.0096 (top1=+0.0012, wdl=-0.0148, aux=-0.0062, cal=-0.0054) | ❌ |
| 16 | Restore value head mean-pool skip connection | 1.173534 (top1=0.4770, wdl=0.6964, aux=0.5463, cal=0.5977) | +0.0068 (top1=+0.0044, wdl=-0.0106, aux=+0.0069, cal=+0.0156) | ❌ |
| 17 | Higher LR + stronger weight decay training regime | 1.183133 (top1=0.4875, wdl=0.6830, aux=0.5516, cal=0.6096) | +0.0164 (top1=+0.0149, wdl=-0.0240, aux=+0.0122, cal=+0.0275) | ✅ |
| 18 | Rebalance calibration loss toward policy pathway | 1.167889 (top1=0.4698, wdl=0.6918, aux=0.5332, cal=0.6139) | -0.0152 (top1=-0.0177, wdl=+0.0088, aux=-0.0184, cal=+0.0043) | ❌ |
| 19 | Focal loss gamma=2.0 for policy head | 1.162769 (top1=0.4710, wdl=0.6925, aux=0.5407, cal=0.5934) | -0.0204 (top1=-0.0165, wdl=+0.0095, aux=-0.0110, cal=-0.0161) | ❌ |
| 20 | Higher LR 1.8× and stronger weight decay 0.01 | 1.180364 (top1=0.4788, wdl=0.6963, aux=0.5457, cal=0.6107) | -0.0028 (top1=-0.0087, wdl=+0.0133, aux=-0.0060, cal=+0.0011) | ❌ |
| 21 | AdamW beta_2 from 0.999 to 0.95 | 1.163385 (top1=0.4685, wdl=0.6978, aux=0.5299, cal=0.5999) | -0.0197 (top1=-0.0189, wdl=+0.0148, aux=-0.0217, cal=-0.0097) | ❌ |
| 22 | Dual-pathway calibration head with policy tokens | 1.168374 (top1=0.4733, wdl=0.6965, aux=0.5412, cal=0.5965) | -0.0148 (top1=-0.0142, wdl=+0.0136, aux=-0.0104, cal=-0.0131) | ❌ |
| 23 | Deeper policy head with second transformer block | 1.156665 (top1=0.4682, wdl=0.6918, aux=0.5349, cal=0.5891) | -0.0265 (top1=-0.0193, wdl=+0.0088, aux=-0.0167, cal=-0.0205) | ❌ |
| 24 | Learnable 2D relative position bias in attention | 0.000000 (top1=0.0000, wdl=0.0000, aux=0.0000, cal=0.0000) | -1.1831 (top1=-0.4875, wdl=-0.6830, aux=-0.5516, cal=-0.6096) | ❌ |
| 25 | [ERROR x5] openrouter API error (status=403 Forbidden): {"err | 0.000000 | — | ⚠️ |
| 2 | [CRASH] Learnable per-head position bias in attention | 0.000000 | — | ❌ |
| 3 | Two-layer nonlinear embedding with unified features | 1.174399 (top1=0.4874, wdl=0.6584, aux=0.5432, cal=0.6229) | +0.0165 (top1=-0.0031, wdl=+0.0341, aux=+0.0048, cal=+0.0039) | ✅ |
| 4 | Fix value head: apply ply-ramp weights + restore skip | 1.163703 (top1=0.4971, wdl=0.6191, aux=0.5473, cal=0.6189) | -0.0107 (top1=+0.0098, wdl=-0.0393, aux=+0.0040, cal=-0.0041) | ❌ |
| 5 | Widen Smolgen attention bias generator bottleneck | 1.171145 (top1=0.4924, wdl=0.6396, aux=0.5474, cal=0.6237) | -0.0033 (top1=+0.0050, wdl=-0.0188, aux=+0.0042, cal=+0.0008) | ❌ |
| 6 | Route calibration head through value-block features | 1.165859 (top1=0.4962, wdl=0.6262, aux=0.5378, cal=0.6225) | -0.0085 (top1=+0.0088, wdl=-0.0322, aux=-0.0054, cal=-0.0004) | ❌ |
| 7 | FiLM-conditioned embedding for Elo-aware input tokens | 1.164077 (top1=0.5001, wdl=0.6225, aux=0.5496, cal=0.6071) | -0.0103 (top1=+0.0127, wdl=-0.0359, aux=+0.0064, cal=-0.0158) | ❌ |
| 8 | Increase trunk depth from 8 to 10 layers | 1.171285 (top1=0.4947, wdl=0.6404, aux=0.5444, cal=0.6188) | -0.0031 (top1=+0.0073, wdl=-0.0180, aux=+0.0012, cal=-0.0041) | ❌ |
| 9 | Compute-neutral wider-shallower architecture 224d×6L×7H | 1.168989 (top1=0.4950, wdl=0.6380, aux=0.5396, cal=0.6176) | -0.0054 (top1=+0.0076, wdl=-0.0204, aux=-0.0036, cal=-0.0054) | ❌ |
| 10 | Enable 5% puzzle data mixing for tactical transfer | 1.181212 (top1=0.4975, wdl=0.6565, aux=0.5576, cal=0.6099) | +0.0068 (top1=+0.0101, wdl=-0.0019, aux=+0.0144, cal=-0.0131) | ❌ |
| 11 | FiLM-conditioned value head for Elo-aware WDL | 1.180152 (top1=0.5003, wdl=0.6481, aux=0.5361, cal=0.6214) | +0.0058 (top1=+0.0129, wdl=-0.0103, aux=-0.0071, cal=-0.0015) | ❌ |
| 12 | Sqrt-scaled warmup for 76% more full-LR training | 1.180067 (top1=0.5012, wdl=0.6337, aux=0.5556, cal=0.6272) | +0.0057 (top1=+0.0138, wdl=-0.0247, aux=+0.0124, cal=+0.0043) | ❌ |
| 13 | Increase attention heads from 6 to 8 | 1.169300 (top1=0.4891, wdl=0.6462, aux=0.5385, cal=0.6235) | -0.0051 (top1=+0.0017, wdl=-0.0122, aux=-0.0048, cal=+0.0006) | ❌ |
| 14 | Concatenated trunk skip connection for value head | 0.931806 (top1=0.3621, wdl=0.5935, aux=0.4566, cal=0.4541) | -0.2426 (top1=-0.1253, wdl=-0.0649, aux=-0.0866, cal=-0.1688) | ❌ |
| 15 | Rebalance GradNorm priorities for value and calibration | 1.135539 (top1=0.4421, wdl=0.6802, aux=0.4994, cal=0.6337) | -0.0389 (top1=-0.0453, wdl=+0.0218, aux=-0.0438, cal=+0.0108) | ❌ |
| 16 | Fix embedding optimizer misconfig from rename | 1.128454 (top1=0.4472, wdl=0.6915, aux=0.5059, cal=0.5858) | -0.0459 (top1=-0.0402, wdl=+0.0331, aux=-0.0374, cal=-0.0372) | ❌ |
| 17 | Grouped Query Attention with 2 KV heads | 0.537733 (top1=0.2238, wdl=0.5061, aux=0.1455, cal=0.0796) | -0.6367 (top1=-0.2636, wdl=-0.1523, aux=-0.3978, cal=-0.5433) | ❌ |
| 18 | Squeeze-and-Excitation channel gating in MLP blocks | 0.907889 (top1=0.3518, wdl=0.5997, aux=0.4435, cal=0.4189) | -0.2665 (top1=-0.1356, wdl=-0.0587, aux=-0.0997, cal=-0.2040) | ❌ |
