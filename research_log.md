# OXI Research Log

| Iter | Description | Score | Δ vs prev best | Kept |
|------|-------------|-------|----------------|------|
| 0 | Baseline (no changes) | 0.000000 (top1=0.0000, wdl=0.0000, aux=0.0000) | — | baseline |
| 0 | [ERROR x5] Bridge request failed: <urlopen error [Errno 61] C | 0.000000 | — | ⚠️ |
| 0 | Disable value-tower-only stage, use LR warm restarts | 0.896838 (top1=0.4609, wdl=0.6588, aux=0.5327) | +0.0095 (top1=-0.0127, wdl=+0.0509, aux=-0.0163) | ❌ |
| 1 | Fix value tower gradient filter name mismatch | 0.888361 (top1=0.4658, wdl=0.6294, aux=0.5393) | +0.0010 (top1=-0.0078, wdl=+0.0215, aux=-0.0097) | ❌ |
| 2 | Reduce transformer depth from 12 to 8 layers | 0.871531 (top1=0.4609, wdl=0.6055, aux=0.5395) | -0.0158 (top1=-0.0127, wdl=-0.0024, aux=-0.0095) | ❌ |
| 3 | Increase LR plateau window size 120→300 | 0.907809 (top1=0.4823, wdl=0.6290, aux=0.5552) | +0.0204 (top1=+0.0087, wdl=+0.0211, aux=+0.0062) | ✅ |
| 4 | Lower LR plateau threshold from 1.5% to 0.8% | 0.935650 (top1=0.4929, wdl=0.6630, aux=0.5565) | +0.0278 (top1=+0.0106, wdl=+0.0340, aux=+0.0013) | ✅ |
| 5 | Increase LR plateau window 300→500 to prevent cascading | 0.928010 (top1=0.4686, wdl=0.7021, aux=0.5417) | -0.0076 (top1=-0.0242, wdl=+0.0391, aux=-0.0148) | ❌ |
| 6 | Softer LR reduction factor 0.7→0.8 | 0.934295 (top1=0.4829, wdl=0.6820, aux=0.5518) | -0.0014 (top1=-0.0099, wdl=+0.0190, aux=-0.0046) | ❌ |

---

**New baseline (2026-04-03):** Warmup was 0.1x (51 steps) — likely causing run-to-run instability due to insufficient warmup with Muon. Bumped to 1.0x (512 steps). Also includes value head attention pooling sqrt scaling fix from run_7. Previous best was run_4 (0.9357).
| 7 | Scale value head attention pooling to fix dead layers | 0.930480 (top1=0.4702, wdl=0.7038, aux=0.5418) | -0.0052 (top1=-0.0227, wdl=+0.0408, aux=-0.0147) | ❌ |
| 8 | [ERROR x5] Bridge request failed: <urlopen error [Errno 61] C | 0.000000 | — | ⚠️ |
| 0 | Fix GradNorm clamp-then-renormalize ordering bug | 0.945956 (top1=0.4935, wdl=0.6823, aux=0.5563) | +0.0301 (top1=+0.0322, wdl=-0.0121, aux=+0.0195) | ✅ |
| 1 | Scale factorized policy logits by 1/sqrt(embed_dim) | 0.929811 (top1=0.4842, wdl=0.6710, aux=0.5506) | -0.0161 (top1=-0.0094, wdl=-0.0113, aux=-0.0056) | ❌ |
| 2 | Composite plateau loss prevents premature LR reduction | 0.940459 (top1=0.4860, wdl=0.6881, aux=0.5518) | -0.0055 (top1=-0.0075, wdl=+0.0058, aux=-0.0045) | ❌ |
| 3 | Connect value_weights to value loss computation | 0.935877 (top1=0.4879, wdl=0.6759, aux=0.5501) | -0.0101 (top1=-0.0056, wdl=-0.0064, aux=-0.0062) | ❌ |
| 4 | Increase attention heads from 6 to 8 | 0.870675 (top1=0.4296, wdl=0.6784, aux=0.5093) | -0.0753 (top1=-0.0639, wdl=-0.0039, aux=-0.0470) | ❌ |
| 5 | [ERROR x5] Bridge request failed: <urlopen error [Errno 61] C | 0.000000 | — | ⚠️ |

---

**Research Loop Reset (2026-04-07):**
The scoring regime has changed and previous baseline scores are no longer directly comparable.

What changed:
- Training now includes Stockfish-backed centipawn-loss calibration on labeled human positions.
- The model has a new CPL auxiliary head, but the policy itself is also trained against policy-implied expected CPL.
- New calibration metrics are logged, including:
  `cp_loss_calibration_overall`
  `cp_loss_labeled_fraction`
  `cp_loss_calibration_beginner`
  `cp_loss_calibration_intermediate`
  `cp_loss_calibration_expert`
- `research_loop.py` now includes calibration in its composite score, with coverage gating based on labeled fraction so sparse calibration batches do not dominate.

State reset performed:
- removed `research_state.json`
- removed `research_runs/baseline`

The next research-loop baseline should be treated as the start of a new series under the calibration-aware objective.
