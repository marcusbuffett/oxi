# Mini Oxi Model

The mini model is a low-latency Oxi preset intended for learnability/autofill
embedding and policy fetches.

## Preset

Use:

```bash
cargo run --release --features "train,backend-tch" -- train --model-size mini
```

The preset is:

- `embed_dim = 128`
- `num_layers = 3`
- `num_heads = 4`
- `smolgen_hidden = 12`
- `smolgen_global_dim = 80`
- `smolgen_gen_size = 80`
- policy-only first pass: value, time-usage, auxiliary, calibration, and
  policy-regret losses default to `0`
- post-training `whitening.json` enabled

This targets well under one tenth of the full model's parameter count. The first
mini preset was `128d/6L/4H`; local policy-forward profiling showed depth was a
poor tradeoff for this serving path, so the default is now the aggressive
`128d/3L/4H` preset. If quality drops too far, `160d/3L/5H` is the next
candidate.

Current measured counts:

- full: `30,034,002` params
- old mini `128d/6L/4H`: `2,647,086` params
- mini `128d/3L/4H`: `1,855,998` params
- ratio vs full: `16.18x` fewer params

Synthetic policy-forward timings on local MPS:

| Config | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `128d/6L/4H` | `8.00ms` | `12.67ms` | `21.06ms` |
| `160d/3L/5H` | `8.47ms` | `8.32ms` | `15.89ms` |
| `128d/3L/4H` | `4.66ms` | `7.16ms` | `10.67ms` |

Explicit CLI flags still override the preset, so width/depth sweeps can use the
same named base config.

## Local Training

```bash
oxi/scripts/train_mini_local.sh ./data ./mini_local
```

By default this runs for one hour (`TIMEOUT=3600`) so WSD has a real budget and
then computes `whitening.json`.

Smoke test:

```bash
TIMEOUT=600 MAX_SAMPLES=50000 oxi/scripts/train_mini_local.sh ./data ./mini_smoke
```

The script writes the checkpoint to `<LOG_DIR>/model` and computes
`whitening.json` there when training exits.

## Serving

The bot server can load both models:

```bash
MODEL_PATH=/models/full \
MINI_MODEL_PATH=/models/mini \
FULL_INFERENCE_WORKERS=2 \
MINI_INFERENCE_WORKERS=2 \
cargo run --release -- server --bind 0.0.0.0:8402
```

Batch `/predict` requests can set `model: "mini"` per item. If
`MINI_MODEL_PATH` is absent, mini requests fall back to the full model.

## Distillation Work Remaining

This patch makes mini a first-class architecture and serving target. True
teacher distillation should add a training objective that loads a frozen full
model and mixes:

- KL from teacher policy logits/probabilities to student policy logits.
- Cosine or MSE imitation of teacher trunk-mean embeddings, preferably before
  whitening.

Keep the removed retrieval-head objectives dead. The embedding target should be
the current trunk-mean serving embedding path.
