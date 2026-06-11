# Oxi — Human-like Chess Engine

Oxi predicts the move a *human* would play — conditioned on their rating, their opponent's rating, the clock, and the phase of the game — rather than the best move. It powers [Chessbook](https://chessbook.com)'s human-like bot opponents and the learnability scoring that decides which opening moves are worth teaching. Built in Rust on the [Burn](https://burn.dev) ML framework, end to end: data acquisition, training, evaluation, and inference.

## Current accuracy

We benchmark on the **Allie test set** (Zhang et al., ICLR 2025): 884,049 positions from 18,239 Lichess 2022 blitz games, downsampled to roughly uniform skill bins, with the first 10 plies and post-time-pressure positions excluded. Our `evaluate-allie` command reproduces the protocol exactly (same 884,049-position denominator), so these numbers are directly comparable:

| Model | Params | Top-1 accuracy |
|-------|--------|----------------|
| Maia-3-79M (ICLR 2026) | 79M | 57.1% |
| Maia-3-23M | 23M | 56.6% |
| Allie-Adaptive-Search | 355M | 55.9% |
| Allie-Policy | 355M | 55.7% |
| Maia-3-5M | 5M | 55.4% |
| **Oxi (mid-training checkpoint)** | **~30M** | **52.3%** |
| Maia-2 | 23M | 52.0% |
| Maia⋆ (original Maia ensemble) | 92M | 51.6% |

The Oxi number is a snapshot from ~1/4 of the way through the current H100 training run (before LR decay), trained on two months of Lichess data with players under 1000 filtered out — the sub-1000 test buckets (12% of the test set) score in the 40s as a result. We'll update when the run completes. No test-set leakage: the run trains on 2023/2026 archives; the test set is 2022 games.

## Architecture

A 320-dim, 16-layer, 10-head transformer over the 64 squares of the board (~scaled up from the previous production model's 192/8/6).

**Board encoding.** Each square is a token with 65 hand-engineered features:

| Group | Count | Contents |
|-------|-------|----------|
| Piece identity | 12 | One-hot piece/color |
| Tactical | 22 | Attackers by role for both sides, attacker counts/material, pins, pin targets, hanging pieces, square control |
| Positional | 25 | Legal-move targets, pawn structure, weak squares, open files, passed pawns, square color, rank/file one-hots |
| Misc | 2 | En passant target, castling rights |
| Recency | 4 | From/to squares of each side's last move, with exponential decay over the previous 5 positions |

The tactical and positional groups are the interesting part: rather than making the network rediscover attack maps and pawn structure from piece placement alone, we compute them with [shakmaty](https://github.com/niklasf/shakmaty) at encode time. Humans see threats and structure directly; the features let the model do the same.

**Global conditioning.** 11 scalar features — both players' Elo, remaining clock time and time ratios for both sides, increment, move number, material imbalance, piece count, and a puzzle flag — are injected into *every* transformer block via FiLM-conditioned RmsNorm: each block's normalization gamma/beta are generated from the globals. This is how one model plays like a 1100 or a 2300 depending on who it's imitating — rating conditioning modulates the entire trunk, not just a head.

**Attention.** Smolgen-style dynamic attention (from the Leela Chess Zero lineage): per-position attention biases are generated from the board content itself, since "which squares should attend to each other" in chess depends on the position, not on fixed distances. Pre-norm residual blocks, RmsNorm, SiLU MLPs.

**Heads.**

- **Policy** — factorized into source-square and target-square projections (64×76 move slots including underpromotions), FiLM-modulated by the globals, with a dedicated transformer block before the head. Masked by legality.
- **Value** — win/draw/loss logits, with its own pre-head transformer block.
- **Time usage** — predicts what fraction of the remaining clock the player spends, supervised from PGN clock annotations.
- **Centipawn loss** — predicts the regret distribution of the played move (how badly a player at this rating errs here).
- **Auxiliary** — side info, mobility, material, and from/to-square heads that exist purely to shape trunk representations.

Each head owns a learnable log-variance for uncertainty-weighted multi-task loss balancing. (The current run trains with policy CE only; the multi-task machinery is used for full runs.)

## Position embeddings & retrieval

The trunk doubles as a position-similarity encoder: mean-pooled trunk activations are extracted as position embeddings, then **ZCA-whitened** (`compute-whitening` produces a `whitening.json` served alongside the model). Raw transformer embeddings are anisotropic — all positions live in a narrow cone dominated by a few directions (side to move, material, phase) — so cosine similarity barely discriminates. Whitening centers the cone and equalizes variance, spreading the cosine range across the informative directions. Chessbook uses these embeddings for KNN retrieval over a user's repertoire when scoring move learnability.

## Training

- **Data**: Lichess standard rated games (streamed `.pgn.zst`, no decompression to disk), filtered to 1000–3000 Elo, ≤200 rating gap, sane time controls, ≥30s on the clock per move. Elo and ply sampling balance the curriculum across rating bands and game phases.
- **Schedule**: warmup–stable–decay (WSD) learning rate. The stable phase can run open-ended (`--lr-hold`); the decay phase can be applied later by resuming the checkpoint with a budget (`--max-samples` or `--timeout`). This replaced GradNorm + reduce-on-plateau — simpler and easier to reason about mid-run.
- **Monitoring**: a Ratatui TUI with live loss/accuracy/calibration breakdowns, plus per-metric logs under `metrics_logs/` for remote runs.

## Quickstart

```bash
# 1. Fetch standard rated Lichess PGNs (downloads .pgn.zst files)
cargo run --release --bin oxi -- download-all --output-dir data/pgn

# 2. Train
cargo run --release --bin oxi -- train \
  --data-path data/pgn \
  --batch-size 1024 \
  --physical-batch-size 64 \
  --num-devices 1
```

- `--resume` continues from the latest checkpoint in `checkpoints/model/`.
- `--disable-tui` for headless systems.

## Command reference

| Command | Purpose |
|---------|---------|
| `train` | Full training loop (PGN dirs/files via `--data-path`; curriculum, checkpointing, LR flags) |
| `download-all` / `download-pgn` | Fetch Lichess monthly archives (all since 2022, or one month) |
| `download-tcec` / `download-puzzles` | Engine games and Lichess puzzle data |
| `create-eval-set` / `quick-eval` / `evaluate` | Build held-out eval sets and score checkpoints |
| `compute-whitening` | Estimate the ZCA whitening transform for retrieval embeddings |
| `create-calibration-db` | Build the centipawn-loss calibration database |
| `inference` | Load a checkpoint and run the inference engine |

Run any subcommand with `--help` for full options.

## Requirements

- Rust 1.75+, a C++ toolchain, and libtorch (Burn downloads a matching build; see the [Burn docs](https://burn.dev) to pin one).
- An NVIDIA GPU with CUDA 12+ (Linux) or Apple Silicon with Metal (macOS). CPU works for smoke tests only.

## Development

```bash
cargo build --release
cargo test
cargo fmt
cargo clippy --all-targets -- -D warnings

# Compare CPU inference latency across architectures
cargo test --release --features backend-tch --test model_size_bench -- --ignored --nocapture
```

## Project layout

```
oxi/
├── src/
│   ├── main.rs                          # CLI entry point
│   ├── config.rs                        # Unified config + feature constants
│   ├── custom_training.rs               # Training loop and metrics
│   ├── model.rs                         # OXIModel and multi-head outputs
│   ├── relative_position_transformer.rs # Transformer blocks, FiLM RmsNorm
│   ├── smolgen.rs                       # Dynamic attention weight generation
│   ├── factorized_policy.rs             # Source/target-factorized policy head
│   ├── encoding.rs, move_encoding.rs    # Board/move feature engineering
│   ├── dataset.rs, pgn_processor.rs     # PGN ingestion and batching
│   ├── inference.rs                     # Inference engine
│   ├── whitening.rs                     # ZCA whitening for retrieval embeddings
│   ├── wsd_scheduler.rs                 # Warmup-stable-decay LR schedule
│   └── ...                              # Metrics, TUI, calibration
├── tests/                               # Integration tests, tensor checks, benchmarks
├── checkpoints/                         # Training outputs (gitignored)
└── Cargo.toml
```
