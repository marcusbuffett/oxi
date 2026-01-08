# Oxi - Human-like Chess Engine

Oxi models human decision making in chess across the full rating spectrum. To our knowledge, it is the most accurate to human play chess AI in the world. The stack is built in Rust on top of the Burn ML framework and ships end-to-end tooling for data acquisition, training, evaluation, and analysis.

## Highlights
- State-of-the-art human move prediction accuracy with Elo-aware conditioning and timing signals.
- Dual-stream board encoder with learned gating between token features and global context.
- Smolgen dynamic positional attention with Peri-LN residual routing (14 layers by default).
- Multi-task heads for policy (64x76 moves), value (win/draw/loss), auxiliary side information, and time usage, each with calibrated uncertainty estimates.
- Adaptive training loop with GradNorm balancing, cosine schedules, curriculum sampling, and a Ratatui dashboard for live metrics.

## Requirements
- Rust 1.75 or newer (install via `rustup`).
- Cargo build tooling and a working C++ toolchain (needed for LibTorch).
- libtorch (CPU, CUDA, or Metal) libraries. Burn will download a matching build automatically; follow the [Burn docs](https://tracel-ai.github.io/burn/) if you need to pin a specific version.
- Optional but recommended: a modern NVIDIA GPU with CUDA 12+ (Linux) or Apple Silicon with Metal (macOS). Training on CPU works for smoke tests but is prohibitively slow for large-scale runs.
- `zstd` CLI utilities if you plan to inspect downloaded PGN archives manually.

## Quickstart

```bash
# 1. Fetch all standard rated Lichess PGNs since 2022 (downloads .pgn.zst files).
cargo run --release --bin oxi -- download-all --output-dir data/pgn

# 2. Train with conservative batch sizes (adjust paths/flags as needed).
cargo run --release --bin oxi -- train \
  --data-path data/pgn \
  --batch-size 1024 \
  --physical-batch-size 64 \
  --num-devices 1
```

- Add `--resume` to continue from the latest checkpoint in `checkpoints/model/`.
- The TUI opens by default; disable it on headless systems with `--disable-tui`.

## Command Reference
- `train`: Full training loop. Accepts PGN directories or single files via `--data-path`. Supports curriculum controls such as `--max-samples`, `--enable-elo-sampling`, and `--checkpoint-interval`.
- `download-all`: Concurrently downloads every standard rated Lichess archive (2022-present) to a target directory.
- `download-pgn`: Fetches a single month (`--year`, `--month`) to a directory. Use `--local` to limit to one file for testing.
- `filter-confident`: Generates a curated dataset of high-confidence positions for bootstrapping or curriculum learning.
- `process-pgn`: Placeholder for pre-processing PGNs into serialized datasets (implementation pending).
- `evaluate`: Stub for offline evaluation; currently prints a placeholder message.
- `download`: Planned pretrained model fetcher; prints available tags today (`blitz`, `rapid`, `classical`).
- `inference`: Loads a checkpoint and prepares the inference engine. Printing of ranked moves is under construction; for now, integrate directly via `InferenceEngine` in code.

Run any subcommand with `--help` for the full list of options and defaults.

## Training Notes
- Data ingestion understands both plain `.pgn` and `.pgn.zst` files. Archives are streamed and decoded on the fly, so you can keep compressed files on disk.
- Elo and ply sampling are enabled by default to emphasize underrepresented positions while keeping a balanced curriculum.
- Checkpoints live under `checkpoints/model/` along with the AdamW optimizer shards and GradNorm state. Copy the entire directory when you want to resume elsewhere.
- The training loop reports detailed accuracy breakdowns (move accuracy, grad norms, value calibration) in the TUI as well as in `train.log`.
- Time usage targets are supervised from the PGN clock data; make sure your source PGNs include clock annotations if you want the time head to learn meaningful patterns.

## Model Architecture
- Board encoding splits each square into per-token features (piece identity, tactical, positional, misc) and recency channels for the latest moves.
- Optional convolutional stem (disabled by default) can be enabled with `--conv-layers`.
- Token and global streams are normalized separately, then fused through learned gates before entering the transformer stack.
- Attention layers use Smolgen dynamic position-dependent biases with Peri-LN (post-residual normalization) in both the attention and MLP paths.
- Policy head outputs 64x76 logits covering from-to square pairs (including underpromotions), masked by legality during training.
- Value head predicts win/draw/loss logits and shares pooled representations with the side-information and time-usage heads.
- Each head owns a learnable log-variance parameter, enabling adaptive loss weighting that works hand-in-hand with GradNorm.

## Development
- Build: `cargo build --release`
- Tests: `cargo test`
- Formatter: `cargo fmt`
- Clippy: `cargo clippy --all-targets -- -D warnings`

Logs land in `train.log`, `error.log`, and the rotating sheets under `sheet-*.txt` for detailed timeline snapshots.

## Project Layout
```
oxi/
|-- src/
|   |-- main.rs                # Unified CLI entry point
|   |-- config.rs              # Training/inference configuration and defaults
|   |-- custom_training.rs     # End-to-end training loop and metrics
|   |-- model.rs               # OXIModel definition and multi-head outputs
|   |-- relative_position_transformer.rs
|   |-- dataset.rs             # PGN ingestion and feature encoding
|   |-- inference.rs           # Inference engine used by the CLI and tests
|   |-- encoding.rs, moves.rs  # Chess-specific feature engineering
|   `-- ...                    # Metrics, schedulers, GradNorm utilities
|-- tests/                     # Integration tests and tensor sanity checks
|-- checkpoints/               # Training outputs (gitignored)
`-- Cargo.toml
```
