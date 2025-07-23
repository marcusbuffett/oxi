# Oxi - Human-like chess engine

Oxi is a chess engine, which predicts human-like chess moves based on player skill levels. It uses the Burn ML framework.

## Features

- Human move prediction
- Neural network architecture similar to Maia 2
- Elo-aware attention mechanism for adapting to player strength
- Training from PGN chess games
- WDL head

## Building

## Usage

### Training a Model

Train a new model from PGN or CSV data. The data will be automatically split into training and validation sets based on the `--train-ratio` parameter (default 0.9 for 90% train, 10% validation).

```bash


cargo run --release --bin oxi -- train --data-path <some_pgn_file_or_directory> --output-dir checkpoints --epochs 1000 --channels 64  --num-blocks 6 --max-samples 2000 --batch-size=128 --num-devices=8
```

### Running Inference

Use a trained model to predict moves:

```bash
# Basic inference with default model
cargo run --release --bin predict -- \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Specify model path and Elo ratings
cargo run --release --bin predict -- \
    --model-path checkpoints/model_epoch_5.mpk \
    --fen "5rk1/2p3p1/p2q1r1p/1p1p2p1/3P1nN1/2P1RP2/PP1Q2PP/4R1K1 w - - 0 31" \
    --elo-self 1600 \
    --elo-oppo 1800 \
    --top-n 10

# Use custom temperature for move sampling
cargo run --release --bin predict -- \
    --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4" \
    --temperature 0.5 \
    --top-n 5

# Use custom model config
cargo run --release --bin predict -- \
    --model-path models/custom_model.mpk \
    --config-path configs/custom_config.yaml \
    --fen "8/8/8/8/8/8/8/8 w - - 0 1"
```

### Example Output

```
Loading model from: checkpoints/model.mpk

Analyzing position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Self Elo: 1600, Opponent Elo: 1600
Temperature: 1.0

Top 10 predicted moves:
Rank  Move     Probability  Value     
----------------------------------------
1     e2e4     0.1523       0.0234    
2     d2d4     0.1456       0.0189    
3     g1f3     0.1234       0.0156    
4     c2c4     0.0987       0.0123    
5     e2e3     0.0876       0.0098    
6     d2d3     0.0765       0.0076    
7     b1c3     0.0654       0.0054    
8     f2f4     0.0543       0.0032    
9     g2g3     0.0432       0.0021    
10    b2b3     0.0321       0.0012    

Position evaluation: 0.0234 (from white's perspective)
Assessment: Position is equal
```

## Model Architecture

The model consists of:

- **ChessResNet encoder**: Configurable residual blocks (default 15) with squeeze-excitation
- **Vision transformer**: 6 layers with Elo-aware attention
- **Output heads**: Policy (1880 moves), value (position evaluation), and side info (141 features)

The only configurable architecture parameters are:
- `num_blocks`: Number of residual blocks in the encoder (default: 15)
- `channels`: Number of channels in the model (default: 256)

## Data Format

The training command accepts a single data source and automatically splits it into training and validation sets. Use the `--train-ratio` parameter to control the split (e.g., 0.9 for 90% training data).


### PGN Files
Standard PGN format with player ratings in the headers:
```
[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "1523"]
[BlackElo "1647"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...
```

## Model Configuration

The model architecture is simplified with only two configurable parameters:

- `--num-blocks`: Number of residual blocks in the CNN encoder (default: 15)
  - Smaller values (3-5) for faster training and testing
  - Larger values (10-20) for better performance
  
- `--channels`: Number of channels in the model (default: 256)
  - Must be divisible by 32 for attention heads
  - Common values: 64, 128, 256, 512

All other architecture parameters are automatically derived from these two values.

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_minimal_save_load

# Run integration tests
cargo test --test smoke_test
```

### Project Structure

```
oxi/
├── src/
│   ├── main.rs           # Training CLI
│   ├── bin/
│   │   └── predict.rs    # Inference CLI
│   ├── model.rs          # Neural network architecture
│   ├── training.rs       # Training loop
│   ├── inference.rs      # Inference engine
│   ├── dataset.rs        # Data loading and batching
│   ├── encoding.rs       # Board encoding
│   ├── moves.rs          # Move encoding/decoding
│   └── config.rs         # Configuration structures
├── tests/
│   └── smoke_test.rs     # Integration tests
└── Cargo.toml
```

