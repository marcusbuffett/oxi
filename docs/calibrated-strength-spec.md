# Calibrated Strength Spec

## Goal

Train Oxi to match human move quality, not just human move identity.

The core target is the human-conditioned regret profile:

- `regret = best_eval_cp - eval_cp(human_move)`
- conditioned on board state, player Elo, game stage, and time context

This avoids the failure mode where the model misses the exact human move but substitutes an implausible blunder.

## Non-Goal

We do not use raw engine-landscape "sharpness" as the primary target.

That signal is too weak as a proxy for human error propensity. A position can be engine-sharp while still being easy for humans because the losing moves are obviously losing.

## Stored Stockfish Labels

We precompute Stockfish labels offline and store them in SQLite.

For each labeled position we store:

- FEN
- human move
- player Elo
- opponent Elo
- ply
- stage
- result from side-to-move perspective
- Stockfish depth
- best evaluation in centipawns
- human move regret
- compressed sparse legal-move regret vector
- regret bin index

The sparse legal-move regret vector is the main expensive artifact. It lets us later compute:

- policy-implied expected regret
- tail probabilities like `P(regret > 100cp)`
- alternate bucketing schemes without re-running Stockfish

## Regret Bins

Version 1 uses an 8-bin categorical target for human regret:

1. `0`
2. `1-10`
3. `11-25`
4. `26-50`
5. `51-100`
6. `101-200`
7. `201-400`
8. `400+`

The model head should output `K=8` logits.

Why categorical:

- captures "usually fine, occasionally awful" behavior
- yields mean regret and blunder rates by summing bin mass
- avoids prematurely choosing a parametric family

## Model Heads

We plan to add:

- `policy_head`
- `regret_head`

The `regret_head` predicts the categorical regret distribution of the human move:

- output shape: `[batch, 8]`
- target: regret bin for the actual human move

Optional later heads:

- tail/blunder thresholds as auxiliary binary heads
- ordinal regret thresholds instead of plain categorical bins

## Losses

On all positions:

- `L_policy_ce`

On Stockfish-labeled positions only:

- `L_regret_ce`
- `L_policy_e_regret`
- optional tail calibration losses

### Direct policy calibration term

Let:

- `pi(a)` be the model policy over legal moves
- `l(a)` be Stockfish regret for move `a`

Then:

- `E_pi[R] = sum_a pi(a) * l(a)`

We compare `E_pi[R]` against human-conditioned regret targets.

Version 1 target:

- actual human move regret for the sample

Version 2 target:

- smoothed bucket target by `(elo, stage, time-control family)`

## Semi-Supervised Training

We will train on many more policy examples than Stockfish-labeled examples.

That is acceptable.

Mechanism:

- every example gets policy CE
- only labeled examples get regret/calibration losses
- unlabeled examples carry masks so those losses are skipped

This keeps the full dataset useful while avoiding live engine calls.

## Research Loop Integration

`research_loop.py` should eventually score both:

- move prediction quality
- calibration quality

Planned calibration metrics:

- model expected regret vs human expected regret
- blunder-rate gap over `100cp`
- blunder-rate gap over `300cp`
- regret-bin cross-entropy on labeled eval set

The acceptance policy should keep top-1 accuracy from regressing badly while rewarding better regret calibration.

## SQLite Schema

### `labeled_positions`

- `id INTEGER PRIMARY KEY`
- `fen TEXT NOT NULL`
- `human_move TEXT NOT NULL`
- `player_elo INTEGER NOT NULL`
- `opponent_elo INTEGER NOT NULL`
- `ply INTEGER NOT NULL`
- `stage TEXT NOT NULL`
- `game_result REAL NOT NULL`
- `stockfish_depth INTEGER NOT NULL`
- `best_eval_cp INTEGER NOT NULL`
- `human_regret_cp REAL NOT NULL`
- `regret_bin INTEGER NOT NULL`
- `move_loss_blob BLOB NOT NULL`
- `created_at_unix INTEGER NOT NULL`

Indexes:

- `(player_elo, ply)`
- `(stage, regret_bin)`
- unique `(fen, human_move, player_elo, opponent_elo, ply, stockfish_depth)`

## Current Implementation Scope

This change set implements:

- the spec
- shared regret-bin definitions
- a SQLite-backed calibration cache writer
- a CLI command to create that cache from PGNs plus Stockfish

The model head and loss changes are left for the next step.
