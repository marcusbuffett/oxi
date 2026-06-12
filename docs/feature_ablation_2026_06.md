# Feature ablation study — 2026-06-11

## Method

Channel zeroing against a trained checkpoint (`oxi feature-ablation`): zero one
named group of input channels at inference time, score the degradation on Allie
test-set positions. Measures what the **trained model relies on** — no
retraining. Run below: step-44k checkpoint of `full_h100_policy_ce_fixed_20260611_114229`
(320d/16L/10H, policy-CE-only), 19,461 positions (400 games), baseline top-1 53.59%.

Caveats:
- Reliance ≠ retraining value. A near-zero channel is redundant *given the other
  inputs*; a retrained model without it would likely adapt. Conversely, reliance
  can overstate: the model never trained with the channel missing.
- Coarse-group rows (ALL) overstate importance vs the sum of their subgroups —
  zeroing many channels at once is a bigger distribution shift. Subgroup rows
  are the actionable ones.

## Results (sorted by reliance)

| group | channels | Δ top-1 | Δ ln p(human move) |
|---|---|---|---|
| piece_identity (ALL) | 12 | −31.50pt | −1.3733 |
| tactical (ALL) | 22 | −19.24pt | −0.8015 |
| attackers_black | 8 | −14.21pt | −0.6316 |
| attackers_white | 8 | −11.95pt | −0.4884 |
| positional (ALL) | 25 | −10.31pt | −0.4057 |
| square_control | 1 | −4.95pt | −0.1981 |
| recency (ALL) | 4 | −2.19pt | −0.0698 |
| rank_onehot | 8 | −1.38pt | −0.0637 |
| recency_black (from/to) | 2 | −1.43pt | −0.0489 |
| castling_right | 1 | −1.01pt | −0.0418 |
| recency_white (from/to) | 2 | −1.13pt | −0.0309 |
| file_onehot | 8 | −0.43pt | −0.0303 |
| hanging | 1 | −0.88pt | −0.0287 |
| mobility (legal_moves_norm) | 1 | −0.29pt | −0.0087 |
| pins (pinned/abs_pin/pin_target) | 3 | −0.12pt | −0.0046 |
| en_passant | 1 | −0.04pt | −0.0020 |
| passed_pawn | 1 | −0.07pt | −0.0020 |
| open_file | 1 | +0.03pt | −0.0004 |
| dark_square | 1 | +0.04pt | −0.0002 |
| weak_squares | 2 | +0.05pt | −0.0002 |
| pawn_structure (iso/back/doubled) | 3 | +0.14pt | −0.0001 |
| pinned_defender | 1 | −0.01pt | −0.0000 |

## Decision (encoding v2)

**Pruned** (13 channels, all ≤0.005 |Δ ln p|): pins (3), pinned_defender,
pawn_structure iso/backward/doubled (3), weak_squares (2), open_file,
passed_pawn, dark_square, en_passant. The trunk evidently reconstructs all of
these from raw piece placement + attack maps. En passant remains available to
the model implicitly via legality masking and the recency channels.

**Kept**: piece identity, both attacker maps, square_control, hanging,
mobility, rank/file one-hots, castling_right, recency. Everything kept measures
≥0.009 |Δ ln p|.

**Added** (same change set): history occupancy planes — 12 piece one-hots for
each of the past 7 positions (84 channels), Maia-3-style, replacing nothing
(recency channels stay). `PREVIOUS_POSITIONS` 5 → 7.

**Added** (follow-up, same branch): two channel families chosen *because* the
ablation showed attack/control inputs dominate:
- **Per-square SEE** (2): resolved static-exchange outcome for each side
  initiating a capture on the square (pawns / 9, signed). Supersedes binary
  `hanging` with the actual attacker/defender-value arithmetic.
- **X-ray attackers** (4): per-side count + material of sliders attacking the
  square through exactly one piece — batteries, pins/skewers, discovered
  attacks. Motivating case: the Sicilian-probe b3 square, where Rb1's cover
  through Qb2 was invisible to the direct-attack channels.

Net: 65 → 142 channels per square.

## Takeaways for future feature work

- The hand-crafted **attack/control** representation is doing enormous work:
  attacker maps ~−12 to −14pt each, square_control −5pt from a single channel.
  Feature-engineering effort should flow toward richer tactical inputs (e.g.
  static-exchange/SEE per square, x-ray attacks), not positional flags.
- The hand-crafted **positional vocabulary** was almost entirely dead weight.
- The trained model's reliance on `attackers_black` (the side-to-move's
  opponent in mirrored frame) exceeds `attackers_white` — threat perception
  outweighs own-attack bookkeeping.

Re-run after any retrain: results live in this file; the channel map is
`CHANNEL_GROUPS` in `src/feature_ablation.rs` and must stay in sync with
`encoding::encode_position`.
