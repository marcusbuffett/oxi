//! Attention-map visualization harness.
//!
//! Runs `InferenceEngine::predict_with_attention_batch` on a hardcoded set of chess
//! positions and writes a single self-contained HTML file showing per-layer
//! attention saliency heatmaps (one 8x8 board per transformer layer, per
//! position). Open the resulting HTML file in a browser.
//!
//! Run with one of:
//!   cargo run --release --example attention_viz --features backend-ndarray
//!   cargo run --release --example attention_viz --features backend-tch
//!
//! The HTML is written to `oxi/attention_viz.html`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use oxi::config::{set_global_config, Config};
use oxi::inference::{
    AttentionLayer, EmbeddingOutput, GlobalFeatures, InferenceEngine, OxiPrediction,
};
use shakmaty::{fen::Fen, san::San, Chess, Color, Position, Role, Square};

/// Minimal view of `params.json` for reading model architecture.
/// Mirrors `load_config_from_model_dir` in `src/main.rs` so this example can
/// stand alone without pulling in that binary-local helper.
#[derive(Debug, Clone, serde::Deserialize)]
struct ModelParams {
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    #[serde(default = "default_smolgen_hidden")]
    smolgen_hidden: usize,
    #[serde(default = "default_smolgen_global_dim")]
    smolgen_global_dim: usize,
    #[serde(default = "default_smolgen_gen_size")]
    smolgen_gen_size: usize,
}

fn default_smolgen_hidden() -> usize {
    24
}
fn default_smolgen_global_dim() -> usize {
    128
}
fn default_smolgen_gen_size() -> usize {
    128
}

fn load_config_from_model_dir(model_dir: &Path) -> Result<Config> {
    let params_path = model_dir.join("params.json");
    let params_str = std::fs::read_to_string(&params_path)
        .with_context(|| format!("reading {}", params_path.display()))?;
    let params: ModelParams = serde_json::from_str(&params_str)
        .with_context(|| format!("parsing {}", params_path.display()))?;
    Ok(Config {
        embed_dim: params.embed_dim,
        num_heads: params.num_heads,
        num_layers: params.num_layers,
        smolgen_hidden: params.smolgen_hidden,
        smolgen_global_dim: params.smolgen_global_dim,
        smolgen_gen_size: params.smolgen_gen_size,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Test positions
// ---------------------------------------------------------------------------

struct TestPosition {
    name: &'static str,
    description: &'static str,
    epd: &'static str,
    elo: f32,
    /// Optional SAN or UCI sequence from the starting position that should
    /// produce `epd`. When set, `verify_test_positions` replays these moves
    /// at startup and compares board + side-to-move + castling rights
    /// against the hardcoded EPD. Use for opening positions. Omit for
    /// tactical/endgame setups that don't have a canonical line.
    setup_moves: Option<&'static str>,
}

fn test_positions() -> Vec<TestPosition> {
    vec![
        TestPosition {
            name: "Italian — Giuoco Piano",
            description: "Classic 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 tabiya.",
            epd: "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            elo: 1600.0,
            setup_moves: Some("e4 e5 Nf3 Nc6 Bc4 Bc5"),
        },
        TestPosition {
            name: "Ruy Lopez — Main line",
            description: "After 1.e4 e5 2.Nf3 Nc6 3.Bb5, mainline Spanish.",
            epd: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            elo: 2000.0,
            setup_moves: Some("e4 e5 Nf3 Nc6 Bb5"),
        },
        TestPosition {
            name: "Sicilian Najdorf",
            description: "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6.",
            epd: "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
            elo: 2200.0,
            setup_moves: Some("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
        },
        TestPosition {
            name: "Queen's Gambit Declined",
            description: "Classical QGD after 1.d4 d5 2.c4 e6 3.Nc3 Nf6.",
            epd: "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
            elo: 1800.0,
            setup_moves: Some("d4 d5 c4 e6 Nc3 Nf6"),
        },
        TestPosition {
            name: "King's Indian — Classical",
            description: "KID mainline with the central tension 6.Be2 e5 — white to move.",
            epd: "rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7",
            elo: 2000.0,
            setup_moves: Some("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5"),
        },
        TestPosition {
            name: "Caro-Kann — Classical",
            description: "After 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5.",
            epd: "rn1qkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 5",
            elo: 1800.0,
            setup_moves: Some("e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
        },
        TestPosition {
            name: "French Defense — Winawer",
            description: "Sharp 1.e4 e6 2.d4 d5 3.Nc3 Bb4 line.",
            epd: "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
            elo: 1900.0,
            setup_moves: Some("e4 e6 d4 d5 Nc3 Bb4"),
        },
        TestPosition {
            name: "Back-rank mate threat (must defend)",
            description: "Black to move. White threatens Ra8#; black must create luft (…h6/…g6) or otherwise defend the back rank.",
            epd: "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
            elo: 1600.0,
            setup_moves: None,
        },
        TestPosition {
            name: "Middlegame pin",
            description: "White bishop on g5 pins the black knight on f6 to the queen on d8.",
            epd: "r1bqkb1r/pppp1ppp/2n2n2/4p1B1/4P3/3P1N2/PPP2PPP/RN1QKB1R b KQkq - 0 4",
            elo: 1700.0,
            setup_moves: None,
        },
        TestPosition {
            name: "Quiet positional middlegame",
            description: "Balanced Queen's Gambit middlegame, both sides developed.",
            epd: "r2q1rk1/pp2bppp/2n1pn2/2pp4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
            elo: 1900.0,
            setup_moves: None,
        },
    ]
}

/// Replay `setup_moves` (when present) from the starting position and verify
/// the resulting board, side-to-move, and castling rights match the hardcoded
/// EPD. Fails fast on any mismatch so the harness refuses to render stale or
/// incorrect positions.
fn verify_test_positions(positions: &[TestPosition]) -> Result<()> {
    for tp in positions {
        let Some(setup_moves) = tp.setup_moves else {
            continue;
        };

        let expected = parse_epd_to_chess(tp.epd)
            .with_context(|| format!("TestPosition '{}': hardcoded EPD is invalid", tp.name))?;

        let mut pos = Chess::default();
        for token in setup_moves.split_whitespace() {
            let san = San::from_ascii(token.as_bytes()).map_err(|e| {
                anyhow::anyhow!(
                    "TestPosition '{}': could not parse SAN token {:?}: {:?}",
                    tp.name,
                    token,
                    e
                )
            })?;
            let m = san.to_move(&pos).map_err(|e| {
                anyhow::anyhow!(
                    "TestPosition '{}': SAN {:?} is not legal in the replayed position: {:?}",
                    tp.name,
                    token,
                    e
                )
            })?;
            pos = pos.play(m).map_err(|e| {
                anyhow::anyhow!(
                    "TestPosition '{}': play({:?}) failed: {:?}",
                    tp.name,
                    token,
                    e
                )
            })?;
        }

        let board_match = pos.board() == expected.board();
        let turn_match = pos.turn() == expected.turn();
        let castles_match = pos.castles().castling_rights() == expected.castles().castling_rights();

        if !(board_match && turn_match && castles_match) {
            let actual_fen = Fen::from_position(&pos, shakmaty::EnPassantMode::Legal).to_string();
            anyhow::bail!(
                "TestPosition '{}': replay of setup_moves produced {}, expected {} \
                 (board_match={}, turn_match={}, castles_match={})",
                tp.name,
                actual_fen,
                tp.epd,
                board_match,
                turn_match,
                castles_match
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attention aggregation
// ---------------------------------------------------------------------------

/// Min-max normalize a `[64]` vector to [0, 1]. If the range is degenerate
/// (all values equal), return all zeros.
fn normalize_layer(v: &[f32; 64]) -> [f32; 64] {
    let mut out = *v;
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in out.iter() {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    let range = mx - mn;
    if range > 1e-8 {
        for s in out.iter_mut() {
            *s = (*s - mn) / range;
        }
    } else {
        // Degenerate: uniform attention. Display as all zero.
        for s in out.iter_mut() {
            *s = 0.0;
        }
    }
    out
}

/// For a single [num_heads, 64, 64] layer, aggregate to a per-square saliency
/// vector [64]: mean across heads, then mean across the query axis.
/// Then min-max normalize to [0, 1].
fn aggregate_layer(layer: &AttentionLayer) -> [f32; 64] {
    let nh = layer.num_heads;
    debug_assert_eq!(layer.data.len(), nh * 64 * 64);

    let mut saliency = [0.0f32; 64];
    for head in 0..nh {
        let head_off = head * 64 * 64;
        for q in 0..64 {
            let row_off = head_off + q * 64;
            for (k, s) in saliency.iter_mut().enumerate() {
                *s += layer.data[row_off + k];
            }
        }
    }

    // Mean across (nh * 64) query-rows.
    let denom = (nh * 64) as f32;
    for s in saliency.iter_mut() {
        *s /= denom;
    }

    normalize_layer(&saliency)
}

/// Un-flip a model-frame `[64]` saliency vector into the absolute frame.
/// Oxi internally mirrors black-to-move positions via a vertical flip
/// (rank reversal) in `mirror_fen`, so `absolute_sq = model_sq ^ 56` (XOR
/// with 56 flips the rank bits, leaving files untouched). This is the same
/// vertical flip — NOT a 180° rotation — so we use `^ 56`, not `63 - sq`.
fn unflip_saliency_for_black_to_move(v: &[f32; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for model_sq in 0..64usize {
        let absolute_sq = model_sq ^ 56;
        out[absolute_sq] = v[model_sq];
    }
    out
}

/// Per-layer top-K linear aggregation: for each layer, rank squares by
/// saliency (highest first) and assign linear weights `(K - rank) / K` for
/// `rank < K`, else 0. Average across layers, then min-max normalize.
fn aggregate_top_k_linear(per_layer: &[[f32; 64]], k: usize) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    if per_layer.is_empty() {
        return out;
    }

    let num_layers = per_layer.len() as f32;
    let k_f = k as f32;
    for layer in per_layer {
        // Rank squares by saliency, highest first.
        let mut idx: [usize; 64] = [0; 64];
        for (i, s) in idx.iter_mut().enumerate() {
            *s = i;
        }
        idx.sort_by(|&a, &b| {
            layer[b]
                .partial_cmp(&layer[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &sq) in idx.iter().enumerate() {
            if rank < k {
                let weight = (k_f - rank as f32) / k_f;
                out[sq] += weight;
            }
        }
    }
    if num_layers > 0.0 {
        for o in out.iter_mut() {
            *o /= num_layers;
        }
    }

    // Final min-max normalize stretches to the full range.
    normalize_layer(&out)
}

// ---------------------------------------------------------------------------
// Embedding similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length float vectors. Returns 0.0 if
/// either vector is all-zero (norm = 0).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Build an N×N cosine similarity matrix over a list of embedding vectors.
fn cosine_sim_matrix(embeds: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeds.len();
    let mut m = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = cosine_sim(&embeds[i], &embeds[j]);
        }
    }
    m
}

/// Min / max / mean of the strictly off-diagonal entries of a square matrix.
fn off_diagonal_stats(m: &[Vec<f32>]) -> (f32, f32, f32) {
    let n = m.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = m[i][j];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
            sum += v as f64;
            count += 1;
        }
    }
    (mn, mx, (sum / count as f64) as f32)
}

/// Map a cosine similarity value onto a background color. The matrix's
/// off-diagonal [min, max] range is stretched onto white → deep red so
/// differences are eyeballable. Diagonal cells (always 1.0) are rendered
/// with a neutral gray caller-side; this function just handles the scale.
fn similarity_color(v: f32, min_val: f32, max_val: f32) -> String {
    let span = (max_val - min_val).max(1e-8);
    let t = ((v - min_val) / span).clamp(0.0, 1.0);
    // Same palette as saliency (white → pale yellow → orange → deep red).
    saliency_color(t)
}

// ---------------------------------------------------------------------------
// HTML rendering
// ---------------------------------------------------------------------------

fn piece_glyph(role: Role, color: Color) -> &'static str {
    match (color, role) {
        (Color::White, Role::King) => "\u{2654}",   // ♔
        (Color::White, Role::Queen) => "\u{2655}",  // ♕
        (Color::White, Role::Rook) => "\u{2656}",   // ♖
        (Color::White, Role::Bishop) => "\u{2657}", // ♗
        (Color::White, Role::Knight) => "\u{2658}", // ♘
        (Color::White, Role::Pawn) => "\u{2659}",   // ♙
        (Color::Black, Role::King) => "\u{265A}",   // ♚
        (Color::Black, Role::Queen) => "\u{265B}",  // ♛
        (Color::Black, Role::Rook) => "\u{265C}",   // ♜
        (Color::Black, Role::Bishop) => "\u{265D}", // ♝
        (Color::Black, Role::Knight) => "\u{265E}", // ♞
        (Color::Black, Role::Pawn) => "\u{265F}",   // ♟
    }
}

/// Map a normalized saliency value in [0,1] to a CSS rgb color.
/// Palette: white (0) → yellow → orange → red (1).
fn saliency_color(t: f32) -> String {
    let t = t.clamp(0.0, 1.0);
    // Piecewise linear interpolation through 4 stops.
    let stops = [
        (0.00f32, (255u8, 255u8, 255u8)), // white
        (0.40, (255, 235, 120)),          // pale yellow
        (0.70, (255, 150, 60)),           // orange
        (1.00, (200, 20, 20)),            // deep red
    ];
    for i in 0..stops.len() - 1 {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t <= t1 {
            let span = (t1 - t0).max(1e-8);
            let a = (t - t0) / span;
            let r = (c0.0 as f32 + a * (c1.0 as f32 - c0.0 as f32)).round() as u8;
            let g = (c0.1 as f32 + a * (c1.1 as f32 - c0.1 as f32)).round() as u8;
            let b = (c0.2 as f32 + a * (c1.2 as f32 - c0.2 as f32)).round() as u8;
            return format!("rgb({},{},{})", r, g, b);
        }
    }
    let (_, c) = stops.last().unwrap();
    format!("rgb({},{},{})", c.0, c.1, c.2)
}

/// Render one small 8x8 board with a saliency overlay.
/// `pieces` is indexed by square (0..=63 in shakmaty's a1=0 convention).
/// Rendered with a1 at bottom-left (white's perspective).
fn render_board(pieces: &[Option<(Role, Color)>; 64], saliency: &[f32; 64], label: &str) -> String {
    let mut out = String::new();
    out.push_str("<div class=\"layer\">");
    out.push_str(&format!("<div class=\"layer-label\">{}</div>", label));
    out.push_str("<div class=\"board\">");
    // rank 7 (top) down to rank 0 (bottom)
    for rank in (0..8).rev() {
        for file in 0..8 {
            let sq = rank * 8 + file;
            let color = saliency_color(saliency[sq]);
            let glyph = pieces[sq]
                .map(|(role, c)| piece_glyph(role, c))
                .unwrap_or("");
            let is_white_piece = matches!(pieces[sq], Some((_, Color::White)));
            let piece_class = if is_white_piece { "p wp" } else { "p bp" };
            out.push_str(&format!(
                "<div class=\"sq\" style=\"background:{}\"><span class=\"{}\">{}</span></div>",
                color, piece_class, glyph
            ));
        }
    }
    out.push_str("</div>"); // board
    out.push_str("</div>"); // layer
    out
}

/// Extract a `[Option<(Role, Color)>; 64]` piece map from a shakmaty `Chess`.
fn pieces_of(pos: &Chess) -> [Option<(Role, Color)>; 64] {
    let mut out: [Option<(Role, Color)>; 64] = [None; 64];
    let board = pos.board();
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            out[sq as usize] = Some((piece.role, piece.color));
        }
    }
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render_position_card(
    tp: &TestPosition,
    pos: &Chess,
    prediction: &OxiPrediction,
    per_layer: &[[f32; 64]],
    similar: &[(String, f32)],
) -> String {
    let pieces = pieces_of(pos);
    let black_to_move = pos.turn() == Color::Black;

    let mut card = String::new();
    card.push_str("<section class=\"card\">");

    // Header
    card.push_str("<div class=\"header\">");
    card.push_str(&format!("<h2>{}</h2>", html_escape(tp.name)));
    card.push_str(&format!(
        "<div class=\"desc\">{}</div>",
        html_escape(tp.description)
    ));
    card.push_str(&format!(
        "<div class=\"epd\"><span class=\"k\">EPD:</span> <code>{}</code></div>",
        html_escape(tp.epd)
    ));
    card.push_str(&format!(
        "<div class=\"meta\">Elo: {} · {}</div>",
        tp.elo as i32,
        if black_to_move {
            "black to move"
        } else {
            "white to move"
        },
    ));
    card.push_str("</div>");

    // Top moves + WDL
    card.push_str("<div class=\"moves\"><span class=\"k\">Top moves:</span> ");
    if prediction.moves.is_empty() {
        card.push_str("<em style=\"color:#999\">none extracted</em>");
    } else {
        for (i, m) in prediction.moves.iter().enumerate() {
            if i > 0 {
                card.push_str(", ");
            }
            card.push_str(&format!(
                "<code>{}</code> <span class=\"p\">{:.1}%</span>",
                html_escape(&m.uci_move),
                m.probability * 100.0
            ));
        }
    }
    card.push_str(&format!(
        " · <span class=\"k\">WDL:</span> <span class=\"wdl\">W {:.0}% / D {:.0}% / L {:.0}%</span>",
        prediction.wdl.win_prob * 100.0,
        prediction.wdl.draw_prob * 100.0,
        prediction.wdl.loss_prob * 100.0,
    ));
    card.push_str("</div>");

    // Most-similar-positions list (from retrieval embedding cosine).
    if !similar.is_empty() {
        card.push_str(
            "<div class=\"row-label\">Most similar positions (by embedding cosine)</div>",
        );
        card.push_str("<ol class=\"similar\">");
        for (name, cos) in similar.iter() {
            card.push_str(&format!(
                "<li><span class=\"sim-name\">{}</span> <span class=\"sim-val\">{:.3}</span></li>",
                html_escape(name),
                cos
            ));
        }
        card.push_str("</ol>");
    }

    // Aggregate row (single board: top-10 linear across layers)
    card.push_str("<div class=\"row-label\">Aggregate</div>");
    card.push_str("<div class=\"layers aggregate\">");
    if !per_layer.is_empty() {
        let sal_topk = aggregate_top_k_linear(per_layer, 10);
        card.push_str(&render_board(&pieces, &sal_topk, "Top-10 linear"));
    }
    card.push_str("</div>");

    // Per-layer row
    card.push_str("<div class=\"row-label\">Per-layer</div>");
    card.push_str("<div class=\"layers\">");
    for (i, sal) in per_layer.iter().enumerate() {
        let label = format!("Layer {}", i);
        card.push_str(&render_board(&pieces, sal, &label));
    }
    card.push_str("</div>");

    card.push_str("</section>");
    card
}

/// Render the N×N cosine similarity grid as an HTML block, to be placed at
/// the top of the page. `names` labels rows and columns; `matrix` has the
/// cosine values (diagonal should be 1.0). Color scale stretches the
/// off-diagonal range onto the usual white → deep red palette; the diagonal
/// is rendered with a neutral gray for readability.
fn render_similarity_matrix(names: &[String], matrix: &[Vec<f32>]) -> String {
    let (mn, mx, mean) = off_diagonal_stats(matrix);

    let mut out = String::new();
    out.push_str("<section class=\"matrix-card\">");
    out.push_str("<h2>Position embedding similarity matrix (retrieval embedding)</h2>");
    out.push_str(&format!(
        "<div class=\"sub\">Cosine similarity between retrieval embeddings across the {} curated positions. Off-diagonal range: min {:.3}, max {:.3}, mean {:.3}. Lower range = embeddings discriminate more strongly. Color scale stretches the off-diagonal range onto white → deep red; the diagonal is always 1.0 (gray).</div>",
        names.len(),
        mn,
        mx,
        mean,
    ));

    out.push_str("<div class=\"matrix-wrap\"><table class=\"matrix\">");
    // Header row
    out.push_str("<tr><th class=\"corner\"></th>");
    for name in names {
        out.push_str(&format!(
            "<th class=\"colh\"><div class=\"colh-inner\">{}</div></th>",
            html_escape(name)
        ));
    }
    out.push_str("</tr>");

    for i in 0..matrix.len() {
        out.push_str("<tr>");
        out.push_str(&format!(
            "<th class=\"rowh\">{}</th>",
            html_escape(&names[i])
        ));
        for j in 0..matrix[i].len() {
            let v = matrix[i][j];
            let (bg, text_color) = if i == j {
                ("#dddddd".to_string(), "#666".to_string())
            } else {
                let bg = similarity_color(v, mn, mx);
                // Readable text: stretch t and darken accordingly.
                let span = (mx - mn).max(1e-8);
                let t = ((v - mn) / span).clamp(0.0, 1.0);
                let txt = if t > 0.6 { "#fff" } else { "#333" };
                (bg, txt.to_string())
            };
            out.push_str(&format!(
                "<td class=\"cell\" style=\"background:{};color:{}\">{:.2}</td>",
                bg, text_color, v
            ));
        }
        out.push_str("</tr>");
    }
    out.push_str("</table></div>");
    out.push_str("</section>");
    out
}

fn render_page(cards: &[String], similarity_section: &str) -> String {
    let css = r#"
:root {
  color-scheme: light;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 20px;
  background: #fafafa;
  color: #222;
}
h1 { margin: 0 0 4px 0; }
.sub { color: #666; font-size: 13px; margin-bottom: 24px; }
.card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card h2 { margin: 0 0 4px 0; font-size: 18px; }
.desc { color: #444; font-size: 14px; margin-bottom: 6px; }
.epd { font-size: 12px; color: #555; margin-bottom: 4px; }
.epd code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
.meta { font-size: 12px; color: #666; margin-bottom: 10px; }
.moves { font-size: 13px; margin-bottom: 12px; color: #333; }
.moves code { background: #f0f4ff; padding: 1px 4px; border-radius: 3px; }
.moves .p { color: #666; font-size: 11px; }
.moves .wdl { color: #444; }
.k { font-weight: 600; color: #555; }
.row-label {
  font-size: 11px;
  font-weight: 700;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 10px 0 6px 0;
}
.layers {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}
.layers.aggregate {
  padding-bottom: 12px;
  border-bottom: 1px dashed #ddd;
  margin-bottom: 4px;
}
.layer {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.layer-label {
  font-size: 11px;
  color: #555;
  margin-bottom: 4px;
  font-weight: 600;
}
.board {
  display: grid;
  grid-template-columns: repeat(8, 34px);
  grid-template-rows: repeat(8, 34px);
  border: 1px solid #333;
}
.sq {
  width: 34px;
  height: 34px;
  border: 0.5px solid #bbb;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}
.p {
  font-size: 22px;
  line-height: 1;
  text-shadow:
    -1px -1px 0 rgba(255,255,255,0.6),
     1px -1px 0 rgba(255,255,255,0.6),
    -1px  1px 0 rgba(255,255,255,0.6),
     1px  1px 0 rgba(255,255,255,0.6);
}
.wp { color: #fff; text-shadow:
    -1px -1px 0 #222,
     1px -1px 0 #222,
    -1px  1px 0 #222,
     1px  1px 0 #222;
}
.bp { color: #111; }
.legend {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #555;
  margin-bottom: 20px;
}
.legend .bar {
  width: 200px;
  height: 10px;
  border: 1px solid #888;
  background: linear-gradient(
    to right,
    rgb(255,255,255) 0%,
    rgb(255,235,120) 40%,
    rgb(255,150,60) 70%,
    rgb(200,20,20) 100%
  );
}
.matrix-card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.matrix-card h2 { margin: 0 0 4px 0; font-size: 18px; }
.matrix-card .sub { color: #555; font-size: 13px; margin-bottom: 12px; }
.matrix-wrap { overflow-x: auto; }
.matrix {
  border-collapse: collapse;
  font-size: 12px;
}
.matrix th, .matrix td {
  border: 1px solid #ccc;
  padding: 6px 8px;
  text-align: center;
  vertical-align: middle;
}
.matrix th.corner { background: transparent; border: none; }
.matrix th.rowh {
  background: #f4f4f4;
  text-align: right;
  font-weight: 600;
  white-space: nowrap;
}
.matrix th.colh {
  background: #f4f4f4;
  font-weight: 600;
  vertical-align: bottom;
  height: 130px;
  min-width: 36px;
  width: 36px;
}
.matrix th.colh .colh-inner {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  white-space: nowrap;
  padding: 4px 0;
}
.matrix td.cell {
  font-variant-numeric: tabular-nums;
  min-width: 48px;
}
.similar {
  margin: 0 0 10px 18px;
  padding: 0;
  font-size: 13px;
  color: #333;
}
.similar li { margin: 2px 0; }
.sim-name { font-weight: 500; }
.sim-val {
  font-variant-numeric: tabular-nums;
  color: #666;
  margin-left: 6px;
}
"#;

    let mut out = String::new();
    out.push_str("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    out.push_str("<title>Oxi Attention Maps</title>");
    out.push_str("<style>");
    out.push_str(css);
    out.push_str("</style></head><body>");
    out.push_str("<h1>Oxi attention maps</h1>");
    out.push_str("<div class=\"sub\">Each per-layer board shows attention saliency (mean over heads, mean over query axis, min-max normalized per layer). The aggregate board combines those per-layer vectors via top-10 rank-based linear weighting. Boards are rendered in the absolute frame (white at bottom, a1 at bottom-left); for black-to-move positions the model-frame attention is un-flipped vertically (model_sq ^ 56) so overlays align with the actual board. The similarity matrix at the top compares positions by the cosine of their retrieval embeddings — an alternative grouping signal to attention top-K fingerprints.</div>");
    out.push_str("<div class=\"legend\"><span>low attention</span><div class=\"bar\"></div><span>high attention</span></div>");
    out.push_str(similarity_section);
    for c in cards {
        out.push_str(c);
    }
    out.push_str("</body></html>");
    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn parse_epd_to_chess(epd: &str) -> Result<Chess> {
    let fen: Fen = epd
        .parse()
        .map_err(|e| anyhow::anyhow!("parsing EPD {:?}: {:?}", epd, e))?;
    let pos: Chess = fen
        .into_position(shakmaty::CastlingMode::Standard)
        .map_err(|e| anyhow::anyhow!("EPD {:?} not a legal position: {:?}", epd, e))?;
    Ok(pos)
}

#[cfg(any(feature = "backend-ndarray", feature = "backend-tch"))]
fn main() -> Result<()> {
    use oxi::inference::InferenceBackend;

    // Assume CWD is repo root (`oxi/`) when run via `cargo run --example ...`.
    let model_dir = PathBuf::from("model");
    if !model_dir.join("params.json").exists() {
        anyhow::bail!(
            "Could not find {}/params.json. Run this from the oxi/ crate root \
             (or adjust the hardcoded model_dir).",
            model_dir.display()
        );
    }

    let positions = test_positions();
    verify_test_positions(&positions).context("test_positions setup_moves verification failed")?;
    println!(
        "Verified {} test positions against setup_moves.",
        positions.len()
    );

    let device = burn::tensor::Device::<InferenceBackend>::default();
    let config = load_config_from_model_dir(&model_dir)?;
    println!(
        "Loaded config: embed_dim={}, num_layers={}, num_heads={}",
        config.embed_dim, config.num_layers, config.num_heads
    );
    // Install global config (model construction / encoding read from it).
    let _ = set_global_config(config.clone());

    let model_mpk = model_dir.join("model");
    let engine =
        InferenceEngine::<InferenceBackend>::from_checkpoint(&model_mpk, config.clone(), device)?;
    println!("Engine loaded from {}", model_mpk.display());

    // Run all positions through one batched forward pass so attention,
    // predictions, and embeddings come from the same pass.
    let t_start = std::time::Instant::now();

    let mut items: Vec<oxi::inference::BatchItem> = Vec::with_capacity(positions.len());
    let mut chess_positions: Vec<Chess> = Vec::with_capacity(positions.len());
    for tp in &positions {
        let chess =
            parse_epd_to_chess(tp.epd).with_context(|| format!("parsing EPD for {}", tp.name))?;
        let globals = GlobalFeatures {
            time_remaining_self: 300,
            time_remaining_oppo: 300,
            base_time: 300,
            increment: 3,
            move_count: 20,
            elo_self: tp.elo as i32,
            elo_oppo: tp.elo as i32,
            is_puzzle: false,
            is_in_check: chess.is_check(),
            total_pieces: chess.board().occupied().count() as u32,
        };
        items.push(oxi::inference::BatchItem {
            positions: vec![chess.clone()],
            previous_moves: Vec::new(),
            globals,
            temperature: 1.0,
            top_k: 5,
        });
        chess_positions.push(chess);
    }

    let results = engine.predict_with_attention_and_embedding_batch(&items)?;
    assert_eq!(results.len(), positions.len());

    // Collect retrieval embeddings.
    let embeddings: Vec<Vec<f32>> = results
        .iter()
        .map(|(_, _, e)| e.embedding.clone())
        .collect();
    let names: Vec<String> = positions.iter().map(|tp| tp.name.to_string()).collect();
    let sim_matrix = cosine_sim_matrix(&embeddings);

    // Per-position top-3 most similar others by cosine.
    let mut per_position_similar: Vec<Vec<(String, f32)>> = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        let mut ranked: Vec<(usize, f32)> = (0..positions.len())
            .filter(|&j| j != i)
            .map(|j| (j, sim_matrix[i][j]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3: Vec<(String, f32)> = ranked
            .into_iter()
            .take(3)
            .map(|(j, v)| (names[j].clone(), v))
            .collect();
        per_position_similar.push(top3);
    }

    // Build per-position cards.
    let mut cards = Vec::with_capacity(positions.len());
    let mut total_layers = 0usize;
    for (idx, tp) in positions.iter().enumerate() {
        let chess = &chess_positions[idx];
        let black_to_move = chess.turn() == Color::Black;
        let (prediction, attn_layers, _embedding): &(
            OxiPrediction,
            Vec<AttentionLayer>,
            EmbeddingOutput,
        ) = &results[idx];
        total_layers += attn_layers.len();

        let per_layer: Vec<[f32; 64]> = attn_layers
            .iter()
            .map(|l| {
                let model_frame = aggregate_layer(l);
                if black_to_move {
                    unflip_saliency_for_black_to_move(&model_frame)
                } else {
                    model_frame
                }
            })
            .collect();

        let card = render_position_card(
            tp,
            chess,
            prediction,
            &per_layer,
            &per_position_similar[idx],
        );
        cards.push(card);
        println!(
            "  · {} — {} layers, top move: {}",
            tp.name,
            attn_layers.len(),
            prediction
                .moves
                .first()
                .map(|m| m.uci_move.as_str())
                .unwrap_or("?")
        );
    }

    let wall_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let (mn, mx, mean) = off_diagonal_stats(&sim_matrix);
    println!(
        "\nEmbedding cosine similarity off-diagonal: min={:.3}, max={:.3}, mean={:.3}",
        mn, mx, mean
    );

    // Print the per-position top-3 similar list for quick inspection.
    for (i, tp) in positions.iter().enumerate() {
        let row: Vec<String> = per_position_similar[i]
            .iter()
            .map(|(name, v)| format!("{} ({:.3})", name, v))
            .collect();
        println!("  {} → {}", tp.name, row.join(", "));
    }

    let similarity_section = render_similarity_matrix(&names, &sim_matrix);
    let html = render_page(&cards, &similarity_section);
    let out_path = PathBuf::from("attention_viz.html");
    std::fs::write(&out_path, html)?;
    println!(
        "\nRendered {} positions × {} layers → {} (open in browser)",
        cards.len(),
        if cards.is_empty() {
            0
        } else {
            total_layers / cards.len().max(1)
        },
        out_path.display()
    );
    println!("Wall time (inference + render): {:.1} ms", wall_ms);

    Ok(())
}

#[cfg(not(any(feature = "backend-ndarray", feature = "backend-tch")))]
fn main() -> Result<()> {
    anyhow::bail!(
        "This example requires the `backend-ndarray` or `backend-tch` feature. Run: \
         cargo run --release --example attention_viz --features backend-tch"
    )
}
