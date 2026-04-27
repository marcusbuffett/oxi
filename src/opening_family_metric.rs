use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

use burn::prelude::*;
use shakmaty::fen::{Epd, Fen};
use shakmaty::{CastlingMode, Chess, EnPassantMode};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::Row;

use crate::dataset::ChessItem;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpeningFamilyNeighborMetrics {
    pub match_rate: f32,
    pub neighbor_pair_count: f32,
    pub position_coverage: f32,
}

static OPENING_FAMILY_LOOKUP: OnceLock<Option<HashMap<String, String>>> = OnceLock::new();

pub fn compute_opening_family_neighbor_metrics<B: Backend>(
    embeddings: Tensor<B, 2>,
    items: &[ChessItem],
    top_k: usize,
) -> OpeningFamilyNeighborMetrics {
    let batch_size = embeddings.dims()[0];
    if batch_size < 2 || items.len() != batch_size || top_k == 0 {
        return OpeningFamilyNeighborMetrics::default();
    }

    let label_sets = items
        .iter()
        .map(|item| {
            let mut labels = item
                .opening_family_labels
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            if labels.is_empty() {
                if let Some(label) = opening_family_label_for_fen(&item.fen) {
                    labels.insert(label);
                }
            }
            labels
        })
        .collect::<Vec<_>>();
    let labeled_positions = label_sets
        .iter()
        .filter(|labels| !labels.is_empty())
        .count();
    let position_coverage = labeled_positions as f32 / batch_size.max(1) as f32;
    if labeled_positions < 2 {
        return OpeningFamilyNeighborMetrics {
            position_coverage,
            ..Default::default()
        };
    }

    let sim = embeddings.clone().matmul(embeddings.transpose());
    let sim_data = sim.to_data();
    let Ok(sim_values) = sim_data.as_slice::<f32>() else {
        return OpeningFamilyNeighborMetrics {
            position_coverage,
            ..Default::default()
        };
    };

    let mut same_family = 0usize;
    let mut neighbor_pairs = 0usize;
    for i in 0..batch_size {
        let query_labels = &label_sets[i];
        if query_labels.is_empty() {
            continue;
        }

        let mut candidates = Vec::with_capacity(labeled_positions.saturating_sub(1));
        for j in 0..batch_size {
            if i == j || label_sets[j].is_empty() {
                continue;
            }
            candidates.push((sim_values[i * batch_size + j], j));
        }
        candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        for (_, j) in candidates.into_iter().take(top_k) {
            neighbor_pairs += 1;
            if query_labels
                .iter()
                .any(|label| label_sets[j].contains(label))
            {
                same_family += 1;
            }
        }
    }

    OpeningFamilyNeighborMetrics {
        match_rate: if neighbor_pairs > 0 {
            same_family as f32 / neighbor_pairs as f32
        } else {
            0.0
        },
        neighbor_pair_count: neighbor_pairs as f32,
        position_coverage,
    }
}

pub fn opening_family_label_for_position(pos: &Chess) -> Option<String> {
    let epd = Epd::from_position(pos, EnPassantMode::Legal).to_string();
    opening_family_label_for_epd(&epd)
}

pub fn opening_family_label_for_fen(fen: &str) -> Option<String> {
    fen_to_epd(fen).and_then(|epd| opening_family_label_for_epd(&epd))
}

pub fn opening_family_label_for_epd(epd: &str) -> Option<String> {
    opening_family_lookup()?.get(epd).cloned()
}

fn opening_family_lookup() -> Option<&'static HashMap<String, String>> {
    OPENING_FAMILY_LOOKUP
        .get_or_init(|| match load_opening_family_lookup() {
            Ok(lookup) if !lookup.is_empty() => {
                tracing::info!(
                    "opening_family_metric: loaded {} ECO opening labels",
                    lookup.len()
                );
                Some(lookup)
            }
            Ok(_) => {
                tracing::warn!("opening_family_metric: ECO lookup was empty");
                None
            }
            Err(err) => {
                tracing::warn!("opening_family_metric: disabled: {err}");
                None
            }
        })
        .as_ref()
}

fn load_opening_family_lookup() -> anyhow::Result<HashMap<String, String>> {
    let database_url = discover_database_url()
        .ok_or_else(|| anyhow::anyhow!("no database URL found in env or ../server/.env"))?;

    let rows = run_sqlx_blocking(async move {
        let options = PgConnectOptions::from_str(&database_url)?;
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(2))
            .connect_with(options);
        let pool = tokio::time::timeout(Duration::from_secs(2), pool).await??;
        let rows = sqlx::query(
            "select epd, code, coalesce(nullif(friendly_name, ''), full_name) as name \
             from eco_codes where hidden = false",
        )
        .fetch_all(&pool)
        .await?;
        pool.close().await;
        anyhow::Ok(rows)
    })?;

    let mut lookup = HashMap::with_capacity(rows.len());
    for row in rows {
        let epd: String = row.try_get("epd")?;
        let code: String = row.try_get("code")?;
        let name: String = row.try_get("name")?;
        lookup.insert(epd, opening_family_label(&code, &name));
    }
    Ok(lookup)
}

fn run_sqlx_blocking<F, T>(future: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(future)
        })
        .join()
        .map_err(|_| anyhow::anyhow!("database loader thread panicked"))?
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(future)
    }
}

fn discover_database_url() -> Option<String> {
    for key in [
        "OPENING_FAMILY_DATABASE_URL",
        "CHESSBOOK_DATABASE_URL",
        "DATABASE_URL",
    ] {
        if let Ok(value) = std::env::var(key) {
            let value = clean_env_value(&value);
            if !value.is_empty() {
                return Some(value);
            }
        }
    }

    read_database_url_from_env_file(Path::new("../server/.env"))
        .or_else(|| read_database_url_from_env_file(Path::new("../server/.dev_secret.env")))
}

fn read_database_url_from_env_file(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    contents.lines().find_map(|line| {
        let line = line.trim();
        let value = line.strip_prefix("DATABASE_URL=")?;
        let value = clean_env_value(value);
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    })
}

fn clean_env_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn fen_to_epd(fen: &str) -> Option<String> {
    let fen: Fen = fen.parse().ok()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard).ok()?;
    Some(Epd::from_position(&pos, EnPassantMode::Legal).to_string())
}

fn opening_family_label(code: &str, name: &str) -> String {
    let base = name
        .split(':')
        .next()
        .unwrap_or(name)
        .split(',')
        .next()
        .unwrap_or(name)
        .trim();
    let base = if base.len() <= 3 { code } else { base };
    base.chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if ch.is_whitespace() || ch == '-' {
                Some(' ')
            } else {
                None
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
