use anyhow::{Context, Result};
use bincode::{Decode, Encode};
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use crate::eval_dataset::SampledPosition;
use crate::move_encoding::encode_move;
use crate::stockfish::StockfishEngine;
use shakmaty::uci::UciMove;

const REGRET_BINS_CP: [f32; 7] = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0];
pub const CALIBRATION_ELO_BUCKETS: [(i32, i32); 8] = [
    (1000, 1200),
    (1200, 1400),
    (1400, 1600),
    (1600, 1800),
    (1800, 2000),
    (2000, 2200),
    (2200, 2400),
    (2400, 2700),
];

pub fn calibration_elo_bucket_label(elo: i32) -> &'static str {
    for (idx, (elo_min, elo_max)) in CALIBRATION_ELO_BUCKETS.iter().enumerate() {
        if elo >= *elo_min && elo < *elo_max {
            return match idx {
                0 => "1000-1200",
                1 => "1200-1400",
                2 => "1400-1600",
                3 => "1600-1800",
                4 => "1800-2000",
                5 => "2000-2200",
                6 => "2200-2400",
                7 => "2400-2700",
                _ => unreachable!(),
            };
        }
    }

    if elo < CALIBRATION_ELO_BUCKETS[0].0 {
        "under-1000"
    } else {
        "2700+"
    }
}

pub fn calibration_skill_band_label(elo: i32) -> &'static str {
    if elo < 1400 {
        "Beginner"
    } else if elo < 2000 {
        "Intermediate"
    } else {
        "Expert"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStage {
    Opening,
    Middlegame,
    Endgame,
}

impl GameStage {
    pub fn from_ply(ply: u32) -> Self {
        if ply < 20 {
            Self::Opening
        } else if ply < 50 {
            Self::Middlegame
        } else {
            Self::Endgame
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Middlegame => "middlegame",
            Self::Endgame => "endgame",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum RegretBin {
    ExactZero,
    Cp1To10,
    Cp11To25,
    Cp26To50,
    Cp51To100,
    Cp101To200,
    Cp201To400,
    Cp400Plus,
}

impl RegretBin {
    pub const COUNT: usize = 8;

    pub fn from_regret_cp(regret_cp: f32) -> Self {
        let regret_cp = regret_cp.max(0.0);
        if regret_cp <= REGRET_BINS_CP[0] {
            Self::ExactZero
        } else if regret_cp <= REGRET_BINS_CP[1] {
            Self::Cp1To10
        } else if regret_cp <= REGRET_BINS_CP[2] {
            Self::Cp11To25
        } else if regret_cp <= REGRET_BINS_CP[3] {
            Self::Cp26To50
        } else if regret_cp <= REGRET_BINS_CP[4] {
            Self::Cp51To100
        } else if regret_cp <= REGRET_BINS_CP[5] {
            Self::Cp101To200
        } else if regret_cp <= REGRET_BINS_CP[6] {
            Self::Cp201To400
        } else {
            Self::Cp400Plus
        }
    }

    pub fn index(self) -> i64 {
        match self {
            Self::ExactZero => 0,
            Self::Cp1To10 => 1,
            Self::Cp11To25 => 2,
            Self::Cp26To50 => 3,
            Self::Cp51To100 => 4,
            Self::Cp101To200 => 5,
            Self::Cp201To400 => 6,
            Self::Cp400Plus => 7,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ExactZero => "0",
            Self::Cp1To10 => "1-10",
            Self::Cp11To25 => "11-25",
            Self::Cp26To50 => "26-50",
            Self::Cp51To100 => "51-100",
            Self::Cp101To200 => "101-200",
            Self::Cp201To400 => "201-400",
            Self::Cp400Plus => "400+",
        }
    }

    pub fn representative_cp(self) -> f32 {
        match self {
            Self::ExactZero => 0.0,
            Self::Cp1To10 => 5.0,
            Self::Cp11To25 => 18.0,
            Self::Cp26To50 => 38.0,
            Self::Cp51To100 => 75.0,
            Self::Cp101To200 => 150.0,
            Self::Cp201To400 => 300.0,
            Self::Cp400Plus => 600.0,
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct SparseMoveLossEntry {
    pub uci_move: String,
    pub regret_cp: f32,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct SparseMoveLossBlob {
    pub entries: Vec<SparseMoveLossEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CalibrationKey {
    pub fen: String,
    pub human_move: String,
}

#[derive(Debug, Clone)]
pub struct CalibrationLabel {
    pub human_regret_cp: f32,
    pub regret_bin: RegretBin,
    pub move_loss_blob: SparseMoveLossBlob,
}

#[derive(Debug, Clone)]
pub struct LabeledPosition {
    pub sampled: SampledPosition,
    pub stage: GameStage,
    pub best_eval_cp: i32,
    pub human_regret_cp: f32,
    pub regret_bin: RegretBin,
    pub move_loss_blob: SparseMoveLossBlob,
}

pub struct CalibrationDb {
    conn: Connection,
}

impl CalibrationDb {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open calibration DB at {}", path.display()))?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS labeled_positions (
                id INTEGER PRIMARY KEY,
                fen TEXT NOT NULL,
                human_move TEXT NOT NULL,
                player_elo INTEGER NOT NULL,
                opponent_elo INTEGER NOT NULL,
                ply INTEGER NOT NULL,
                stage TEXT NOT NULL,
                game_result REAL NOT NULL,
                stockfish_depth INTEGER NOT NULL,
                best_eval_cp INTEGER NOT NULL,
                human_regret_cp REAL NOT NULL,
                regret_bin INTEGER NOT NULL,
                move_loss_blob BLOB NOT NULL,
                created_at_unix INTEGER NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_labeled_positions_unique
            ON labeled_positions (fen, human_move, player_elo, opponent_elo, ply, stockfish_depth);

            CREATE INDEX IF NOT EXISTS idx_labeled_positions_elo_ply
            ON labeled_positions (player_elo, ply);

            CREATE INDEX IF NOT EXISTS idx_labeled_positions_stage_regret
            ON labeled_positions (stage, regret_bin);
            ",
        )?;
        Ok(())
    }

    pub fn insert_labeled_position(
        &self,
        position: &LabeledPosition,
        stockfish_depth: u32,
    ) -> Result<()> {
        let blob = serialize_move_loss_blob(&position.move_loss_blob)?;
        self.conn.execute(
            "
            INSERT OR REPLACE INTO labeled_positions (
                fen,
                human_move,
                player_elo,
                opponent_elo,
                ply,
                stage,
                game_result,
                stockfish_depth,
                best_eval_cp,
                human_regret_cp,
                regret_bin,
                move_loss_blob,
                created_at_unix
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, unixepoch())
            ",
            params![
                position.sampled.fen,
                position.sampled.human_move,
                position.sampled.player_elo,
                position.sampled.opponent_elo,
                position.sampled.ply,
                position.stage.as_str(),
                position.sampled.game_result,
                stockfish_depth as i64,
                position.best_eval_cp,
                position.human_regret_cp,
                position.regret_bin.index(),
                blob,
            ],
        )?;
        Ok(())
    }

    pub fn count_positions(&self) -> Result<i64> {
        let count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM labeled_positions", [], |row| {
                    row.get(0)
                })?;
        Ok(count)
    }

    pub fn load_existing_keys(&self, stockfish_depth: u32) -> Result<HashSet<CalibrationKey>> {
        let mut stmt = self.conn.prepare(
            "
            SELECT
                fen,
                human_move,
                player_elo,
                opponent_elo,
                ply
            FROM labeled_positions
            WHERE stockfish_depth = ?1
            ",
        )?;

        let rows = stmt.query_map(params![stockfish_depth as i64], |row| {
            Ok(CalibrationKey {
                fen: row.get(0)?,
                human_move: row.get(1)?,
            })
        })?;

        let mut keys = HashSet::new();
        for row in rows {
            keys.insert(row?);
        }
        Ok(keys)
    }

    pub fn load_lookup(path: &Path) -> Result<Arc<HashMap<CalibrationKey, CalibrationLabel>>> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open calibration DB at {}", path.display()))?;
        let mut stmt = conn.prepare(
            "
            SELECT
                fen,
                human_move,
                player_elo,
                opponent_elo,
                ply,
                human_regret_cp,
                regret_bin,
                move_loss_blob
            FROM labeled_positions
            ",
        )?;

        let rows = stmt.query_map([], |row| {
            let compressed: Vec<u8> = row.get(7)?;
            let blob = deserialize_move_loss_blob(&compressed).map_err(|err| {
                rusqlite::Error::FromSqlConversionFailure(
                    compressed.len(),
                    rusqlite::types::Type::Blob,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        err.to_string(),
                    )),
                )
            })?;
            Ok((
                CalibrationKey {
                    fen: row.get(0)?,
                    human_move: row.get(1)?,
                },
                CalibrationLabel {
                    human_regret_cp: row.get(5)?,
                    regret_bin: regret_bin_from_index(row.get::<_, i64>(6)?),
                    move_loss_blob: blob,
                },
            ))
        })?;

        let mut map = HashMap::new();
        for row in rows {
            let (key, value) = row?;
            map.insert(key, value);
        }
        Ok(Arc::new(map))
    }
}

pub fn label_sampled_position(
    engine: &mut StockfishEngine,
    sampled: SampledPosition,
    depth: u32,
) -> Result<LabeledPosition> {
    let eval = engine.evaluate_position_at_depth(&sampled.fen, depth)?;
    let best_eval_cp = eval.best_eval_cp();
    let move_loss_blob = SparseMoveLossBlob {
        entries: eval
            .move_evals
            .iter()
            .map(|entry| SparseMoveLossEntry {
                uci_move: entry.uci_move.clone(),
                regret_cp: (best_eval_cp - entry.score_cp).max(0) as f32,
            })
            .collect(),
    };

    let human_regret_cp = eval.ecl_of_move(&sampled.human_move).with_context(|| {
        format!(
            "Human move {} missing from Stockfish eval for position {}",
            sampled.human_move, sampled.fen
        )
    })?;

    Ok(LabeledPosition {
        stage: GameStage::from_ply(sampled.ply),
        best_eval_cp,
        regret_bin: RegretBin::from_regret_cp(human_regret_cp),
        human_regret_cp,
        move_loss_blob,
        sampled,
    })
}

fn serialize_move_loss_blob(blob: &SparseMoveLossBlob) -> Result<Vec<u8>> {
    let encoded = bincode::encode_to_vec(blob, bincode::config::standard())
        .context("Failed to bincode-encode move loss blob")?;
    let compressed =
        zstd::encode_all(Cursor::new(encoded), 3).context("Failed to compress move loss blob")?;
    Ok(compressed)
}

fn deserialize_move_loss_blob(bytes: &[u8]) -> Result<SparseMoveLossBlob> {
    let decompressed =
        zstd::decode_all(Cursor::new(bytes)).context("Failed to decompress move loss blob")?;
    let (blob, _): (SparseMoveLossBlob, usize) =
        bincode::decode_from_slice(&decompressed, bincode::config::standard())
            .context("Failed to decode move loss blob")?;
    Ok(blob)
}

fn regret_bin_from_index(index: i64) -> RegretBin {
    match index {
        0 => RegretBin::ExactZero,
        1 => RegretBin::Cp1To10,
        2 => RegretBin::Cp11To25,
        3 => RegretBin::Cp26To50,
        4 => RegretBin::Cp51To100,
        5 => RegretBin::Cp101To200,
        6 => RegretBin::Cp201To400,
        _ => RegretBin::Cp400Plus,
    }
}

pub fn calibration_key_for_sample(fen: &str, human_move: &str) -> CalibrationKey {
    CalibrationKey {
        fen: fen.to_string(),
        human_move: human_move.to_string(),
    }
}

pub fn dense_move_losses(blob: &SparseMoveLossBlob, legal_moves: usize) -> Vec<f32> {
    let mut dense = vec![0.0; legal_moves];
    for entry in &blob.entries {
        let Ok(uci_move) = entry.uci_move.parse::<UciMove>() else {
            continue;
        };
        let Some((from_idx, to_idx)) = encode_move(&uci_move) else {
            continue;
        };
        let move_idx = from_idx as usize * 76 + to_idx as usize;
        if move_idx < dense.len() {
            dense[move_idx] = entry.regret_cp.max(0.0);
        }
    }
    dense
}
