use oxi::config::{
    elo_keep_probability, elo_keep_probability_with_boost, set_global_config, Config, MAX_ELO,
    MIN_ELO,
};
use pgn_reader::{BufferedReader, RawHeader, Skip, Visitor};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::PathBuf;

struct EloExtractor {
    white_elo: Option<i32>,
    black_elo: Option<i32>,
    games: Vec<(i32, i32)>,
}

impl EloExtractor {
    fn new() -> Self {
        Self {
            white_elo: None,
            black_elo: None,
            games: Vec::new(),
        }
    }
}

impl Visitor for EloExtractor {
    type Result = ();

    fn begin_game(&mut self) {
        self.white_elo = None;
        self.black_elo = None;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let Ok(value_str) = value.decode_utf8() else {
            return;
        };

        match key {
            b"WhiteElo" => {
                self.white_elo = value_str.parse::<i32>().ok();
            }
            b"BlackElo" => {
                self.black_elo = value_str.parse::<i32>().ok();
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        if let (Some(white), Some(black)) = (self.white_elo, self.black_elo) {
            self.games.push((white, black));
        }
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {}
}

fn bucket_elo(elo: i32) -> i32 {
    (elo / 100) * 100
}

fn enhanced_elo_keep_probability(avg_elo: f64, priority_boost: f64) -> f64 {
    let base_prob = elo_keep_probability(avg_elo);
    if avg_elo >= 2000.0 {
        let boost_factor = 1.0 + priority_boost * ((avg_elo - 2000.0) / 500.0).min(1.0);
        (base_prob * boost_factor).min(1.0)
    } else {
        base_prob
    }
}

#[test]
fn analyze_elo_distribution_from_lichess_pgn() {
    let config = Config::default();
    let _ = set_global_config(config);

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pgn_path = manifest_dir.join("../data/lichess_db_standard_rated_2023-01.pgn.zst");

    if !pgn_path.exists() {
        println!("Skipping test: PGN file not found at {:?}", pgn_path);
        println!("Expected file: lichess_db_standard_rated_2023-01.pgn.zst in ../data/");
        return;
    }

    println!("\n=== Analyzing ELO Distribution from Lichess PGN ===\n");
    println!("File: {:?}", pgn_path);

    let file = std::fs::File::open(&pgn_path).expect("Failed to open PGN file");
    let buf_reader = BufReader::with_capacity(1024 * 1024, file);
    let decoder = zstd::stream::read::Decoder::new(buf_reader).expect("Failed to create decoder");
    let mut reader = BufferedReader::new(decoder);

    let mut extractor = EloExtractor::new();
    let max_games = 100_000;

    println!("Reading up to {} games...\n", max_games);

    while extractor.games.len() < max_games {
        match reader.read_game(&mut extractor) {
            Ok(None) => break,
            Ok(Some(_)) => {}
            Err(_) => continue,
        }
    }

    let games = &extractor.games;
    println!("Total games analyzed: {}\n", games.len());

    let mut raw_distribution: HashMap<i32, usize> = HashMap::new();
    for (white, black) in games {
        let avg_elo = (*white + *black) / 2;
        let bucket = bucket_elo(avg_elo);
        *raw_distribution.entry(bucket).or_insert(0) += 1;
    }

    let mut sampled_distribution: HashMap<i32, f64> = HashMap::new();
    for (white, black) in games {
        let avg_elo = (*white + *black) as f64 / 2.0;
        let bucket = bucket_elo(avg_elo as i32);
        let keep_prob = elo_keep_probability(avg_elo);
        *sampled_distribution.entry(bucket).or_insert(0.0) += keep_prob;
    }

    let mut buckets: Vec<i32> = raw_distribution.keys().cloned().collect();
    buckets.sort();

    println!("=== RAW ELO Distribution (Average of WhiteElo + BlackElo) ===\n");
    println!(
        "{:>12} {:>10} {:>8} {:>12} {:>10} {:>12}",
        "ELO Range", "Raw Count", "Raw %", "Keep Prob", "Expected", "Expected %"
    );
    println!("{}", "-".repeat(70));

    let total_raw: usize = raw_distribution.values().sum();
    let total_sampled: f64 = sampled_distribution.values().sum();

    for bucket in &buckets {
        let raw_count = raw_distribution.get(bucket).unwrap_or(&0);
        let raw_pct = (*raw_count as f64 / total_raw as f64) * 100.0;
        let expected = sampled_distribution.get(bucket).unwrap_or(&0.0);
        let expected_pct = (*expected / total_sampled) * 100.0;
        let keep_prob = elo_keep_probability((*bucket + 50) as f64);

        println!(
            "{:>5}-{:<5} {:>10} {:>7.1}% {:>11.3} {:>11.1} {:>11.1}%",
            bucket,
            bucket + 99,
            raw_count,
            raw_pct,
            keep_prob,
            expected,
            expected_pct
        );
    }

    println!("\n{}", "-".repeat(70));
    println!(
        "{:>12} {:>10} {:>8} {:>12} {:>10.1}",
        "TOTAL", total_raw, "100.0%", "", total_sampled
    );

    println!("\n=== Focus: Advanced/Expert Ranges (2000+) ===\n");

    let advanced_raw: usize = buckets
        .iter()
        .filter(|b| **b >= 2000)
        .map(|b| raw_distribution.get(b).unwrap_or(&0))
        .sum();
    let advanced_sampled: f64 = buckets
        .iter()
        .filter(|b| **b >= 2000)
        .map(|b| *sampled_distribution.get(b).unwrap_or(&0.0))
        .sum();

    let advanced_raw_pct = (advanced_raw as f64 / total_raw as f64) * 100.0;
    let advanced_sampled_pct = (advanced_sampled / total_sampled) * 100.0;

    println!(
        "2000+ ELO games in raw data:      {} ({:.1}%)",
        advanced_raw, advanced_raw_pct
    );
    println!(
        "2000+ ELO games after sampling:   {:.0} ({:.1}%)",
        advanced_sampled, advanced_sampled_pct
    );
    println!(
        "Improvement factor:               {:.2}x",
        advanced_sampled_pct / advanced_raw_pct
    );

    println!("\n=== Keep Probability by ELO (current formula) ===\n");
    println!("{:>8} {:>12}", "ELO", "Keep Prob");
    println!("{}", "-".repeat(25));
    for elo in (1000..=2500).step_by(100) {
        let prob = elo_keep_probability(elo as f64);
        println!("{:>8} {:>11.4}", elo, prob);
    }

    println!("\n=== Proposed: Enhanced Priority for 2000+ ===\n");
    println!("Current formula flattens but doesn't BOOST high ELO.");
    println!("Suggestion: Add a priority multiplier for advanced ranges.\n");

    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "ELO", "Current", "Boost 2x", "Boost 4x"
    );
    println!("{}", "-".repeat(55));
    for elo in (1000..=2500).step_by(100) {
        let current = elo_keep_probability(elo as f64);
        let boost_2x = enhanced_elo_keep_probability(elo as f64, 1.0);
        let boost_4x = enhanced_elo_keep_probability(elo as f64, 3.0);
        println!(
            "{:>8} {:>11.4} {:>14.4} {:>14.4}",
            elo, current, boost_2x, boost_4x
        );
    }

    println!("\n=== Expected Distribution with 4x Boost for 2000+ ===\n");

    let mut boosted_distribution: HashMap<i32, f64> = HashMap::new();
    for (white, black) in games {
        let avg_elo = (*white + *black) as f64 / 2.0;
        let bucket = bucket_elo(avg_elo as i32);
        let keep_prob = enhanced_elo_keep_probability(avg_elo, 3.0);
        *boosted_distribution.entry(bucket).or_insert(0.0) += keep_prob;
    }

    let total_boosted: f64 = boosted_distribution.values().sum();

    println!(
        "{:>12} {:>10} {:>12} {:>12}",
        "ELO Range", "Current %", "Boosted %", "Change"
    );
    println!("{}", "-".repeat(50));

    for bucket in &buckets {
        let current = sampled_distribution.get(bucket).unwrap_or(&0.0);
        let current_pct = (*current / total_sampled) * 100.0;
        let boosted = boosted_distribution.get(bucket).unwrap_or(&0.0);
        let boosted_pct = (*boosted / total_boosted) * 100.0;
        let change = boosted_pct - current_pct;

        println!(
            "{:>5}-{:<5} {:>9.1}% {:>11.1}% {:>+11.1}%",
            bucket,
            bucket + 99,
            current_pct,
            boosted_pct,
            change
        );
    }

    let advanced_boosted: f64 = buckets
        .iter()
        .filter(|b| **b >= 2000)
        .map(|b| *boosted_distribution.get(b).unwrap_or(&0.0))
        .sum();
    let advanced_boosted_pct = (advanced_boosted / total_boosted) * 100.0;

    println!("\n2000+ ELO representation:");
    println!("  Current (no boost): {:.1}%", advanced_sampled_pct);
    println!("  With 4x boost: {:.1}%", advanced_boosted_pct);
}

#[test]
fn test_current_elo_sampling_curve() {
    let config = Config::default();
    let _ = set_global_config(config);

    println!("\n=== Current ELO Sampling Curve Analysis ===\n");

    println!("Constants:");
    println!("  MIN_ELO: {}", MIN_ELO);
    println!("  MAX_ELO: {}", MAX_ELO);
    println!("  Distribution mean: 1672");
    println!("  Distribution std: 404");
    println!("  Flattening factor: 0.05\n");

    println!("Effect of current formula:");
    println!("  - Games at 1000 ELO: heavily downsampled (graduated_prob near 0)");
    println!("  - Games at 1672 ELO (peak): moderately downsampled (flatten_prob ~0.05)");
    println!("  - Games at 2000+ ELO: kept at higher rate (flatten_prob approaches 1.0)\n");

    println!("Problem: While flattening helps, 2000+ games are still:");
    println!("  1. Rare in raw data (~5% of games)");
    println!("  2. Not actively PRIORITIZED, just less downsampled\n");

    println!("Proposed solution:");
    println!("  Add --elo-priority-boost flag to multiply keep probability for 2000+ games");
}
