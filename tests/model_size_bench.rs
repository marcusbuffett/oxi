//! Compares CPU inference latency of the production model architecture
//! (192d/8L/6H) against the new one (320d/16L/10H) at bot-realistic batch
//! sizes. Weights are random — latency does not depend on weight values.
//!
//! Run with:
//!   cargo test --release --features backend-tch --test model_size_bench -- --ignored --nocapture

use std::time::Instant;

use burn::prelude::*;
use oxi::config::{Config, FEATURES_PER_TOKEN, NUM_GLOBALS};
use oxi::model::OXIModel;

type B = burn_tch::LibTorch<f32>;

fn bench_config(label: &str, embed_dim: usize, num_layers: usize, num_heads: usize) -> Vec<f64> {
    let device = burn_tch::LibTorchDevice::Cpu;
    let config = Config {
        embed_dim,
        num_layers,
        num_heads,
        ..Default::default()
    };
    let _ = oxi::config::set_global_config(config.clone());
    let model = OXIModel::<B>::new(&device, &config);
    println!("{label} params={}", model.num_params());

    let mut results = Vec::new();
    for &batch in &[1usize, 16, 64] {
        let board = Tensor::<B, 3>::random(
            [batch, 64, FEATURES_PER_TOKEN],
            burn::tensor::Distribution::Default,
            &device,
        );
        let globals = Tensor::<B, 2>::random(
            [batch, NUM_GLOBALS],
            burn::tensor::Distribution::Default,
            &device,
        );

        // Warmup
        for _ in 0..3 {
            let _ = model.forward(board.clone(), globals.clone());
        }

        let iters = 10;
        let start = Instant::now();
        for _ in 0..iters {
            let (policy, _, _, _) = model.forward(board.clone(), globals.clone());
            // Force evaluation
            let _ = policy.into_data();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("{label} batch={batch}: {ms:.1} ms/forward");
        results.push(ms);
    }
    results
}

fn bench_policy_config(
    label: &str,
    embed_dim: usize,
    num_layers: usize,
    num_heads: usize,
) -> Vec<f64> {
    #[cfg(target_os = "macos")]
    let device = burn_tch::LibTorchDevice::Mps;
    #[cfg(not(target_os = "macos"))]
    let device = burn_tch::LibTorchDevice::Cpu;

    let config = Config {
        embed_dim,
        num_layers,
        num_heads,
        smolgen_hidden: (embed_dim / 10).max(8),
        smolgen_global_dim: (embed_dim / 2).max(64),
        smolgen_gen_size: (embed_dim / 2).max(64),
        ..Default::default()
    };
    let _ = oxi::config::set_global_config(config.clone());
    let model = OXIModel::<B>::new(&device, &config);
    println!("{label} params={}", model.num_params());

    let mut results = Vec::new();
    for &batch in &[1usize, 16, 64] {
        let board = Tensor::<B, 3>::random(
            [batch, 64, FEATURES_PER_TOKEN],
            burn::tensor::Distribution::Default,
            &device,
        );
        let globals = Tensor::<B, 2>::random(
            [batch, NUM_GLOBALS],
            burn::tensor::Distribution::Default,
            &device,
        );

        for _ in 0..5 {
            let _ = model
                .forward_policy(board.clone(), globals.clone())
                .into_data();
        }

        let iters = 20;
        let start = Instant::now();
        for _ in 0..iters {
            let policy = model.forward_policy(board.clone(), globals.clone());
            let _ = policy.into_data();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("{label} policy batch={batch}: {ms:.2} ms/forward");
        results.push(ms);
    }
    results
}

#[test]
#[ignore]
fn bench_old_vs_new_model_cpu() {
    let old = bench_config("old 192d/8L/6H ", 192, 8, 6);
    let new = bench_config("new 320d/16L/10H", 320, 16, 10);
    for (i, &batch) in [1, 16, 64].iter().enumerate() {
        println!("batch={batch}: ratio new/old = {:.2}x", new[i] / old[i]);
    }
}

#[test]
#[ignore]
fn bench_mini_depth_width_policy_only() {
    let cases = [
        ("full 320d/16L/10H", 320, 16, 10),
        ("mini 128d/6L/4H", 128, 6, 4),
        ("mini 160d/6L/5H", 160, 6, 5),
        ("wide 160d/3L/5H", 160, 3, 5),
        ("wide 192d/3L/6H", 192, 3, 6),
        ("thin 128d/3L/4H", 128, 3, 4),
    ];

    for (label, embed_dim, layers, heads) in cases {
        bench_policy_config(label, embed_dim, layers, heads);
    }
}
