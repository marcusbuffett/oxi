use burn::lr_scheduler::LrScheduler;
use oxi::cosine_scheduler::CosineAnnealingWithWarmup;

#[test]
fn test_warmup_linear_increase() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    // Test warmup phase - should increase linearly
    // Note: step() increments internal counter first, then calculates LR
    // So first call to step() uses internal_step=1

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);
    let lr_1 = scheduler.step(); // internal step = 1
    assert!(
        (lr_1 - max_lr * 0.01).abs() < 0.00001,
        "lr_1={}, expected ~{}",
        lr_1,
        max_lr * 0.01
    );

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);
    for _ in 0..24 {
        scheduler.step();
    }
    let lr_25 = scheduler.step(); // internal step = 25
    assert!(
        (lr_25 - max_lr * 0.25).abs() < 0.00001,
        "lr_25={}, expected ~{}",
        lr_25,
        max_lr * 0.25
    );

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);
    for _ in 0..49 {
        scheduler.step();
    }
    let lr_50 = scheduler.step(); // internal step = 50
    assert!(
        (lr_50 - max_lr * 0.50).abs() < 0.00001,
        "lr_50={}, expected ~{}",
        lr_50,
        max_lr * 0.50
    );

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);
    for _ in 0..99 {
        scheduler.step();
    }
    let lr_100 = scheduler.step(); // internal step = 100
    assert!(
        (lr_100 - max_lr).abs() < 0.00001,
        "lr_100={}, expected {}",
        lr_100,
        max_lr
    );
}

#[test]
fn test_cosine_annealing_after_warmup() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Skip warmup
    for _ in 0..warmup_steps {
        scheduler.step();
    }

    let lr_after_warmup = scheduler.step();

    // After warmup, should be close to max_lr and start decreasing
    assert!(
        lr_after_warmup < max_lr,
        "lr_after_warmup={}, should be < max_lr={}",
        lr_after_warmup,
        max_lr
    );
    assert!(
        lr_after_warmup > max_lr * 0.98,
        "lr_after_warmup={}, should be close to max_lr={}",
        lr_after_warmup,
        max_lr
    );
}

#[test]
fn test_reaches_min_lr_at_end() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Step through all training
    let mut final_lr = 0.0;
    for _ in 0..total_steps {
        final_lr = scheduler.step();
    }

    // At the end, should be close to min_lr
    assert!(
        (final_lr - min_lr).abs() < 0.000001,
        "final_lr={}, expected ~{}",
        final_lr,
        min_lr
    );
}

#[test]
fn test_lr_monotonic_increase_during_warmup() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    let mut prev_lr = 0.0;
    for _ in 0..warmup_steps {
        let lr = scheduler.step();
        assert!(
            lr > prev_lr,
            "LR should increase during warmup: {} <= {}",
            lr,
            prev_lr
        );
        prev_lr = lr;
    }
}

#[test]
fn test_lr_monotonic_decrease_during_annealing() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Skip warmup
    for _ in 0..warmup_steps {
        scheduler.step();
    }

    // During annealing, LR should decrease
    let mut prev_lr = scheduler.step();
    for _ in (warmup_steps + 1)..total_steps {
        let lr = scheduler.step();
        assert!(
            lr < prev_lr,
            "LR should decrease during annealing: {} >= {}",
            lr,
            prev_lr
        );
        prev_lr = lr;
    }
}

#[test]
fn test_with_gradient_accumulation_scenario() {
    // Simulating a real training scenario with gradient accumulation
    let grad_accumulation_steps = 4;
    let warmup_optimizer_steps = 2000; // User wants 2000 optimizer steps of warmup
    let total_batches = 12000;

    // Scheduler configuration (as in custom_training.rs)
    let warmup_batches = warmup_optimizer_steps * grad_accumulation_steps; // 8000 batches
    let max_lr = 0.0001;
    let min_lr = 0.000001;

    let mut scheduler =
        CosineAnnealingWithWarmup::new(warmup_batches, total_batches, max_lr, min_lr);

    // Step through batches
    let mut lr_at_warmup_end = 0.0;
    for i in 0..total_batches {
        let lr = scheduler.step();

        if i + 1 == warmup_batches {
            lr_at_warmup_end = lr;
        }
    }

    // At the end of warmup (8000 batches), should be at max_lr
    assert!(
        (lr_at_warmup_end - max_lr).abs() < 0.000001,
        "lr_at_warmup_end={}, expected {}",
        lr_at_warmup_end,
        max_lr
    );
}

#[test]
fn test_no_warmup() {
    let warmup_steps = 0;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // First step should already be in annealing phase
    let lr_0 = scheduler.step();
    assert!(
        lr_0 < max_lr,
        "With no warmup, first LR should be less than max_lr"
    );
    assert!(lr_0 > min_lr, "First LR should be greater than min_lr");
}

#[test]
fn test_warmup_equals_total_steps() {
    // Edge case: warmup takes the entire training
    let warmup_steps = 1000;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Should just do warmup for all steps
    let mut prev_lr = 0.0;
    for _ in 0..warmup_steps {
        let lr = scheduler.step();
        assert!(
            lr > prev_lr,
            "LR should increase throughout when warmup = total_steps"
        );
        prev_lr = lr;
    }

    // Final LR should be close to max_lr
    assert!(
        (prev_lr - max_lr).abs() < 0.000001,
        "final_lr={}, expected {}",
        prev_lr,
        max_lr
    );
}

#[test]
fn test_scheduler_range_bounds() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Check all LRs are within bounds
    for _ in 0..total_steps {
        let lr = scheduler.step();
        assert!(lr >= 0.0, "LR should never be negative: {}", lr);
        assert!(
            lr <= max_lr * 1.01,
            "LR should not exceed max_lr: {} > {}",
            lr,
            max_lr
        );
        // Allow some tolerance for floating point
        assert!(
            lr >= min_lr * 0.99 || lr == 0.0,
            "LR should not go below min_lr: {} < {}",
            lr,
            min_lr
        );
    }
}

#[test]
fn test_steps_beyond_total() {
    let warmup_steps = 100;
    let total_steps = 1000;
    let max_lr = 0.001;
    let min_lr = 0.00001;

    let mut scheduler = CosineAnnealingWithWarmup::new(warmup_steps, total_steps, max_lr, min_lr);

    // Step beyond total_steps
    for _ in 0..total_steps {
        scheduler.step();
    }

    // Continue stepping - should stay at min_lr
    let lr_1 = scheduler.step();
    let lr_2 = scheduler.step();
    let lr_3 = scheduler.step();

    assert!(
        (lr_1 - min_lr).abs() < 0.000001,
        "After total_steps, LR should stay at min_lr"
    );
    assert!(
        (lr_2 - min_lr).abs() < 0.000001,
        "After total_steps, LR should stay at min_lr"
    );
    assert!(
        (lr_3 - min_lr).abs() < 0.000001,
        "After total_steps, LR should stay at min_lr"
    );
}

#[test]
fn test_realistic_training_scenario() {
    // Realistic scenario: 100k training samples, batch size 8, grad accumulation 4
    // Effective batch size: 32
    // Total batches: 100k / 8 = 12,500
    // Optimizer steps: 12,500 / 4 = 3,125
    // Warmup: 500 optimizer steps = 2,000 batches

    let physical_batch_size = 8;
    let grad_accumulation = 4;
    let train_size = 100_000;
    let warmup_optimizer_steps = 500;

    let total_batches = train_size / physical_batch_size; // 12,500
    let warmup_batches = warmup_optimizer_steps * grad_accumulation; // 2,000

    let max_lr = 0.0001;
    let min_lr = 0.000001;

    let mut scheduler =
        CosineAnnealingWithWarmup::new(warmup_batches, total_batches, max_lr, min_lr);

    // Simulate training
    let mut optimizer_step_count = 0;
    let mut lr_at_each_optimizer_step = Vec::new();

    for batch_idx in 0..total_batches {
        let lr = scheduler.step();

        // Record LR at each optimizer step
        if (batch_idx + 1) % grad_accumulation == 0 {
            optimizer_step_count += 1;
            lr_at_each_optimizer_step.push((optimizer_step_count, lr));
        }
    }

    // Verify warmup: at step 500, should be at max_lr
    let (step, lr) = lr_at_each_optimizer_step[499];
    assert_eq!(step, 500);
    assert!(
        (lr - max_lr).abs() < 0.000001,
        "After warmup, lr={}, expected {}",
        lr,
        max_lr
    );

    // Verify annealing: LR should decrease after warmup
    let (_, lr_501) = lr_at_each_optimizer_step[500];
    assert!(lr_501 < max_lr, "After warmup, LR should start decreasing");

    // Verify final LR is close to min_lr
    let (final_step, final_lr) = lr_at_each_optimizer_step.last().unwrap();
    assert_eq!(*final_step, 3125);
    assert!(
        (final_lr - min_lr).abs() < 0.00001,
        "Final lr={}, expected ~{}",
        final_lr,
        min_lr
    );
}
