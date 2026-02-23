#![cfg(feature = "bytecode")]

use echidna::{record, BReverse};
use num_traits::Float;
use std::path::Path;

/// Helper: compute gradient without checkpointing (record all steps into one tape).
fn grad_naive(
    step: impl Fn(&[BReverse<f64>]) -> Vec<BReverse<f64>>,
    loss: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>,
    x0: &[f64],
    num_steps: usize,
) -> Vec<f64> {
    let (mut tape, _) = record(
        |x| {
            let mut state: Vec<BReverse<f64>> = x.to_vec();
            for _ in 0..num_steps {
                state = step(&state);
            }
            loss(&state)
        },
        x0,
    );
    tape.gradient(x0)
}

#[test]
fn single_step() {
    let x0 = [2.0_f64, 3.0];
    let step = |x: &[BReverse<f64>]| vec![x[0] * x[1], x[0] + x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] + x[1];

    let g_naive = grad_naive(step, loss, &x0, 1);
    let g_ckpt = echidna::grad_checkpointed(step, loss, &x0, 1, 1);

    for i in 0..2 {
        assert!(
            (g_naive[i] - g_ckpt[i]).abs() < 1e-10,
            "mismatch at {}: naive={}, ckpt={}",
            i,
            g_naive[i],
            g_ckpt[i]
        );
    }
}

#[test]
fn linear_step() {
    // step(x) = 2*x (element-wise), loss(x) = sum(x)
    // After n steps: state = 2^n * x0
    // loss = sum(2^n * x0) = 2^n * sum(x0)
    // gradient = 2^n * [1, 1, ...]
    let x0 = [1.0_f64, 1.0];
    let num_steps = 5;
    let step = |x: &[BReverse<f64>]| {
        let two = BReverse::constant(2.0);
        x.iter().map(|&xi| xi * two).collect()
    };
    let loss = |x: &[BReverse<f64>]| {
        let mut s = x[0];
        for i in 1..x.len() {
            s = s + x[i];
        }
        s
    };

    let g = echidna::grad_checkpointed(step, loss, &x0, num_steps, 2);
    let expected = 2.0_f64.powi(num_steps as i32);

    for i in 0..2 {
        assert!(
            (g[i] - expected).abs() < 1e-8,
            "gradient[{}]: expected {}, got {}",
            i,
            expected,
            g[i]
        );
    }
}

#[test]
fn nonlinear_step_vs_finite_diff() {
    // step(x) = [sin(x[0])*x[1], x[0]+x[1]*x[1]]
    // Compare checkpointed gradient against finite differences
    let x0 = [0.5_f64, 0.3];
    let num_steps = 3;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g = echidna::grad_checkpointed(step, loss, &x0, num_steps, 2);

    // Finite difference
    let h = 1e-6;
    for i in 0..2 {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += h;
        xm[i] -= h;

        let fp = simulate_primal(&xp, num_steps);
        let fm = simulate_primal(&xm, num_steps);

        let lp = fp[0] * fp[0] + fp[1];
        let lm = fm[0] * fm[0] + fm[1];

        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (g[i] - fd).abs() < 1e-4,
            "gradient[{}]: ckpt={}, fd={}",
            i,
            g[i],
            fd
        );
    }
}

fn simulate_primal(x0: &[f64], num_steps: usize) -> Vec<f64> {
    let mut state = x0.to_vec();
    for _ in 0..num_steps {
        let new = vec![state[0].sin() * state[1], state[0] + state[1] * state[1]];
        state = new;
    }
    state
}

#[test]
fn checkpoint_count_independence() {
    // Gradient must be identical regardless of num_checkpoints
    let x0 = [0.5_f64, 0.3];
    let num_steps = 8;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];

    let g1 = echidna::grad_checkpointed(step, |x| x[0] + x[1], &x0, num_steps, 1);
    let g2 = echidna::grad_checkpointed(step, |x| x[0] + x[1], &x0, num_steps, 2);
    let g4 = echidna::grad_checkpointed(step, |x| x[0] + x[1], &x0, num_steps, 4);
    let gn = echidna::grad_checkpointed(step, |x| x[0] + x[1], &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!((g1[i] - g2[i]).abs() < 1e-10, "g1 vs g2 at {}", i);
        assert!((g1[i] - g4[i]).abs() < 1e-10, "g1 vs g4 at {}", i);
        assert!((g1[i] - gn[i]).abs() < 1e-10, "g1 vs gn at {}", i);
    }
}

#[test]
#[should_panic(expected = "step must preserve dimension")]
fn dimension_preservation_assert() {
    let x0 = [1.0_f64, 2.0];
    let step = |x: &[BReverse<f64>]| vec![x[0] + x[1]]; // Reduces dimension!
    let loss = |x: &[BReverse<f64>]| x[0];

    echidna::grad_checkpointed(step, loss, &x0, 2, 1);
}

#[test]
fn euler_step_ode() {
    // Simple ODE: dx/dt = -x, Euler step: x_{k+1} = x_k + h*(-x_k) = (1-h)*x_k
    // After n steps: x_n = (1-h)^n * x_0
    // loss = sum(x_n)
    // gradient = (1-h)^n * [1, 1]
    let x0 = [1.0_f64, 2.0];
    let h = 0.1;
    let num_steps = 10;

    let step = move |x: &[BReverse<f64>]| {
        let dt = BReverse::constant(h);
        x.iter().map(|&xi| xi - dt * xi).collect()
    };
    let loss = |x: &[BReverse<f64>]| {
        let mut s = x[0];
        for i in 1..x.len() {
            s = s + x[i];
        }
        s
    };

    let g = echidna::grad_checkpointed(step, loss, &x0, num_steps, 3);
    let expected = (1.0 - h).powi(num_steps as i32);

    for i in 0..2 {
        assert!(
            (g[i] - expected).abs() < 1e-8,
            "gradient[{}]: expected {}, got {}",
            i,
            expected,
            g[i]
        );
    }
}

#[test]
fn zero_steps() {
    // 0 steps: gradient of loss(x0) directly
    let x0 = [2.0_f64, 3.0];
    let step = |x: &[BReverse<f64>]| x.to_vec();
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1] * x[1];

    let g = echidna::grad_checkpointed(step, loss, &x0, 0, 1);
    // gradient of x^2 + y^2 = [2x, 2y] = [4, 6]
    assert!((g[0] - 4.0).abs() < 1e-10);
    assert!((g[1] - 6.0).abs() < 1e-10);
}

#[test]
fn matches_naive_multi_step() {
    // Verify checkpointed gradient matches naive (all-in-one-tape) gradient
    let x0 = [0.5_f64, 0.3];
    let num_steps = 6;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_naive = grad_naive(step, loss, &x0, num_steps);
    let g_ckpt = echidna::grad_checkpointed(step, loss, &x0, num_steps, 3);

    for i in 0..2 {
        assert!(
            (g_naive[i] - g_ckpt[i]).abs() < 1e-10,
            "mismatch at {}: naive={}, ckpt={}",
            i,
            g_naive[i],
            g_ckpt[i]
        );
    }
}

#[test]
fn revolve_vs_equal_spacing_identical_gradient() {
    // The gradient must be identical regardless of the internal scheduling
    // strategy. This test uses various checkpoint counts.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 12;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    // Reference: store-all (num_checkpoints >= num_steps)
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for ckpts in [1, 2, 3, 4, 6, 8, 11] {
        let g = echidna::grad_checkpointed(step, loss, &x0, num_steps, ckpts);
        for i in 0..2 {
            assert!(
                (g_ref[i] - g[i]).abs() < 1e-10,
                "gradient mismatch with {} ckpts at {}: ref={}, got={}",
                ckpts,
                i,
                g_ref[i],
                g[i]
            );
        }
    }
}

#[test]
fn large_step_count() {
    // Test with a larger step count to exercise revolve scheduling
    let x0 = [1.0_f64, 0.5];
    let num_steps = 50;

    let step = |x: &[BReverse<f64>]| {
        let half = BReverse::constant(0.5_f64);
        vec![
            x[0].sin() * half + x[1] * half,
            x[0] * half + x[1].cos() * half,
        ]
    };
    let loss = |x: &[BReverse<f64>]| x[0] + x[1];

    let g_all = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);
    let g_few = echidna::grad_checkpointed(step, loss, &x0, num_steps, 5);

    for i in 0..2 {
        assert!(
            (g_all[i] - g_few[i]).abs() < 1e-8,
            "large step mismatch at {}: all={}, few={}",
            i,
            g_all[i],
            g_few[i]
        );
    }
}

// ══════════════════════════════════════════════
//  Online checkpointing tests (R9a)
// ══════════════════════════════════════════════

#[test]
fn online_matches_offline() {
    let x0 = [0.5_f64, 0.3];
    let num_steps = 10;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_offline = echidna::grad_checkpointed(step, loss, &x0, num_steps, 4);
    let g_online = echidna::grad_checkpointed_online(
        step,
        |_, step_idx| step_idx >= num_steps,
        loss,
        &x0,
        4,
    );

    for i in 0..2 {
        assert!(
            (g_offline[i] - g_online[i]).abs() < 1e-10,
            "online vs offline mismatch at {}: offline={}, online={}",
            i,
            g_offline[i],
            g_online[i]
        );
    }
}

#[test]
fn online_convergence() {
    // Run until ||state|| < tolerance, then verify gradient vs finite diff.
    let x0 = [1.0_f64, 2.0];
    let h = 0.1;
    let tol = 0.5;

    let step = move |x: &[BReverse<f64>]| {
        let dt = BReverse::constant(h);
        x.iter().map(|&xi| xi - dt * xi).collect()
    };
    let stop = move |state: &[f64], _: usize| {
        let norm: f64 = state.iter().map(|&s| s * s).sum::<f64>().sqrt();
        norm < tol
    };
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1] * x[1];

    let g = echidna::grad_checkpointed_online(step, stop, loss, &x0, 4);

    // Compute how many steps until convergence.
    let mut state = x0.to_vec();
    let mut steps = 0;
    loop {
        state = state.iter().map(|&s| s * (1.0 - h)).collect();
        steps += 1;
        let norm: f64 = state.iter().map(|&s| s * s).sum::<f64>().sqrt();
        if norm < tol {
            break;
        }
    }

    // Finite diff verification.
    let eps = 1e-6;
    for i in 0..2 {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += eps;
        xm[i] -= eps;

        let factor = (1.0 - h).powi(steps as i32);
        let sp: Vec<f64> = xp.iter().map(|&v| v * factor).collect();
        let sm: Vec<f64> = xm.iter().map(|&v| v * factor).collect();
        let lp: f64 = sp.iter().map(|&v| v * v).sum();
        let lm: f64 = sm.iter().map(|&v| v * v).sum();
        let fd = (lp - lm) / (2.0 * eps);

        assert!(
            (g[i] - fd).abs() < 1e-4,
            "online convergence gradient[{}]: ad={}, fd={}",
            i,
            g[i],
            fd
        );
    }
}

#[test]
fn online_single_step() {
    let x0 = [2.0_f64, 3.0];
    let step = |x: &[BReverse<f64>]| vec![x[0] * x[1], x[0] + x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] + x[1];

    let g = echidna::grad_checkpointed_online(step, |_, step_idx| step_idx >= 1, loss, &x0, 2);
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, 1, 1);

    for i in 0..2 {
        assert!(
            (g[i] - g_ref[i]).abs() < 1e-10,
            "online single step mismatch at {}",
            i
        );
    }
}

#[test]
fn online_stop_at_zero() {
    // Stop predicate returns true at step 0 => gradient of loss(x0).
    let x0 = [2.0_f64, 3.0];
    let step = |x: &[BReverse<f64>]| x.to_vec();
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1] * x[1];

    let g = echidna::grad_checkpointed_online(step, |_, _| true, loss, &x0, 2);
    // gradient of x^2 + y^2 = [2x, 2y] = [4, 6]
    assert!((g[0] - 4.0).abs() < 1e-10);
    assert!((g[1] - 6.0).abs() < 1e-10);
}

#[test]
fn online_exact_fill() {
    // Step count exactly fills the buffer without triggering thinning.
    // With num_checkpoints=5 and spacing=1, buffer holds [0,1,2,3,4].
    // Stop at step 4 means 4 entries in buffer[1..] + buffer[0] = 5 total.
    // Buffer fills at exactly capacity, no thinning needed.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 4;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_online = echidna::grad_checkpointed_online(
        step,
        |_, step_idx| step_idx >= num_steps,
        loss,
        &x0,
        5,
    );
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!(
            (g_online[i] - g_ref[i]).abs() < 1e-10,
            "exact fill mismatch at {}: online={}, ref={}",
            i,
            g_online[i],
            g_ref[i]
        );
    }
}

#[test]
fn online_thinning_stress() {
    // Many steps with few checkpoints to exercise multiple thinning rounds.
    let x0 = [1.0_f64, 0.5];
    let num_steps = 200;

    let step = |x: &[BReverse<f64>]| {
        let half = BReverse::constant(0.5_f64);
        vec![
            x[0].sin() * half + x[1] * half,
            x[0] * half + x[1].cos() * half,
        ]
    };
    let loss = |x: &[BReverse<f64>]| x[0] + x[1];

    let g_online = echidna::grad_checkpointed_online(
        step,
        |_, step_idx| step_idx >= num_steps,
        loss,
        &x0,
        3,
    );
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!(
            (g_online[i] - g_ref[i]).abs() < 1e-8,
            "thinning stress mismatch at {}: online={}, ref={}",
            i,
            g_online[i],
            g_ref[i]
        );
    }
}

#[test]
#[should_panic(expected = "online checkpointing requires at least 2")]
fn online_panics_on_insufficient_checkpoints() {
    let x0 = [1.0_f64];
    echidna::grad_checkpointed_online(
        |x: &[BReverse<f64>]| x.to_vec(),
        |_, _| true,
        |x: &[BReverse<f64>]| x[0],
        &x0,
        1,
    );
}

// ══════════════════════════════════════════════
//  Checkpoint placement hints tests (R9c)
// ══════════════════════════════════════════════

#[test]
fn hints_matches_unhinted() {
    // Required positions that happen to match what Revolve would choose.
    // Result should be identical to the base grad_checkpointed.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 10;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_base = echidna::grad_checkpointed(step, loss, &x0, num_steps, 4);
    // The gradient must be correct regardless of which positions we require.
    let g_hints =
        echidna::grad_checkpointed_with_hints(step, loss, &x0, num_steps, 4, &[3, 6]);

    for i in 0..2 {
        assert!(
            (g_base[i] - g_hints[i]).abs() < 1e-10,
            "hints vs base mismatch at {}",
            i
        );
    }
}

#[test]
fn hints_single_required() {
    let x0 = [0.5_f64, 0.3];
    let num_steps = 8;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_hints =
        echidna::grad_checkpointed_with_hints(step, loss, &x0, num_steps, 4, &[4]);
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!(
            (g_ref[i] - g_hints[i]).abs() < 1e-10,
            "single required mismatch at {}",
            i
        );
    }
}

#[test]
fn hints_all_required() {
    // All positions required = store all = maximum accuracy.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 5;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let required: Vec<usize> = (1..num_steps).collect();
    let g_hints = echidna::grad_checkpointed_with_hints(
        step,
        loss,
        &x0,
        num_steps,
        num_steps,
        &required,
    );
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!(
            (g_ref[i] - g_hints[i]).abs() < 1e-10,
            "all required mismatch at {}",
            i
        );
    }
}

#[test]
fn hints_empty() {
    // Empty required list = equivalent to grad_checkpointed.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 8;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_hints =
        echidna::grad_checkpointed_with_hints(step, loss, &x0, num_steps, 3, &[]);
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, 3);

    for i in 0..2 {
        assert!(
            (g_ref[i] - g_hints[i]).abs() < 1e-10,
            "empty hints mismatch at {}",
            i
        );
    }
}

#[test]
fn hints_out_of_range() {
    // Positions at 0 and >= num_steps are silently ignored.
    let x0 = [0.5_f64, 0.3];
    let num_steps = 6;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let g_hints = echidna::grad_checkpointed_with_hints(
        step,
        loss,
        &x0,
        num_steps,
        3,
        &[0, 3, 6, 100],
    );
    // Only position 3 is valid; 0, 6, 100 are out of range.
    let g_ref = echidna::grad_checkpointed(step, loss, &x0, num_steps, num_steps);

    for i in 0..2 {
        assert!(
            (g_ref[i] - g_hints[i]).abs() < 1e-10,
            "out of range hints mismatch at {}",
            i
        );
    }
}

#[test]
#[should_panic(expected = "required positions")]
fn hints_panics_on_too_many_required() {
    let x0 = [1.0_f64];
    echidna::grad_checkpointed_with_hints(
        |x: &[BReverse<f64>]| x.to_vec(),
        |x: &[BReverse<f64>]| x[0],
        &x0,
        10,
        2,
        &[1, 2, 3], // 3 required > 2 checkpoints
    );
}

// ══════════════════════════════════════════════
//  Disk-backed checkpointing tests (R9b)
// ══════════════════════════════════════════════

#[test]
fn disk_matches_memory() {
    let x0 = [0.5_f64, 0.3];
    let num_steps = 10;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] * x[0] + x[1];

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let g_disk =
        echidna::grad_checkpointed_disk(step, loss, &x0, num_steps, 3, dir.path());
    let g_mem = echidna::grad_checkpointed(step, loss, &x0, num_steps, 3);

    for i in 0..2 {
        assert!(
            (g_disk[i] - g_mem[i]).abs() < 1e-10,
            "disk vs memory mismatch at {}: disk={}, mem={}",
            i,
            g_disk[i],
            g_mem[i]
        );
    }
}

#[test]
fn disk_cleanup() {
    let x0 = [0.5_f64, 0.3];
    let num_steps = 5;

    let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
    let loss = |x: &[BReverse<f64>]| x[0] + x[1];

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let _ = echidna::grad_checkpointed_disk(step, loss, &x0, num_steps, 3, dir.path());

    // Verify no checkpoint files remain.
    let remaining: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("ckpt_"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        remaining.is_empty(),
        "checkpoint files not cleaned up: {:?}",
        remaining.iter().map(|e| e.file_name()).collect::<Vec<_>>()
    );
}

#[test]
fn disk_cleanup_on_panic() {
    let x0 = [0.5_f64, 0.3];
    let num_steps = 5;

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let dir_path = dir.path().to_path_buf();

    // Use catch_unwind to test panic cleanup.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let step = |x: &[BReverse<f64>]| vec![x[0].sin() * x[1], x[0] + x[1] * x[1]];
        let loss = |_x: &[BReverse<f64>]| {
            panic!("intentional panic in loss");
        };
        echidna::grad_checkpointed_disk(step, loss, &x0, num_steps, 3, &dir_path);
    }));

    assert!(result.is_err(), "should have panicked");

    // Verify checkpoint files are cleaned up despite panic.
    let remaining: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("ckpt_"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        remaining.is_empty(),
        "checkpoint files not cleaned up after panic: {:?}",
        remaining.iter().map(|e| e.file_name()).collect::<Vec<_>>()
    );
}

#[test]
fn disk_large_state() {
    // 1000-dim state to verify correctness with large vectors.
    let dim = 1000;
    let x0: Vec<f64> = (0..dim).map(|i| 0.001 * i as f64).collect();
    let num_steps = 5;

    let step = |x: &[BReverse<f64>]| {
        let scale = BReverse::constant(0.99_f64);
        x.iter().map(|&xi| xi * scale).collect()
    };
    let loss = |x: &[BReverse<f64>]| {
        let mut s = BReverse::constant(0.0_f64);
        for &xi in x {
            s = s + xi * xi;
        }
        s
    };

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let g_disk =
        echidna::grad_checkpointed_disk(step, loss, &x0, num_steps, 3, dir.path());
    let g_mem = echidna::grad_checkpointed(step, loss, &x0, num_steps, 3);

    for i in 0..dim {
        assert!(
            (g_disk[i] - g_mem[i]).abs() < 1e-10,
            "disk large state mismatch at {}: disk={}, mem={}",
            i,
            g_disk[i],
            g_mem[i]
        );
    }
}

#[test]
#[should_panic(expected = "checkpoint directory does not exist")]
fn disk_panics_on_missing_dir() {
    let x0 = [1.0_f64];
    echidna::grad_checkpointed_disk(
        |x: &[BReverse<f64>]| x.to_vec(),
        |x: &[BReverse<f64>]| x[0],
        &x0,
        1,
        1,
        Path::new("/nonexistent/dir/that/does/not/exist"),
    );
}
