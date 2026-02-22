#![cfg(feature = "bytecode")]

use echidna::{record, BReverse};
use num_traits::Float;

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
