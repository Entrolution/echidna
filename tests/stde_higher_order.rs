#![cfg(feature = "stde")]

use approx::assert_relative_eq;
use echidna::{BReverse, BytecodeTape, Scalar};

fn record_fn(f: impl FnOnce(&[BReverse<f64>]) -> BReverse<f64>, x: &[f64]) -> BytecodeTape<f64> {
    let (tape, _) = echidna::record(f, x);
    tape
}

// ══════════════════════════════════════════════
//  Test functions
// ══════════════════════════════════════════════

/// f(x,y) = x^2 + y^2
fn sum_of_squares<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] + x[1] * x[1]
}

/// f(x,y,z) = x^2*y + y^3
fn cubic_mix<T: Scalar>(x: &[T]) -> T {
    x[0] * x[0] * x[1] + x[1] * x[1] * x[1]
}

/// f(x) = exp(x) — all derivatives equal exp(x).
fn exp_1d<T: Scalar>(x: &[T]) -> T {
    x[0].exp()
}

/// f(x,y) = x^4 + y^4
fn quartic<T: Scalar>(x: &[T]) -> T {
    let x0 = x[0];
    let y0 = x[1];
    x0 * x0 * x0 * x0 + y0 * y0 * y0 * y0
}

// ══════════════════════════════════════════════
//  17. Higher-order diagonal estimation
// ══════════════════════════════════════════════

#[test]
fn diagonal_kth_order_exp() {
    // ∂^k(exp(x))/∂x^k = exp(x) for k=2,3,4,5
    let tape = record_fn(exp_1d, &[1.0]);
    let expected = 1.0_f64.exp();

    for k in 2..=5 {
        let (val, diag) = echidna::stde::diagonal_kth_order(&tape, &[1.0], k);
        assert_relative_eq!(val, expected, epsilon = 1e-10);
        assert_eq!(diag.len(), 1);
        // Tolerance relaxes with k (higher-order coefficients accumulate error)
        let tol = 10.0_f64.powi(-(12 - k as i32));
        assert_relative_eq!(diag[0], expected, epsilon = tol);
    }
}

#[test]
fn diagonal_kth_order_polynomial() {
    // f(x,y) = x^4 + y^4
    // ∂^4f/∂x^4 = 24, ∂^4f/∂y^4 = 24
    // ∂^5f/∂x^5 = 0, ∂^5f/∂y^5 = 0
    let tape = record_fn(quartic, &[2.0, 3.0]);

    let (_, diag4) = echidna::stde::diagonal_kth_order(&tape, &[2.0, 3.0], 4);
    assert_eq!(diag4.len(), 2);
    assert_relative_eq!(diag4[0], 24.0, epsilon = 1e-6);
    assert_relative_eq!(diag4[1], 24.0, epsilon = 1e-6);

    let (_, diag5) = echidna::stde::diagonal_kth_order(&tape, &[2.0, 3.0], 5);
    assert_eq!(diag5.len(), 2);
    assert_relative_eq!(diag5[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(diag5[1], 0.0, epsilon = 1e-4);
}

#[test]
fn diagonal_kth_order_matches_hessian_diagonal() {
    // k=2 case must match existing hessian_diagonal
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_hd, diag_hd) = echidna::stde::hessian_diagonal(&tape, &x);
    let (val_dk, diag_dk) = echidna::stde::diagonal_kth_order(&tape, &x, 2);

    assert_relative_eq!(val_hd, val_dk, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_hd[j], diag_dk[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_kth_order_stochastic_full_sample() {
    // Full sample (all indices) should give exact sum
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (_, diag) = echidna::stde::diagonal_kth_order(&tape, &x, 4);
    let exact_sum: f64 = diag.iter().sum(); // 24 + 24 = 48

    let all_indices: Vec<usize> = (0..x.len()).collect();
    let result = echidna::stde::diagonal_kth_order_stochastic(&tape, &x, 4, &all_indices);

    assert_relative_eq!(result.estimate, exact_sum, epsilon = 1e-4);
}

#[test]
fn diagonal_kth_order_stochastic_scaling() {
    // Subset estimate should have n/|J| scaling
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    // Sample only index 0: estimate = n/|J| * mean = 2/1 * 24 = 48
    let result = echidna::stde::diagonal_kth_order_stochastic(&tape, &x, 4, &[0]);
    assert_relative_eq!(result.estimate, 48.0, epsilon = 1e-4);
}

#[test]
#[should_panic(expected = "k must be >= 2")]
fn diagonal_kth_order_k_too_small() {
    let tape = record_fn(exp_1d, &[1.0]);
    let _ = echidna::stde::diagonal_kth_order(&tape, &[1.0], 1);
}

#[test]
#[should_panic(expected = "k must be <= 18")]
fn diagonal_kth_order_k_too_large() {
    // k=19 is the first rejected order: 19! exceeds 2^53 and is inexact in f64.
    let tape = record_fn(exp_1d, &[1.0]);
    let _ = echidna::stde::diagonal_kth_order(&tape, &[1.0], 19);
}

// ══════════════════════════════════════════════
//  18. Parabolic diffusion
// ══════════════════════════════════════════════

#[test]
fn parabolic_diffusion_identity_sigma() {
    // σ=I reduces to standard Laplacian: ½ tr(I · H · I) = ½ tr(H)
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let e0: Vec<f64> = vec![1.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0];
    let cols: Vec<&[f64]> = vec![&e0, &e1];

    let (value, diffusion) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    assert_relative_eq!(value, 5.0, epsilon = 1e-10);
    // H = [[2, 0], [0, 2]], tr(H) = 4, ½ tr(H) = 2
    assert_relative_eq!(diffusion, 2.0, epsilon = 1e-10);
}

#[test]
fn parabolic_diffusion_diagonal_sigma() {
    // σ = diag(a₁, a₂): ½ tr(σσ^T H) = ½ Σ a_i² ∂²u/∂x_i²
    // f(x,y) = x²y + y³ at (1,2): H diag = [2y, 6y] = [4, 12]
    // σ = diag(2, 3): ½(4*4 + 9*12) = ½(16 + 108) = 62
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    // σ = diag(2, 3, 0.5) — columns of diagonal matrix
    let c0: Vec<f64> = vec![2.0, 0.0, 0.0];
    let c1: Vec<f64> = vec![0.0, 3.0, 0.0];
    let c2: Vec<f64> = vec![0.0, 0.0, 0.5];
    let cols: Vec<&[f64]> = vec![&c0, &c1, &c2];

    let (_, diffusion) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    // H diag = [4, 12, 0], σ = diag(2,3,0.5)
    // ½(4*4 + 9*12 + 0.25*0) = ½(16 + 108) = 62
    assert_relative_eq!(diffusion, 62.0, epsilon = 1e-10);
}

#[test]
fn parabolic_diffusion_stochastic_unbiased() {
    // Full sample (all indices) matches exact
    let tape = record_fn(sum_of_squares, &[1.0, 2.0]);
    let x = [1.0, 2.0];

    let e0: Vec<f64> = vec![1.0, 0.0];
    let e1: Vec<f64> = vec![0.0, 1.0];
    let cols: Vec<&[f64]> = vec![&e0, &e1];

    let (_, exact) = echidna::stde::parabolic_diffusion(&tape, &x, &cols);
    let result = echidna::stde::parabolic_diffusion_stochastic(&tape, &x, &cols, &[0, 1]);

    assert_relative_eq!(result.estimate, exact, epsilon = 1e-10);
}

// ══════════════════════════════════════════════
//  19. Const-generic diagonal_kth_order_const
// ══════════════════════════════════════════════

#[test]
fn diagonal_const_matches_dyn_k3() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 2);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_const_matches_dyn_k4() {
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 4>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 3);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-6);
    }
}

#[test]
fn diagonal_const_matches_dyn_k5() {
    let tape = record_fn(quartic, &[2.0, 3.0]);
    let x = [2.0, 3.0];

    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 5>(&tape, &x);
    let (val_d, diag_d) = echidna::stde::diagonal_kth_order(&tape, &x, 4);

    assert_relative_eq!(val_c, val_d, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_c[j], diag_d[j], epsilon = 1e-4);
    }
}

#[test]
fn diagonal_const_matches_hessian_diagonal() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_hd, diag_hd) = echidna::stde::hessian_diagonal(&tape, &x);
    let (val_c, diag_c) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);

    assert_relative_eq!(val_hd, val_c, epsilon = 1e-10);
    for j in 0..x.len() {
        assert_relative_eq!(diag_hd[j], diag_c[j], epsilon = 1e-10);
    }
}

#[test]
fn diagonal_const_with_buf_reuse() {
    let tape = record_fn(cubic_mix, &[1.0, 2.0, 3.0]);
    let x = [1.0, 2.0, 3.0];

    let (val_a, diag_a) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &x);

    let mut buf = Vec::new();
    let (val_b, diag_b) =
        echidna::stde::diagonal_kth_order_const_with_buf::<_, 3>(&tape, &x, &mut buf);
    let (val_c, diag_c) =
        echidna::stde::diagonal_kth_order_const_with_buf::<_, 3>(&tape, &x, &mut buf);

    assert_relative_eq!(val_a, val_b, epsilon = 1e-14);
    assert_relative_eq!(val_b, val_c, epsilon = 1e-14);
    for j in 0..x.len() {
        assert_relative_eq!(diag_a[j], diag_b[j], epsilon = 1e-14);
        assert_relative_eq!(diag_b[j], diag_c[j], epsilon = 1e-14);
    }
}

#[test]
fn diagonal_const_exp_1d() {
    // ∂^k(exp(x))/∂x^k = exp(x) for all k
    let tape = record_fn(exp_1d, &[1.0]);
    let expected = 1.0_f64.exp();

    let (val3, diag3) = echidna::stde::diagonal_kth_order_const::<_, 3>(&tape, &[1.0]);
    assert_relative_eq!(val3, expected, epsilon = 1e-10);
    assert_relative_eq!(diag3[0], expected, epsilon = 1e-10);

    let (_, diag4) = echidna::stde::diagonal_kth_order_const::<_, 4>(&tape, &[1.0]);
    assert_relative_eq!(diag4[0], expected, epsilon = 1e-8);

    let (_, diag5) = echidna::stde::diagonal_kth_order_const::<_, 5>(&tape, &[1.0]);
    assert_relative_eq!(diag5[0], expected, epsilon = 1e-7);
}
