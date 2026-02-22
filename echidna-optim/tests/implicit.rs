use echidna::record_multi;
use echidna_optim::linalg::lu_solve;
use echidna_optim::{implicit_adjoint, implicit_jacobian, implicit_tangent};

/// Simple Newton root-finder for testing: solve F(z, x) = 0 for z given fixed x.
///
/// Uses `J_z · delta = -F` where J_z is the state block of the Jacobian.
/// Returns the converged z* or panics if convergence fails.
fn newton_root_find(
    tape: &mut echidna::BytecodeTape<f64>,
    z_init: &[f64],
    x: &[f64],
    num_states: usize,
) -> Vec<f64> {
    let mut z = z_init.to_vec();
    let max_iter = 100;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let mut inputs = z.clone();
        inputs.extend_from_slice(x);

        // Evaluate F and get Jacobian
        let jac = tape.jacobian(&inputs);

        // Get residual values
        tape.forward(&inputs);
        let residual = tape.output_values();

        // Check convergence
        let norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if norm < tol {
            return z;
        }

        // Extract F_z (state block of Jacobian)
        let f_z: Vec<Vec<f64>> = jac.iter().map(|row| row[..num_states].to_vec()).collect();

        // Solve J_z · delta = -F
        let neg_res: Vec<f64> = residual.iter().map(|r| -r).collect();
        let delta = lu_solve(&f_z, &neg_res).expect("Singular Jacobian in Newton root-find");

        for i in 0..num_states {
            z[i] += delta[i];
        }
    }

    panic!("Newton root-finder did not converge");
}

// ============================================================
// Test 1: Linear system F(z,x) = Az + Bx
// ============================================================

#[test]
fn linear_system_jacobian() {
    // F(z, x) = A*z + B*x where A = [[2,1],[1,3]], B = I
    // At equilibrium F(z*,x) = 0 => z* = -A^{-1} B x
    // dz*/dx = -A^{-1} B = -A^{-1}
    //
    // A^{-1} = 1/5 * [[3,-1],[-1,2]]
    // So dz*/dx = -1/5 * [[3,-1],[-1,2]] = [[-0.6, 0.2],[0.2, -0.4]]
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];

            let two = v[0] - v[0] + v[0] - v[0]; // 0, used to build constants
            let one = x0 / x0; // trick: 1.0

            // F0 = 2*z0 + z1 + x0
            let f0 = (one + one) * z0 + z1 + x0;
            // F1 = z0 + 3*z1 + x1
            let f1 = z0 + (one + one + one) * z1 + x1;
            let _ = two;
            vec![f0, f1]
        },
        &[0.0_f64, 0.0, 1.0, 1.0], // dummy recording point
    );

    // Find z* for x = [1, 1]: A z* = -B x => z* = -A^{-1} [1,1]
    // z* = -1/5 * [3-1, -1+2] = -1/5 * [2, 1] = [-0.4, -0.2]
    let x = [1.0, 1.0];
    let z_star = [-0.4, -0.2];

    let jac = implicit_jacobian(&mut tape, &z_star, &x, 2).unwrap();

    // Expected: dz*/dx = -A^{-1} = [[-0.6, 0.2],[0.2, -0.4]]
    assert!(
        (jac[0][0] - (-0.6)).abs() < 1e-10,
        "jac[0][0] = {}",
        jac[0][0]
    );
    assert!(
        (jac[0][1] - 0.2).abs() < 1e-10,
        "jac[0][1] = {}",
        jac[0][1]
    );
    assert!(
        (jac[1][0] - 0.2).abs() < 1e-10,
        "jac[1][0] = {}",
        jac[1][0]
    );
    assert!(
        (jac[1][1] - (-0.4)).abs() < 1e-10,
        "jac[1][1] = {}",
        jac[1][1]
    );
}

// ============================================================
// Test 2: Scalar nonlinear F(z,x) = z^3 - x
// ============================================================

#[test]
fn scalar_nonlinear() {
    // F(z,x) = z^3 - x, at x=8, z*=2
    // dz*/dx = 1/(3z*^2) = 1/12
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            vec![z * z * z - x]
        },
        &[2.0_f64, 8.0],
    );

    let z_star = [2.0];
    let x = [8.0];

    let jac = implicit_jacobian(&mut tape, &z_star, &x, 1).unwrap();
    let expected = 1.0 / 12.0;
    assert!(
        (jac[0][0] - expected).abs() < 1e-10,
        "dz*/dx = {}, expected {}",
        jac[0][0],
        expected
    );
}

// ============================================================
// Test 3: Cross-validation — tangent vs jacobian columns
// ============================================================

#[test]
fn tangent_matches_jacobian_columns() {
    // F(z, x) = [z0^2 + z1 - x0, z0*z1 - x1]
    // At z* = [1, 1], x = [2, 1]: F = [1+1-2, 1-1] = [0, 0] ✓
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            vec![z0 * z0 + z1 - x0, z0 * z1 - x1]
        },
        &[1.0_f64, 1.0, 2.0, 1.0],
    );

    let z_star = [1.0, 1.0];
    let x = [2.0, 1.0];
    let num_states = 2;

    let jac = implicit_jacobian(&mut tape, &z_star, &x, num_states).unwrap();

    // tangent with e_0 = [1, 0] should give column 0 of jacobian
    let t0 = implicit_tangent(&mut tape, &z_star, &x, &[1.0, 0.0], num_states).unwrap();
    assert!(
        (t0[0] - jac[0][0]).abs() < 1e-10,
        "tangent e0[0] = {}, jac[0][0] = {}",
        t0[0],
        jac[0][0]
    );
    assert!(
        (t0[1] - jac[1][0]).abs() < 1e-10,
        "tangent e0[1] = {}, jac[1][0] = {}",
        t0[1],
        jac[1][0]
    );

    // tangent with e_1 = [0, 1] should give column 1 of jacobian
    let t1 = implicit_tangent(&mut tape, &z_star, &x, &[0.0, 1.0], num_states).unwrap();
    assert!(
        (t1[0] - jac[0][1]).abs() < 1e-10,
        "tangent e1[0] = {}, jac[0][1] = {}",
        t1[0],
        jac[0][1]
    );
    assert!(
        (t1[1] - jac[1][1]).abs() < 1e-10,
        "tangent e1[1] = {}, jac[1][1] = {}",
        t1[1],
        jac[1][1]
    );
}

// ============================================================
// Test 4: Cross-validation — adjoint vs jacobian^T rows
// ============================================================

#[test]
fn adjoint_matches_jacobian_transpose() {
    // Same system as tangent test
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            vec![z0 * z0 + z1 - x0, z0 * z1 - x1]
        },
        &[1.0_f64, 1.0, 2.0, 1.0],
    );

    let z_star = [1.0, 1.0];
    let x = [2.0, 1.0];
    let num_states = 2;

    let jac = implicit_jacobian(&mut tape, &z_star, &x, num_states).unwrap();

    // adjoint with e_0 = [1, 0] should give row 0 of jacobian^T = column 0 transpose
    // i.e., (dz*/dx)^T · [1,0] = [jac[0][0], jac[0][1]]... wait:
    // jac is m×n. adjoint returns (dz*/dx)^T · z_bar which is n-vector.
    // (dz*/dx)^T is n×m. With z_bar = e_i (length m), result is column i of (dz*/dx)^T = row i of dz*/dx.
    let a0 = implicit_adjoint(&mut tape, &z_star, &x, &[1.0, 0.0], num_states).unwrap();
    assert!(
        (a0[0] - jac[0][0]).abs() < 1e-10,
        "adjoint e0[0] = {}, jac[0][0] = {}",
        a0[0],
        jac[0][0]
    );
    assert!(
        (a0[1] - jac[0][1]).abs() < 1e-10,
        "adjoint e0[1] = {}, jac[0][1] = {}",
        a0[1],
        jac[0][1]
    );

    let a1 = implicit_adjoint(&mut tape, &z_star, &x, &[0.0, 1.0], num_states).unwrap();
    assert!(
        (a1[0] - jac[1][0]).abs() < 1e-10,
        "adjoint e1[0] = {}, jac[1][0] = {}",
        a1[0],
        jac[1][0]
    );
    assert!(
        (a1[1] - jac[1][1]).abs() < 1e-10,
        "adjoint e1[1] = {}, jac[1][1] = {}",
        a1[1],
        jac[1][1]
    );
}

// ============================================================
// Test 5: Singular F_z returns None
// ============================================================

#[test]
fn singular_fz_returns_none() {
    // F(z, x) = [z0 + z1 - x, 2*z0 + 2*z1 - 2*x]
    // F_z = [[1,1],[2,2]] which is singular (rank 1)
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x = v[2];
            let one = x / x;
            let two = one + one;
            vec![z0 + z1 - x, two * z0 + two * z1 - two * x]
        },
        &[0.5_f64, 0.5, 1.0],
    );

    let z_star = [0.5, 0.5]; // F(z*, x) = [0, 0] ✓
    let x = [1.0];

    assert!(implicit_jacobian(&mut tape, &z_star, &x, 2).is_none());
    assert!(implicit_tangent(&mut tape, &z_star, &x, &[1.0], 2).is_none());
    assert!(implicit_adjoint(&mut tape, &z_star, &x, &[1.0, 0.0], 2).is_none());
}

// ============================================================
// Test 6: Finite differences — 2D nonlinear system
// ============================================================

#[test]
fn finite_differences_2d_nonlinear() {
    // F(z0, z1, x0, x1) = [z0^2 + z1 - x0, z0*z1 - x1]
    // At x = [2, 1], z* = [1, 1]
    let make_tape = || {
        record_multi(
            |v| {
                let z0 = v[0];
                let z1 = v[1];
                let x0 = v[2];
                let x1 = v[3];
                vec![z0 * z0 + z1 - x0, z0 * z1 - x1]
            },
            &[1.0_f64, 1.0, 2.0, 1.0],
        )
        .0
    };

    let x = [2.0, 1.0];
    let z_star = [1.0, 1.0];
    let num_states = 2;

    // Analytical implicit Jacobian
    let mut tape = make_tape();
    let jac = implicit_jacobian(&mut tape, &z_star, &x, num_states).unwrap();

    // Finite difference approximation
    let h = 1e-5;
    let n = x.len();
    let m = num_states;
    let mut fd_jac = vec![vec![0.0; n]; m];

    for j in 0..n {
        // Perturb x[j] forward and backward
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[j] += h;
        x_minus[j] -= h;

        // Find z* for perturbed x
        let mut tape_plus = make_tape();
        let z_plus = newton_root_find(&mut tape_plus, &z_star, &x_plus, num_states);

        let mut tape_minus = make_tape();
        let z_minus = newton_root_find(&mut tape_minus, &z_star, &x_minus, num_states);

        for i in 0..m {
            fd_jac[i][j] = (z_plus[i] - z_minus[i]) / (2.0 * h);
        }
    }

    // Compare at tolerance 1e-4
    for i in 0..m {
        for j in 0..n {
            assert!(
                (jac[i][j] - fd_jac[i][j]).abs() < 1e-4,
                "jac[{}][{}] = {}, fd = {}, diff = {}",
                i,
                j,
                jac[i][j],
                fd_jac[i][j],
                (jac[i][j] - fd_jac[i][j]).abs()
            );
        }
    }
}
