use echidna::record_multi;
use echidna_optim::{
    implicit_tangent, piggyback_adjoint_solve, piggyback_forward_adjoint_solve,
    piggyback_tangent_solve, piggyback_tangent_step, piggyback_tangent_step_with_buf,
};

// ============================================================
// Test 1: tangent_step_linear — single step correctness
// ============================================================

#[test]
fn tangent_step_linear() {
    // G(z, x) = 0.5*z + x (scalar)
    // From z=0, x=3: z_new = 0.5*0 + 3 = 3
    // With ż=0, ẋ=1: ż_new = 0.5*0 + 1 = 1
    let (tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x); // 0.5
            vec![half * z + x]
        },
        &[0.0_f64, 3.0],
    );

    let (z_new, z_dot_new) = piggyback_tangent_step(&tape, &[0.0], &[3.0], &[0.0], &[1.0], 1);

    assert!((z_new[0] - 3.0).abs() < 1e-12, "z_new = {}", z_new[0]);
    assert!(
        (z_dot_new[0] - 1.0).abs() < 1e-12,
        "z_dot_new = {}",
        z_dot_new[0]
    );
}

// ============================================================
// Test 2: tangent_solve_linear_contraction
// ============================================================

#[test]
fn tangent_solve_linear_contraction() {
    // G(z, x) = 0.5*z + x, fixed point z* = 2*x
    // dz*/dx = 2, so ẋ=1 => ż*=2
    let (tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![half * z + x]
        },
        &[0.0_f64, 3.0],
    );

    let result = piggyback_tangent_solve(&tape, &[0.0], &[3.0], &[1.0], 1, 200, 1e-12);
    let (z_star, z_dot_star, iters) = result.expect("should converge");

    assert!(
        (z_star[0] - 6.0).abs() < 1e-10,
        "z* = {}, expected 6",
        z_star[0]
    );
    assert!(
        (z_dot_star[0] - 2.0).abs() < 1e-8,
        "ż* = {}, expected 2",
        z_dot_star[0]
    );
    assert!(iters > 0, "should take at least 1 iteration");
}

// ============================================================
// Test 3: tangent_solve_2d_contraction
// ============================================================

#[test]
fn tangent_solve_2d_contraction() {
    // G([z0,z1], [x0,x1]) = [0.4*z0 + x0, 0.3*z1 + x1]
    // Fixed points: z0* = x0/0.6, z1* = x1/0.7
    // dz0*/dx0 = 1/0.6, dz1*/dx1 = 1/0.7, cross-terms = 0
    let (tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            let one = x0 / x0;
            let pt4 = (one + one) / (one + one + one + one + one); // 2/5 = 0.4
            let pt3 = (one + one + one) / (one + one + one + one + one + one + one + one + one + one); // 3/10 = 0.3
            vec![pt4 * z0 + x0, pt3 * z1 + x1]
        },
        &[0.0_f64, 0.0, 1.2, 2.1],
    );

    let x = [1.2, 2.1];

    // Direction dx0=1, dx1=0
    let result = piggyback_tangent_solve(&tape, &[0.0, 0.0], &x, &[1.0, 0.0], 2, 200, 1e-12);
    let (z_star, z_dot, _) = result.expect("should converge");

    let expected_z0 = 1.2 / 0.6;
    let expected_z1 = 2.1 / 0.7;
    assert!(
        (z_star[0] - expected_z0).abs() < 1e-9,
        "z0* = {}, expected {}",
        z_star[0],
        expected_z0
    );
    assert!(
        (z_star[1] - expected_z1).abs() < 1e-9,
        "z1* = {}, expected {}",
        z_star[1],
        expected_z1
    );
    assert!(
        (z_dot[0] - 1.0 / 0.6).abs() < 1e-7,
        "dz0*/dx0 = {}, expected {}",
        z_dot[0],
        1.0 / 0.6
    );
    assert!(
        z_dot[1].abs() < 1e-7,
        "dz1*/dx0 = {}, expected 0",
        z_dot[1]
    );

    // Direction dx0=0, dx1=1
    let result2 = piggyback_tangent_solve(&tape, &[0.0, 0.0], &x, &[0.0, 1.0], 2, 200, 1e-12);
    let (_, z_dot2, _) = result2.expect("should converge");

    assert!(
        z_dot2[0].abs() < 1e-7,
        "dz0*/dx1 = {}, expected 0",
        z_dot2[0]
    );
    assert!(
        (z_dot2[1] - 1.0 / 0.7).abs() < 1e-7,
        "dz1*/dx1 = {}, expected {}",
        z_dot2[1],
        1.0 / 0.7
    );
}

// ============================================================
// Test 4: adjoint_solve_linear_contraction
// ============================================================

#[test]
fn adjoint_solve_linear_contraction() {
    // G(z, x) = 0.5*z + x, z* = 2*x = 6 at x=3
    // z̄=1 => x̄ = dz*/dx = 2
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![half * z + x]
        },
        &[6.0_f64, 3.0],
    );

    let result = piggyback_adjoint_solve(&mut tape, &[6.0], &[3.0], &[1.0], 1, 200, 1e-12);
    let (x_bar, iters) = result.expect("should converge");

    assert!(
        (x_bar[0] - 2.0).abs() < 1e-8,
        "x̄ = {}, expected 2",
        x_bar[0]
    );
    assert!(iters > 0);
}

// ============================================================
// Test 5: adjoint vs tangent transpose (2D)
// ============================================================

#[test]
fn adjoint_vs_tangent_transpose() {
    // G([z0,z1], [x0,x1]) = [0.4*z0 + x0, 0.3*z1 + x1]
    // z0* = x0/0.6, z1* = x1/0.7
    // Jacobian dz*/dx = diag(1/0.6, 1/0.7)
    // adjoint(z̄) should equal (dz*/dx)^T · z̄
    let make_tape = || {
        record_multi(
            |v| {
                let z0 = v[0];
                let z1 = v[1];
                let x0 = v[2];
                let x1 = v[3];
                let one = x0 / x0;
                let pt4 = (one + one) / (one + one + one + one + one);
                let pt3 = (one + one + one)
                    / (one + one + one + one + one + one + one + one + one + one);
                vec![pt4 * z0 + x0, pt3 * z1 + x1]
            },
            &[0.0_f64, 0.0, 1.2, 2.1],
        )
    };

    let x = [1.2, 2.1];
    let z_star = [1.2 / 0.6, 2.1 / 0.7];

    // Get tangent Jacobian columns
    let (tape, _) = make_tape();
    let (_, col0, _) =
        piggyback_tangent_solve(&tape, &[0.0, 0.0], &x, &[1.0, 0.0], 2, 200, 1e-12)
            .expect("should converge");
    let (_, col1, _) =
        piggyback_tangent_solve(&tape, &[0.0, 0.0], &x, &[0.0, 1.0], 2, 200, 1e-12)
            .expect("should converge");

    // Get adjoint rows
    let (mut tape_a, _) = make_tape();
    let (row0, _) =
        piggyback_adjoint_solve(&mut tape_a, &z_star, &x, &[1.0, 0.0], 2, 200, 1e-12)
            .expect("should converge");
    let (row1, _) =
        piggyback_adjoint_solve(&mut tape_a, &z_star, &x, &[0.0, 1.0], 2, 200, 1e-12)
            .expect("should converge");

    // adjoint(e_i)[j] should equal tangent(e_j)[i] (transpose relationship)
    // row0 = (dz*/dx)^T · [1,0] = [dz0*/dx0, dz0*/dx1] = [col0[0], col1[0]]
    assert!(
        (row0[0] - col0[0]).abs() < 1e-7,
        "row0[0]={}, col0[0]={}",
        row0[0],
        col0[0]
    );
    assert!(
        (row0[1] - col1[0]).abs() < 1e-7,
        "row0[1]={}, col1[0]={}",
        row0[1],
        col1[0]
    );
    assert!(
        (row1[0] - col0[1]).abs() < 1e-7,
        "row1[0]={}, col0[1]={}",
        row1[0],
        col0[1]
    );
    assert!(
        (row1[1] - col1[1]).abs() < 1e-7,
        "row1[1]={}, col1[1]={}",
        row1[1],
        col1[1]
    );
}

// ============================================================
// Test 6: cross-validate piggyback tangent with IFT
// ============================================================

#[test]
fn cross_validate_with_ift() {
    // Step tape: G(z, x) = 0.5*z + x (scalar contraction)
    // Residual tape: F(z, x) = z - G(z, x) = z - 0.5*z - x = 0.5*z - x
    // At z*=6, x=3: F = 0.5*6 - 3 = 0 ✓
    // IFT on F: dz*/dx = -F_z^{-1} F_x = -(0.5)^{-1} * (-1) = 2
    let (step_tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![half * z + x]
        },
        &[6.0_f64, 3.0],
    );

    let (mut residual_tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![z - (half * z + x)] // z - G(z, x)
        },
        &[6.0_f64, 3.0],
    );

    // Piggyback tangent
    let (_, z_dot_pb, _) =
        piggyback_tangent_solve(&step_tape, &[0.0], &[3.0], &[1.0], 1, 200, 1e-12)
            .expect("piggyback should converge");

    // IFT tangent
    let z_dot_ift =
        implicit_tangent(&mut residual_tape, &[6.0], &[3.0], &[1.0], 1).expect("IFT should work");

    assert!(
        (z_dot_pb[0] - z_dot_ift[0]).abs() < 1e-7,
        "piggyback ż*={}, IFT ż*={}",
        z_dot_pb[0],
        z_dot_ift[0]
    );
}

// ============================================================
// Test 7: tangent_step_buffer_reuse
// ============================================================

#[test]
fn tangent_step_buffer_reuse() {
    // Two calls with the same buffer should give identical results
    let (tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![half * z + x]
        },
        &[1.0_f64, 3.0],
    );

    let mut buf = Vec::new();

    let (z1, zd1) =
        piggyback_tangent_step_with_buf(&tape, &[1.0], &[3.0], &[0.5], &[1.0], 1, &mut buf);
    let (z2, zd2) =
        piggyback_tangent_step_with_buf(&tape, &[1.0], &[3.0], &[0.5], &[1.0], 1, &mut buf);

    assert_eq!(z1[0], z2[0]);
    assert_eq!(zd1[0], zd2[0]);
}

// ============================================================
// Test 8: adjoint_non_convergent
// ============================================================

#[test]
fn adjoint_non_convergent() {
    // G(z, x) = 2*z + x — ||G_z|| = 2 > 1, not a contraction
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let two = (x + x) / x; // 2.0
            vec![two * z + x]
        },
        &[1.0_f64, 1.0],
    );

    // z* doesn't really exist for this system, but test that adjoint detects divergence
    let result = piggyback_adjoint_solve(&mut tape, &[1.0], &[1.0], &[1.0], 1, 100, 1e-12);
    assert!(result.is_none(), "should not converge for non-contraction");
}

// ============================================================
// Test 9: forward_adjoint_solve_linear
// ============================================================

#[test]
fn forward_adjoint_solve_linear() {
    // G(z, x) = 0.5*z + x, z* = 2*x = 6 at x=3
    // z̄=1 => x̄ = dz*/dx = 2
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let half = x / (x + x);
            vec![half * z + x]
        },
        &[0.0_f64, 3.0],
    );

    let result =
        piggyback_forward_adjoint_solve(&mut tape, &[0.0], &[3.0], &[1.0], 1, 200, 1e-12);
    let (z_star, x_bar, iters) = result.expect("should converge");

    assert!(
        (z_star[0] - 6.0).abs() < 1e-10,
        "z* = {}, expected 6",
        z_star[0]
    );
    assert!(
        (x_bar[0] - 2.0).abs() < 1e-8,
        "x̄ = {}, expected 2",
        x_bar[0]
    );
    assert!(iters > 0);
}

// ============================================================
// Test 10: forward_adjoint_solve_2d
// ============================================================

#[test]
fn forward_adjoint_solve_2d() {
    // G([z0,z1], [x0,x1]) = [0.4*z0 + x0, 0.3*z1 + x1]
    // z0* = x0/0.6, z1* = x1/0.7
    // dz*/dx = diag(1/0.6, 1/0.7)
    // z̄=[1,0] => x̄ = (dz*/dx)^T · [1,0] = [1/0.6, 0]
    let (mut tape, _) = record_multi(
        |v| {
            let z0 = v[0];
            let z1 = v[1];
            let x0 = v[2];
            let x1 = v[3];
            let one = x0 / x0;
            let pt4 = (one + one) / (one + one + one + one + one);
            let pt3 =
                (one + one + one) / (one + one + one + one + one + one + one + one + one + one);
            vec![pt4 * z0 + x0, pt3 * z1 + x1]
        },
        &[0.0_f64, 0.0, 1.2, 2.1],
    );

    let result = piggyback_forward_adjoint_solve(
        &mut tape,
        &[0.0, 0.0],
        &[1.2, 2.1],
        &[1.0, 0.0],
        2,
        200,
        1e-12,
    );
    let (z_star, x_bar, _) = result.expect("should converge");

    assert!(
        (z_star[0] - 1.2 / 0.6).abs() < 1e-9,
        "z0* = {}, expected {}",
        z_star[0],
        1.2 / 0.6
    );
    assert!(
        (z_star[1] - 2.1 / 0.7).abs() < 1e-9,
        "z1* = {}, expected {}",
        z_star[1],
        2.1 / 0.7
    );
    assert!(
        (x_bar[0] - 1.0 / 0.6).abs() < 1e-7,
        "x̄[0] = {}, expected {}",
        x_bar[0],
        1.0 / 0.6
    );
    assert!(
        x_bar[1].abs() < 1e-7,
        "x̄[1] = {}, expected 0",
        x_bar[1]
    );
}

// ============================================================
// Test 11: forward_adjoint_matches_sequential
// ============================================================

#[test]
fn forward_adjoint_matches_sequential() {
    // Interleaved should give the same result as tangent_solve + adjoint_solve
    let make_tape = || {
        record_multi(
            |v| {
                let z0 = v[0];
                let z1 = v[1];
                let x0 = v[2];
                let x1 = v[3];
                let one = x0 / x0;
                let pt4 = (one + one) / (one + one + one + one + one);
                let pt3 = (one + one + one)
                    / (one + one + one + one + one + one + one + one + one + one);
                vec![pt4 * z0 + x0, pt3 * z1 + x1]
            },
            &[0.0_f64, 0.0, 1.2, 2.1],
        )
    };

    let x = [1.2, 2.1];
    let z_bar = [1.0, 0.5];

    // Sequential: adjoint after convergence
    let (mut tape_seq, _) = make_tape();
    let (z_star_seq, _, _) =
        piggyback_tangent_solve(&tape_seq, &[0.0, 0.0], &x, &[1.0, 0.0], 2, 200, 1e-12)
            .expect("tangent should converge");
    let (x_bar_seq, _) =
        piggyback_adjoint_solve(&mut tape_seq, &z_star_seq, &x, &z_bar, 2, 200, 1e-12)
            .expect("adjoint should converge");

    // Interleaved
    let (mut tape_int, _) = make_tape();
    let (z_star_int, x_bar_int, _) = piggyback_forward_adjoint_solve(
        &mut tape_int,
        &[0.0, 0.0],
        &x,
        &z_bar,
        2,
        200,
        1e-12,
    )
    .expect("interleaved should converge");

    for i in 0..2 {
        assert!(
            (z_star_int[i] - z_star_seq[i]).abs() < 1e-9,
            "z*[{}]: interleaved={}, sequential={}",
            i,
            z_star_int[i],
            z_star_seq[i]
        );
    }
    for j in 0..2 {
        assert!(
            (x_bar_int[j] - x_bar_seq[j]).abs() < 1e-7,
            "x̄[{}]: interleaved={}, sequential={}",
            j,
            x_bar_int[j],
            x_bar_seq[j]
        );
    }
}

// ============================================================
// Test 12: forward_adjoint_non_convergent
// ============================================================

#[test]
fn forward_adjoint_non_convergent() {
    // G(z, x) = 2*z + x — not a contraction
    let (mut tape, _) = record_multi(
        |v| {
            let z = v[0];
            let x = v[1];
            let two = (x + x) / x;
            vec![two * z + x]
        },
        &[1.0_f64, 1.0],
    );

    let result =
        piggyback_forward_adjoint_solve(&mut tape, &[0.0], &[1.0], &[1.0], 1, 100, 1e-12);
    assert!(
        result.is_none(),
        "should not converge for non-contraction"
    );
}
