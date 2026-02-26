use echidna::Scalar;
use echidna_optim::{
    lbfgs, newton, trust_region, LbfgsConfig, NewtonConfig, TapeObjective, TerminationReason,
    TrustRegionConfig,
};

fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let hundred = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(100.0).unwrap());
    let mut sum = T::zero();
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

fn make_tape() -> echidna::BytecodeTape<f64> {
    let (tape, _) = echidna::record(rosenbrock, &[0.0_f64, 0.0]);
    tape
}

#[test]
fn lbfgs_tape_rosenbrock() {
    let tape = make_tape();
    let mut obj = TapeObjective::new(tape);
    let config = LbfgsConfig::default();
    let result = lbfgs(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", result.x[1]);
    assert!(result.gradient_norm < 1e-8);
}

#[test]
fn newton_tape_rosenbrock() {
    let tape = make_tape();
    let mut obj = TapeObjective::new(tape);
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", result.x[1]);
    assert!(result.gradient_norm < 1e-8);
}

#[test]
fn trust_region_tape_rosenbrock() {
    let tape = make_tape();
    let mut obj = TapeObjective::new(tape);
    let config = TrustRegionConfig {
        convergence: echidna_optim::ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(
        result.termination,
        TerminationReason::GradientNorm,
        "terminated with {:?} after {} iters, grad_norm={}",
        result.termination,
        result.iterations,
        result.gradient_norm
    );
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", result.x[1]);
}

#[test]
fn newton_fewer_iters_than_lbfgs() {
    let tape_n = make_tape();
    let tape_l = make_tape();

    let mut obj_n = TapeObjective::new(tape_n);
    let result_n = newton(&mut obj_n, &[0.0, 0.0], &NewtonConfig::default());

    let mut obj_l = TapeObjective::new(tape_l);
    let result_l = lbfgs(&mut obj_l, &[0.0, 0.0], &LbfgsConfig::default());

    assert_eq!(result_n.termination, TerminationReason::GradientNorm);
    assert_eq!(result_l.termination, TerminationReason::GradientNorm);

    // Newton should converge in fewer iterations than L-BFGS on this small problem
    assert!(
        result_n.iterations < result_l.iterations,
        "Newton: {} iters, L-BFGS: {} iters",
        result_n.iterations,
        result_l.iterations
    );
}

#[test]
fn optimized_tape_still_works() {
    let (mut tape, _) = echidna::record(rosenbrock, &[0.0_f64, 0.0]);
    tape.optimize();

    let mut obj = TapeObjective::new(tape);
    let result = lbfgs(&mut obj, &[0.0, 0.0], &LbfgsConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-6, "x[1] = {}", result.x[1]);
}

#[test]
fn all_solvers_agree_on_minimum() {
    let x0 = &[-1.0, 2.0];

    let tape1 = make_tape();
    let mut obj1 = TapeObjective::new(tape1);
    let r1 = lbfgs(&mut obj1, x0, &LbfgsConfig::default());

    let tape2 = make_tape();
    let mut obj2 = TapeObjective::new(tape2);
    let r2 = newton(&mut obj2, x0, &NewtonConfig::default());

    let tape3 = make_tape();
    let mut obj3 = TapeObjective::new(tape3);
    let r3 = trust_region(
        &mut obj3,
        x0,
        &TrustRegionConfig {
            convergence: echidna_optim::ConvergenceParams {
                max_iter: 200,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    for (name, result) in [("L-BFGS", &r1), ("Newton", &r2), ("Trust-region", &r3)] {
        assert_eq!(
            result.termination,
            TerminationReason::GradientNorm,
            "{} did not converge: {:?}",
            name,
            result.termination
        );
        assert!(
            (result.x[0] - 1.0).abs() < 1e-5,
            "{}: x[0] = {}",
            name,
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-5,
            "{}: x[1] = {}",
            name,
            result.x[1]
        );
    }
}
