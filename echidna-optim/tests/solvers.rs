use echidna::Scalar;
use echidna_optim::{
    lbfgs, newton, trust_region, ConvergenceParams, LbfgsConfig, NewtonConfig, Objective,
    OptimResult, TapeObjective, TerminationReason, TrustRegionConfig,
};

// ============================================================
// Test objectives
// ============================================================

/// f(x) = 0.5 * sum(x_i^2). Minimum at origin, value 0.
struct Quadratic {
    dim: usize,
}

impl Objective<f64> for Quadratic {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f: f64 = 0.5 * x.iter().map(|&xi| xi * xi).sum::<f64>();
        let g: Vec<f64> = x.to_vec();
        (f, g)
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let n = x.len();
        let mut h = vec![vec![0.0; n]; n];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        (f, g, h)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = x.to_vec();
        let hv = v.to_vec(); // H = I, so Hv = v
        (g, hv)
    }
}

/// f(x) = 0.5 * (a*x0^2 + b*x1^2). Ill-conditioned when a/b >> 1.
struct IllConditionedQuadratic {
    a: f64,
    b: f64,
}

impl Objective<f64> for IllConditionedQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = 0.5 * (self.a * x[0] * x[0] + self.b * x[1] * x[1]);
        let g = vec![self.a * x[0], self.b * x[1]];
        (f, g)
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let h = vec![vec![self.a, 0.0], vec![0.0, self.b]];
        (f, g, h)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = vec![self.a * x[0], self.b * x[1]];
        let hv = vec![self.a * v[0], self.b * v[1]];
        (g, hv)
    }
}

/// Rosenbrock: f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2. Minimum at (1,1), value 0.
struct Rosenbrock2D;

impl Objective<f64> for Rosenbrock2D {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        let g0 = -2.0 * a - 400.0 * x[0] * b;
        let g1 = 200.0 * b;
        (f, vec![g0, g1])
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        let g0 = -2.0 * a - 400.0 * x[0] * b;
        let g1 = 200.0 * b;

        let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
        let h01 = -400.0 * x[0];
        let h11 = 200.0;

        (f, vec![g0, g1], vec![vec![h00, h01], vec![h01, h11]])
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
        let h01 = -400.0 * x[0];
        let h11 = 200.0;

        let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let g1 = 200.0 * (x[1] - x[0] * x[0]);

        let hv0 = h00 * v[0] + h01 * v[1];
        let hv1 = h01 * v[0] + h11 * v[1];

        (vec![g0, g1], vec![hv0, hv1])
    }
}

fn assert_near_origin(result: &OptimResult<f64>, tol: f64) {
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(xi.abs() < tol, "x[{}] = {}, expected ~0", i, xi);
    }
    assert!(result.value < tol, "f = {}, expected ~0", result.value);
}

// ============================================================
// Newton: simple quadratic
// ============================================================

#[test]
fn newton_quadratic_2d() {
    let mut obj = Quadratic { dim: 2 };
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    // Newton on a quadratic converges in 1 iteration
    assert_eq!(
        result.iterations, 1,
        "Newton should solve quadratic in 1 step"
    );
    assert_near_origin(&result, 1e-10);
}

#[test]
fn newton_quadratic_4d() {
    let mut obj = Quadratic { dim: 4 };
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[10.0, -7.0, 3.0, -1.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 1);
    assert_near_origin(&result, 1e-10);
}

#[test]
fn newton_ill_conditioned() {
    let mut obj = IllConditionedQuadratic { a: 1000.0, b: 1.0 };
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[5.0, -3.0], &config);

    // Newton with exact Hessian is unaffected by conditioning
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 1);
    assert_near_origin(&result, 1e-10);
}

#[test]
fn newton_at_optimum() {
    let mut obj = Quadratic { dim: 2 };
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.func_evals, 1); // single eval to check gradient
}

#[test]
fn newton_max_iter_zero() {
    let mut obj = Quadratic { dim: 2 };
    let config = NewtonConfig {
        convergence: ConvergenceParams {
            max_iter: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = newton(&mut obj, &[1.0, 1.0], &config);
    assert_eq!(result.termination, TerminationReason::NumericalError);
    assert_eq!(result.iterations, 0);
}

// ============================================================
// L-BFGS: simple quadratic
// ============================================================

#[test]
fn lbfgs_quadratic_2d() {
    let mut obj = Quadratic { dim: 2 };
    let config = LbfgsConfig::default();
    let result = lbfgs(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn lbfgs_quadratic_8d() {
    let mut obj = Quadratic { dim: 8 };
    let config = LbfgsConfig::default();
    let result = lbfgs(
        &mut obj,
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
        &config,
    );

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn lbfgs_ill_conditioned() {
    let mut obj = IllConditionedQuadratic { a: 1000.0, b: 1.0 };
    let config = LbfgsConfig::default();
    let result = lbfgs(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-4);
}

#[test]
fn lbfgs_memory_1() {
    let mut obj = Quadratic { dim: 2 };
    let config = LbfgsConfig {
        memory: 1,
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn lbfgs_at_optimum() {
    let mut obj = Quadratic { dim: 3 };
    let config = LbfgsConfig::default();
    let result = lbfgs(&mut obj, &[0.0, 0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 0);
}

#[test]
fn lbfgs_max_iter_zero() {
    let mut obj = Quadratic { dim: 2 };
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[1.0, 1.0], &config);
    assert_eq!(result.termination, TerminationReason::NumericalError);
}

#[test]
fn lbfgs_memory_zero() {
    let mut obj = Quadratic { dim: 2 };
    let config = LbfgsConfig {
        memory: 0,
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[1.0, 1.0], &config);
    assert_eq!(result.termination, TerminationReason::NumericalError);
}

// ============================================================
// Trust-region: simple quadratic
// ============================================================

#[test]
fn trust_region_quadratic_2d() {
    let mut obj = Quadratic { dim: 2 };
    let config = TrustRegionConfig::default();
    let result = trust_region(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn trust_region_quadratic_8d() {
    let mut obj = Quadratic { dim: 8 };
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(
        &mut obj,
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
        &config,
    );

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn trust_region_ill_conditioned() {
    let mut obj = IllConditionedQuadratic { a: 1000.0, b: 1.0 };
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[5.0, -3.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-4);
}

#[test]
fn trust_region_at_optimum() {
    let mut obj = Quadratic { dim: 2 };
    let config = TrustRegionConfig::default();
    let result = trust_region(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 0);
}

#[test]
fn trust_region_small_radius() {
    let mut obj = Quadratic { dim: 2 };
    let config = TrustRegionConfig {
        initial_radius: 0.01,
        max_radius: 100.0,
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[5.0, -3.0], &config);

    // Should still converge, just take more iterations
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn trust_region_max_iter_zero() {
    let mut obj = Quadratic { dim: 2 };
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[1.0, 1.0], &config);
    assert_eq!(result.termination, TerminationReason::NumericalError);
}

#[test]
fn trust_region_zero_radius() {
    let mut obj = Quadratic { dim: 2 };
    let config = TrustRegionConfig {
        initial_radius: 0.0,
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[1.0, 1.0], &config);
    assert_eq!(result.termination, TerminationReason::NumericalError);
}

// ============================================================
// Convergence criteria
// ============================================================

#[test]
fn func_tol_terminates_lbfgs() {
    // Use Rosenbrock (takes many iterations) so func_tol can fire
    let mut obj = Rosenbrock2D;
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 1000,
            grad_tol: 0.0,  // disable gradient termination
            step_tol: 0.0,  // disable step termination
            func_tol: 1e-2, // generous func_tol — fires when progress stalls
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::FunctionChange);
    // Should have made progress toward the minimum
    assert!(result.value < 1.0, "f = {}", result.value);
}

#[test]
fn max_iter_terminates_lbfgs() {
    // Use Rosenbrock so the solver doesn't converge before max_iter
    let mut obj = Rosenbrock2D;
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 3,
            grad_tol: 0.0,
            step_tol: 0.0,
            func_tol: 0.0,
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(result.termination, TerminationReason::MaxIterations);
    assert_eq!(result.iterations, 3);
}

#[test]
fn func_tol_terminates_newton() {
    // Use Rosenbrock so Newton takes multiple iterations
    let mut obj = Rosenbrock2D;
    let config = NewtonConfig {
        convergence: ConvergenceParams {
            max_iter: 1000,
            grad_tol: 0.0,
            step_tol: 0.0,
            func_tol: 1e-2,
        },
        ..Default::default()
    };
    let result = newton(&mut obj, &[0.0, 0.0], &config);

    assert_eq!(
        result.termination,
        TerminationReason::FunctionChange,
        "terminated with {:?}",
        result.termination
    );
}

// ============================================================
// Cross-solver agreement on quadratic
// ============================================================

#[test]
fn all_solvers_agree_quadratic() {
    let x0 = &[3.0, -4.0, 1.0];

    let mut obj1 = Quadratic { dim: 3 };
    let r1 = lbfgs(&mut obj1, x0, &LbfgsConfig::default());

    let mut obj2 = Quadratic { dim: 3 };
    let r2 = newton(&mut obj2, x0, &NewtonConfig::default());

    let mut obj3 = Quadratic { dim: 3 };
    let r3 = trust_region(
        &mut obj3,
        x0,
        &TrustRegionConfig {
            convergence: ConvergenceParams {
                max_iter: 200,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    for (name, r) in [("L-BFGS", &r1), ("Newton", &r2), ("Trust-region", &r3)] {
        assert_eq!(
            r.termination,
            TerminationReason::GradientNorm,
            "{} did not converge: {:?}",
            name,
            r.termination
        );
        assert_near_origin(r, 1e-6);
    }
}

// ============================================================
// TapeObjective tests (higher-dimensional, tape-based)
// ============================================================

fn quadratic_tape<T: Scalar>(x: &[T]) -> T {
    let half = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(0.5).unwrap());
    let mut sum = T::zero();
    for &xi in x {
        sum = sum + xi * xi;
    }
    half * sum
}

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

#[test]
fn tape_lbfgs_quadratic_6d() {
    let x0 = [1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let (tape, _) = echidna::record(quadratic_tape, &x0);
    let mut obj = TapeObjective::new(tape);
    let result = lbfgs(&mut obj, &x0, &LbfgsConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn tape_newton_quadratic_6d() {
    let x0 = [1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let (tape, _) = echidna::record(quadratic_tape, &x0);
    let mut obj = TapeObjective::new(tape);
    let result = newton(&mut obj, &x0, &NewtonConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 1);
    assert_near_origin(&result, 1e-10);
}

#[test]
fn tape_trust_region_quadratic_6d() {
    let x0 = [1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let (tape, _) = echidna::record(quadratic_tape, &x0);
    let mut obj = TapeObjective::new(tape);
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &x0, &config);

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

#[test]
fn tape_rosenbrock_4d_newton() {
    let x0 = [0.0_f64, 0.0, 0.0, 0.0];
    let (tape, _) = echidna::record(rosenbrock, &x0);
    let mut obj = TapeObjective::new(tape);
    let result = newton(&mut obj, &x0, &NewtonConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    for i in 0..4 {
        assert!(
            (result.x[i] - 1.0).abs() < 1e-5,
            "x[{}] = {}, expected 1.0",
            i,
            result.x[i]
        );
    }
}

#[test]
fn tape_rosenbrock_4d_lbfgs() {
    let x0 = [0.0_f64, 0.0, 0.0, 0.0];
    let (tape, _) = echidna::record(rosenbrock, &x0);
    let mut obj = TapeObjective::new(tape);
    let result = lbfgs(&mut obj, &x0, &LbfgsConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    for i in 0..4 {
        assert!(
            (result.x[i] - 1.0).abs() < 1e-4,
            "x[{}] = {}, expected 1.0",
            i,
            result.x[i]
        );
    }
}

#[test]
fn tape_rosenbrock_4d_trust_region() {
    let x0 = [0.0_f64, 0.0, 0.0, 0.0];
    let (tape, _) = echidna::record(rosenbrock, &x0);
    let mut obj = TapeObjective::new(tape);
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 500,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &x0, &config);

    assert_eq!(
        result.termination,
        TerminationReason::GradientNorm,
        "terminated with {:?} after {} iters",
        result.termination,
        result.iterations
    );
    for i in 0..4 {
        assert!(
            (result.x[i] - 1.0).abs() < 1e-4,
            "x[{}] = {}, expected 1.0",
            i,
            result.x[i]
        );
    }
}

// ============================================================
// TapeObjective with optimized tape
// ============================================================

#[test]
fn optimized_tape_quadratic() {
    let x0 = [2.0_f64, -3.0, 4.0];
    let (mut tape, _) = echidna::record(quadratic_tape, &x0);
    tape.optimize();

    let mut obj = TapeObjective::new(tape);
    let result = lbfgs(&mut obj, &x0, &LbfgsConfig::default());

    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_near_origin(&result, 1e-6);
}

// ============================================================
// func_evals tracking
// ============================================================

#[test]
fn tape_objective_counts_evals() {
    let x0 = [1.0_f64, 1.0];
    let (tape, _) = echidna::record(quadratic_tape, &x0);
    let mut obj = TapeObjective::new(tape);
    assert_eq!(obj.func_evals(), 0);

    let _ = lbfgs(&mut obj, &x0, &LbfgsConfig::default());
    assert!(obj.func_evals() > 0, "should have counted evaluations");
}

// ============================================================
// Edge cases: near-singular, highly ill-conditioned, saddle
// ============================================================

/// f(x,y) = x^4 + y^2. Hessian at origin has zero eigenvalue in x.
struct NearSingularHessian;

impl Objective<f64> for NearSingularHessian {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = x[0].powi(4) + x[1] * x[1];
        let g = vec![4.0 * x[0].powi(3), 2.0 * x[1]];
        (f, g)
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let h = vec![vec![12.0 * x[0] * x[0], 0.0], vec![0.0, 2.0]];
        (f, g, h)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = vec![4.0 * x[0].powi(3), 2.0 * x[1]];
        let hv = vec![12.0 * x[0] * x[0] * v[0], 2.0 * v[1]];
        (g, hv)
    }
}

/// f(x,y) = 0.5*(1e6*x^2 + y^2). Highly ill-conditioned (1e6:1 ratio).
struct HighlyIllConditioned;

impl Objective<f64> for HighlyIllConditioned {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = 0.5 * (1e6 * x[0] * x[0] + x[1] * x[1]);
        let g = vec![1e6 * x[0], x[1]];
        (f, g)
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let h = vec![vec![1e6, 0.0], vec![0.0, 1.0]];
        (f, g, h)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = vec![1e6 * x[0], x[1]];
        let hv = vec![1e6 * v[0], v[1]];
        (g, hv)
    }
}

/// f(x,y) = x^2 - y^2 + y^4. Saddle at origin, minima at y = ±1/√2.
struct SaddleNearby;

impl Objective<f64> for SaddleNearby {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = x[0] * x[0] - x[1] * x[1] + x[1].powi(4);
        let g = vec![2.0 * x[0], -2.0 * x[1] + 4.0 * x[1].powi(3)];
        (f, g)
    }

    fn eval_hessian(&mut self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let (f, g) = self.eval_grad(x);
        let h = vec![vec![2.0, 0.0], vec![0.0, -2.0 + 12.0 * x[1] * x[1]]];
        (f, g, h)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = vec![2.0 * x[0], -2.0 * x[1] + 4.0 * x[1].powi(3)];
        let hv = vec![2.0 * v[0], (-2.0 + 12.0 * x[1] * x[1]) * v[1]];
        (g, hv)
    }
}

#[test]
fn newton_near_singular_hessian() {
    let mut obj = NearSingularHessian;
    let config = NewtonConfig {
        convergence: ConvergenceParams {
            max_iter: 100,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = newton(&mut obj, &[1.0, 1.0], &config);

    // Should converge to origin despite singular Hessian there
    assert!(
        matches!(
            result.termination,
            TerminationReason::GradientNorm | TerminationReason::StepSize
        ),
        "Newton near-singular: terminated with {:?}",
        result.termination
    );
    assert!(result.x[0].abs() < 1e-2, "x[0] = {}", result.x[0]);
    assert!(result.x[1].abs() < 1e-6, "x[1] = {}", result.x[1]);
}

#[test]
fn lbfgs_near_singular_hessian() {
    let mut obj = NearSingularHessian;
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[1.0, 1.0], &config);

    assert!(
        matches!(
            result.termination,
            TerminationReason::GradientNorm | TerminationReason::StepSize
        ),
        "L-BFGS near-singular: terminated with {:?}",
        result.termination
    );
    assert!(result.x[0].abs() < 1e-2, "x[0] = {}", result.x[0]);
    assert!(result.x[1].abs() < 1e-6, "x[1] = {}", result.x[1]);
}

#[test]
fn newton_highly_ill_conditioned() {
    let mut obj = HighlyIllConditioned;
    let config = NewtonConfig::default();
    let result = newton(&mut obj, &[1.0, 1.0], &config);

    // Newton with exact Hessian handles even extreme conditioning
    assert_eq!(result.termination, TerminationReason::GradientNorm);
    assert_eq!(result.iterations, 1);
    assert_near_origin(&result, 1e-8);
}

#[test]
fn lbfgs_highly_ill_conditioned() {
    let mut obj = HighlyIllConditioned;
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 500,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&mut obj, &[1.0, 1.0], &config);

    assert_eq!(
        result.termination,
        TerminationReason::GradientNorm,
        "L-BFGS 1e6:1 ill-conditioned: terminated with {:?}",
        result.termination
    );
    assert_near_origin(&result, 1e-3);
}

#[test]
fn trust_region_highly_ill_conditioned() {
    let mut obj = HighlyIllConditioned;
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 500,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[1.0, 1.0], &config);

    assert_eq!(
        result.termination,
        TerminationReason::GradientNorm,
        "Trust-region 1e6:1 ill-conditioned: terminated with {:?}",
        result.termination
    );
    assert_near_origin(&result, 1e-3);
}

#[test]
fn newton_avoids_saddle() {
    // Start near the saddle at origin but offset in y. Newton should converge
    // to a local minimum at (0, ±1/√2), not the saddle.
    let mut obj = SaddleNearby;
    let config = NewtonConfig {
        convergence: ConvergenceParams {
            max_iter: 100,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = newton(&mut obj, &[0.1, 0.5], &config);

    assert!(
        matches!(
            result.termination,
            TerminationReason::GradientNorm | TerminationReason::StepSize
        ),
        "Newton saddle: terminated with {:?}",
        result.termination
    );
    // Should be at x=0, |y| ≈ 1/√2 ≈ 0.7071
    assert!(result.x[0].abs() < 1e-4, "x[0] = {}", result.x[0]);
    let y_expected = 1.0 / 2.0_f64.sqrt();
    assert!(
        (result.x[1].abs() - y_expected).abs() < 1e-4,
        "|y| = {}, expected {}",
        result.x[1].abs(),
        y_expected
    );
}

#[test]
fn trust_region_avoids_saddle() {
    let mut obj = SaddleNearby;
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut obj, &[0.1, 0.5], &config);

    assert!(
        matches!(
            result.termination,
            TerminationReason::GradientNorm | TerminationReason::StepSize
        ),
        "Trust-region saddle: terminated with {:?}",
        result.termination
    );
    assert!(result.x[0].abs() < 1e-4, "x[0] = {}", result.x[0]);
    let y_expected = 1.0 / 2.0_f64.sqrt();
    assert!(
        (result.x[1].abs() - y_expected).abs() < 1e-4,
        "|y| = {}, expected {}",
        result.x[1].abs(),
        y_expected
    );
}

// ============================================================
// Bug hunt regression tests
// ============================================================

// ── #9: NaN gradient detection ──

/// Objective that always returns NaN value and gradient.
struct NanGradObj;

impl Objective<f64> for NanGradObj {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, _x: &[f64]) -> (f64, Vec<f64>) {
        (f64::NAN, vec![f64::NAN; 2])
    }

    fn eval_hessian(&mut self, _x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        (
            f64::NAN,
            vec![f64::NAN; 2],
            vec![vec![f64::NAN; 2]; 2],
        )
    }

    fn hvp(&mut self, _x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (vec![f64::NAN; 2], v.to_vec())
    }
}

#[test]
fn regression_9_trust_region_nan_gradient_terminates() {
    let config = TrustRegionConfig::default();
    let result = trust_region(&mut NanGradObj, &[1.0, 1.0], &config);
    assert_eq!(
        result.termination,
        TerminationReason::NumericalError,
        "Trust region should terminate with NumericalError on NaN gradient, got {:?}",
        result.termination
    );
}

#[test]
fn regression_9_lbfgs_nan_gradient_terminates() {
    let config = LbfgsConfig::default();
    let result = lbfgs(&mut NanGradObj, &[1.0, 1.0], &config);
    assert_eq!(
        result.termination,
        TerminationReason::NumericalError,
        "L-BFGS should terminate with NumericalError on NaN gradient, got {:?}",
        result.termination
    );
}

#[test]
fn regression_9_newton_nan_gradient_terminates() {
    let config = NewtonConfig::default();
    let result = newton(&mut NanGradObj, &[1.0, 1.0], &config);
    assert_eq!(
        result.termination,
        TerminationReason::NumericalError,
        "Newton should terminate with NumericalError on NaN gradient, got {:?}",
        result.termination
    );
}

// ── #10: Negative predicted reduction ──

/// f(x) = -x^2 : negative-definite Hessian. Trust region should handle this
/// without expanding the radius (predicted reduction would be negative).
struct NegDefQuadratic;

impl Objective<f64> for NegDefQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let f = -(x[0] * x[0] + x[1] * x[1]);
        let g = vec![-2.0 * x[0], -2.0 * x[1]];
        (f, g)
    }

    fn hvp(&mut self, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let g = vec![-2.0 * x[0], -2.0 * x[1]];
        let hv = vec![-2.0 * v[0], -2.0 * v[1]];
        (g, hv)
    }
}

#[test]
fn regression_10_trust_region_negative_predicted_reduction() {
    // f(x) = -x^2 has no local minimum (it's concave). The trust-region solver
    // should terminate sensibly (not loop forever or produce NaN).
    let config = TrustRegionConfig {
        convergence: ConvergenceParams {
            max_iter: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = trust_region(&mut NegDefQuadratic, &[1.0, 1.0], &config);
    // Should terminate without panicking. We don't require a specific reason
    // since there is no local minimum, but the result must be finite.
    assert!(
        result.value.is_finite(),
        "Trust region on concave f should produce finite value, got {}",
        result.value
    );
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(xi.is_finite(), "x[{}] should be finite, got {}", i, xi);
    }
}

// ── #27: L-BFGS gamma overflow on flat objective ──

/// Very flat objective: f(x) = epsilon * sum(x_i^2). Gradients are tiny,
/// which can cause gamma = y^T s / (y^T y) to overflow when y is near-zero.
struct VeryFlatQuadratic;

impl Objective<f64> for VeryFlatQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn eval_grad(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let eps = 1e-30;
        let f = eps * 0.5 * x.iter().map(|&xi| xi * xi).sum::<f64>();
        let g: Vec<f64> = x.iter().map(|&xi| eps * xi).collect();
        (f, g)
    }
}

#[test]
fn regression_27_lbfgs_gamma_overflow_flat_objective() {
    let config = LbfgsConfig {
        convergence: ConvergenceParams {
            max_iter: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = lbfgs(&mut VeryFlatQuadratic, &[1.0, 1.0], &config);
    // Should terminate without NaN values
    assert!(
        !result.value.is_nan(),
        "L-BFGS on very flat objective should not produce NaN value"
    );
    for (i, &xi) in result.x.iter().enumerate() {
        assert!(!xi.is_nan(), "x[{}] should not be NaN", i);
    }
}
