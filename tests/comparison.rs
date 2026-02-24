//! Cross-check tests: verify ad-trait gradient matches echidna.

#![cfg(feature = "bytecode")]

use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ReverseAD};
use ad_trait::function_engine::FunctionEngine;
use ad_trait::AD;

use echidna::Scalar;

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

fn rosenbrock_ad<T: AD>(x: &[T]) -> T {
    let one = T::constant(1.0);
    let hundred = T::constant(100.0);
    let mut sum = T::constant(0.0);
    for i in 0..x.len() - 1 {
        let t1 = one - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum = sum + t1 * t1 + hundred * t2 * t2;
    }
    sum
}

#[derive(Clone)]
struct RosenbrockAD {
    n: usize,
}

impl<T: AD> DifferentiableFunctionTrait<T> for RosenbrockAD {
    const NAME: &'static str = "Rosenbrock";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![rosenbrock_ad(inputs)]
    }

    fn num_inputs(&self) -> usize {
        self.n
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn make_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.01 * i as f64).collect()
}

#[test]
fn ad_trait_forward_gradient_matches_echidna() {
    let n = 4;
    let x = make_input(n);

    let echidna_grad = echidna::grad(|v| rosenbrock(v), &x);

    let rosen_std = RosenbrockAD { n };
    let rosen_fwd = rosen_std.clone();
    let engine = FunctionEngine::new(rosen_std, rosen_fwd, ForwardAD::new());
    let (_values, jac) = engine.derivative(&x);

    for i in 0..n {
        let ad_trait_gi = jac[(0, i)];
        let echidna_gi = echidna_grad[i];
        assert!(
            (ad_trait_gi - echidna_gi).abs() < 1e-8,
            "forward mismatch at index {}: ad_trait={}, echidna={}",
            i,
            ad_trait_gi,
            echidna_gi
        );
    }
}

#[test]
fn ad_trait_reverse_gradient_matches_echidna() {
    let n = 4;
    let x = make_input(n);

    let echidna_grad = echidna::grad(|v| rosenbrock(v), &x);

    let rosen_std = RosenbrockAD { n };
    let rosen_rev = rosen_std.clone();
    let engine = FunctionEngine::new(rosen_std, rosen_rev, ReverseAD::new());
    let (_values, jac) = engine.derivative(&x);

    for i in 0..n {
        let ad_trait_gi = jac[(0, i)];
        let echidna_gi = echidna_grad[i];
        assert!(
            (ad_trait_gi - echidna_gi).abs() < 1e-8,
            "reverse mismatch at index {}: ad_trait={}, echidna={}",
            i,
            ad_trait_gi,
            echidna_gi
        );
    }
}
