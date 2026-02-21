# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-21

### Added

#### Core Types
- `Dual<F>` forward-mode dual number with all 30+ elemental operations
- `Reverse<F>` reverse-mode AD variable (12 bytes for f64, `Copy`)
- `Float` marker trait for `f32`/`f64`
- `Scalar` trait for writing AD-generic code
- Type aliases: `Dual64`, `Dual32`, `Reverse64`, `Reverse32`

#### Tape
- Adept-style two-stack tape with precomputed partial derivatives
- Thread-local active tape with RAII guard (`TapeGuard`)
- Constant sentinel (`u32::MAX`) to avoid tape bloat from literals
- Zero-adjoint skipping in the reverse sweep

#### API
- `grad(f, x)` — gradient via reverse mode
- `jvp(f, x, v)` — Jacobian-vector product via forward mode
- `vjp(f, x, w)` — vector-Jacobian product via reverse mode
- `jacobian(f, x)` — full Jacobian via forward mode

#### Elemental Operations
- Powers: `recip`, `sqrt`, `cbrt`, `powi`, `powf`
- Exp/Log: `exp`, `exp2`, `exp_m1`, `ln`, `log2`, `log10`, `ln_1p`, `log`
- Trig: `sin`, `cos`, `tan`, `sin_cos`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Misc: `abs`, `signum`, `floor`, `ceil`, `round`, `trunc`, `fract`, `mul_add`, `hypot`

#### Trait Implementations
- `num-traits`: `Float`, `Zero`, `One`, `Num`, `Signed`, `FloatConst`, `FromPrimitive`, `ToPrimitive`, `NumCast`
- `std::ops`: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Rem` with assign variants
- Mixed scalar ops (`Dual<f64> + f64`, `f64 * Reverse<f64>`, etc.)

#### Testing
- 94 tests: forward mode, reverse mode, API, and cross-validation
- Every elemental validated against central finite differences
- Forward-vs-reverse cross-validation on Rosenbrock, Beale, Ackley, Booth, and more
- Criterion benchmarks for forward overhead and reverse gradient

[Unreleased]: https://github.com/Entrolution/echidna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Entrolution/echidna/releases/tag/v0.1.0
