use echidna::Scalar;

// ─── Rosenbrock ────────────────────────────────────────────────────────────

pub fn rosenbrock<T: Scalar>(x: &[T]) -> T {
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

pub fn rosenbrock_f64(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = 1.0 - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        sum += t1 * t1 + 100.0 * t2 * t2;
    }
    sum
}

// ─── Tridiagonal ───────────────────────────────────────────────────────────

pub fn tridiagonal<T: Scalar>(x: &[T]) -> T {
    let mut sum = x[0] - x[0]; // zero with correct type
    for i in 0..x.len() - 1 {
        sum = sum + x[i] * x[i + 1];
    }
    sum
}

// ─── Rastrigin ─────────────────────────────────────────────────────────────
// f(x) = 10n + Σ[x_i² - 10·cos(2π·x_i)]
// Diagonal Hessian, exercises sin/cos.

pub fn rastrigin<T: Scalar>(x: &[T]) -> T {
    let ten = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(10.0).unwrap());
    let two_pi = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(
        2.0 * std::f64::consts::PI,
    ).unwrap());
    let n = T::from_f(<T::Float as num_traits::FromPrimitive>::from_usize(x.len()).unwrap());
    let mut sum = ten * n;
    for &xi in x {
        sum = sum + xi * xi - ten * (two_pi * xi).cos();
    }
    sum
}

// ─── Neural Network Layer ──────────────────────────────────────────────────
// f(x) = Σ_j sigmoid(Σ_i w_ji·x_i + b_j), 4 hidden units
// Deterministic weights: w_ji = sin(j*N+i+1) / (N+1), b_j = 0.1*(j+1)
// Dense Jacobian, exercises exp/add/mul.

pub fn nn_layer<T: Scalar>(x: &[T]) -> T {
    let n = x.len();
    let one = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(1.0).unwrap());
    let scale = 1.0 / (n as f64 + 1.0);
    let mut total = T::zero();
    for j in 0..4_usize {
        let mut z = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(
            0.1 * (j as f64 + 1.0),
        ).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let w = (j * n + i + 1) as f64;
            let w = w.sin() * scale;
            let w = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(w).unwrap());
            z = z + w * xi;
        }
        // sigmoid(z) = 1 / (1 + exp(-z))
        total = total + one / (one + (-z).exp());
    }
    total
}

// ─── PDE Poisson Residual (scalar) ─────────────────────────────────────────
// 1D Poisson residual: Σ_i r_i², where r_i = -u_{i-1} + 2u_i - u_{i+1} - h²
// x = interior node values, Dirichlet BCs u_0 = u_{N+1} = 0, h = 1/(N+1)

pub fn pde_poisson<T: Scalar>(x: &[T]) -> T {
    let n = x.len();
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(h * h).unwrap());
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    let zero = T::zero();
    let mut sum = T::zero();
    for i in 0..n {
        let u_prev = if i == 0 { zero } else { x[i - 1] };
        let u_next = if i == n - 1 { zero } else { x[i + 1] };
        let r = two * x[i] - u_prev - u_next - h2;
        sum = sum + r * r;
    }
    sum
}

// ─── PDE Poisson Residual (vector) ─────────────────────────────────────────
// Returns each residual r_i individually (for Jacobian benchmarks).

pub fn pde_poisson_vec<T: Scalar>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(h * h).unwrap());
    let two = T::from_f(<T::Float as num_traits::FromPrimitive>::from_f64(2.0).unwrap());
    let zero = T::zero();
    (0..n)
        .map(|i| {
            let u_prev = if i == 0 { zero } else { x[i - 1] };
            let u_next = if i == n - 1 { zero } else { x[i + 1] };
            two * x[i] - u_prev - u_next - h2
        })
        .collect()
}

// ─── Finite Differences ────────────────────────────────────────────────────

pub fn finite_diff_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    grad
}

// ─── Helpers ───────────────────────────────────────────────────────────────

pub fn make_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.01 * i as f64).collect()
}

pub fn make_direction(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.1 * (i + 1) as f64).collect()
}
