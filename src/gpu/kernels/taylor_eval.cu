// Batched second-order Taylor forward propagation on CUDA.
//
// One thread per batch element. Each thread propagates a Taylor jet with
// 3 coefficients (c0=primal, c1=first-order, c2=second-order) through the
// tape. Used for GPU-accelerated STDE.
//
// Convention: c[k] = f^(k)(t₀) / k!  (normalized Taylor coefficients).
//
// Templated on FLOAT_TYPE (float or double) via preprocessor define
// injected before NVRTC compilation.

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif

typedef FLOAT_TYPE F;

// OpCode constants (must match OpCode #[repr(u8)] discriminants)
#define OP_INPUT  0u
#define OP_CONST  1u
#define OP_ADD    2u
#define OP_SUB    3u
#define OP_MUL    4u
#define OP_DIV    5u
#define OP_REM    6u
#define OP_POWF   7u
#define OP_ATAN2  8u
#define OP_HYPOT  9u
#define OP_MAX    10u
#define OP_MIN    11u
#define OP_NEG    12u
#define OP_RECIP  13u
#define OP_SQRT   14u
#define OP_CBRT   15u
#define OP_POWI   16u
#define OP_EXP    17u
#define OP_EXP2   18u
#define OP_EXPM1  19u
#define OP_LN     20u
#define OP_LOG2   21u
#define OP_LOG10  22u
#define OP_LN1P   23u
#define OP_SIN    24u
#define OP_COS    25u
#define OP_TAN    26u
#define OP_ASIN   27u
#define OP_ACOS   28u
#define OP_ATAN   29u
#define OP_SINH   30u
#define OP_COSH   31u
#define OP_TANH   32u
#define OP_ASINH  33u
#define OP_ACOSH  34u
#define OP_ATANH  35u
#define OP_ABS    36u
#define OP_SIGNUM 37u
#define OP_FLOOR  38u
#define OP_CEIL   39u
#define OP_ROUND  40u
#define OP_TRUNC  41u
#define OP_FRACT  42u

// ── Math helpers ──
__device__ F _sign(F x) { return (x > F(0)) - (x < F(0)); }
__device__ F _cbrt_f(F x) { return copysign(pow(fabs(x), F(1.0/3.0)), x); }
__device__ F _fract(F x) { return x - floor(x); }

// ── Jet3: 3 Taylor coefficients ──
struct Jet3 {
    F c0, c1, c2;
    __device__ Jet3() : c0(0), c1(0), c2(0) {}
    __device__ Jet3(F a, F b, F c) : c0(a), c1(b), c2(c) {}
};

// ── Arithmetic jet operations ──

__device__ Jet3 jet_add(Jet3 a, Jet3 b) {
    return Jet3(a.c0+b.c0, a.c1+b.c1, a.c2+b.c2);
}

__device__ Jet3 jet_sub(Jet3 a, Jet3 b) {
    return Jet3(a.c0-b.c0, a.c1-b.c1, a.c2-b.c2);
}

__device__ Jet3 jet_neg(Jet3 a) {
    return Jet3(-a.c0, -a.c1, -a.c2);
}

// Cauchy product
__device__ Jet3 jet_mul(Jet3 a, Jet3 b) {
    return Jet3(
        a.c0*b.c0,
        a.c0*b.c1 + a.c1*b.c0,
        a.c0*b.c2 + a.c1*b.c1 + a.c2*b.c0
    );
}

// Recursive division
__device__ Jet3 jet_div(Jet3 a, Jet3 b) {
    F inv_b0 = F(1) / b.c0;
    F c0 = a.c0 * inv_b0;
    F c1 = (a.c1 - b.c1*c0) * inv_b0;
    F c2 = (a.c2 - b.c1*c1 - b.c2*c0) * inv_b0;
    return Jet3(c0, c1, c2);
}

// Reciprocal
__device__ Jet3 jet_recip(Jet3 a) {
    F c0 = F(1) / a.c0;
    F c1 = -(a.c1*c0) * c0;
    F c2 = -(a.c1*c1 + a.c2*c0) * c0;
    return Jet3(c0, c1, c2);
}

__device__ Jet3 jet_scale(Jet3 a, F s) {
    return Jet3(a.c0*s, a.c1*s, a.c2*s);
}

// ── Transcendental jet operations ──

// exp: c[k] = (1/k) * Σ j*a[j]*c[k-j]
__device__ Jet3 jet_exp(Jet3 a) {
    F c0 = exp(a.c0);
    F c1 = a.c1 * c0;
    F c2 = F(0.5) * (a.c1*c1 + F(2)*a.c2*c0);
    return Jet3(c0, c1, c2);
}

// ln: c[k] = (a[k] - (1/k)*Σ j*c[j]*a[k-j]) / a[0]
__device__ Jet3 jet_ln(Jet3 a) {
    F inv_a0 = F(1) / a.c0;
    F c0 = log(a.c0);
    F c1 = a.c1 * inv_a0;
    F c2 = (a.c2 - F(0.5)*c1*a.c1) * inv_a0;
    return Jet3(c0, c1, c2);
}

// sqrt: c[k] = (a[k] - Σ c[j]*c[k-j]) / (2*c[0])
__device__ Jet3 jet_sqrt(Jet3 a) {
    F c0 = sqrt(a.c0);
    F inv_2c0 = F(0.5) / c0;
    F c1 = a.c1 * inv_2c0;
    F c2 = (a.c2 - c1*c1) * inv_2c0;
    return Jet3(c0, c1, c2);
}

// sin_cos: coupled recurrence, writes both sin and cos jets
__device__ void jet_sin_cos(Jet3 a, Jet3& s, Jet3& co) {
    F sv, cv;
    sincos(a.c0, &sv, &cv);
    s.c0 = sv; co.c0 = cv;
    s.c1 = a.c1 * cv;
    co.c1 = -a.c1 * sv;
    s.c2 = F(0.5) * (a.c1*co.c1 + F(2)*a.c2*cv);
    co.c2 = -F(0.5) * (a.c1*s.c1 + F(2)*a.c2*sv);
}

// sinh_cosh: coupled recurrence with positive signs
__device__ void jet_sinh_cosh(Jet3 a, Jet3& sh, Jet3& ch) {
    sh.c0 = sinh(a.c0); ch.c0 = cosh(a.c0);
    sh.c1 = a.c1 * ch.c0;
    ch.c1 = a.c1 * sh.c0;
    sh.c2 = F(0.5) * (a.c1*ch.c1 + F(2)*a.c2*ch.c0);
    ch.c2 = F(0.5) * (a.c1*sh.c1 + F(2)*a.c2*sh.c0);
}

// tan: c' = a' * (1+c²)
__device__ Jet3 jet_tan(Jet3 a) {
    F c0 = tan(a.c0);
    F s0 = F(1) + c0*c0;
    F c1 = a.c1 * s0;
    F s1 = F(2) * c0 * c1;
    F c2 = F(0.5) * (a.c1*s1 + F(2)*a.c2*s0);
    return Jet3(c0, c1, c2);
}

// tanh: c' = a' * (1-c²)
__device__ Jet3 jet_tanh(Jet3 a) {
    F c0 = tanh(a.c0);
    F s0 = F(1) - c0*c0;
    F c1 = a.c1 * s0;
    F s1 = F(-2) * c0 * c1;
    F c2 = F(0.5) * (a.c1*s1 + F(2)*a.c2*s0);
    return Jet3(c0, c1, c2);
}

// Integration step: c[k] = (1/k) * Σ j*a[j]*g[k-j]
// Returns (c1, c2) given derivative-factor jet g
__device__ void integrate3(F a1, F a2, F g0, F g1, F& c1, F& c2) {
    c1 = a1 * g0;
    c2 = F(0.5) * (a1*g1 + F(2)*a2*g0);
}

// atan: c' = a' / (1+a²)
__device__ Jet3 jet_atan(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(F(1)+asq.c0, asq.c1, asq.c2);
    Jet3 g = jet_recip(d);
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(atan(a.c0), c1, c2);
}

// asin: c' = a' / sqrt(1-a²)
__device__ Jet3 jet_asin(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(F(1)-asq.c0, -asq.c1, -asq.c2);
    Jet3 g = jet_recip(jet_sqrt(d));
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(asin(a.c0), c1, c2);
}

// acos: π/2 - asin (negate higher coefficients)
__device__ Jet3 jet_acos(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(F(1)-asq.c0, -asq.c1, -asq.c2);
    Jet3 g = jet_recip(jet_sqrt(d));
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(acos(a.c0), -c1, -c2);
}

// asinh: c' = a' / sqrt(1+a²)
__device__ Jet3 jet_asinh(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(F(1)+asq.c0, asq.c1, asq.c2);
    Jet3 g = jet_recip(jet_sqrt(d));
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(asinh(a.c0), c1, c2);
}

// acosh: c' = a' / sqrt(a²-1)
__device__ Jet3 jet_acosh(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(asq.c0-F(1), asq.c1, asq.c2);
    Jet3 g = jet_recip(jet_sqrt(d));
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(acosh(a.c0), c1, c2);
}

// atanh: c' = a' / (1-a²)
__device__ Jet3 jet_atanh(Jet3 a) {
    Jet3 asq = jet_mul(a, a);
    Jet3 d = Jet3(F(1)-asq.c0, -asq.c1, -asq.c2);
    Jet3 g = jet_recip(d);
    F c1, c2;
    integrate3(a.c1, a.c2, g.c0, g.c1, c1, c2);
    return Jet3(atanh(a.c0), c1, c2);
}

// ════════════════════════════════════════════════════════════════════
// Taylor forward 2nd-order kernel
// ════════════════════════════════════════════════════════════════════
extern "C" __global__ void taylor_forward_2nd(
    const unsigned int* __restrict__ opcodes,
    const unsigned int* __restrict__ arg0,
    const unsigned int* __restrict__ arg1,
    const F* __restrict__ constants,
    const F* __restrict__ primal_inputs,
    const F* __restrict__ direction_seeds,
    F* __restrict__ jets,            // [B * nv * 3] interleaved
    F* __restrict__ jet_outputs,     // [B * n_out * 3]
    const unsigned int* __restrict__ output_indices,
    unsigned int num_ops,
    unsigned int num_inputs,
    unsigned int num_variables,
    unsigned int num_outputs,
    unsigned int batch_size
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;

    unsigned int j_base = bid * num_variables * 3u;
    unsigned int in_base = bid * num_inputs;

    // Initialize all variables from constants: (const_val, 0, 0)
    for (unsigned int i = 0; i < num_variables; i++) {
        unsigned int off = j_base + i * 3u;
        jets[off + 0u] = constants[i];
        jets[off + 1u] = F(0);
        jets[off + 2u] = F(0);
    }

    // Set input variables: (primal, direction_seed, 0)
    for (unsigned int i = 0; i < num_inputs; i++) {
        unsigned int off = j_base + i * 3u;
        jets[off + 0u] = primal_inputs[in_base + i];
        jets[off + 1u] = direction_seeds[in_base + i];
    }

    // Walk the tape
    for (unsigned int i = num_inputs; i < num_ops; i++) {
        unsigned int op = opcodes[i];
        if (op == OP_CONST) continue;

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];

        // Read operand A
        unsigned int a_off = j_base + ai * 3u;
        Jet3 a(jets[a_off], jets[a_off+1u], jets[a_off+2u]);

        Jet3 r;

        switch (op) {
            // ── Binary arithmetic ──
            case OP_ADD: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = jet_add(a, b); break;
            }
            case OP_SUB: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = jet_sub(a, b); break;
            }
            case OP_MUL: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = jet_mul(a, b); break;
            }
            case OP_DIV: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = jet_div(a, b); break;
            }
            case OP_REM: {
                F bval = jets[j_base + bi * 3u];
                r = Jet3(fmod(a.c0, bval), F(0), F(0)); break;
            }
            case OP_POWF: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                Jet3 lna = jet_ln(a);
                Jet3 product = jet_mul(b, lna);
                r = jet_exp(product);
                r.c0 = pow(a.c0, b.c0); break;
            }
            case OP_ATAN2: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                Jet3 ratio = jet_div(a, b);
                Jet3 at = jet_atan(ratio);
                r = Jet3(atan2(a.c0, b.c0), at.c1, at.c2); break;
            }
            case OP_HYPOT: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                Jet3 asq = jet_mul(a, a);
                Jet3 bsq = jet_mul(b, b);
                Jet3 s = jet_add(asq, bsq);
                r = jet_sqrt(s);
                r.c0 = hypot(a.c0, b.c0); break;
            }
            case OP_MAX: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = (a.c0 >= b.c0) ? a : b; break;
            }
            case OP_MIN: {
                unsigned int b_off = j_base + bi * 3u;
                Jet3 b(jets[b_off], jets[b_off+1u], jets[b_off+2u]);
                r = (a.c0 <= b.c0) ? a : b; break;
            }

            // ── Unary arithmetic ──
            case OP_NEG: r = jet_neg(a); break;
            case OP_RECIP: r = jet_recip(a); break;
            case OP_SQRT: r = jet_sqrt(a); break;
            case OP_CBRT: {
                F s = _sign(a.c0);
                Jet3 abs_a(fabs(a.c0), s*a.c1, s*a.c2);
                Jet3 lna = jet_ln(abs_a);
                Jet3 third = jet_scale(lna, F(1.0/3.0));
                Jet3 e = jet_exp(third);
                r = Jet3(s*e.c0, s*e.c1, s*e.c2); break;
            }
            case OP_POWI: {
                int n = (int)bi;
                F fn = F(n);
                if (n == 0) {
                    r = Jet3(F(1), F(0), F(0));
                } else if (n == 1) {
                    r = a;
                } else {
                    Jet3 lna = jet_ln(a);
                    Jet3 nlna = jet_scale(lna, fn);
                    r = jet_exp(nlna);
                    r.c0 = pow(a.c0, fn);
                }
                break;
            }

            // ── Transcendental ──
            case OP_EXP: r = jet_exp(a); break;
            case OP_EXP2: {
                F ln2 = log(F(2));
                Jet3 scaled = jet_scale(a, ln2);
                r = jet_exp(scaled);
                r.c0 = exp2(a.c0); break;
            }
            case OP_EXPM1: {
                r = jet_exp(a);
                r.c0 = expm1(a.c0); break;
            }
            case OP_LN: r = jet_ln(a); break;
            case OP_LOG2: {
                F inv_ln2 = F(1) / log(F(2));
                r = jet_ln(a);
                r.c0 = log2(a.c0);
                r.c1 *= inv_ln2;
                r.c2 *= inv_ln2; break;
            }
            case OP_LOG10: {
                F inv_ln10 = F(1) / log(F(10));
                r = jet_ln(a);
                r.c0 = log10(a.c0);
                r.c1 *= inv_ln10;
                r.c2 *= inv_ln10; break;
            }
            case OP_LN1P: {
                Jet3 one_plus_a(F(1)+a.c0, a.c1, a.c2);
                r = jet_ln(one_plus_a);
                r.c0 = log1p(a.c0); break;
            }
            case OP_SIN: {
                Jet3 sj, cj;
                jet_sin_cos(a, sj, cj);
                r = sj; break;
            }
            case OP_COS: {
                Jet3 sj, cj;
                jet_sin_cos(a, sj, cj);
                r = cj; break;
            }
            case OP_TAN: r = jet_tan(a); break;
            case OP_ASIN: r = jet_asin(a); break;
            case OP_ACOS: r = jet_acos(a); break;
            case OP_ATAN: r = jet_atan(a); break;
            case OP_SINH: {
                Jet3 sh, ch;
                jet_sinh_cosh(a, sh, ch);
                r = sh; break;
            }
            case OP_COSH: {
                Jet3 sh, ch;
                jet_sinh_cosh(a, sh, ch);
                r = ch; break;
            }
            case OP_TANH: r = jet_tanh(a); break;
            case OP_ASINH: r = jet_asinh(a); break;
            case OP_ACOSH: r = jet_acosh(a); break;
            case OP_ATANH: r = jet_atanh(a); break;

            // ── Nonsmooth ──
            case OP_ABS: {
                F s = _sign(a.c0);
                r = Jet3(fabs(a.c0), s*a.c1, s*a.c2); break;
            }
            case OP_SIGNUM: r = Jet3(_sign(a.c0), F(0), F(0)); break;
            case OP_FLOOR:  r = Jet3(floor(a.c0), F(0), F(0)); break;
            case OP_CEIL:   r = Jet3(ceil(a.c0), F(0), F(0)); break;
            case OP_ROUND:  r = Jet3(round(a.c0), F(0), F(0)); break;
            case OP_TRUNC:  r = Jet3(trunc(a.c0), F(0), F(0)); break;
            case OP_FRACT:  r = Jet3(_fract(a.c0), a.c1, a.c2); break;
            default: break;
        }

        // Write result jet
        unsigned int r_off = j_base + i * 3u;
        jets[r_off + 0u] = r.c0;
        jets[r_off + 1u] = r.c1;
        jets[r_off + 2u] = r.c2;
    }

    // Write output jets
    unsigned int out_base = bid * num_outputs * 3u;
    for (unsigned int j = 0; j < num_outputs; j++) {
        unsigned int oi = output_indices[j];
        unsigned int src = j_base + oi * 3u;
        unsigned int dst = out_base + j * 3u;
        jet_outputs[dst + 0u] = jets[src + 0u];
        jet_outputs[dst + 1u] = jets[src + 1u];
        jet_outputs[dst + 2u] = jets[src + 2u];
    }
}
