// Batched second-order Taylor forward propagation on GPU.
//
// One thread per batch element. Each thread propagates a Taylor jet with
// 3 coefficients (c0=primal, c1=first-order, c2=second-order) through the
// tape. Used for GPU-accelerated STDE: Laplacian estimation, Hessian
// diagonal, and directional second derivatives.
//
// Convention: c[k] = f^(k)(t₀) / k!  (normalized Taylor coefficients).
//
// Memory layout: interleaved [c0, c1, c2] per variable per batch element.

// ── OpCode constants ──
const OP_INPUT:  u32 = 0u;
const OP_CONST:  u32 = 1u;
const OP_ADD:    u32 = 2u;
const OP_SUB:    u32 = 3u;
const OP_MUL:    u32 = 4u;
const OP_DIV:    u32 = 5u;
const OP_REM:    u32 = 6u;
const OP_POWF:   u32 = 7u;
const OP_ATAN2:  u32 = 8u;
const OP_HYPOT:  u32 = 9u;
const OP_MAX:    u32 = 10u;
const OP_MIN:    u32 = 11u;
const OP_NEG:    u32 = 12u;
const OP_RECIP:  u32 = 13u;
const OP_SQRT:   u32 = 14u;
const OP_CBRT:   u32 = 15u;
const OP_POWI:   u32 = 16u;
const OP_EXP:    u32 = 17u;
const OP_EXP2:   u32 = 18u;
const OP_EXPM1:  u32 = 19u;
const OP_LN:     u32 = 20u;
const OP_LOG2:   u32 = 21u;
const OP_LOG10:  u32 = 22u;
const OP_LN1P:   u32 = 23u;
const OP_SIN:    u32 = 24u;
const OP_COS:    u32 = 25u;
const OP_TAN:    u32 = 26u;
const OP_ASIN:   u32 = 27u;
const OP_ACOS:   u32 = 28u;
const OP_ATAN:   u32 = 29u;
const OP_SINH:   u32 = 30u;
const OP_COSH:   u32 = 31u;
const OP_TANH:   u32 = 32u;
const OP_ASINH:  u32 = 33u;
const OP_ACOSH:  u32 = 34u;
const OP_ATANH:  u32 = 35u;
const OP_ABS:    u32 = 36u;
const OP_SIGNUM: u32 = 37u;
const OP_FLOOR:  u32 = 38u;
const OP_CEIL:   u32 = 39u;
const OP_ROUND:  u32 = 40u;
const OP_TRUNC:  u32 = 41u;
const OP_FRACT:  u32 = 42u;

struct TapeMeta {
    num_ops: u32,
    num_inputs: u32,
    num_variables: u32,
    num_outputs: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Tape data (bind group 0) ──
@group(0) @binding(0) var<storage, read> opcodes: array<u32>;
@group(0) @binding(1) var<storage, read> arg0: array<u32>;
@group(0) @binding(2) var<storage, read> arg1: array<u32>;
@group(0) @binding(3) var<storage, read> constants: array<f32>;
@group(0) @binding(4) var<uniform> tape_meta: TapeMeta;
@group(0) @binding(5) var<storage, read> output_indices: array<u32>;

// ── I/O buffers (bind group 1) ──
// binding 0: primal inputs [B * num_inputs]
@group(1) @binding(0) var<storage, read> primal_inputs: array<f32>;
// binding 1: direction seeds [B * num_inputs] (c1 for input variables)
@group(1) @binding(1) var<storage, read> direction_seeds: array<f32>;
// binding 2: jets working buffer [B * num_variables * 3] (interleaved c0,c1,c2)
@group(1) @binding(2) var<storage, read_write> jets: array<f32>;
// binding 3: jet outputs [B * num_outputs * 3]
@group(1) @binding(3) var<storage, read_write> jet_outputs: array<f32>;

// ── Helpers for missing WGSL builtins ──
fn sinh_f(x: f32) -> f32 { return (exp(x) - exp(-x)) * 0.5; }
fn cosh_f(x: f32) -> f32 { return (exp(x) + exp(-x)) * 0.5; }
fn asinh_f(x: f32) -> f32 { return log(x + sqrt(x * x + 1.0)); }
fn acosh_f(x: f32) -> f32 { return log(x + sqrt(x * x - 1.0)); }
fn atanh_f(x: f32) -> f32 { return 0.5 * log((1.0 + x) / (1.0 - x)); }

// ── Jet3 type: 3 Taylor coefficients ──
struct Jet3 { c0: f32, c1: f32, c2: f32, }

fn jet_const(v: f32) -> Jet3 { return Jet3(v, 0.0, 0.0); }

// ── Arithmetic jet operations ──

fn jet_add(a: Jet3, b: Jet3) -> Jet3 {
    return Jet3(a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2);
}

fn jet_sub(a: Jet3, b: Jet3) -> Jet3 {
    return Jet3(a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2);
}

fn jet_neg(a: Jet3) -> Jet3 {
    return Jet3(-a.c0, -a.c1, -a.c2);
}

// Cauchy product: c[k] = Σ_{j=0}^{k} a[j]*b[k-j]
fn jet_mul(a: Jet3, b: Jet3) -> Jet3 {
    return Jet3(
        a.c0 * b.c0,
        a.c0 * b.c1 + a.c1 * b.c0,
        a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0,
    );
}

// Recursive division: c[k] = (a[k] - Σ_{j=1}^{k} b[j]*c[k-j]) / b[0]
fn jet_div(a: Jet3, b: Jet3) -> Jet3 {
    var c: Jet3;
    let inv_b0 = 1.0 / b.c0;
    c.c0 = a.c0 * inv_b0;
    c.c1 = (a.c1 - b.c1 * c.c0) * inv_b0;
    c.c2 = (a.c2 - b.c1 * c.c1 - b.c2 * c.c0) * inv_b0;
    return c;
}

// Reciprocal: c = 1/a
fn jet_recip(a: Jet3) -> Jet3 {
    var c: Jet3;
    c.c0 = 1.0 / a.c0;
    c.c1 = -(a.c1 * c.c0) * c.c0;
    c.c2 = -(a.c1 * c.c1 + a.c2 * c.c0) * c.c0;
    return c;
}

fn jet_scale(a: Jet3, s: f32) -> Jet3 {
    return Jet3(a.c0 * s, a.c1 * s, a.c2 * s);
}

// ── Transcendental jet operations ──

// exp: c[k] = (1/k) * Σ_{j=1}^{k} j*a[j]*c[k-j]
fn jet_exp(a: Jet3) -> Jet3 {
    var c: Jet3;
    c.c0 = exp(a.c0);
    c.c1 = a.c1 * c.c0;
    c.c2 = 0.5 * (a.c1 * c.c1 + 2.0 * a.c2 * c.c0);
    return c;
}

// ln: c[k] = (a[k] - (1/k)*Σ_{j=1}^{k-1} j*c[j]*a[k-j]) / a[0]
fn jet_ln(a: Jet3) -> Jet3 {
    var c: Jet3;
    let inv_a0 = 1.0 / a.c0;
    c.c0 = log(a.c0);
    c.c1 = a.c1 * inv_a0;
    c.c2 = (a.c2 - 0.5 * c.c1 * a.c1) * inv_a0;
    return c;
}

// sqrt: c[k] = (a[k] - Σ_{j=1}^{k-1} c[j]*c[k-j]) / (2*c[0])
fn jet_sqrt(a: Jet3) -> Jet3 {
    var c: Jet3;
    c.c0 = sqrt(a.c0);
    let inv_2c0 = 0.5 / c.c0;
    c.c1 = a.c1 * inv_2c0;
    c.c2 = (a.c2 - c.c1 * c.c1) * inv_2c0;
    return c;
}

// sin_cos: coupled recurrence, returns (sin_jet, cos_jet)
// s[k] = (1/k) * Σ j*a[j]*co[k-j], co[k] = -(1/k) * Σ j*a[j]*s[k-j]
struct JetPair { a: Jet3, b: Jet3, }

fn jet_sin_cos(a: Jet3) -> JetPair {
    var s: Jet3;
    var co: Jet3;
    s.c0 = sin(a.c0);
    co.c0 = cos(a.c0);
    s.c1 = a.c1 * co.c0;
    co.c1 = -a.c1 * s.c0;
    s.c2 = 0.5 * (a.c1 * co.c1 + 2.0 * a.c2 * co.c0);
    co.c2 = -0.5 * (a.c1 * s.c1 + 2.0 * a.c2 * s.c0);
    return JetPair(s, co);
}

// sinh_cosh: coupled recurrence with positive signs
fn jet_sinh_cosh(a: Jet3) -> JetPair {
    var sh: Jet3;
    var ch: Jet3;
    sh.c0 = sinh_f(a.c0);
    ch.c0 = cosh_f(a.c0);
    sh.c1 = a.c1 * ch.c0;
    ch.c1 = a.c1 * sh.c0;
    sh.c2 = 0.5 * (a.c1 * ch.c1 + 2.0 * a.c2 * ch.c0);
    ch.c2 = 0.5 * (a.c1 * sh.c1 + 2.0 * a.c2 * sh.c0);
    return JetPair(sh, ch);
}

// tan: c' = a' * (1+c²), scratch s = 1+c²
fn jet_tan(a: Jet3) -> Jet3 {
    var c: Jet3;
    c.c0 = tan(a.c0);
    var s0 = 1.0 + c.c0 * c.c0;
    c.c1 = a.c1 * s0;
    let s1 = 2.0 * c.c0 * c.c1;
    c.c2 = 0.5 * (a.c1 * s1 + 2.0 * a.c2 * s0);
    return c;
}

// tanh: c' = a' * (1-c²), scratch s = 1-c²
fn jet_tanh(a: Jet3) -> Jet3 {
    var c: Jet3;
    c.c0 = tanh(a.c0);
    var s0 = 1.0 - c.c0 * c.c0;
    c.c1 = a.c1 * s0;
    let s1 = -2.0 * c.c0 * c.c1;
    c.c2 = 0.5 * (a.c1 * s1 + 2.0 * a.c2 * s0);
    return c;
}

// Integration step: c[k] = (1/k) * Σ j*a[j]*g[k-j]
// Returns (c1, c2) given derivative-factor jet g
fn integrate3(a1: f32, a2: f32, g0: f32, g1: f32) -> vec2<f32> {
    return vec2<f32>(
        a1 * g0,
        0.5 * (a1 * g1 + 2.0 * a2 * g0),
    );
}

// atan: c' = a' / (1+a²)
fn jet_atan(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(1.0 + asq.c0, asq.c1, asq.c2);
    let g = jet_recip(d);
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(atan(a.c0), hi.x, hi.y);
}

// asin: c' = a' / sqrt(1-a²)
fn jet_asin(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(1.0 - asq.c0, -asq.c1, -asq.c2);
    let g = jet_recip(jet_sqrt(d));
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(asin(a.c0), hi.x, hi.y);
}

// acos: π/2 - asin (negate higher coefficients)
fn jet_acos(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(1.0 - asq.c0, -asq.c1, -asq.c2);
    let g = jet_recip(jet_sqrt(d));
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(acos(a.c0), -hi.x, -hi.y);
}

// asinh: c' = a' / sqrt(1+a²)
fn jet_asinh(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(1.0 + asq.c0, asq.c1, asq.c2);
    let g = jet_recip(jet_sqrt(d));
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(asinh_f(a.c0), hi.x, hi.y);
}

// acosh: c' = a' / sqrt(a²-1)
fn jet_acosh(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(asq.c0 - 1.0, asq.c1, asq.c2);
    let g = jet_recip(jet_sqrt(d));
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(acosh_f(a.c0), hi.x, hi.y);
}

// atanh: c' = a' / (1-a²)
fn jet_atanh(a: Jet3) -> Jet3 {
    let asq = jet_mul(a, a);
    let d = Jet3(1.0 - asq.c0, -asq.c1, -asq.c2);
    let g = jet_recip(d);
    let hi = integrate3(a.c1, a.c2, g.c0, g.c1);
    return Jet3(atanh_f(a.c0), hi.x, hi.y);
}

// ── Main compute kernel ──

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= tape_meta.batch_size {
        return;
    }

    let nv = tape_meta.num_variables;
    let ni = tape_meta.num_inputs;
    let num_ops = tape_meta.num_ops;
    let n_out = tape_meta.num_outputs;

    let j_base = bid * nv * 3u;  // jets base for this batch element

    // Initialize all variables from constants: (const_val, 0, 0)
    for (var i = 0u; i < nv; i = i + 1u) {
        let off = j_base + i * 3u;
        jets[off + 0u] = constants[i];
        jets[off + 1u] = 0.0;
        jets[off + 2u] = 0.0;
    }

    // Set input variables: (primal, direction_seed, 0)
    let in_base = bid * ni;
    for (var i = 0u; i < ni; i = i + 1u) {
        let off = j_base + i * 3u;
        jets[off + 0u] = primal_inputs[in_base + i];
        jets[off + 1u] = direction_seeds[in_base + i];
        // c2 stays 0
    }

    // Walk the tape
    for (var i = ni; i < num_ops; i = i + 1u) {
        let op = opcodes[i];
        if op == OP_CONST {
            continue;
        }

        let a_idx = arg0[i];
        let b_idx = arg1[i];

        // Read operand A
        let a_off = j_base + a_idx * 3u;
        let a = Jet3(jets[a_off], jets[a_off + 1u], jets[a_off + 2u]);

        var r = Jet3(0.0, 0.0, 0.0);

        switch op {
            // ── Binary arithmetic ──
            case 2u /* ADD */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                r = jet_add(a, b);
            }
            case 3u /* SUB */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                r = jet_sub(a, b);
            }
            case 4u /* MUL */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                r = jet_mul(a, b);
            }
            case 5u /* DIV */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                r = jet_div(a, b);
            }
            case 6u /* REM */: {
                // No meaningful Taylor rule — zero higher-order coefficients
                let b_val = jets[j_base + b_idx * 3u];
                r = Jet3(a.c0 - trunc(a.c0 / b_val) * b_val, 0.0, 0.0);
            }
            case 7u /* POWF */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                // c = exp(b * ln(a))
                let lna = jet_ln(a);
                let product = jet_mul(b, lna);
                r = jet_exp(product);
                r.c0 = pow(a.c0, b.c0);
            }
            case 8u /* ATAN2 */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                // atan2(a, b) = atan(a/b) with quadrant fix
                let ratio = jet_div(a, b);
                let at = jet_atan(ratio);
                r = Jet3(atan2(a.c0, b.c0), at.c1, at.c2);
            }
            case 9u /* HYPOT */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                // hypot(a, b) = sqrt(a²+b²)
                let asq = jet_mul(a, a);
                let bsq = jet_mul(b, b);
                let sum = jet_add(asq, bsq);
                r = jet_sqrt(sum);
                r.c0 = sqrt(a.c0 * a.c0 + b.c0 * b.c0);
            }
            case 10u /* MAX */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                if a.c0 >= b.c0 { r = a; } else { r = b; }
            }
            case 11u /* MIN */: {
                let b_off = j_base + b_idx * 3u;
                let b = Jet3(jets[b_off], jets[b_off + 1u], jets[b_off + 2u]);
                if a.c0 <= b.c0 { r = a; } else { r = b; }
            }

            // ── Unary arithmetic ──
            case 12u /* NEG */: { r = jet_neg(a); }
            case 13u /* RECIP */: { r = jet_recip(a); }
            case 14u /* SQRT */: { r = jet_sqrt(a); }
            case 15u /* CBRT */: {
                // cbrt(a) = sign(a) * exp(ln(|a|)/3)
                let s = sign(a.c0);
                let abs_a = Jet3(abs(a.c0), s * a.c1, s * a.c2);
                let lna = jet_ln(abs_a);
                let third = jet_scale(lna, 1.0 / 3.0);
                let e = jet_exp(third);
                r = Jet3(s * e.c0, s * e.c1, s * e.c2);
            }
            case 16u /* POWI */: {
                // Exponent n is stored in arg1 as bitcast i32, NOT a buffer index
                let n = f32(bitcast<i32>(b_idx));
                if n == 0.0 {
                    r = Jet3(1.0, 0.0, 0.0);
                } else if n == 1.0 {
                    r = a;
                } else {
                    // exp(n * ln(a))
                    let lna = jet_ln(a);
                    let nlna = jet_scale(lna, n);
                    r = jet_exp(nlna);
                    r.c0 = pow(a.c0, n);
                }
            }

            // ── Transcendental (unary) ──
            case 17u /* EXP */: { r = jet_exp(a); }
            case 18u /* EXP2 */: {
                // exp2(a) = exp(a * ln(2))
                let ln2 = log(2.0);
                let scaled = jet_scale(a, ln2);
                r = jet_exp(scaled);
                r.c0 = exp2(a.c0);
            }
            case 19u /* EXPM1 */: {
                // expm1(a) = exp(a) - 1 (higher coefficients same as exp)
                r = jet_exp(a);
                r.c0 = exp(a.c0) - 1.0;
            }
            case 20u /* LN */: { r = jet_ln(a); }
            case 21u /* LOG2 */: {
                // log2(a) = ln(a) / ln(2)
                r = jet_ln(a);
                let inv_ln2 = 1.0 / log(2.0);
                r.c0 = log2(a.c0);
                r.c1 = r.c1 * inv_ln2;
                r.c2 = r.c2 * inv_ln2;
            }
            case 22u /* LOG10 */: {
                // log10(a) = ln(a) / ln(10)
                r = jet_ln(a);
                let inv_ln10 = 1.0 / log(10.0);
                r.c0 = log(a.c0) * inv_ln10;
                r.c1 = r.c1 * inv_ln10;
                r.c2 = r.c2 * inv_ln10;
            }
            case 23u /* LN1P */: {
                // ln(1+a)
                let one_plus_a = Jet3(1.0 + a.c0, a.c1, a.c2);
                r = jet_ln(one_plus_a);
                r.c0 = log(1.0 + a.c0);
            }
            case 24u /* SIN */: {
                let sc = jet_sin_cos(a);
                r = sc.a;
            }
            case 25u /* COS */: {
                let sc = jet_sin_cos(a);
                r = sc.b;
            }
            case 26u /* TAN */: { r = jet_tan(a); }
            case 27u /* ASIN */: { r = jet_asin(a); }
            case 28u /* ACOS */: { r = jet_acos(a); }
            case 29u /* ATAN */: { r = jet_atan(a); }
            case 30u /* SINH */: {
                let sc = jet_sinh_cosh(a);
                r = sc.a;
            }
            case 31u /* COSH */: {
                let sc = jet_sinh_cosh(a);
                r = sc.b;
            }
            case 32u /* TANH */: { r = jet_tanh(a); }
            case 33u /* ASINH */: { r = jet_asinh(a); }
            case 34u /* ACOSH */: { r = jet_acosh(a); }
            case 35u /* ATANH */: { r = jet_atanh(a); }

            // ── Nonsmooth ──
            case 36u /* ABS */: {
                let s = sign(a.c0);
                r = Jet3(abs(a.c0), s * a.c1, s * a.c2);
            }
            case 37u, 38u, 39u, 40u, 41u /* SIGNUM..TRUNC */: {
                // Zero higher-order coefficients
                var val = 0.0f;
                switch op {
                    case 37u: { val = sign(a.c0); }
                    case 38u: { val = floor(a.c0); }
                    case 39u: { val = ceil(a.c0); }
                    case 40u: { val = round(a.c0); }
                    case 41u: { val = trunc(a.c0); }
                    default: {}
                }
                r = Jet3(val, 0.0, 0.0);
            }
            case 42u /* FRACT */: {
                // fract has same derivatives as the argument
                r = Jet3(fract(a.c0), a.c1, a.c2);
            }
            default: {}
        }

        // Write result jet
        let r_off = j_base + i * 3u;
        jets[r_off + 0u] = r.c0;
        jets[r_off + 1u] = r.c1;
        jets[r_off + 2u] = r.c2;
    }

    // Write output jets
    let out_base = bid * n_out * 3u;
    for (var j = 0u; j < n_out; j = j + 1u) {
        let oi = output_indices[j];
        let src = j_base + oi * 3u;
        let dst = out_base + j * 3u;
        jet_outputs[dst + 0u] = jets[src + 0u];
        jet_outputs[dst + 1u] = jets[src + 1u];
        jet_outputs[dst + 2u] = jets[src + 2u];
    }
}
