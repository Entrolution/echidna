// Batched forward tangent (JVP) evaluation on GPU.
//
// One thread per batch element. Each thread propagates both primal values and
// tangent derivatives through the tape using the forward-mode chain rule:
//   unary f(a):    tangent = f'(a) * a_tangent
//   binary f(a,b): tangent = df/da * a_tangent + df/db * b_tangent
//
// Used for sparse Jacobian: dispatch C colors in parallel, each with different
// tangent seeds.

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

const UNUSED: u32 = 0xFFFFFFFFu;

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
// binding 0: primal inputs [B * num_inputs] (same x for all colors, or different per batch)
@group(1) @binding(0) var<storage, read> primal_inputs: array<f32>;
// binding 1: tangent seeds [B * num_inputs] (different per color/batch element)
@group(1) @binding(1) var<storage, read> tangent_seeds: array<f32>;
// binding 2: primals working buffer [B * num_variables]
@group(1) @binding(2) var<storage, read_write> primals: array<f32>;
// binding 3: tangents working buffer [B * num_variables]
@group(1) @binding(3) var<storage, read_write> tangents: array<f32>;
// binding 4: tangent outputs [B * num_outputs]
@group(1) @binding(4) var<storage, read_write> tangent_outputs: array<f32>;

fn sinh_f(x: f32) -> f32 { return (exp(x) - exp(-x)) * 0.5; }
fn cosh_f(x: f32) -> f32 { return (exp(x) + exp(-x)) * 0.5; }

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

    let p_base = bid * nv; // primals base
    let t_base = bid * nv; // tangents base

    // Initialize primals from constants, tangents to zero
    for (var i = 0u; i < nv; i = i + 1u) {
        primals[p_base + i] = constants[i];
        tangents[t_base + i] = 0.0;
    }

    // Set input primals and tangent seeds
    let in_base = bid * ni;
    for (var i = 0u; i < ni; i = i + 1u) {
        primals[p_base + i] = primal_inputs[in_base + i];
        tangents[t_base + i] = tangent_seeds[in_base + i];
    }

    // Walk the tape: compute primals and propagate tangents
    for (var i = ni; i < num_ops; i = i + 1u) {
        let op = opcodes[i];
        if op == OP_CONST {
            continue;
        }

        let a_idx = arg0[i];
        let b_idx = arg1[i];

        let a = primals[p_base + a_idx];
        let at = tangents[t_base + a_idx];

        var r = 0.0f;
        var rt = 0.0f;

        switch op {
            case 2u /* ADD */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = a + b;
                rt = at + bt;
            }
            case 3u /* SUB */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = a - b;
                rt = at - bt;
            }
            case 4u /* MUL */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = a * b;
                rt = b * at + a * bt;
            }
            case 5u /* DIV */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = a / b;
                let inv = 1.0 / b;
                rt = inv * at - a * inv * inv * bt;
            }
            case 6u /* REM */: {
                r = a - trunc(a / primals[p_base + b_idx]) * primals[p_base + b_idx];
                rt = at; // d(a%b)/da = 1
            }
            case 7u /* POWF */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = pow(a, b);
                rt = r * (b / a * at + log(a) * bt);
            }
            case 8u /* ATAN2 */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = atan2(a, b);
                let d = a * a + b * b;
                rt = (b * at - a * bt) / d;
            }
            case 9u /* HYPOT */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                r = sqrt(a * a + b * b);
                rt = (a * at + b * bt) / r;
            }
            case 10u /* MAX */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                if a >= b { r = a; rt = at; } else { r = b; rt = bt; }
            }
            case 11u /* MIN */: {
                let b = primals[p_base + b_idx];
                let bt = tangents[t_base + b_idx];
                if a <= b { r = a; rt = at; } else { r = b; rt = bt; }
            }

            // Unary
            case 12u /* NEG */: { r = -a; rt = -at; }
            case 13u /* RECIP */: { r = 1.0 / a; rt = -at / (a * a); }
            case 14u /* SQRT */: { r = sqrt(a); rt = at / (2.0 * r); }
            case 15u /* CBRT */: {
                let s = sign(a);
                r = s * pow(abs(a), 1.0 / 3.0);
                rt = at / (3.0 * r * r);
            }
            case 16u /* POWI */: {
                let n = f32(bitcast<i32>(b_idx));
                r = pow(a, n);
                rt = n * pow(a, n - 1.0) * at;
            }
            case 17u /* EXP */: { r = exp(a); rt = r * at; }
            case 18u /* EXP2 */: { r = exp2(a); rt = r * log(2.0) * at; }
            case 19u /* EXPM1 */: { r = exp(a) - 1.0; rt = (r + 1.0) * at; }
            case 20u /* LN */: { r = log(a); rt = at / a; }
            case 21u /* LOG2 */: { r = log2(a); rt = at / (a * log(2.0)); }
            case 22u /* LOG10 */: { r = log(a) / log(10.0); rt = at / (a * log(10.0)); }
            case 23u /* LN1P */: { r = log(1.0 + a); rt = at / (1.0 + a); }
            case 24u /* SIN */: { r = sin(a); rt = cos(a) * at; }
            case 25u /* COS */: { r = cos(a); rt = -sin(a) * at; }
            case 26u /* TAN */: { r = tan(a); let c = cos(a); rt = at / (c * c); }
            case 27u /* ASIN */: { r = asin(a); rt = at / sqrt(1.0 - a * a); }
            case 28u /* ACOS */: { r = acos(a); rt = -at / sqrt(1.0 - a * a); }
            case 29u /* ATAN */: { r = atan(a); rt = at / (1.0 + a * a); }
            case 30u /* SINH */: { r = sinh_f(a); rt = cosh_f(a) * at; }
            case 31u /* COSH */: { r = cosh_f(a); rt = sinh_f(a) * at; }
            case 32u /* TANH */: { r = tanh(a); let c = cosh_f(a); rt = at / (c * c); }
            case 33u /* ASINH */: { r = log(a + sqrt(a * a + 1.0)); rt = at / sqrt(a * a + 1.0); }
            case 34u /* ACOSH */: { r = log(a + sqrt(a * a - 1.0)); rt = at / sqrt(a * a - 1.0); }
            case 35u /* ATANH */: { r = 0.5 * log((1.0 + a) / (1.0 - a)); rt = at / (1.0 - a * a); }
            case 36u /* ABS */: { r = abs(a); rt = sign(a) * at; }
            case 37u, 38u, 39u, 40u, 41u /* SIGNUM..TRUNC */: {
                // Zero derivative ops
                switch op {
                    case 37u: { r = sign(a); }
                    case 38u: { r = floor(a); }
                    case 39u: { r = ceil(a); }
                    case 40u: { r = round(a); }
                    case 41u: { r = trunc(a); }
                    default: {}
                }
                rt = 0.0;
            }
            case 42u /* FRACT */: { r = fract(a); rt = at; }
            default: {}
        }

        primals[p_base + i] = r;
        tangents[t_base + i] = rt;
    }

    // Write tangent outputs
    let out_base = bid * n_out;
    for (var j = 0u; j < n_out; j = j + 1u) {
        let oi = output_indices[j];
        tangent_outputs[out_base + j] = tangents[t_base + oi];
    }
}
