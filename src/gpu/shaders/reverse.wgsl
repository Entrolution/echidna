// Batched reverse (adjoint) sweep of a BytecodeTape on GPU.
//
// One compute thread per batch element. Each thread walks the tape in reverse,
// accumulating adjoints. The values buffer must already be populated by the
// forward shader.

// ── OpCode constants (must match OpCode repr(u8) discriminants) ──
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

// ── Tape data (bind group 0) ──
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

@group(0) @binding(0) var<storage, read> opcodes: array<u32>;
@group(0) @binding(1) var<storage, read> arg0: array<u32>;
@group(0) @binding(2) var<storage, read> arg1: array<u32>;
@group(0) @binding(3) var<storage, read> constants: array<f32>;
@group(0) @binding(4) var<uniform> tape_meta: TapeMeta;
@group(0) @binding(5) var<storage, read> output_indices: array<u32>;

// ── I/O buffers (bind group 1) ──
// binding 0: values [B * num_variables] (from forward pass, read-only here)
@group(1) @binding(0) var<storage, read> values: array<f32>;
// binding 1: adjoints [B * num_variables] (working memory, read-write)
@group(1) @binding(1) var<storage, read_write> adjoints: array<f32>;
// binding 2: grad_out [B * num_inputs] (output gradients)
@group(1) @binding(2) var<storage, read_write> grad_out: array<f32>;

// ── Main kernel ──

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_id = gid.x;
    if batch_id >= tape_meta.batch_size {
        return;
    }

    let num_vars = tape_meta.num_variables;
    let num_ops = tape_meta.num_ops;
    let num_in = tape_meta.num_inputs;
    let n_out = tape_meta.num_outputs;

    // Base offsets into per-thread buffers.
    let v_base = batch_id * num_vars; // values base
    let a_base = batch_id * num_vars; // adjoints base

    // Initialize adjoints to zero.
    for (var i = 0u; i < num_vars; i = i + 1u) {
        adjoints[a_base + i] = 0.0;
    }

    // Seed: set adjoint of the output to 1.0.
    // For single-output tapes, seed the first (only) output.
    let seed_idx = output_indices[0];
    adjoints[a_base + seed_idx] = 1.0;

    // Reverse sweep.
    for (var ii = 0u; ii < num_ops; ii = ii + 1u) {
        let i = num_ops - 1u - ii;
        let adj = adjoints[a_base + i];

        // Skip zero adjoints and structural ops.
        if adj == 0.0 {
            continue;
        }

        let op = opcodes[i];
        if op == OP_INPUT || op == OP_CONST {
            continue;
        }

        // Clear this adjoint (it's been consumed).
        adjoints[a_base + i] = 0.0;

        let a_idx = arg0[i];
        let b_idx = arg1[i];
        let a = values[v_base + a_idx];
        let r = values[v_base + i];

        // Compute reverse partials (da, db) and accumulate.
        var da = 0.0f;
        var db = 0.0f;

        switch op {
            // Binary
            case 2u /* ADD */: { da = 1.0; db = 1.0; }
            case 3u /* SUB */: { da = 1.0; db = -1.0; }
            case 4u /* MUL */: {
                let b = values[v_base + b_idx];
                da = b; db = a;
            }
            case 5u /* DIV */: {
                let b = values[v_base + b_idx];
                let inv = 1.0 / b;
                da = inv;
                db = -a * inv * inv;
            }
            case 6u /* REM */: { da = 1.0; db = 0.0; }
            case 7u /* POWF */: {
                let b = values[v_base + b_idx];
                da = b * pow(a, b - 1.0);
                db = r * log(a);
            }
            case 8u /* ATAN2 */: {
                let b = values[v_base + b_idx];
                let denom = a * a + b * b;
                da = b / denom;
                db = -a / denom;
            }
            case 9u /* HYPOT */: {
                let b = values[v_base + b_idx];
                da = a / r;
                db = b / r;
            }
            case 10u /* MAX */: {
                let b = values[v_base + b_idx];
                if a >= b { da = 1.0; db = 0.0; } else { da = 0.0; db = 1.0; }
            }
            case 11u /* MIN */: {
                let b = values[v_base + b_idx];
                if a <= b { da = 1.0; db = 0.0; } else { da = 0.0; db = 1.0; }
            }

            // Unary
            case 12u /* NEG */: { da = -1.0; }
            case 13u /* RECIP */: { let inv = 1.0 / a; da = -inv * inv; }
            case 14u /* SQRT */: { da = 0.5 / r; }
            case 15u /* CBRT */: { da = 1.0 / (3.0 * r * r); }
            case 16u /* POWI */: {
                let n = f32(bitcast<i32>(b_idx));
                da = n * pow(a, n - 1.0);
            }

            // Exp/Log
            case 17u /* EXP */: { da = r; }
            case 18u /* EXP2 */: { da = r * log(2.0); }
            case 19u /* EXPM1 */: { da = r + 1.0; }
            case 20u /* LN */: { da = 1.0 / a; }
            case 21u /* LOG2 */: { da = 1.0 / (a * log(2.0)); }
            case 22u /* LOG10 */: { da = 1.0 / (a * log(10.0)); }
            case 23u /* LN1P */: { da = 1.0 / (1.0 + a); }

            // Trig
            case 24u /* SIN */: { da = cos(a); }
            case 25u /* COS */: { da = -sin(a); }
            case 26u /* TAN */: { let c = cos(a); da = 1.0 / (c * c); }
            case 27u /* ASIN */: { da = 1.0 / sqrt(1.0 - a * a); }
            case 28u /* ACOS */: { da = -1.0 / sqrt(1.0 - a * a); }
            case 29u /* ATAN */: { da = 1.0 / (1.0 + a * a); }

            // Hyperbolic
            case 30u /* SINH */: { da = cosh(a); }
            case 31u /* COSH */: { da = sinh(a); }
            case 32u /* TANH */: { let c = cosh(a); da = 1.0 / (c * c); }
            case 33u /* ASINH */: { da = 1.0 / sqrt(a * a + 1.0); }
            case 34u /* ACOSH */: { da = 1.0 / sqrt(a * a - 1.0); }
            case 35u /* ATANH */: { da = 1.0 / (1.0 - a * a); }

            // Misc
            case 36u /* ABS */: { da = sign(a); }
            case 37u, 38u, 39u, 40u, 41u /* SIGNUM..TRUNC */: { da = 0.0; }
            case 42u /* FRACT */: { da = 1.0; }

            default: {}
        }

        // Accumulate
        adjoints[a_base + a_idx] = adjoints[a_base + a_idx] + da * adj;
        if b_idx != UNUSED && op != OP_POWI {
            adjoints[a_base + b_idx] = adjoints[a_base + b_idx] + db * adj;
        }
    }

    // Write gradients: input adjoints → grad_out.
    let g_base = batch_id * num_in;
    for (var i = 0u; i < num_in; i = i + 1u) {
        grad_out[g_base + i] = adjoints[a_base + i];
    }
}

// Manual implementations for WGSL builtins not available.
fn sinh(x: f32) -> f32 {
    return (exp(x) - exp(-x)) * 0.5;
}

fn cosh(x: f32) -> f32 {
    return (exp(x) + exp(-x)) * 0.5;
}
