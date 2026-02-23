// Batched forward evaluation of a BytecodeTape on GPU.
//
// One compute thread per batch element. Each thread walks the tape sequentially,
// maintaining a private section of the values buffer.

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
@group(1) @binding(0) var<storage, read> inputs: array<f32>;
@group(1) @binding(1) var<storage, read_write> values: array<f32>;
@group(1) @binding(2) var<storage, read_write> outputs: array<f32>;

// ── Manual implementations for functions not in WGSL ──

fn cbrt_f32(x: f32) -> f32 {
    // cbrt(x) = sign(x) * |x|^(1/3)
    let s = sign(x);
    return s * pow(abs(x), 1.0 / 3.0);
}

fn expm1_f32(x: f32) -> f32 {
    // For small |x|, exp(x)-1 loses precision, but f32 limits make this acceptable.
    return exp(x) - 1.0;
}

fn ln1p_f32(x: f32) -> f32 {
    return log(1.0 + x);
}

fn sinh_f32(x: f32) -> f32 {
    return (exp(x) - exp(-x)) * 0.5;
}

fn cosh_f32(x: f32) -> f32 {
    return (exp(x) + exp(-x)) * 0.5;
}

fn asinh_f32(x: f32) -> f32 {
    // asinh(x) = ln(x + sqrt(x^2 + 1))
    return log(x + sqrt(x * x + 1.0));
}

fn acosh_f32(x: f32) -> f32 {
    // acosh(x) = ln(x + sqrt(x^2 - 1))
    return log(x + sqrt(x * x - 1.0));
}

fn atanh_f32(x: f32) -> f32 {
    // atanh(x) = 0.5 * ln((1+x)/(1-x))
    return 0.5 * log((1.0 + x) / (1.0 - x));
}

fn hypot_f32(a: f32, b: f32) -> f32 {
    return sqrt(a * a + b * b);
}

fn rem_f32(a: f32, b: f32) -> f32 {
    // Rust's % is remainder (truncated), matching: a - trunc(a/b) * b
    return a - trunc(a / b) * b;
}

fn recip_f32(x: f32) -> f32 {
    return 1.0 / x;
}

fn log10_f32(x: f32) -> f32 {
    return log(x) / log(10.0);
}

fn signum_f32(x: f32) -> f32 {
    if x > 0.0 { return 1.0; }
    if x < 0.0 { return -1.0; }
    return 0.0;
}

fn powi_f32(base: f32, exp_bits: u32) -> f32 {
    // The exponent is stored as i32 reinterpreted as u32.
    let n = bitcast<i32>(exp_bits);
    return pow(base, f32(n));
}

// ── Main kernel ──

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_id = gid.x;
    // Guard: skip threads beyond the batch size (last workgroup padding).
    if batch_id >= tape_meta.batch_size {
        return;
    }
    let num_vars = tape_meta.num_variables;
    let num_in = tape_meta.num_inputs;
    let num_ops = tape_meta.num_ops;
    let n_out = tape_meta.num_outputs;

    // Base offset into the per-thread values section.
    let base = batch_id * num_vars;

    // Initialize values: copy constants, then overwrite input slots from inputs buffer.
    for (var i = 0u; i < num_vars; i = i + 1u) {
        values[base + i] = constants[i];
    }

    // Overwrite input slots with this batch element's inputs.
    let input_base = batch_id * num_in;
    for (var i = 0u; i < num_in; i = i + 1u) {
        values[base + i] = inputs[input_base + i];
    }

    // Walk the tape.
    for (var i = num_in; i < num_ops; i = i + 1u) {
        let op = opcodes[i];

        // Skip Const entries — already initialized from constants buffer.
        if op == OP_CONST {
            continue;
        }

        let a_idx = arg0[i];
        let b_idx = arg1[i];

        let a = values[base + a_idx];
        var b = 0.0f;
        if b_idx != UNUSED {
            b = values[base + b_idx];
        }

        var r = 0.0f;

        switch op {
            case 2u /* ADD */: { r = a + b; }
            case 3u /* SUB */: { r = a - b; }
            case 4u /* MUL */: { r = a * b; }
            case 5u /* DIV */: { r = a / b; }
            case 6u /* REM */: { r = rem_f32(a, b); }
            case 7u /* POWF */: { r = pow(a, b); }
            case 8u /* ATAN2 */: { r = atan2(a, b); }
            case 9u /* HYPOT */: { r = hypot_f32(a, b); }
            case 10u /* MAX */: { r = max(a, b); }
            case 11u /* MIN */: { r = min(a, b); }
            case 12u /* NEG */: { r = -a; }
            case 13u /* RECIP */: { r = recip_f32(a); }
            case 14u /* SQRT */: { r = sqrt(a); }
            case 15u /* CBRT */: { r = cbrt_f32(a); }
            case 16u /* POWI */: { r = powi_f32(a, b_idx); }
            case 17u /* EXP */: { r = exp(a); }
            case 18u /* EXP2 */: { r = exp2(a); }
            case 19u /* EXPM1 */: { r = expm1_f32(a); }
            case 20u /* LN */: { r = log(a); }
            case 21u /* LOG2 */: { r = log2(a); }
            case 22u /* LOG10 */: { r = log10_f32(a); }
            case 23u /* LN1P */: { r = ln1p_f32(a); }
            case 24u /* SIN */: { r = sin(a); }
            case 25u /* COS */: { r = cos(a); }
            case 26u /* TAN */: { r = tan(a); }
            case 27u /* ASIN */: { r = asin(a); }
            case 28u /* ACOS */: { r = acos(a); }
            case 29u /* ATAN */: { r = atan(a); }
            case 30u /* SINH */: { r = sinh_f32(a); }
            case 31u /* COSH */: { r = cosh_f32(a); }
            case 32u /* TANH */: { r = tanh(a); }
            case 33u /* ASINH */: { r = asinh_f32(a); }
            case 34u /* ACOSH */: { r = acosh_f32(a); }
            case 35u /* ATANH */: { r = atanh_f32(a); }
            case 36u /* ABS */: { r = abs(a); }
            case 37u /* SIGNUM */: { r = signum_f32(a); }
            case 38u /* FLOOR */: { r = floor(a); }
            case 39u /* CEIL */: { r = ceil(a); }
            case 40u /* ROUND */: { r = round(a); }
            case 41u /* TRUNC */: { r = trunc(a); }
            case 42u /* FRACT */: { r = fract(a); }
            default: { r = 0.0; }
        }

        values[base + i] = r;
    }

    // Write outputs.
    let out_base = batch_id * n_out;
    for (var j = 0u; j < n_out; j = j + 1u) {
        let oi = output_indices[j];
        outputs[out_base + j] = values[base + oi];
    }
}
