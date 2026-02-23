// Forward-over-reverse (HVP) shader for sparse Hessian computation.
//
// Each thread performs:
// 1. Forward tangent pass: compute (primals, tangents) for all tape entries
// 2. Reverse adjoint sweep with Dual adjoints: adj_re and adj_eps
//    adj_re → gradient, adj_eps → Hessian-vector product
//
// For adjoint accumulation with Dual partials:
//   adj_re[a] += da_re * adj_re[i]
//   adj_eps[a] += da_re * adj_eps[i] + da_eps * adj_re[i]
// where da = Dual(da_re, da_eps) is the tangent of the reverse partial.

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

@group(0) @binding(0) var<storage, read> opcodes: array<u32>;
@group(0) @binding(1) var<storage, read> arg0: array<u32>;
@group(0) @binding(2) var<storage, read> arg1: array<u32>;
@group(0) @binding(3) var<storage, read> constants: array<f32>;
@group(0) @binding(4) var<uniform> tape_meta: TapeMeta;
@group(0) @binding(5) var<storage, read> output_indices: array<u32>;

// I/O: bind group 1
// 0: primal_inputs [B * num_inputs]
@group(1) @binding(0) var<storage, read> primal_inputs: array<f32>;
// 1: tangent_seeds [B * num_inputs]
@group(1) @binding(1) var<storage, read> tangent_seeds: array<f32>;
// 2: primals [B * num_variables]
@group(1) @binding(2) var<storage, read_write> primals: array<f32>;
// 3: tangents [B * num_variables]
@group(1) @binding(3) var<storage, read_write> tans: array<f32>;
// 4: adj_re [B * num_variables]
@group(1) @binding(4) var<storage, read_write> adj_re: array<f32>;
// 5: adj_eps [B * num_variables]
@group(1) @binding(5) var<storage, read_write> adj_eps: array<f32>;
// 6: grad_out [B * num_inputs]
@group(1) @binding(6) var<storage, read_write> grad_out: array<f32>;
// 7: hvp_out [B * num_inputs]
@group(1) @binding(7) var<storage, read_write> hvp_out: array<f32>;

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
    let base = bid * nv;

    // ──── Phase 1: Forward tangent pass ────
    for (var i = 0u; i < nv; i = i + 1u) {
        primals[base + i] = constants[i];
        tans[base + i] = 0.0;
    }
    let in_base = bid * ni;
    for (var i = 0u; i < ni; i = i + 1u) {
        primals[base + i] = primal_inputs[in_base + i];
        tans[base + i] = tangent_seeds[in_base + i];
    }

    for (var i = ni; i < num_ops; i = i + 1u) {
        let op = opcodes[i];
        if op == OP_CONST { continue; }
        let ai = arg0[i];
        let bi = arg1[i];
        let a = primals[base + ai];
        let at = tans[base + ai];
        var r = 0.0f;
        var rt = 0.0f;

        switch op {
            case 2u: { let b = primals[base+bi]; let bt = tans[base+bi]; r=a+b; rt=at+bt; }
            case 3u: { let b = primals[base+bi]; let bt = tans[base+bi]; r=a-b; rt=at-bt; }
            case 4u: { let b = primals[base+bi]; let bt = tans[base+bi]; r=a*b; rt=b*at+a*bt; }
            case 5u: { let b = primals[base+bi]; let bt = tans[base+bi]; r=a/b; let inv=1.0/b; rt=inv*at-a*inv*inv*bt; }
            case 6u: { r=a-trunc(a/primals[base+bi])*primals[base+bi]; rt=at; }
            case 7u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=pow(a,b); rt=r*(b/a*at+log(a)*bt); }
            case 8u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=atan2(a,b); let d=a*a+b*b; rt=(b*at-a*bt)/d; }
            case 9u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=sqrt(a*a+b*b); rt=(a*at+b*bt)/r; }
            case 10u: { let b=primals[base+bi]; let bt=tans[base+bi]; if a>=b {r=a;rt=at;} else {r=b;rt=bt;} }
            case 11u: { let b=primals[base+bi]; let bt=tans[base+bi]; if a<=b {r=a;rt=at;} else {r=b;rt=bt;} }
            case 12u: { r=-a; rt=-at; }
            case 13u: { r=1.0/a; rt=-at/(a*a); }
            case 14u: { r=sqrt(a); rt=at/(2.0*r); }
            case 15u: { let s=sign(a); r=s*pow(abs(a),1.0/3.0); rt=at/(3.0*r*r); }
            case 16u: { let n=f32(bitcast<i32>(bi)); r=pow(a,n); rt=n*pow(a,n-1.0)*at; }
            case 17u: { r=exp(a); rt=r*at; }
            case 18u: { r=exp2(a); rt=r*log(2.0)*at; }
            case 19u: { r=exp(a)-1.0; rt=(r+1.0)*at; }
            case 20u: { r=log(a); rt=at/a; }
            case 21u: { r=log2(a); rt=at/(a*log(2.0)); }
            case 22u: { r=log(a)/log(10.0); rt=at/(a*log(10.0)); }
            case 23u: { r=log(1.0+a); rt=at/(1.0+a); }
            case 24u: { r=sin(a); rt=cos(a)*at; }
            case 25u: { r=cos(a); rt=-sin(a)*at; }
            case 26u: { r=tan(a); let c=cos(a); rt=at/(c*c); }
            case 27u: { r=asin(a); rt=at/sqrt(1.0-a*a); }
            case 28u: { r=acos(a); rt=-at/sqrt(1.0-a*a); }
            case 29u: { r=atan(a); rt=at/(1.0+a*a); }
            case 30u: { r=sinh_f(a); rt=cosh_f(a)*at; }
            case 31u: { r=cosh_f(a); rt=sinh_f(a)*at; }
            case 32u: { r=tanh(a); let c=cosh_f(a); rt=at/(c*c); }
            case 33u: { r=log(a+sqrt(a*a+1.0)); rt=at/sqrt(a*a+1.0); }
            case 34u: { r=log(a+sqrt(a*a-1.0)); rt=at/sqrt(a*a-1.0); }
            case 35u: { r=0.5*log((1.0+a)/(1.0-a)); rt=at/(1.0-a*a); }
            case 36u: { r=abs(a); rt=sign(a)*at; }
            case 37u: { r=sign(a); rt=0.0; }
            case 38u: { r=floor(a); rt=0.0; }
            case 39u: { r=ceil(a); rt=0.0; }
            case 40u: { r=round(a); rt=0.0; }
            case 41u: { r=trunc(a); rt=0.0; }
            case 42u: { r=fract(a); rt=at; }
            default: {}
        }
        primals[base + i] = r;
        tans[base + i] = rt;
    }

    // ──── Phase 2: Reverse sweep with Dual adjoints ────
    for (var i = 0u; i < nv; i = i + 1u) {
        adj_re[base + i] = 0.0;
        adj_eps[base + i] = 0.0;
    }
    // Seed output adjoint
    let seed_idx = output_indices[0];
    adj_re[base + seed_idx] = 1.0;

    for (var ii = 0u; ii < num_ops; ii = ii + 1u) {
        let i = num_ops - 1u - ii;
        let ar = adj_re[base + i];
        let ae = adj_eps[base + i];
        if ar == 0.0 && ae == 0.0 { continue; }

        let op = opcodes[i];
        if op == OP_INPUT || op == OP_CONST { continue; }

        adj_re[base + i] = 0.0;
        adj_eps[base + i] = 0.0;

        let ai = arg0[i];
        let bi = arg1[i];
        let a = primals[base + ai];
        let at = tans[base + ai];
        let r = primals[base + i];

        // Compute Dual reverse partials: (da_re, da_eps, db_re, db_eps)
        var da_re = 0.0f;
        var da_eps = 0.0f;
        var db_re = 0.0f;
        var db_eps = 0.0f;

        switch op {
            case 2u /* ADD */: { da_re=1.0; db_re=1.0; }
            case 3u /* SUB */: { da_re=1.0; db_re=-1.0; }
            case 4u /* MUL */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                da_re=b; da_eps=bt; db_re=a; db_eps=at;
            }
            case 5u /* DIV */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                let inv=1.0/b;
                da_re=inv; da_eps=-bt*inv*inv;
                db_re=-a*inv*inv; db_eps=-at*inv*inv+2.0*a*bt*inv*inv*inv;
            }
            case 6u /* REM */: { da_re=1.0; }
            case 7u /* POWF */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                // da = b*a^(b-1)
                let ab1 = pow(a, b-1.0);
                da_re = b * ab1;
                // da_eps = d/dε[b*a^(b-1)] = bt*a^(b-1) + b*(b-1)*a^(b-2)*at + b*a^(b-1)*log(a)*bt
                // Actually: d/dε[(b+bt*ε)*(a+at*ε)^(b+bt*ε-1)]
                // First order: bt*ab1 + b*((b-1)/a*at + log(a)*bt)*ab1
                da_eps = bt*ab1 + b*ab1*((b-1.0)/a*at + log(a)*bt);
                // db = r*ln(a)
                let la = log(a);
                db_re = r * la;
                let rt = tans[base+i];
                db_eps = rt*la + r*at/a;
            }
            case 8u /* ATAN2 */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                let d=a*a+b*b;
                let d2 = d*d;
                let dd = 2.0*(a*at+b*bt);
                da_re=b/d; da_eps=(bt*d-b*dd)/d2;
                db_re=-a/d; db_eps=(-at*d+a*dd)/d2;
            }
            case 9u /* HYPOT */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                // da=a/r, db=b/r
                let r2=r*r; let rt2=tans[base+i];
                da_re=a/r; da_eps=(at*r-a*rt2)/(r2);
                db_re=b/r; db_eps=(bt*r-b*rt2)/(r2);
            }
            case 10u /* MAX */: {
                let b=primals[base+bi];
                if a>=b { da_re=1.0; } else { db_re=1.0; }
            }
            case 11u /* MIN */: {
                let b=primals[base+bi];
                if a<=b { da_re=1.0; } else { db_re=1.0; }
            }

            // Unary ops: da_re = f'(a), da_eps = f''(a)*at
            case 12u /* NEG */: { da_re=-1.0; }
            case 13u /* RECIP */: { let inv=1.0/a; da_re=-inv*inv; da_eps=2.0*at*inv*inv*inv; }
            case 14u /* SQRT */: { da_re=0.5/r; da_eps=-0.25*at/(a*r); }
            case 15u /* CBRT */: { let rr=r*r; da_re=1.0/(3.0*rr); da_eps=-2.0*at/(9.0*rr*r); }
            case 16u /* POWI */: {
                let n=f32(bitcast<i32>(bi));
                da_re=n*pow(a,n-1.0);
                da_eps=n*(n-1.0)*pow(a,n-2.0)*at;
            }
            case 17u /* EXP */: { da_re=r; da_eps=r*at; }
            case 18u /* EXP2 */: { let l2=log(2.0); da_re=r*l2; da_eps=r*l2*l2*at; }
            case 19u /* EXPM1 */: { da_re=r+1.0; da_eps=(r+1.0)*at; }
            case 20u /* LN */: { da_re=1.0/a; da_eps=-at/(a*a); }
            case 21u /* LOG2 */: { let l2=log(2.0); da_re=1.0/(a*l2); da_eps=-at/(a*a*l2); }
            case 22u /* LOG10 */: { let l10=log(10.0); da_re=1.0/(a*l10); da_eps=-at/(a*a*l10); }
            case 23u /* LN1P */: { let t=1.0+a; da_re=1.0/t; da_eps=-at/(t*t); }
            case 24u /* SIN */: { da_re=cos(a); da_eps=-sin(a)*at; }
            case 25u /* COS */: { da_re=-sin(a); da_eps=-cos(a)*at; }
            case 26u /* TAN */: { let c=cos(a); let s=1.0/(c*c); da_re=s; da_eps=2.0*tan(a)*s*at; }
            case 27u /* ASIN */: { let t=sqrt(1.0-a*a); da_re=1.0/t; da_eps=a*at/(t*t*t); }
            case 28u /* ACOS */: { let t=sqrt(1.0-a*a); da_re=-1.0/t; da_eps=-a*at/(t*t*t); }
            case 29u /* ATAN */: { let t=1.0+a*a; da_re=1.0/t; da_eps=-2.0*a*at/(t*t); }
            case 30u /* SINH */: { da_re=cosh_f(a); da_eps=sinh_f(a)*at; }
            case 31u /* COSH */: { da_re=sinh_f(a); da_eps=cosh_f(a)*at; }
            case 32u /* TANH */: { let c=cosh_f(a); let s=1.0/(c*c); da_re=s; da_eps=-2.0*tanh(a)*s*at; }
            case 33u /* ASINH */: { let t=sqrt(a*a+1.0); da_re=1.0/t; da_eps=-a*at/(t*t*t); }
            case 34u /* ACOSH */: { let t=sqrt(a*a-1.0); da_re=1.0/t; da_eps=-a*at/(t*t*t); }
            case 35u /* ATANH */: { let t=1.0-a*a; da_re=1.0/t; da_eps=2.0*a*at/(t*t); }
            case 36u /* ABS */: { da_re=sign(a); }
            case 37u, 38u, 39u, 40u, 41u: { /* zero derivative */ }
            case 42u /* FRACT */: { da_re=1.0; }
            default: {}
        }

        // Dual accumulation: adj[arg] += Dual(da_re, da_eps) * Dual(ar, ae)
        adj_re[base + ai] += da_re * ar;
        adj_eps[base + ai] += da_re * ae + da_eps * ar;

        if bi != UNUSED && op != OP_POWI {
            adj_re[base + bi] += db_re * ar;
            adj_eps[base + bi] += db_re * ae + db_eps * ar;
        }
    }

    // Write gradient and HVP outputs
    let g_base = bid * ni;
    for (var i = 0u; i < ni; i = i + 1u) {
        grad_out[g_base + i] = adj_re[base + i];
        hvp_out[g_base + i] = adj_eps[base + i];
    }
}
