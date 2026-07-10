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

fn powf_real(base: f32, b: f32) -> f32 {
    // WGSL `pow(x, y)` is undefined for x < 0 (naga lowers it to
    // `exp2(y*log2(x))`, and `log2(negative) = NaN`). Rust/C `powf` define
    // x^y for x < 0 only when y is an integer: sign(x)^y * |x|^y. A
    // non-integer exponent at a negative base is NaN — the same as on CPU.
    // 0^0 = 1 (matches CPU/C `powf`); naga lowers `pow(0,0)` to
    // `exp2(0*log2(0)) = exp2(NaN) = NaN`, so guard it explicitly.
    if base == 0.0 && b == 0.0 { return 1.0; }
    if base >= 0.0 { return pow(base, b); }
    let rb = round(b);
    if rb != b { return bitcast<f32>(0x7fc00000u); }
    let mag = pow(abs(base), b);
    if (i32(rb) & 1) != 0 { return -mag; }
    return mag;
}

// Precision-preserving EXPM1 / LN1P primals for small |x|, matching
// forward.wgsl helpers. `exp(x) - 1` and `log(1 + x)` cancel
// catastrophically as x → 0; the Taylor-series shortcut avoids that.
fn expm1_f32(x: f32) -> f32 {
    if abs(x) < 1e-4 { return x + 0.5 * x * x; }
    return exp(x) - 1.0;
}
fn ln1p_f32(x: f32) -> f32 {
    if abs(x) < 1e-4 { return x - 0.5 * x * x; }
    return log(1.0 + x);
}

fn abs_deriv_f32(x: f32) -> f32 {
    // Unified abs' convention (matches kernels::abs_deriv): 0 at the kink
    // (value-based, so +0 and -0 agree), sign(x) elsewhere, NaN at NaN. The NaN
    // test inspects the bits — `x != x` is unreliable under Metal fast-math.
    let b = bitcast<u32>(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return x; }
    if (x == 0.0) { return 0.0; }
    return select(1.0, -1.0, (b & 0x80000000u) != 0u);
}

fn signum_f32(x: f32) -> f32 {
    // Rust f32::signum: -1 for -0.0 (sign bit), +1 for +0.0/positive, NaN at NaN.
    // `x >= 0.0` wrongly maps -0.0 to +1; inspect the sign bit. Bitcast NaN test
    // since `x != x` is unreliable under Metal fast-math.
    let b = bitcast<u32>(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return x; }
    return select(1.0, -1.0, (b & 0x80000000u) != 0u);
}

// Overflow-safe hypot with IEEE Inf handling — same helper as
// forward.wgsl / tangent_forward.wgsl so the primal stays bit-matched
// across kernels. The naive sqrt(a*a + b*b) overflows to Inf for |a| or
// |b| above ~1.8e19 even where the true hypot is representable.
fn hypot_f32(a: f32, b: f32) -> f32 {
    let ax = abs(a);
    let ay = abs(b);
    let inf = bitcast<f32>(0x7f800000u);
    if ax == inf || ay == inf { return inf; }
    let mx = max(ax, ay);
    let mn = min(ax, ay);
    if mx == 0.0 { return 0.0; }
    let r = mn / mx;
    return mx * sqrt(1.0 + r * r);
}

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
            // DIV tangent factored as r*inv (r = a/b) to match
            // tangent_forward.wgsl and the CUDA kernel. WGSL left-
            // associativity evaluates r*inv*bt as (r*inv)*bt, never
            // forming inv*inv, so the form is overflow-safe.
            case 5u: { let b = primals[base+bi]; let bt = tans[base+bi]; r=a/b; let inv=1.0/b; rt=inv*at-r*inv*bt; }
            // REM is exact only for |a/b| < 2^24 (f32 mantissa) — see rem_f32 in
            // forward.wgsl; CPU/CUDA use exact fmod.
            case 6u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=a-trunc(a/b)*b; rt=at-trunc(a/b)*bt; }
            case 7u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=powf_real(a,b); let dx=select(select(b*r/a*at, b*powf_real(a,b-1.0)*at, a==0.0), 0.0, b==0.0 || at==0.0); let dy=select(r*log(a)*bt, 0.0, r==0.0 || a<=0.0 || bt==0.0); rt=dx+dy; }
            case 8u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=atan2(a,b); let mx=max(abs(a),abs(b)); if mx==0.0 {rt=0.0;} else {let au=a/mx; let bu=b/mx; let d=mx*(au*au+bu*bu); rt=(bu*at-au*bt)/d;} }
            // HYPOT primal via the overflow-safe helper; the tangent numerator
            // a*at + b*bt is left un-rescaled (it overflows only when the true
            // tangent magnitude does).
            case 9u: { let b=primals[base+bi]; let bt=tans[base+bi]; r=hypot_f32(a,b); if r==0.0 {rt=0.0;} else {rt=(a*at+b*bt)/r;} }
            case 10u: { let b=primals[base+bi]; let bt=tans[base+bi]; let bb=bitcast<u32>(b); let bn=((bb>>23u)&0xffu)==0xffu && (bb&0x7fffffu)!=0u; if a>=b || bn {r=a;rt=at;} else {r=b;rt=bt;} }
            case 11u: { let b=primals[base+bi]; let bt=tans[base+bi]; let bb=bitcast<u32>(b); let bn=((bb>>23u)&0xffu)==0xffu && (bb&0x7fffffu)!=0u; if a<=b || bn {r=a;rt=at;} else {r=b;rt=bt;} }
            case 12u: { r=-a; rt=-at; }
            case 13u: { r=1.0/a; rt=-at/(a*a); }
            case 14u: { r=sqrt(a); rt=at/(2.0*r); }
            case 15u: { let s=sign(a); r=s*pow(abs(a),1.0/3.0); rt=at/(3.0*r*r); }
            case 16u: { let e=bitcast<i32>(bi); let n=f32(e); r=powf_real(a,n); rt=select(n*powf_real(a,n-1.0)*at, 0.0, e==0); }
            case 17u: { r=exp(a); rt=r*at; }
            case 18u: { r=exp2(a); rt=r*log(2.0)*at; }
            case 19u: { r=expm1_f32(a); rt=(r+1.0)*at; }
            case 20u: { r=log(a); rt=select(bitcast<f32>(0x7fc00000u), at/a, a >= 0.0); }
            case 21u: { r=log2(a); rt=select(bitcast<f32>(0x7fc00000u), at/(a*log(2.0)), a >= 0.0); }
            case 22u: { r=log(a)/log(10.0); rt=select(bitcast<f32>(0x7fc00000u), at/(a*log(10.0)), a >= 0.0); }
            case 23u: { r=ln1p_f32(a); rt=select(bitcast<f32>(0x7fc00000u), at/(1.0+a), a >= -1.0); }
            case 24u: { r=sin(a); rt=cos(a)*at; }
            case 25u: { r=cos(a); rt=-sin(a)*at; }
            case 26u: { r=tan(a); let c=cos(a); rt=at/(c*c); }
            case 27u: { r=asin(a); rt=at/sqrt((1.0-a)*(1.0+a)); }
            case 28u: { r=acos(a); rt=-at/sqrt((1.0-a)*(1.0+a)); }
            case 29u: {
                let aa = abs(a); r = atan(a);
                if aa > 1e8 { let inv = 1.0 / a; rt = at * inv * inv / (1.0 + inv * inv); }
                else        { rt = at / (1.0 + a * a); }
            }
            case 30u: { r=sinh_f(a); rt=cosh_f(a)*at; }
            case 31u: { r=cosh_f(a); rt=sinh_f(a)*at; }
            case 32u: { r=tanh(a); let c=cosh_f(a); rt=at/(c*c); }
            case 33u: { let ax=abs(a); if ax>1e8 {let inv=1.0/a; let rr=log(ax)+log(1.0+sqrt(1.0+inv*inv)); r=select(-rr,rr,a>=0.0); rt=at*abs(inv)/sqrt(1.0+inv*inv);} else {r=select(-log(ax+sqrt(ax*ax+1.0)), log(ax+sqrt(ax*ax+1.0)), a>=0.0); rt=at/sqrt(a*a+1.0);} }
            case 34u: { if a < 1.0 { let n=bitcast<f32>(0x7fc00000u); r=n; rt=n; } else if abs(a)>1e8 {let inv=1.0/a; r=log(a)+log(1.0+sqrt(1.0-inv*inv)); rt=at*abs(inv)/sqrt(1.0-inv*inv);} else {r=log(a+sqrt((a-1.0)*(a+1.0))); rt=at/sqrt((a-1.0)*(a+1.0));} }
            case 35u: { r=0.5*log((1.0+a)/(1.0-a)); rt=select(bitcast<f32>(0x7fc00000u), at/((1.0-a)*(1.0+a)), a >= -1.0 && a <= 1.0); }
            case 36u: { r=abs(a); rt=abs_deriv_f32(a)*at; }
            case 37u: { r = signum_f32(a); rt=0.0; }
            case 38u: { r=floor(a); rt=0.0; }
            case 39u: { r=ceil(a); rt=0.0; }
            case 40u: { let t=trunc(a); r=select(t, t + select(-1.0, 1.0, a >= 0.0), abs(a - t) >= 0.5); rt=0.0; }
            case 41u: { r=trunc(a); rt=0.0; }
            case 42u: { r=a-trunc(a); rt=at; }
            default: {}
        }
        primals[base + i] = r;
        // Structural-zero tangent convention — see tangent_forward.wgsl
        // (uniform across all unary ops, matching the CPU chain rule).
        let unary_singular = op >= 12u && op <= 42u; // NEG..FRACT
        if at == 0.0 && unary_singular {
            rt = 0.0;
        }
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
                // Factor through `r = a/b` to drop one `inv` from each
                // higher-order term: `-a*inv²` → `-r*inv`, and
                // `2*a*inv³` → `2*r*inv²`. One factor of `inv*inv` still
                // remains in the eps-eps terms (unavoidable second
                // derivative), but `inv³` is eliminated.
                da_re=inv; da_eps=-bt*inv*inv;
                db_re=-r*inv; db_eps=-at*inv*inv+2.0*r*bt*inv*inv;
            }
            case 6u /* REM */: {
                let b=primals[base+bi];
                da_re=1.0;
                db_re=-trunc(a/b);
                // db_eps = 0 since trunc has zero derivative a.e.
            }
            case 7u /* POWF */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                let ab1 = powf_real(a, b-1.0);
                if b == 0.0 {
                    // a^0 = 1 is constant in a → every base-direction derivative
                    // is 0, in BOTH the value and eps parts. Matches CPU
                    // `reverse_partials`, which returns da = 0 at b==0.
                    da_re = 0.0;
                    da_eps = 0.0;
                } else {
                    da_re = b * ab1;
                    // Second-order ∂²/∂a² = b(b-1)*a^(b-2). For a <= 0 the closed
                    // form `b*a^(b-1)*(b-1)/a` evaluates 0*Inf at a=0; use the
                    // division-free equivalent, finite for b >= 2 (and
                    // algebraically identical for a < 0 integer b). It is 0 for
                    // b == 1 (linear in a) — short-circuit so it doesn't hit
                    // 0*Inf. (a=0 ∧ b==1 is doubly degenerate: the mixed
                    // ∂²/∂a∂b term carries ln(0), non-finite on CPU and GPU
                    // alike, so it is not asserted.) The a>0 branch is unchanged.
                    if a <= 0.0 {
                        let daa = select(b*(b-1.0)*powf_real(a, b-2.0)*at, 0.0, b == 1.0 || at == 0.0);
                        // Per-direction structural-zero guards: ab1 can be
                        // non-finite at a singular base, so each term is
                        // zeroed with its own direction component.
                        da_eps = select(bt*ab1, 0.0, bt == 0.0) + daa;
                    } else {
                        da_eps = select(bt*ab1, 0.0, bt == 0.0)
                            + select(b*ab1*((b-1.0)/a)*at, 0.0, at == 0.0)
                            + select(b*ab1*log(a)*bt, 0.0, bt == 0.0);
                    }
                }
                let rr = primals[base+i]; // r = a^b from forward pass
                if rr == 0.0 || a <= 0.0 {
                    db_re = 0.0;
                    db_eps = 0.0;
                } else {
                    let la = log(a);
                    let rt = tans[base+i];
                    db_re = rr * la;
                    // rt is direction-consistent from phase 1 (zero for a
                    // fully-constant direction); the at-term needs its own
                    // guard because rr can be non-finite on overflow.
                    db_eps = rt*la + select(rr*at/a, 0.0, at == 0.0);
                }
            }
            case 8u /* ATAN2 */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                // Normalize by max(|a|,|b|) so a²+b² doesn't overflow in f32
                // even when a*a alone would. Let au = a/mx, bu = b/mx, s =
                // au² + bu² ∈ [1, 2], ms = mx·s. Then:
                //   a² + b² = mx · ms
                //   da_re  = b/(a²+b²) = bu/ms
                //   da_eps = bt/(a²+b²) − bu·dd_over_mx/ms²
                //          = (bt/mx − bu·dd_over_mx/ms) / ms
                // where dd_over_mx = 2·(au·at + bu·bt) is bounded for
                // fixed tangent magnitudes. Expressing in this form avoids
                // the explicit mx² that would overflow for |mx| > sqrt(f32::MAX).
                let mx = max(abs(a), abs(b));
                if mx == 0.0 {
                    da_re = 0.0; da_eps = 0.0; db_re = 0.0; db_eps = 0.0;
                } else {
                    let au = a / mx;
                    let bu = b / mx;
                    let s = au * au + bu * bu;
                    let ms = mx * s;
                    let dd_over_mx = 2.0 * (au * at + bu * bt);
                    da_re = bu / ms;
                    db_re = -au / ms;
                    da_eps = (bt / mx - bu * dd_over_mx / ms) / ms;
                    db_eps = (-at / mx + au * dd_over_mx / ms) / ms;
                }
            }
            case 9u /* HYPOT */: {
                let b=primals[base+bi]; let bt=tans[base+bi];
                if r == 0.0 {
                    da_re=0.0; da_eps=0.0; db_re=0.0; db_eps=0.0;
                } else {
                    let r2=r*r; let rt2=tans[base+i];
                    da_re=a/r; da_eps=(at*r-a*rt2)/(r2);
                    db_re=b/r; db_eps=(bt*r-b*rt2)/(r2);
                }
            }
            // NaN routing uses the bit-pattern test (as Phase 1 above and every
            // other kernel): `b != b` can be folded to false under Metal
            // fast-math, which would route the adjoint to the wrong operand.
            case 10u /* MAX */: {
                let b=primals[base+bi];
                let bb=bitcast<u32>(b);
                let bn=((bb>>23u)&0xffu)==0xffu && (bb&0x7fffffu)!=0u;
                if a>=b || bn { da_re=1.0; } else { db_re=1.0; }
            }
            case 11u /* MIN */: {
                let b=primals[base+bi];
                let bb=bitcast<u32>(b);
                let bn=((bb>>23u)&0xffu)==0xffu && (bb&0x7fffffu)!=0u;
                if a<=b || bn { da_re=1.0; } else { db_re=1.0; }
            }

            // Unary ops: da_re = f'(a), da_eps = f''(a)*at
            case 12u /* NEG */: { da_re=-1.0; }
            case 13u /* RECIP */: { let inv=1.0/a; da_re=-inv*inv; da_eps=2.0*at*inv*inv*inv; }
            case 14u /* SQRT */: { da_re=0.5/r; da_eps=-0.25*at/(a*r); }
            // f''(a) = -2/(9*r^5) where r = cbrt(a)
            case 15u /* CBRT */: { let rr=r*r; da_re=1.0/(3.0*rr); da_eps=-2.0*at/(9.0*rr*rr*r); }
            case 16u /* POWI */: {
                let e=bitcast<i32>(bi);
                if e == 0 { da_re=0.0; da_eps=0.0; }
                else if e == 1 {
                    // f(a)=a, f'=1, f''=0. The general formula evaluates
                    // `pow(a, -1) → Inf` at a=0, giving `0 * Inf * at = NaN`.
                    // Short-circuit to the mathematically exact zero second
                    // derivative. Mirrors the CUDA fix in tape_eval.cu.
                    da_re=1.0; da_eps=0.0;
                } else {
                    let n=f32(e); da_re=n*powf_real(a,n-1.0); da_eps=n*(n-1.0)*powf_real(a,n-2.0)*at;
                }
            }
            case 17u /* EXP */: { da_re=r; da_eps=r*at; }
            case 18u /* EXP2 */: { let l2=log(2.0); da_re=r*l2; da_eps=r*l2*l2*at; }
            case 19u /* EXPM1 */: { da_re=r+1.0; da_eps=(r+1.0)*at; }
            case 20u /* LN */: { if (a >= 0.0) { da_re=1.0/a; da_eps=-at/(a*a); } else { let n=bitcast<f32>(0x7fc00000u); da_re=n; da_eps=n; } }
            case 21u /* LOG2 */: { if (a >= 0.0) { let l2=log(2.0); da_re=1.0/(a*l2); da_eps=-at/(a*a*l2); } else { let n=bitcast<f32>(0x7fc00000u); da_re=n; da_eps=n; } }
            case 22u /* LOG10 */: { if (a >= 0.0) { let l10=log(10.0); da_re=1.0/(a*l10); da_eps=-at/(a*a*l10); } else { let n=bitcast<f32>(0x7fc00000u); da_re=n; da_eps=n; } }
            case 23u /* LN1P */: { if (a >= -1.0) { let t=1.0+a; da_re=1.0/t; da_eps=-at/(t*t); } else { let n=bitcast<f32>(0x7fc00000u); da_re=n; da_eps=n; } }
            case 24u /* SIN */: { da_re=cos(a); da_eps=-sin(a)*at; }
            case 25u /* COS */: { da_re=-sin(a); da_eps=-cos(a)*at; }
            case 26u /* TAN */: { let c=cos(a); let s=1.0/(c*c); da_re=s; da_eps=2.0*tan(a)*s*at; }
            case 27u /* ASIN */: { let t=sqrt((1.0-a)*(1.0+a)); da_re=1.0/t; da_eps=a*at/(t*t*t); }
            case 28u /* ACOS */: { let t=sqrt((1.0-a)*(1.0+a)); da_re=-1.0/t; da_eps=-a*at/(t*t*t); }
            case 29u /* ATAN */: {
                let aa = abs(a);
                if aa > 1e8 {
                    let inv = 1.0 / a;
                    let h = 1.0 + inv * inv;
                    da_re = inv * inv / h;
                    da_eps = -2.0 * inv * inv * inv * at / (h * h);
                } else {
                    let t = 1.0 + a * a;
                    da_re = 1.0 / t;
                    da_eps = -2.0 * a * at / (t * t);
                }
            }
            case 30u /* SINH */: { da_re=cosh_f(a); da_eps=sinh_f(a)*at; }
            case 31u /* COSH */: { da_re=sinh_f(a); da_eps=cosh_f(a)*at; }
            case 32u /* TANH */: { let c=cosh_f(a); let s=1.0/(c*c); da_re=s; da_eps=-2.0*tanh(a)*s*at; }
            case 33u /* ASINH */: {
                // For |a| > 1e8 use inv-based formula to avoid a*a+1 overflow.
                if abs(a) > 1e8 {
                    let inv = 1.0 / a;
                    // d/dx asinh = 1/sqrt(1+x²) = |1/x|/sqrt(1+1/x²) for large |x|
                    let denom = sqrt(1.0 + inv * inv);
                    da_re = abs(inv) / denom;
                    // d²/dx² asinh = -x/(1+x²)^(3/2). Rewrite via inv = 1/x:
                    //   = -sign(x)·|inv|³/denom³ = -sign(x)·inv²·|inv|/denom³
                    let denom3 = denom * denom * denom;
                    da_eps = -a * at * inv * inv * abs(inv) / denom3;
                } else {
                    let t = sqrt(a * a + 1.0);
                    da_re = 1.0 / t;
                    da_eps = -a * at / (t * t * t);
                }
            }
            case 34u /* ACOSH */: {
                if a < 1.0 {
                    // Out of domain (acosh domain a >= 1): both HVP terms NaN.
                    // Matches kernels::acosh_deriv; strict `< 1` keeps a==1 singular.
                    let n = bitcast<f32>(0x7fc00000u);
                    da_re = n;
                    da_eps = n;
                } else if abs(a) > 1e8 {
                    let inv = 1.0 / a;
                    let denom = sqrt(1.0 - inv * inv);
                    da_re = abs(inv) / denom;
                    let denom3 = denom * denom * denom;
                    da_eps = -a * at * inv * inv * abs(inv) / denom3;
                } else {
                    // Factored form (a-1)(a+1) avoids cancellation
                    // near a=1; matches kernels::acosh_deriv. Both
                    // first-order (1/t) and second-order (-a*at/t³)
                    // benefit from the precision improvement.
                    let t = sqrt((a - 1.0) * (a + 1.0));
                    da_re = 1.0 / t;
                    da_eps = -a * at / (t * t * t);
                }
            }
            case 35u /* ATANH */: { if (a >= -1.0 && a <= 1.0) { let t=(1.0-a)*(1.0+a); da_re=1.0/t; da_eps=2.0*a*at/(t*t); } else { let n=bitcast<f32>(0x7fc00000u); da_re=n; da_eps=n; } }
            case 36u /* ABS */: { da_re = abs_deriv_f32(a); }
            case 37u, 38u, 39u, 40u, 41u: { /* zero derivative */ }
            case 42u /* FRACT */: { da_re=1.0; }
            default: {}
        }

        // Structural-zero direction component: for the unary arms da_eps is
        // f''(a)·at, so a zero direction keeps the second-order contribution
        // exactly zero even at singular primals (0/0 or 0*Inf otherwise).
        // Mirrors the forward-phase guard; POWF is guarded per-term in its
        // arm. The first-order multiplier da_re does not involve the
        // direction and is untouched.
        let unary_singular2 = op >= 12u && op <= 42u; // NEG..FRACT
        if at == 0.0 && unary_singular2 {
            da_eps = 0.0;
        }

        // Dual accumulation: adj[arg] += Dual(da_re, da_eps) * Dual(ar, ae).
        // Zero-multiplier convention (see kernels/mod.rs): the PAIR guard
        // mirrors the CPU sweep's is_all_zero(Dual(da_re, da_eps)) — an
        // all-zero dual partial absorbs any adjoint, while a partial with
        // zero primal but live second-order component still accumulates.
        if da_re != 0.0 || da_eps != 0.0 {
            adj_re[base + ai] += da_re * ar;
            adj_eps[base + ai] += da_re * ae + da_eps * ar;
        }

        if bi != UNUSED && op != OP_POWI && (db_re != 0.0 || db_eps != 0.0) {
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
