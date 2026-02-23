// Batched tape evaluation kernels for echidna GPU backend.
//
// Templated on FLOAT_TYPE (float or double) via preprocessor define
// injected before NVRTC compilation.
//
// Kernels:
//   forward_eval   — batched forward evaluation
//   reverse_sweep  — batched reverse adjoint sweep (after forward)
//   tangent_forward — batched forward tangent (JVP)
//   tangent_reverse — forward-over-reverse (HVP) for sparse Hessian

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

#define UNUSED 0xFFFFFFFFu

// Math helpers — use the right precision
__device__ F _sign(F x) { return (x > F(0)) - (x < F(0)); }
__device__ F _cbrt(F x) { return copysign(pow(fabs(x), F(1.0/3.0)), x); }
__device__ F _fract(F x) { return x - floor(x); }

// ════════════════════════════════════════════════════════════════════
// Forward evaluation kernel
// ════════════════════════════════════════════════════════════════════
extern "C" __global__ void forward_eval(
    const unsigned int* __restrict__ opcodes,
    const unsigned int* __restrict__ arg0,
    const unsigned int* __restrict__ arg1,
    const F* __restrict__ constants,
    const F* __restrict__ inputs,
    F* __restrict__ values,
    F* __restrict__ outputs,
    const unsigned int* __restrict__ output_indices,
    unsigned int num_ops,
    unsigned int num_inputs,
    unsigned int num_variables,
    unsigned int num_outputs,
    unsigned int batch_size
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;

    unsigned int v_base = bid * num_variables;
    unsigned int in_base = bid * num_inputs;

    // Initialize from constants
    for (unsigned int i = 0; i < num_variables; i++) {
        values[v_base + i] = constants[i];
    }

    // Set input values
    for (unsigned int i = 0; i < num_inputs; i++) {
        values[v_base + i] = inputs[in_base + i];
    }

    // Evaluate tape
    for (unsigned int i = num_inputs; i < num_ops; i++) {
        unsigned int op = opcodes[i];
        if (op == OP_CONST) continue;

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];
        F a = values[v_base + ai];
        F r = F(0);

        switch (op) {
            case OP_ADD:    r = a + values[v_base + bi]; break;
            case OP_SUB:    r = a - values[v_base + bi]; break;
            case OP_MUL:    r = a * values[v_base + bi]; break;
            case OP_DIV:    r = a / values[v_base + bi]; break;
            case OP_REM:    r = fmod(a, values[v_base + bi]); break;
            case OP_POWF:   r = pow(a, values[v_base + bi]); break;
            case OP_ATAN2:  r = atan2(a, values[v_base + bi]); break;
            case OP_HYPOT:  r = hypot(a, values[v_base + bi]); break;
            case OP_MAX:    r = fmax(a, values[v_base + bi]); break;
            case OP_MIN:    r = fmin(a, values[v_base + bi]); break;
            case OP_NEG:    r = -a; break;
            case OP_RECIP:  r = F(1) / a; break;
            case OP_SQRT:   r = sqrt(a); break;
            case OP_CBRT:   r = _cbrt(a); break;
            case OP_POWI: {
                int n = (int)bi;
                r = pow(a, F(n));
                break;
            }
            case OP_EXP:    r = exp(a); break;
            case OP_EXP2:   r = exp2(a); break;
            case OP_EXPM1:  r = expm1(a); break;
            case OP_LN:     r = log(a); break;
            case OP_LOG2:   r = log2(a); break;
            case OP_LOG10:  r = log10(a); break;
            case OP_LN1P:   r = log1p(a); break;
            case OP_SIN:    r = sin(a); break;
            case OP_COS:    r = cos(a); break;
            case OP_TAN:    r = tan(a); break;
            case OP_ASIN:   r = asin(a); break;
            case OP_ACOS:   r = acos(a); break;
            case OP_ATAN:   r = atan(a); break;
            case OP_SINH:   r = sinh(a); break;
            case OP_COSH:   r = cosh(a); break;
            case OP_TANH:   r = tanh(a); break;
            case OP_ASINH:  r = asinh(a); break;
            case OP_ACOSH:  r = acosh(a); break;
            case OP_ATANH:  r = atanh(a); break;
            case OP_ABS:    r = fabs(a); break;
            case OP_SIGNUM: r = _sign(a); break;
            case OP_FLOOR:  r = floor(a); break;
            case OP_CEIL:   r = ceil(a); break;
            case OP_ROUND:  r = round(a); break;
            case OP_TRUNC:  r = trunc(a); break;
            case OP_FRACT:  r = _fract(a); break;
            default: break;
        }
        values[v_base + i] = r;
    }

    // Write outputs
    unsigned int out_base = bid * num_outputs;
    for (unsigned int j = 0; j < num_outputs; j++) {
        outputs[out_base + j] = values[v_base + output_indices[j]];
    }
}

// ════════════════════════════════════════════════════════════════════
// Reverse adjoint sweep kernel
// ════════════════════════════════════════════════════════════════════
extern "C" __global__ void reverse_sweep(
    const unsigned int* __restrict__ opcodes,
    const unsigned int* __restrict__ arg0,
    const unsigned int* __restrict__ arg1,
    const F* __restrict__ values,      // from forward pass
    F* __restrict__ adjoints,
    F* __restrict__ grad_out,
    const unsigned int* __restrict__ output_indices,
    unsigned int num_ops,
    unsigned int num_inputs,
    unsigned int num_variables,
    unsigned int batch_size
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;

    unsigned int v_base = bid * num_variables;
    unsigned int a_base = bid * num_variables;

    // Zero adjoints
    for (unsigned int i = 0; i < num_variables; i++) {
        adjoints[a_base + i] = F(0);
    }

    // Seed output
    unsigned int seed_idx = output_indices[0];
    adjoints[a_base + seed_idx] = F(1);

    // Reverse sweep
    for (unsigned int ii = 0; ii < num_ops; ii++) {
        unsigned int i = num_ops - 1 - ii;
        F adj = adjoints[a_base + i];
        if (adj == F(0)) continue;

        unsigned int op = opcodes[i];
        if (op == OP_INPUT || op == OP_CONST) continue;

        adjoints[a_base + i] = F(0);

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];
        F a = values[v_base + ai];
        F r = values[v_base + i];

        F da = F(0), db = F(0);

        switch (op) {
            case OP_ADD: da = F(1); db = F(1); break;
            case OP_SUB: da = F(1); db = F(-1); break;
            case OP_MUL: { F b = values[v_base+bi]; da = b; db = a; break; }
            case OP_DIV: {
                F b = values[v_base+bi];
                F inv = F(1)/b;
                da = inv; db = -a*inv*inv; break;
            }
            case OP_REM: da = F(1); break;
            case OP_POWF: {
                F b = values[v_base+bi];
                da = b * pow(a, b-F(1));
                db = r * log(a); break;
            }
            case OP_ATAN2: {
                F b = values[v_base+bi];
                F d = a*a + b*b;
                da = b/d; db = -a/d; break;
            }
            case OP_HYPOT: {
                F b = values[v_base+bi];
                da = a/r; db = b/r; break;
            }
            case OP_MAX: {
                F b = values[v_base+bi];
                if (a >= b) { da = F(1); } else { db = F(1); } break;
            }
            case OP_MIN: {
                F b = values[v_base+bi];
                if (a <= b) { da = F(1); } else { db = F(1); } break;
            }
            case OP_NEG:    da = F(-1); break;
            case OP_RECIP:  { F inv = F(1)/a; da = -inv*inv; break; }
            case OP_SQRT:   da = F(0.5)/r; break;
            case OP_CBRT:   da = F(1)/(F(3)*r*r); break;
            case OP_POWI: {
                int n = (int)bi;
                da = F(n) * pow(a, F(n)-F(1)); break;
            }
            case OP_EXP:    da = r; break;
            case OP_EXP2:   da = r * log(F(2)); break;
            case OP_EXPM1:  da = r + F(1); break;
            case OP_LN:     da = F(1)/a; break;
            case OP_LOG2:   da = F(1)/(a*log(F(2))); break;
            case OP_LOG10:  da = F(1)/(a*log(F(10))); break;
            case OP_LN1P:   da = F(1)/(F(1)+a); break;
            case OP_SIN:    da = cos(a); break;
            case OP_COS:    da = -sin(a); break;
            case OP_TAN:    { F c = cos(a); da = F(1)/(c*c); break; }
            case OP_ASIN:   da = F(1)/sqrt(F(1)-a*a); break;
            case OP_ACOS:   da = F(-1)/sqrt(F(1)-a*a); break;
            case OP_ATAN:   da = F(1)/(F(1)+a*a); break;
            case OP_SINH:   da = cosh(a); break;
            case OP_COSH:   da = sinh(a); break;
            case OP_TANH:   { F c = cosh(a); da = F(1)/(c*c); break; }
            case OP_ASINH:  da = F(1)/sqrt(a*a+F(1)); break;
            case OP_ACOSH:  da = F(1)/sqrt(a*a-F(1)); break;
            case OP_ATANH:  da = F(1)/(F(1)-a*a); break;
            case OP_ABS:    da = _sign(a); break;
            case OP_SIGNUM: case OP_FLOOR: case OP_CEIL:
            case OP_ROUND:  case OP_TRUNC: da = F(0); break;
            case OP_FRACT:  da = F(1); break;
            default: break;
        }

        adjoints[a_base + ai] += da * adj;
        if (bi != UNUSED && op != OP_POWI) {
            adjoints[a_base + bi] += db * adj;
        }
    }

    // Write gradients
    unsigned int g_base = bid * num_inputs;
    for (unsigned int i = 0; i < num_inputs; i++) {
        grad_out[g_base + i] = adjoints[a_base + i];
    }
}

// ════════════════════════════════════════════════════════════════════
// Forward tangent (JVP) kernel
// ════════════════════════════════════════════════════════════════════
extern "C" __global__ void tangent_forward(
    const unsigned int* __restrict__ opcodes,
    const unsigned int* __restrict__ arg0,
    const unsigned int* __restrict__ arg1,
    const F* __restrict__ constants,
    const F* __restrict__ primal_inputs,
    const F* __restrict__ tangent_seeds,
    F* __restrict__ primals,
    F* __restrict__ tangents,
    F* __restrict__ tangent_outputs,
    const unsigned int* __restrict__ output_indices,
    unsigned int num_ops,
    unsigned int num_inputs,
    unsigned int num_variables,
    unsigned int num_outputs,
    unsigned int batch_size
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;

    unsigned int base = bid * num_variables;
    unsigned int in_base = bid * num_inputs;

    // Initialize
    for (unsigned int i = 0; i < num_variables; i++) {
        primals[base + i] = constants[i];
        tangents[base + i] = F(0);
    }
    for (unsigned int i = 0; i < num_inputs; i++) {
        primals[base + i] = primal_inputs[in_base + i];
        tangents[base + i] = tangent_seeds[in_base + i];
    }

    // Forward pass with tangent propagation
    for (unsigned int i = num_inputs; i < num_ops; i++) {
        unsigned int op = opcodes[i];
        if (op == OP_CONST) continue;

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];
        F a = primals[base + ai];
        F at = tangents[base + ai];
        F r = F(0), rt = F(0);

        switch (op) {
            case OP_ADD: { F b=primals[base+bi]; F bt=tangents[base+bi]; r=a+b; rt=at+bt; break; }
            case OP_SUB: { F b=primals[base+bi]; F bt=tangents[base+bi]; r=a-b; rt=at-bt; break; }
            case OP_MUL: { F b=primals[base+bi]; F bt=tangents[base+bi]; r=a*b; rt=b*at+a*bt; break; }
            case OP_DIV: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                r=a/b; F inv=F(1)/b; rt=inv*at-a*inv*inv*bt; break;
            }
            case OP_REM: r=fmod(a, primals[base+bi]); rt=at; break;
            case OP_POWF: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                r=pow(a,b); rt=r*(b/a*at+log(a)*bt); break;
            }
            case OP_ATAN2: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                r=atan2(a,b); F d=a*a+b*b; rt=(b*at-a*bt)/d; break;
            }
            case OP_HYPOT: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                r=hypot(a,b); rt=(a*at+b*bt)/r; break;
            }
            case OP_MAX: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                if(a>=b){r=a;rt=at;}else{r=b;rt=bt;} break;
            }
            case OP_MIN: {
                F b=primals[base+bi]; F bt=tangents[base+bi];
                if(a<=b){r=a;rt=at;}else{r=b;rt=bt;} break;
            }
            case OP_NEG:   r=-a; rt=-at; break;
            case OP_RECIP: r=F(1)/a; rt=-at/(a*a); break;
            case OP_SQRT:  r=sqrt(a); rt=at/(F(2)*r); break;
            case OP_CBRT:  r=_cbrt(a); rt=at/(F(3)*r*r); break;
            case OP_POWI: {
                int n = (int)bi;
                F fn = F(n);
                r=pow(a,fn); rt=fn*pow(a,fn-F(1))*at; break;
            }
            case OP_EXP:   r=exp(a); rt=r*at; break;
            case OP_EXP2:  r=exp2(a); rt=r*log(F(2))*at; break;
            case OP_EXPM1: r=expm1(a); rt=(r+F(1))*at; break;
            case OP_LN:    r=log(a); rt=at/a; break;
            case OP_LOG2:  r=log2(a); rt=at/(a*log(F(2))); break;
            case OP_LOG10: r=log10(a); rt=at/(a*log(F(10))); break;
            case OP_LN1P:  r=log1p(a); rt=at/(F(1)+a); break;
            case OP_SIN:   r=sin(a); rt=cos(a)*at; break;
            case OP_COS:   r=cos(a); rt=-sin(a)*at; break;
            case OP_TAN:   r=tan(a); { F c=cos(a); rt=at/(c*c); } break;
            case OP_ASIN:  r=asin(a); rt=at/sqrt(F(1)-a*a); break;
            case OP_ACOS:  r=acos(a); rt=-at/sqrt(F(1)-a*a); break;
            case OP_ATAN:  r=atan(a); rt=at/(F(1)+a*a); break;
            case OP_SINH:  r=sinh(a); rt=cosh(a)*at; break;
            case OP_COSH:  r=cosh(a); rt=sinh(a)*at; break;
            case OP_TANH:  r=tanh(a); { F c=cosh(a); rt=at/(c*c); } break;
            case OP_ASINH: r=asinh(a); rt=at/sqrt(a*a+F(1)); break;
            case OP_ACOSH: r=acosh(a); rt=at/sqrt(a*a-F(1)); break;
            case OP_ATANH: r=atanh(a); rt=at/(F(1)-a*a); break;
            case OP_ABS:   r=fabs(a); rt=_sign(a)*at; break;
            case OP_SIGNUM: r=_sign(a); rt=F(0); break;
            case OP_FLOOR:  r=floor(a); rt=F(0); break;
            case OP_CEIL:   r=ceil(a); rt=F(0); break;
            case OP_ROUND:  r=round(a); rt=F(0); break;
            case OP_TRUNC:  r=trunc(a); rt=F(0); break;
            case OP_FRACT:  r=_fract(a); rt=at; break;
            default: break;
        }
        primals[base + i] = r;
        tangents[base + i] = rt;
    }

    // Write tangent outputs
    unsigned int out_base = bid * num_outputs;
    for (unsigned int j = 0; j < num_outputs; j++) {
        tangent_outputs[out_base + j] = tangents[base + output_indices[j]];
    }
}

// ════════════════════════════════════════════════════════════════════
// Forward-over-reverse (HVP) kernel for sparse Hessian
// ════════════════════════════════════════════════════════════════════
extern "C" __global__ void tangent_reverse(
    const unsigned int* __restrict__ opcodes,
    const unsigned int* __restrict__ arg0,
    const unsigned int* __restrict__ arg1,
    const F* __restrict__ constants,
    const F* __restrict__ primal_inputs,
    const F* __restrict__ tangent_seeds,
    F* __restrict__ primals,
    F* __restrict__ tans,
    F* __restrict__ adj_re,
    F* __restrict__ adj_eps,
    F* __restrict__ grad_out,
    F* __restrict__ hvp_out,
    const unsigned int* __restrict__ output_indices,
    unsigned int num_ops,
    unsigned int num_inputs,
    unsigned int num_variables,
    unsigned int batch_size
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;

    unsigned int base = bid * num_variables;
    unsigned int in_base = bid * num_inputs;

    // Phase 1: Forward tangent pass
    for (unsigned int i = 0; i < num_variables; i++) {
        primals[base + i] = constants[i];
        tans[base + i] = F(0);
    }
    for (unsigned int i = 0; i < num_inputs; i++) {
        primals[base + i] = primal_inputs[in_base + i];
        tans[base + i] = tangent_seeds[in_base + i];
    }

    for (unsigned int i = num_inputs; i < num_ops; i++) {
        unsigned int op = opcodes[i];
        if (op == OP_CONST) continue;

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];
        F a = primals[base + ai];
        F at = tans[base + ai];
        F r = F(0), rt = F(0);

        switch (op) {
            case OP_ADD: { F b=primals[base+bi]; F bt=tans[base+bi]; r=a+b; rt=at+bt; break; }
            case OP_SUB: { F b=primals[base+bi]; F bt=tans[base+bi]; r=a-b; rt=at-bt; break; }
            case OP_MUL: { F b=primals[base+bi]; F bt=tans[base+bi]; r=a*b; rt=b*at+a*bt; break; }
            case OP_DIV: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                r=a/b; F inv=F(1)/b; rt=inv*at-a*inv*inv*bt; break;
            }
            case OP_REM: r=fmod(a, primals[base+bi]); rt=at; break;
            case OP_POWF: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                r=pow(a,b); rt=r*(b/a*at+log(a)*bt); break;
            }
            case OP_ATAN2: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                r=atan2(a,b); F d=a*a+b*b; rt=(b*at-a*bt)/d; break;
            }
            case OP_HYPOT: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                r=hypot(a,b); rt=(a*at+b*bt)/r; break;
            }
            case OP_MAX: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                if(a>=b){r=a;rt=at;}else{r=b;rt=bt;} break;
            }
            case OP_MIN: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                if(a<=b){r=a;rt=at;}else{r=b;rt=bt;} break;
            }
            case OP_NEG:   r=-a; rt=-at; break;
            case OP_RECIP: r=F(1)/a; rt=-at/(a*a); break;
            case OP_SQRT:  r=sqrt(a); rt=at/(F(2)*r); break;
            case OP_CBRT:  r=_cbrt(a); rt=at/(F(3)*r*r); break;
            case OP_POWI: {
                int n = (int)bi;
                F fn = F(n);
                r=pow(a,fn); rt=fn*pow(a,fn-F(1))*at; break;
            }
            case OP_EXP:   r=exp(a); rt=r*at; break;
            case OP_EXP2:  r=exp2(a); rt=r*log(F(2))*at; break;
            case OP_EXPM1: r=expm1(a); rt=(r+F(1))*at; break;
            case OP_LN:    r=log(a); rt=at/a; break;
            case OP_LOG2:  r=log2(a); rt=at/(a*log(F(2))); break;
            case OP_LOG10: r=log10(a); rt=at/(a*log(F(10))); break;
            case OP_LN1P:  r=log1p(a); rt=at/(F(1)+a); break;
            case OP_SIN:   r=sin(a); rt=cos(a)*at; break;
            case OP_COS:   r=cos(a); rt=-sin(a)*at; break;
            case OP_TAN:   r=tan(a); { F c=cos(a); rt=at/(c*c); } break;
            case OP_ASIN:  r=asin(a); rt=at/sqrt(F(1)-a*a); break;
            case OP_ACOS:  r=acos(a); rt=-at/sqrt(F(1)-a*a); break;
            case OP_ATAN:  r=atan(a); rt=at/(F(1)+a*a); break;
            case OP_SINH:  r=sinh(a); rt=cosh(a)*at; break;
            case OP_COSH:  r=cosh(a); rt=sinh(a)*at; break;
            case OP_TANH:  r=tanh(a); { F c=cosh(a); rt=at/(c*c); } break;
            case OP_ASINH: r=asinh(a); rt=at/sqrt(a*a+F(1)); break;
            case OP_ACOSH: r=acosh(a); rt=at/sqrt(a*a-F(1)); break;
            case OP_ATANH: r=atanh(a); rt=at/(F(1)-a*a); break;
            case OP_ABS:   r=fabs(a); rt=_sign(a)*at; break;
            case OP_SIGNUM: r=_sign(a); rt=F(0); break;
            case OP_FLOOR:  r=floor(a); rt=F(0); break;
            case OP_CEIL:   r=ceil(a); rt=F(0); break;
            case OP_ROUND:  r=round(a); rt=F(0); break;
            case OP_TRUNC:  r=trunc(a); rt=F(0); break;
            case OP_FRACT:  r=_fract(a); rt=at; break;
            default: break;
        }
        primals[base + i] = r;
        tans[base + i] = rt;
    }

    // Phase 2: Reverse sweep with Dual adjoints
    for (unsigned int i = 0; i < num_variables; i++) {
        adj_re[base + i] = F(0);
        adj_eps[base + i] = F(0);
    }
    unsigned int seed_idx = output_indices[0];
    adj_re[base + seed_idx] = F(1);

    for (unsigned int ii = 0; ii < num_ops; ii++) {
        unsigned int i = num_ops - 1 - ii;
        F ar = adj_re[base + i];
        F ae = adj_eps[base + i];
        if (ar == F(0) && ae == F(0)) continue;

        unsigned int op = opcodes[i];
        if (op == OP_INPUT || op == OP_CONST) continue;

        adj_re[base + i] = F(0);
        adj_eps[base + i] = F(0);

        unsigned int ai = arg0[i];
        unsigned int bi = arg1[i];
        F a = primals[base + ai];
        F at = tans[base + ai];
        F r = primals[base + i];

        F da_re = F(0), da_eps = F(0);
        F db_re = F(0), db_eps = F(0);

        switch (op) {
            case OP_ADD: da_re=F(1); db_re=F(1); break;
            case OP_SUB: da_re=F(1); db_re=F(-1); break;
            case OP_MUL: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                da_re=b; da_eps=bt; db_re=a; db_eps=at; break;
            }
            case OP_DIV: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                F inv=F(1)/b;
                da_re=inv; da_eps=-bt*inv*inv;
                db_re=-a*inv*inv; db_eps=-at*inv*inv+F(2)*a*bt*inv*inv*inv; break;
            }
            case OP_REM: da_re=F(1); break;
            case OP_POWF: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                F ab1 = pow(a, b-F(1));
                da_re = b * ab1;
                da_eps = bt*ab1 + b*ab1*((b-F(1))/a*at + log(a)*bt);
                F la = log(a);
                db_re = r * la;
                F rt2 = tans[base+i];
                db_eps = rt2*la + r*at/a; break;
            }
            case OP_ATAN2: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                F d=a*a+b*b; F d2=d*d;
                F dd=F(2)*(a*at+b*bt);
                da_re=b/d; da_eps=(bt*d-b*dd)/d2;
                db_re=-a/d; db_eps=(-at*d+a*dd)/d2; break;
            }
            case OP_HYPOT: {
                F b=primals[base+bi]; F bt=tans[base+bi];
                F r2=r*r; F rt2=tans[base+i];
                da_re=a/r; da_eps=(at*r-a*rt2)/r2;
                db_re=b/r; db_eps=(bt*r-b*rt2)/r2; break;
            }
            case OP_MAX: {
                F b=primals[base+bi];
                if(a>=b){da_re=F(1);}else{db_re=F(1);} break;
            }
            case OP_MIN: {
                F b=primals[base+bi];
                if(a<=b){da_re=F(1);}else{db_re=F(1);} break;
            }
            case OP_NEG:   da_re=F(-1); break;
            case OP_RECIP: { F inv=F(1)/a; da_re=-inv*inv; da_eps=F(2)*at*inv*inv*inv; break; }
            case OP_SQRT:  da_re=F(0.5)/r; da_eps=F(-0.25)*at/(a*r); break;
            case OP_CBRT:  { F rr=r*r; da_re=F(1)/(F(3)*rr); da_eps=F(-2)*at/(F(9)*rr*r); break; }
            case OP_POWI: {
                int n = (int)bi;
                F fn = F(n);
                da_re=fn*pow(a,fn-F(1));
                da_eps=fn*(fn-F(1))*pow(a,fn-F(2))*at; break;
            }
            case OP_EXP:    da_re=r; da_eps=r*at; break;
            case OP_EXP2:   { F l2=log(F(2)); da_re=r*l2; da_eps=r*l2*l2*at; break; }
            case OP_EXPM1:  da_re=r+F(1); da_eps=(r+F(1))*at; break;
            case OP_LN:     da_re=F(1)/a; da_eps=-at/(a*a); break;
            case OP_LOG2:   { F l2=log(F(2)); da_re=F(1)/(a*l2); da_eps=-at/(a*a*l2); break; }
            case OP_LOG10:  { F l10=log(F(10)); da_re=F(1)/(a*l10); da_eps=-at/(a*a*l10); break; }
            case OP_LN1P:   { F t=F(1)+a; da_re=F(1)/t; da_eps=-at/(t*t); break; }
            case OP_SIN:    da_re=cos(a); da_eps=-sin(a)*at; break;
            case OP_COS:    da_re=-sin(a); da_eps=-cos(a)*at; break;
            case OP_TAN:    { F c=cos(a); F s=F(1)/(c*c); da_re=s; da_eps=F(2)*tan(a)*s*at; break; }
            case OP_ASIN:   { F t=sqrt(F(1)-a*a); da_re=F(1)/t; da_eps=a*at/(t*t*t); break; }
            case OP_ACOS:   { F t=sqrt(F(1)-a*a); da_re=F(-1)/t; da_eps=-a*at/(t*t*t); break; }
            case OP_ATAN:   { F t=F(1)+a*a; da_re=F(1)/t; da_eps=F(-2)*a*at/(t*t); break; }
            case OP_SINH:   da_re=cosh(a); da_eps=sinh(a)*at; break;
            case OP_COSH:   da_re=sinh(a); da_eps=cosh(a)*at; break;
            case OP_TANH:   { F c=cosh(a); F s=F(1)/(c*c); da_re=s; da_eps=F(-2)*tanh(a)*s*at; break; }
            case OP_ASINH:  { F t=sqrt(a*a+F(1)); da_re=F(1)/t; da_eps=-a*at/(t*t*t); break; }
            case OP_ACOSH:  { F t=sqrt(a*a-F(1)); da_re=F(1)/t; da_eps=-a*at/(t*t*t); break; }
            case OP_ATANH:  { F t=F(1)-a*a; da_re=F(1)/t; da_eps=F(2)*a*at/(t*t); break; }
            case OP_ABS:    da_re=_sign(a); break;
            case OP_SIGNUM: case OP_FLOOR: case OP_CEIL:
            case OP_ROUND:  case OP_TRUNC: break;
            case OP_FRACT:  da_re=F(1); break;
            default: break;
        }

        adj_re[base + ai] += da_re * ar;
        adj_eps[base + ai] += da_re * ae + da_eps * ar;
        if (bi != UNUSED && op != OP_POWI) {
            adj_re[base + bi] += db_re * ar;
            adj_eps[base + bi] += db_re * ae + db_eps * ar;
        }
    }

    // Write gradient and HVP outputs
    unsigned int g_base = bid * num_inputs;
    for (unsigned int i = 0; i < num_inputs; i++) {
        grad_out[g_base + i] = adj_re[base + i];
        hvp_out[g_base + i] = adj_eps[base + i];
    }
}
