//! Source-level convention gates (structural regression guards).
//!
//! These scan the checked-in sources for patterns the codebase has banned
//! after real bugs:
//!
//! - `== T::zero()` / `!= T::zero()` in tangent-generic sweep code — the
//!   comparison sees only the primal of a tangent-carrying type; semantic
//!   zero tests must use `IsAllZero` (four wrong-second-derivative bugs
//!   came from this).
//! - bare `x != x` NaN tests in WGSL — Metal fast-math may fold the
//!   comparison; NaN tests must inspect the bits (`is_nan_f32` or an
//!   inline bit test). CUDA sources are exempt: NVRTC compiles IEEE-strict
//!   here (no fast-math, `fmad=false`).

use std::fs;
use std::path::Path;

fn src(rel: &str) -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    fs::read_to_string(Path::new(root).join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// Strip `//` line comments and single-line `/* … */` block comments so a
/// commented-out banned pattern cannot false-positive the gates.
fn strip_comments(line: &str) -> String {
    let mut code = line.split("//").next().unwrap_or("").to_string();
    while let (Some(start), Some(end)) = (code.find("/*"), code.find("*/")) {
        if end > start {
            code.replace_range(start..end + 2, "");
        } else {
            break;
        }
    }
    code
}

#[test]
fn tangent_generic_code_uses_is_all_zero_not_eq() {
    // Files whose sweeps are generic over tangent-carrying T.
    for rel in [
        "src/bytecode_tape/tangent.rs",
        "src/bytecode_tape/taylor.rs",
    ] {
        let text = src(rel);
        for (i, line) in text.lines().enumerate() {
            let code = strip_comments(line);
            assert!(
                !code.contains("== T::zero()") && !code.contains("!= T::zero()"),
                "{rel}:{}: comparison against T::zero() in tangent-generic \
                 code compares only the primal — use IsAllZero::is_all_zero \
                 (or, if primal-only semantics are genuinely intended, move \
                 the code out of the scanned files listed above and \
                 document why)",
                i + 1
            );
        }
    }
}

#[test]
fn wgsl_nan_tests_inspect_bits() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders");
    for entry in fs::read_dir(&shader_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for (i, line) in text.lines().enumerate() {
            let code = strip_comments(line);
            // The banned idiom: comparing a value against itself for NaN.
            let banned = code.contains("x != x") || code.contains("v != v");
            assert!(
                !banned,
                "{}:{}: bare self-inequality NaN test in WGSL — Metal \
                 fast-math may fold it; inspect the bits instead \
                 (see is_nan_f32)",
                path.display(),
                i + 1
            );
        }
    }
}
