use std::cell::Cell;

use crate::dual::Dual;
use crate::float::Float;

use super::BytecodeTape;

thread_local! {
    static BTAPE_F32: Cell<*mut BytecodeTape<f32>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_F64: Cell<*mut BytecodeTape<f64>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_DUAL_F32: Cell<*mut BytecodeTape<Dual<f32>>> = const { Cell::new(std::ptr::null_mut()) };
    static BTAPE_DUAL_F64: Cell<*mut BytecodeTape<Dual<f64>>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Trait to select the correct thread-local for a given float type.
///
/// Implemented for `f32`, `f64`, `Dual<f32>`, and `Dual<f64>`, enabling
/// `BReverse<F>` to be used with these base types.
pub trait BtapeThreadLocal: Float {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>>;
}

impl BtapeThreadLocal for f32 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F32
    }
}

impl BtapeThreadLocal for f64 {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_F64
    }
}

impl BtapeThreadLocal for Dual<f32> {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_DUAL_F32
    }
}

impl BtapeThreadLocal for Dual<f64> {
    fn btape_cell() -> &'static std::thread::LocalKey<Cell<*mut BytecodeTape<Self>>> {
        &BTAPE_DUAL_F64
    }
}

/// Access the active bytecode tape for the current thread.
/// Panics if no tape is active.
#[inline]
pub fn with_active_btape<F: BtapeThreadLocal, R>(f: impl FnOnce(&mut BytecodeTape<F>) -> R) -> R {
    F::btape_cell().with(|cell| {
        let ptr = cell.get();
        assert!(
            !ptr.is_null(),
            "No active bytecode tape. Use echidna::record() to record a function."
        );
        // SAFETY: BtapeGuard guarantees validity for the duration of the
        // recording scope, single-threaded via thread-local.
        let tape = unsafe { &mut *ptr };
        f(tape)
    })
}

/// RAII guard that sets a bytecode tape as the thread-local active tape.
pub struct BtapeGuard<F: BtapeThreadLocal> {
    prev: *mut BytecodeTape<F>,
}

impl<F: BtapeThreadLocal> BtapeGuard<F> {
    /// Activate `tape` as the thread-local bytecode tape.
    pub fn new(tape: &mut BytecodeTape<F>) -> Self {
        let prev = F::btape_cell().with(|cell| {
            let prev = cell.get();
            cell.set(tape as *mut BytecodeTape<F>);
            prev
        });
        BtapeGuard { prev }
    }
}

impl<F: BtapeThreadLocal> Drop for BtapeGuard<F> {
    fn drop(&mut self) {
        F::btape_cell().with(|cell| {
            cell.set(self.prev);
        });
    }
}
