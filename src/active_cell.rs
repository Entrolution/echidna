//! Shared machinery for thread-local "active pointer" access.
//!
//! `Tape` and `BytecodeTape` each publish a per-float-type thread-local
//! `Cell<*mut T>` that an RAII guard points at a live recording, plus a
//! per-type borrow flag that rejects reentrant access. The flag handling
//! and the pointer dereference are identical for both tapes and
//! safety-critical, so they live here once. The activation guards
//! themselves stay with their tape types — their drop semantics differ
//! (`BtapeGuard` enforces LIFO with a hard assert; `TapeGuard` does not).

use std::cell::Cell;
use std::thread::LocalKey;

/// RAII set/clear of a thread-local borrow flag; panics on reentrance.
struct ReentranceGuard {
    cell: &'static LocalKey<Cell<bool>>,
}

impl ReentranceGuard {
    fn new(cell: &'static LocalKey<Cell<bool>>, fn_name: &str) -> Self {
        cell.with(|b| {
            assert!(
                !b.get(),
                "reentrant {fn_name} call detected — this would create aliased &mut references"
            );
            b.set(true);
        });
        ReentranceGuard { cell }
    }
}

impl Drop for ReentranceGuard {
    fn drop(&mut self) {
        self.cell.with(|b| b.set(false));
    }
}

/// Run `f` on the active `T` behind a thread-local pointer cell.
///
/// Panics with `missing_msg` when no `T` is active, and rejects reentrant
/// calls via `borrow_cell` (naming the entry point `fn_name` in the panic).
///
/// # Safety contract (callers)
///
/// A non-null pointer in `ptr_cell` must have been installed by an RAII
/// guard whose lifetime ties it to a live `&mut T` on the stack frame that
/// constructed the guard (`TapeGuard`, `BtapeGuard`) — the borrow checker
/// then rejects any program in which the guard outlives its referent.
#[inline]
pub(crate) fn with_active_ptr<T, R>(
    ptr_cell: &'static LocalKey<Cell<*mut T>>,
    borrow_cell: &'static LocalKey<Cell<bool>>,
    fn_name: &str,
    missing_msg: &str,
    f: impl FnOnce(&mut T) -> R,
) -> R {
    let _guard = ReentranceGuard::new(borrow_cell, fn_name);
    ptr_cell.with(|cell| {
        let ptr = cell.get();
        assert!(!ptr.is_null(), "{missing_msg}");
        // SAFETY: per the caller contract above, a non-null pointer was
        // installed by a lifetime-tied RAII guard, so it references a live
        // `T`. Access is single-threaded via the thread-local, and the
        // ReentranceGuard above rejects nested calls, so no aliased `&mut`
        // references can form.
        let target = unsafe { &mut *ptr };
        f(target)
    })
}
