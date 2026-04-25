--------------------------- MODULE BinomialBeta ---------------------------
(*
 * Shared operator module for the Revolve checkpointing specifications.
 *
 * Defines the binomial coefficient function Beta(s, c) and the optimal
 * advance computation used by the Revolve schedule algorithm.
 *
 * This module has no variables or state machine — it is EXTENDS'd by
 * other specs that need these operators.
 *
 * Rust correspondence:
 *   Beta           -> beta()              in src/checkpoint.rs
 *   OptimalAdvance -> optimal_advance()   in src/checkpoint.rs
 *)

LOCAL INSTANCE Naturals
LOCAL INSTANCE Sequences
LOCAL INSTANCE FiniteSets

---------------------------------------------------------------------------
(* Utility operators *)
---------------------------------------------------------------------------

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

(*
 * Sum of a sequence of naturals.
 *)
RECURSIVE SumSeq(_)
SumSeq(s) ==
    IF s = << >> THEN 0
    ELSE Head(s) + SumSeq(Tail(s))

---------------------------------------------------------------------------
(* Standard binomial coefficient C(n, k) via Pascal's rule *)
---------------------------------------------------------------------------

RECURSIVE Choose(_, _)
Choose(n, k) ==
    IF k = 0 THEN 1
    ELSE IF k = n THEN 1
    ELSE IF n = 0 THEN 0
    ELSE Choose(n - 1, k - 1) + Choose(n - 1, k)

---------------------------------------------------------------------------
(* Beta(s, c) — Revolve binomial function *)
---------------------------------------------------------------------------
(*
 * Beta(s, c) represents the maximum number of forward steps that can be
 * reversed with s recomputations and c checkpoint slots.
 *
 * For c >= 1: Beta(s, c) = C(s+c, c), the standard binomial coefficient.
 * For c = 0:  Beta(s, 0) = s + 1 (linear scan, no checkpoints).
 *
 * The c=0 case is a convention from the Revolve algorithm (Griewank &
 * Walther, 2000), not the standard binomial C(s, 0) = 1.
 *
 * The Rust implementation uses a multiplicative formula with overflow
 * detection; here we use Choose() via Pascal's rule, which is safe for
 * the small parameter bounds used in model checking.
 *)
Beta(s, c) ==
    IF c = 0 THEN s + 1
    ELSE Choose(s + c, c)

---------------------------------------------------------------------------
(* Optimal advance distance *)
---------------------------------------------------------------------------
(*
 * OptimalAdvance(steps, c) computes how far to advance before placing
 * a checkpoint, given `steps` remaining forward steps and `c` available
 * checkpoint slots.
 *
 * Finds the smallest t such that Beta(t, c) >= steps, then returns
 * Beta(t-1, c-1) clamped to [1, steps-1].
 *
 * Rust correspondence: optimal_advance() in src/checkpoint.rs
 *)
OptimalAdvance(steps, c) ==
    IF c = 0 \/ steps <= 1
    THEN steps
    ELSE
        LET t == CHOOSE t \in 1..steps : Beta(t, c) >= steps
                                         /\ (t = 1 \/ Beta(t - 1, c) < steps)
        IN  IF t > 0 /\ c > 0
            THEN Min(Max(Beta(t - 1, c - 1), 1), steps - 1)
            ELSE 1

---------------------------------------------------------------------------
(* Verification: ASSUME assertions *)
---------------------------------------------------------------------------
(*
 * Checked by TLC at startup when any spec EXTENDS this module.
 * Correspond to the Rust test beta_base_cases in tests/checkpoint.rs.
 *)
ASSUME Beta(0, 0) = 1
ASSUME Beta(0, 1) = 1
ASSUME Beta(0, 3) = 1
ASSUME Beta(1, 0) = 2
ASSUME Beta(3, 0) = 4
ASSUME Beta(1, 1) = 2
ASSUME Beta(2, 2) = 6
ASSUME Beta(3, 2) = 10

(* Spot checks for OptimalAdvance *)
ASSUME OptimalAdvance(1, 1) = 1
ASSUME OptimalAdvance(2, 1) = 1

==========================================================================
