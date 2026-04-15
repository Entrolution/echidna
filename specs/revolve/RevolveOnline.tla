--------------------------- MODULE RevolveOnline ---------------------------
(*
 * Formal specification of the online gradient checkpointing algorithm
 * used in echidna.
 *
 * Unlike base Revolve, the total step count is unknown at the start.
 * A nondeterministic stop condition models the black-box closure — the
 * spec verifies that invariants hold for ALL possible stopping points.
 *
 * The buffer uses periodic thinning: when full, it keeps buffer[0]
 * (pinned initial state) and every other remaining entry, then doubles
 * the spacing. This guarantees O(log N) recomputation overhead.
 *
 * Rust correspondence: grad_checkpointed_online() in src/checkpoint.rs
 *)

EXTENDS BinomialBeta, Naturals, Sequences, FiniteSets

CONSTANTS
    MaxSteps,        \* Upper bound for model checking (not known at runtime)
    NumCheckpoints   \* Buffer capacity (>= 2)

ASSUME MaxSteps >= 1
ASSUME NumCheckpoints >= 2

---------------------------------------------------------------------------
(* Helper operators *)
---------------------------------------------------------------------------

(*
 * Select every other element from seq starting at 1-based index startIdx.
 * Used to implement the thinning logic.
 *)
RECURSIVE SelectEveryOther(_, _)
SelectEveryOther(seq, startIdx) ==
    IF startIdx > Len(seq) THEN << >>
    ELSE << seq[startIdx] >> \o SelectEveryOther(seq, startIdx + 2)

(*
 * Thinning: keep buffer[1] (step 0, pinned), then from buffer[2..],
 * skip the first and keep every other.
 *
 * Rust: buffer[0] + buffer[1..].iter().skip(1).step_by(2)
 *
 * In 0-based Rust indices, this keeps: {0, 2, 4, 6, ...}
 * In 1-based TLA+ indices, this keeps: {1, 3, 5, 7, ...}
 *
 * Worked example:
 *   Rust [0, 1, 2, 3, 4] -> [0, 2, 4]     (spacing 1 -> 2)
 *   TLA+ <<0, 1, 2, 3, 4>> -> <<0, 2, 4>> (indices 1, 3, 5)
 *)
ThinBuffer(buf) ==
    << buf[1] >> \o SelectEveryOther(buf, 3)

(*
 * Determine the end of a backward segment given its index.
 *)
SegEnd(seg, buffer, totalSteps) ==
    IF seg < Len(buffer)
    THEN buffer[seg + 1]
    ELSE totalSteps

---------------------------------------------------------------------------
(* Variables *)
---------------------------------------------------------------------------

VARIABLES
    buffer,          \* Sequence of checkpoint step indices
    spacing,         \* Current spacing (always a power of 2)
    stepIndex,       \* Current step counter
    stopped,         \* Whether the stop predicate has fired
    phase,           \* "forward" | "backward" | "done"
    segIndex,        \* Backward pass segment counter
    coveredSteps     \* Steps covered by backward pass

vars == <<buffer, spacing, stepIndex, stopped, phase, segIndex, coveredSteps>>

---------------------------------------------------------------------------
(* Initial state *)
---------------------------------------------------------------------------

Init ==
    /\ buffer = << 0 >>        \* Pinned: step 0
    /\ spacing = 1
    /\ stepIndex = 0
    /\ stopped = FALSE
    /\ phase = "forward"
    /\ segIndex = 0
    /\ coveredSteps = {}

---------------------------------------------------------------------------
(* Forward phase: advance, save, maybe stop, maybe thin *)
---------------------------------------------------------------------------

(*
 * One forward step. The Rust code's loop body has this order:
 *   1. Advance (step_index += 1)
 *   2. Save checkpoint if on spacing grid
 *   3. Check stop condition — if true, break BEFORE thinning
 *   4. Thin if buffer is full
 *
 * We model stop as a nondeterministic choice: the environment may stop
 * at any step >= 1. This means TLC explores ALL possible stopping points.
 *)

\* Step forward and save, then stop.
ForwardAndStop ==
    /\ phase = "forward"
    /\ ~stopped
    /\ stepIndex < MaxSteps
    /\ LET nextStep == stepIndex + 1
           onGrid  == (nextStep % spacing) = 0
           newBuf  == IF onGrid THEN Append(buffer, nextStep)
                      ELSE buffer
       IN
       /\ stepIndex' = nextStep
       /\ buffer' = newBuf
       /\ stopped' = TRUE
       /\ UNCHANGED <<spacing, phase, segIndex, coveredSteps>>

\* Step forward and save, continue (don't stop), then thin if needed.
ForwardAndContinue ==
    /\ phase = "forward"
    /\ ~stopped
    /\ stepIndex < MaxSteps - 1   \* Can't continue past MaxSteps-1 (will do last step as stop)
    /\ LET nextStep == stepIndex + 1
           onGrid  == (nextStep % spacing) = 0
           bufAfterSave == IF onGrid THEN Append(buffer, nextStep)
                           ELSE buffer
           \* Thin if buffer is at capacity (only when NOT stopping)
           needsThin == Len(bufAfterSave) >= NumCheckpoints
           bufAfterThin == IF needsThin THEN ThinBuffer(bufAfterSave)
                           ELSE bufAfterSave
           newSpacing == IF needsThin THEN spacing * 2
                         ELSE spacing
       IN
       /\ stepIndex' = nextStep
       /\ buffer' = bufAfterThin
       /\ spacing' = newSpacing
       /\ stopped' = FALSE
       /\ UNCHANGED <<phase, segIndex, coveredSteps>>

\* Force stop at MaxSteps (bound for model checking).
ForwardForceStop ==
    /\ phase = "forward"
    /\ ~stopped
    /\ stepIndex = MaxSteps
    /\ stopped' = TRUE
    /\ UNCHANGED <<buffer, spacing, stepIndex, phase, segIndex, coveredSteps>>

\* Transition from forward to backward after stopping.
ForwardDone ==
    /\ phase = "forward"
    /\ stopped
    /\ phase' = "backward"
    /\ segIndex' = Len(buffer)
    /\ UNCHANGED <<buffer, spacing, stepIndex, stopped, coveredSteps>>

---------------------------------------------------------------------------
(* Backward phase *)
---------------------------------------------------------------------------

BackwardSegment ==
    /\ phase = "backward"
    /\ segIndex >= 1
    /\ LET ckptStep == buffer[segIndex]
           segEnd   == SegEnd(segIndex, buffer, stepIndex)
       IN
       /\ coveredSteps' = coveredSteps \union (ckptStep .. (segEnd - 1))
       /\ segIndex' = segIndex - 1
    /\ UNCHANGED <<buffer, spacing, stepIndex, stopped, phase>>

BackwardDone ==
    /\ phase = "backward"
    /\ segIndex = 0
    /\ phase' = "done"
    /\ UNCHANGED <<buffer, spacing, stepIndex, stopped, segIndex, coveredSteps>>

---------------------------------------------------------------------------
(* Next-state relation *)
---------------------------------------------------------------------------

Next ==
    \/ ForwardAndStop
    \/ ForwardAndContinue
    \/ ForwardForceStop
    \/ ForwardDone
    \/ BackwardSegment
    \/ BackwardDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

---------------------------------------------------------------------------
(* Invariants *)
---------------------------------------------------------------------------

(*
 * SAFETY: Buffer never exceeds capacity.
 * Rust: thinning triggers when buffer.len() >= num_checkpoints
 *)
BufferBudget ==
    Len(buffer) <= NumCheckpoints

(*
 * SAFETY: Buffer[1] is always step 0 (pinned initial state).
 * Rust: buffer[0] = (0, x0), preserved during thinning.
 *)
PinnedOrigin ==
    Len(buffer) >= 1 => buffer[1] = 0

(*
 * SAFETY: Spacing is always a power of 2.
 * Rust: spacing starts at 1 and is only ever multiplied by 2.
 *)
SpacingPowerOf2 ==
    \E k \in 0..30 : spacing = 2^k

(*
 * SAFETY: All buffer entries after index 1 are multiples of spacing.
 * This is the uniformity guarantee maintained by thinning.
 * Rust: checkpoints are saved only when step_index.is_multiple_of(spacing)
 *)
UniformSpacing ==
    \A i \in 2..Len(buffer) : (buffer[i] % spacing) = 0

(*
 * SAFETY: Buffer entries are strictly increasing.
 *)
BufferSorted ==
    \A i, j \in 1..Len(buffer) : i < j => buffer[i] < buffer[j]

(*
 * COMPLETENESS: When done, backward pass covers all steps [0, stepIndex-1].
 * Only checked in terminal state.
 *)
CompletenessProperty ==
    phase = "done" => coveredSteps = 0 .. (stepIndex - 1)

---------------------------------------------------------------------------
(* Temporal properties *)
---------------------------------------------------------------------------

Termination == <>(phase = "done")

==========================================================================
