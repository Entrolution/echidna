------------------------------ MODULE Revolve ------------------------------
(*
 * Formal specification of the base Revolve (binomial) gradient checkpointing
 * algorithm used in echidna.
 *
 * Models the full lifecycle:
 *   Phase 1 (Schedule):  Compute optimal checkpoint positions via recursive
 *                         interval splitting (mirrors schedule_recursive).
 *   Phase 2 (Forward):   Walk forward through all steps, storing state at
 *                         checkpoint positions (mirrors grad_checkpointed).
 *   Phase 3 (Backward):  Process segments in reverse, marking covered steps
 *                         (mirrors backward_from_checkpoints).
 *
 * State vectors are abstracted to step indices. We verify the bookkeeping
 * — which steps are checkpointed, which segments are covered by the
 * backward pass — not numerical values.
 *
 * The backward pass is deliberately coarse: each segment is modelled as an
 * atomic action that marks steps as covered. The real implementation
 * recomputes forward within each segment, but uses only a local buffer
 * (not additional checkpoint slots), so this abstraction is safe.
 *
 * Rust correspondence: grad_checkpointed() in src/checkpoint.rs
 *)

EXTENDS BinomialBeta, Naturals, Sequences, FiniteSets

CONSTANTS
    NumSteps,        \* Total forward steps (>= 2)
    NumCheckpoints   \* Available checkpoint slots (>= 1, <= NumSteps)

ASSUME NumSteps >= 2
ASSUME NumCheckpoints >= 1
ASSUME NumCheckpoints <= NumSteps

---------------------------------------------------------------------------
(* Variables *)
---------------------------------------------------------------------------

VARIABLES
    phase,              \* "schedule" | "forward" | "backward" | "done"
    positions,          \* Set of checkpoint positions computed by schedule
    workStack,          \* Stack of <<start, end, slots>> for schedule phase
    currentStep,        \* Forward pass loop counter
    storedCheckpoints,  \* Sequence of step indices stored during forward pass
    segIndex,           \* Backward pass segment counter (counts down)
    coveredSteps        \* Set of steps whose VJP has been computed

vars == <<phase, positions, workStack, currentStep,
          storedCheckpoints, segIndex, coveredSteps>>

---------------------------------------------------------------------------
(* Helper operators *)
---------------------------------------------------------------------------

(*
 * For the backward pass: determine the end of a segment given its index
 * in storedCheckpoints. The last segment ends at NumSteps.
 *)
SegEnd(seg) ==
    IF seg < Len(storedCheckpoints)
    THEN storedCheckpoints[seg + 1]
    ELSE NumSteps

---------------------------------------------------------------------------
(* Initial state *)
---------------------------------------------------------------------------

Init ==
    /\ phase = "schedule"
    /\ positions = {}
    /\ workStack = << <<0, NumSteps, NumCheckpoints>> >>
    /\ currentStep = 0
    /\ storedCheckpoints = << 0 >>   \* Step 0 (initial state) always stored
    /\ segIndex = 0
    /\ coveredSteps = {}

---------------------------------------------------------------------------
(* Phase 1: Schedule — recursive interval splitting *)
---------------------------------------------------------------------------
(*
 * Each step pops one work item from the stack, computes the optimal split,
 * adds the split to positions, and pushes two sub-intervals.
 *
 * Mirrors schedule_recursive() in src/checkpoint.rs.
 *)

ScheduleStep ==
    /\ phase = "schedule"
    /\ workStack # << >>
    /\ LET item  == Head(workStack)
           rest  == Tail(workStack)
           start == item[1]
           end   == item[2]
           slots == item[3]
           steps == end - start
       IN
       IF steps <= 1 \/ slots = 0
       THEN
           \* Base case: no split needed, just pop
           /\ workStack' = rest
           /\ positions' = positions
       ELSE
           LET advance == OptimalAdvance(steps, slots)
               split   == start + advance
           IN
           IF split > start /\ split < end
           THEN
               /\ positions' = positions \union {split}
               \* Push left sub-interval (slots - 1) then right (slots)
               /\ workStack' = Append(
                      Append(rest, <<start, split, slots - 1>>),
                      <<split, end, slots>>)
           ELSE
               /\ workStack' = rest
               /\ positions' = positions
    /\ UNCHANGED <<phase, currentStep, storedCheckpoints, segIndex, coveredSteps>>

(*
 * Transition from schedule phase to forward phase when the work stack
 * is empty.
 *)
ScheduleDone ==
    /\ phase = "schedule"
    /\ workStack = << >>
    /\ phase' = "forward"
    /\ UNCHANGED <<positions, workStack, currentStep, storedCheckpoints,
                   segIndex, coveredSteps>>

---------------------------------------------------------------------------
(* Phase 2: Forward pass *)
---------------------------------------------------------------------------
(*
 * Walk steps 0..NumSteps-1. After each step, if the next step index is a
 * checkpoint position and we haven't exceeded the budget, store it.
 *
 * Mirrors the forward loop in grad_checkpointed().
 *)

ForwardStep ==
    /\ phase = "forward"
    /\ currentStep < NumSteps
    /\ currentStep' = currentStep + 1
    /\ LET nextStep == currentStep + 1
       IN
       IF nextStep < NumSteps
          /\ nextStep \in positions
          /\ Len(storedCheckpoints) < NumCheckpoints + 1  \* +1 for pinned step 0
       THEN
           storedCheckpoints' = Append(storedCheckpoints, nextStep)
       ELSE
           storedCheckpoints' = storedCheckpoints
    /\ UNCHANGED <<phase, positions, workStack, segIndex, coveredSteps>>

(*
 * Transition from forward to backward when all steps are done.
 *)
ForwardDone ==
    /\ phase = "forward"
    /\ currentStep = NumSteps
    /\ phase' = "backward"
    /\ segIndex' = Len(storedCheckpoints)
    /\ UNCHANGED <<positions, workStack, currentStep, storedCheckpoints,
                   coveredSteps>>

---------------------------------------------------------------------------
(* Phase 3: Backward pass *)
---------------------------------------------------------------------------
(*
 * Process segments in reverse order. Each segment covers steps from
 * its checkpoint to the start of the next segment (or NumSteps for the
 * last segment).
 *
 * Mirrors backward_from_checkpoints() in src/checkpoint.rs.
 *)

BackwardSegment ==
    /\ phase = "backward"
    /\ segIndex >= 1
    /\ LET ckptStep == storedCheckpoints[segIndex]
           segEnd   == SegEnd(segIndex)
       IN
       /\ coveredSteps' = coveredSteps \union (ckptStep .. (segEnd - 1))
       /\ segIndex' = segIndex - 1
    /\ UNCHANGED <<phase, positions, workStack, currentStep, storedCheckpoints>>

(*
 * Transition to done when all segments have been processed.
 *)
BackwardDone ==
    /\ phase = "backward"
    /\ segIndex = 0
    /\ phase' = "done"
    /\ UNCHANGED <<positions, workStack, currentStep, storedCheckpoints,
                   segIndex, coveredSteps>>

---------------------------------------------------------------------------
(* Next-state relation *)
---------------------------------------------------------------------------

Next ==
    \/ ScheduleStep
    \/ ScheduleDone
    \/ ForwardStep
    \/ ForwardDone
    \/ BackwardSegment
    \/ BackwardDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

---------------------------------------------------------------------------
(* Invariants *)
---------------------------------------------------------------------------

(*
 * SAFETY: Checkpoint budget is never exceeded.
 * storedCheckpoints includes pinned step 0, so the limit is
 * NumCheckpoints + 1.
 *
 * Rust: all_positions.truncate(num_checkpoints) in grad_checkpointed
 *)
BudgetInvariant ==
    Len(storedCheckpoints) <= NumCheckpoints + 1

(*
 * SAFETY: All computed positions are within valid range [1, NumSteps-1].
 *
 * Rust: next_step < num_steps guard in forward loop
 *)
PositionRangeInvariant ==
    \A p \in positions : p >= 1 /\ p <= NumSteps - 1

(*
 * SAFETY: Stored checkpoints have strictly increasing step indices.
 * This must hold after the forward phase completes.
 *
 * Rust: positions are computed from a sorted, deduped Vec;
 *       forward pass inserts in order.
 *)
SortedCheckpoints ==
    phase \in {"backward", "done"} =>
        \A i, j \in 1..Len(storedCheckpoints) :
            i < j => storedCheckpoints[i] < storedCheckpoints[j]

(*
 * SAFETY: Step 0 (initial state) is always the first checkpoint.
 *
 * Rust: checkpoints.push((0, x0.to_vec())) is always first.
 *)
InitialStateStored ==
    Len(storedCheckpoints) >= 1 => storedCheckpoints[1] = 0

(*
 * SAFETY (schedule phase): Work stack entries are valid intervals.
 *)
WorkStackBoundsInvariant ==
    phase = "schedule" =>
        \A i \in 1..Len(workStack) :
            /\ workStack[i][1] >= 0
            /\ workStack[i][2] <= NumSteps
            /\ workStack[i][1] < workStack[i][2]
            /\ workStack[i][3] >= 0

(*
 * COMPLETENESS: When done, the backward pass has covered every step
 * in [0, NumSteps-1]. This is the key safety property: no step is missed.
 *)
CompletenessProperty ==
    phase = "done" => coveredSteps = 0 .. (NumSteps - 1)

---------------------------------------------------------------------------
(* Temporal properties *)
---------------------------------------------------------------------------

(*
 * LIVENESS: The algorithm always terminates.
 *)
Termination == <>(phase = "done")

==========================================================================
