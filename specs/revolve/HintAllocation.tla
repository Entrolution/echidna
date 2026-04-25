--------------------------- MODULE HintAllocation --------------------------
(*
 * Formal specification of the hint-based checkpoint placement algorithm
 * used in echidna.
 *
 * Verifies that for ANY valid set of required checkpoint positions, the
 * combined allocation (required + Revolve-scheduled) satisfies:
 *   - All required positions are included
 *   - Total positions do not exceed the checkpoint budget
 *   - Free slots are distributed exactly via largest-remainder
 *   - All positions are within valid range
 *
 * The required set is chosen nondeterministically, so TLC exhaustively
 * checks all valid inputs.
 *
 * Rust correspondence: grad_checkpointed_with_hints() in src/checkpoint.rs
 *)

EXTENDS BinomialBeta, Naturals, Sequences, FiniteSets

CONSTANTS
    NumSteps,        \* Total forward steps (>= 2)
    NumCheckpoints   \* Available checkpoint slots (>= 1)

ASSUME NumSteps >= 2
ASSUME NumCheckpoints >= 1

---------------------------------------------------------------------------
(* Helper operators *)
---------------------------------------------------------------------------

(*
 * All valid subsets of {1..NumSteps-1} with size <= NumCheckpoints.
 * These are the possible required position sets.
 *)
ValidRequiredSets ==
    { S \in SUBSET (1 .. (NumSteps - 1)) : Cardinality(S) <= NumCheckpoints }

(*
 * Convert a set to a sorted sequence (ascending).
 *)
RECURSIVE SetToSortedSeq(_, _)
SetToSortedSeq(S, lo) ==
    IF S = {} THEN << >>
    ELSE LET smallest == CHOOSE x \in S : \A y \in S : x <= y
         IN  << smallest >> \o SetToSortedSeq(S \ {smallest}, smallest + 1)

SortedSeq(S) == SetToSortedSeq(S, 0)

(*
 * Sequence of interval lengths from boundary points.
 * boundaries is a sorted sequence: <<0, r1, r2, ..., rk, NumSteps>>
 * Returns <<r1-0, r2-r1, ..., NumSteps-rk>>
 *)
IntervalLengths(boundaries) ==
    [i \in 1..(Len(boundaries) - 1) |->
        boundaries[i + 1] - boundaries[i]]

(*
 * Largest-remainder allocation (Hamilton's method).
 *
 * Distributes `total` items across buckets proportionally to `weights`.
 * Uses integer arithmetic throughout to avoid floating-point issues.
 *
 * NOTE: Tie-breaking may differ from the Rust implementation which uses
 * f64 remainders. The spec verifies structural properties (sum, range)
 * rather than identical output in tie cases.
 *)
LargestRemainderAlloc(total, weights) ==
    LET n == Len(weights)
        weightSum == SumSeq(weights)
        \* Integer quotients (floor division)
        floors == [i \in 1..n |-> (weights[i] * total) \div weightSum]
        allocated == SumSeq(floors)
        remaining == total - allocated
        \* Fractional remainders as scaled integers for comparison.
        \* remainder_i = (weights[i] * total) mod weightSum
        \* (This avoids floating-point: we compare w_i*total mod weightSum)
        remainders == [i \in 1..n |-> (weights[i] * total) % weightSum]
        \* Assign +1 to the `remaining` indices with the largest remainders.
        \* Use CHOOSE to pick a valid assignment (any tie-breaking is fine).
        bonusSet == CHOOSE S \in SUBSET (1..n) :
                        /\ Cardinality(S) = remaining
                        /\ \A i \in S, j \in (1..n) \ S :
                            remainders[i] >= remainders[j]
    IN  [i \in 1..n |-> floors[i] + (IF i \in bonusSet THEN 1 ELSE 0)]

(*
 * Compute Revolve schedule positions for a sub-interval [start, end)
 * with the given number of checkpoint slots.
 *
 * Uses the same recursive schedule logic as Revolve.tla, but computed
 * eagerly (not as a state machine) since this is a deterministic function.
 *)
RECURSIVE RevolvePositions(_, _, _, _)
RevolvePositions(start, end, slots, acc) ==
    LET steps == end - start
    IN
    IF steps <= 1 \/ slots = 0 THEN acc
    ELSE
        LET advance == OptimalAdvance(steps, slots)
            split   == start + advance
        IN
        IF split > start /\ split < end
        THEN
            LET acc2 == acc \union {split}
                acc3 == RevolvePositions(start, split, slots - 1, acc2)
            IN  RevolvePositions(split, end, slots, acc3)
        ELSE acc

(*
 * Truncate a set of positions to at most `limit` elements.
 * Since sets are unordered in TLA+, we pick any subset of the right size.
 * In practice, the Revolve schedule produces at most `slots` positions.
 *)
TruncateSet(S, limit) ==
    IF Cardinality(S) <= limit THEN S
    ELSE CHOOSE T \in SUBSET S : Cardinality(T) = limit

---------------------------------------------------------------------------
(* Variables *)
---------------------------------------------------------------------------

VARIABLES
    required,        \* Set of required positions (nondeterministic input)
    allocation,      \* Sequence of slot counts per interval
    mergedPositions, \* Final set of all checkpoint positions
    phase            \* "init" | "allocated" | "done"

vars == <<required, allocation, mergedPositions, phase>>

---------------------------------------------------------------------------
(* Initial state: choose any valid required set *)
---------------------------------------------------------------------------

Init ==
    /\ required \in ValidRequiredSets
    /\ allocation = << >>
    /\ mergedPositions = {}
    /\ phase = "init"

---------------------------------------------------------------------------
(* Compute allocation and merged positions *)
---------------------------------------------------------------------------

(*
 * Compute the slot allocation and merged positions in one step.
 * This mirrors the deterministic computation in grad_checkpointed_with_hints.
 *)
Allocate ==
    /\ phase = "init"
    /\ LET reqSeq == SortedSeq(required)
           \* Build boundaries: <<0, r1, r2, ..., rk, NumSteps>>
           boundaries == << 0 >> \o reqSeq \o << NumSteps >>
           lengths == IntervalLengths(boundaries)
           free == NumCheckpoints - Cardinality(required)
       IN
       IF free = 0 \/ SumSeq(lengths) = 0
       THEN
           \* No free slots: just use required positions
           /\ allocation' = [i \in 1..Len(lengths) |-> 0]
           /\ mergedPositions' = required
           /\ phase' = "done"
       ELSE
           LET alloc == LargestRemainderAlloc(free, lengths)
               \* Run Revolve on each sub-interval and merge
               intervals == [i \in 1..(Len(boundaries) - 1) |->
                               <<boundaries[i], boundaries[i + 1]>>]
               revolvePositions ==
                   UNION { LET start == intervals[i][1]
                               end   == intervals[i][2]
                               subSteps == end - start
                               subSlots == alloc[i]
                               rawPos == RevolvePositions(start, end, subSlots, {})
                           IN  TruncateSet(rawPos, subSlots)
                         : i \in 1..Len(alloc) }
           IN
           /\ allocation' = alloc
           /\ mergedPositions' = required \union revolvePositions
           /\ phase' = "done"
    /\ UNCHANGED <<required>>

---------------------------------------------------------------------------
(* Next-state relation *)
---------------------------------------------------------------------------

Next == Allocate

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

---------------------------------------------------------------------------
(* Invariants *)
---------------------------------------------------------------------------

(*
 * SAFETY: All required positions are included in the final set.
 * Rust: all_positions starts as required.iter().copied().collect()
 *)
RequiredIncluded ==
    phase = "done" => required \subseteq mergedPositions

(*
 * SAFETY: Total positions do not exceed the checkpoint budget.
 * Rust: the combined set is bounded by num_checkpoints.
 *)
BudgetRespected ==
    phase = "done" => Cardinality(mergedPositions) <= NumCheckpoints

(*
 * SAFETY: Allocation sums to the number of free slots exactly.
 * Rust: largest_remainder_alloc returns values summing to `total`.
 *)
AllocationExact ==
    phase = "done" /\ allocation # << >> =>
        SumSeq(allocation) = NumCheckpoints - Cardinality(required)

(*
 * SAFETY: All positions are within valid range.
 *)
AllInRange ==
    \A p \in mergedPositions : p >= 1 /\ p <= NumSteps - 1

---------------------------------------------------------------------------
(* Temporal properties *)
---------------------------------------------------------------------------

Termination == <>(phase = "done")

==========================================================================
