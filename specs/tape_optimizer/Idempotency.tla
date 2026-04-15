------------------------------ MODULE Idempotency ------------------------------
(*
 * Formal verification that the tape optimizer is idempotent:
 *   optimize(optimize(tape)) = optimize(tape)
 * for all structurally valid tapes.
 *
 * CSE and DCE are defined as pure functional TLA+ operators (no variables).
 * The state machine just builds a nondeterministic valid tape, then the
 * invariant checks idempotency by computing both sides and comparing.
 *
 * The functional operators mirror the stepwise state machine in
 * TapeOptimizer.tla but compute the result in one shot, enabling
 * direct equality comparison.
 *
 * Rust correspondence: optimize() in src/bytecode_tape/optimize.rs
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    MaxTapeLen,
    NumInputs

ASSUME MaxTapeLen >= NumInputs + 1
ASSUME NumInputs >= 1

---------------------------------------------------------------------------
(* Shared definitions (same as TapeOptimizer.tla) *)
---------------------------------------------------------------------------

OpKind == {"Input", "Const", "Unary", "BinComm", "BinNonComm"}
UNUSED == MaxTapeLen + 100
FullDomain == 0 .. (MaxTapeLen - 1)

IsLeaf(op) == op \in {"Input", "Const"}
Min2(a, b) == IF a <= b THEN a ELSE b
Max2(a, b) == IF a >= b THEN a ELSE b

CSEKey(op, a, b) ==
    IF b = UNUSED THEN <<op, a, UNUSED>>
    ELSE IF op = "BinComm" THEN <<op, Min2(a, b), Max2(a, b)>>
    ELSE <<op, a, b>>

SeenLookup(key, seenSet) ==
    LET matches == { pair \in seenSet : pair[1] = key }
    IN IF matches = {} THEN UNUSED
       ELSE (CHOOSE pair \in matches : TRUE)[2]

ValidBuildEntries(i) ==
    IF i < NumInputs
    THEN { <<"Input", UNUSED, UNUSED>> }
    ELSE LET refs == 0 .. (i - 1)
         IN  { <<"Const", UNUSED, UNUSED>> }
             \union { <<"Unary", a, UNUSED>> : a \in refs }
             \union { <<"BinComm", a, b>> : a \in refs, b \in refs }
             \union { <<"BinNonComm", a, b>> : a \in refs, b \in refs }

---------------------------------------------------------------------------
(* Functional CSE *)
---------------------------------------------------------------------------
(*
 * Forward scan building the CSE remap table, then remap output index.
 * Returns <<opcodes, newArgs, numEntries, newOutputIdx>>.
 *
 * The remap pass on args is skipped (it is idempotent -- proven by
 * CSERemapIdempotent in TapeOptimizer.tla). The output index IS
 * remapped since it is not touched during the scan.
 *)

RECURSIVE FnCSEScan(_, _, _, _, _, _)
FnCSEScan(ops, curArgs, n, pos, curRemap, curSeen) ==
    IF pos = n
    THEN <<curRemap, curArgs>>
    ELSE IF IsLeaf(ops[pos])
    THEN FnCSEScan(ops, curArgs, n, pos + 1, curRemap, curSeen)
    ELSE
        LET a0 == curArgs[pos][1]
            b0 == curArgs[pos][2]
            a == curRemap[a0]
            b == IF b0 # UNUSED THEN curRemap[b0] ELSE UNUSED
            key == CSEKey(ops[pos], a, b)
            existing == SeenLookup(key, curSeen)
            newArgs == [curArgs EXCEPT ![pos] = <<a, b>>]
        IN IF existing # UNUSED
           THEN FnCSEScan(ops, newArgs, n, pos + 1,
                          [curRemap EXCEPT ![pos] = existing], curSeen)
           ELSE FnCSEScan(ops, newArgs, n, pos + 1,
                          curRemap, curSeen \union {<<key, pos>>})

FnCSE(ops, as, n, out) ==
    LET initRemap == [i \in FullDomain |-> i]
        result == FnCSEScan(ops, as, n, 0, initRemap, {})
        rm == result[1]
        newArgs == result[2]
    IN <<ops, newArgs, n, rm[out]>>

---------------------------------------------------------------------------
(* Functional DCE *)
---------------------------------------------------------------------------
(*
 * Three sub-steps matching dce_compact() in optimize.rs:
 *   1. Mark reachability (worklist DFS from output, inputs pre-marked)
 *   2. Build compaction remap (old index -> new index)
 *   3. Compact (copy reachable entries with remapped args)
 *
 * Returns <<newOps, newArgs, newNumEntries, newOutputIdx>>.
 *)

RECURSIVE FnDCEMark(_, _, _)
FnDCEMark(as, stack, reach) ==
    IF stack = << >>
    THEN reach
    ELSE
        LET idx == Head(stack)
            rest == Tail(stack)
        IN IF reach[idx]
           THEN FnDCEMark(as, rest, reach)
           ELSE
               LET a == as[idx][1]
                   b == as[idx][2]
                   pA == IF a # UNUSED /\ ~reach[a] THEN <<a>> ELSE << >>
                   pB == IF b # UNUSED /\ ~reach[b] THEN <<b>> ELSE << >>
               IN FnDCEMark(as, pA \o pB \o rest,
                             [reach EXCEPT ![idx] = TRUE])

RECURSIVE FnBuildDCERemap(_, _, _, _, _)
FnBuildDCERemap(reach, n, pos, nextIdx, rm) ==
    IF pos = n THEN <<rm, nextIdx>>
    ELSE IF ~reach[pos]
    THEN FnBuildDCERemap(reach, n, pos + 1, nextIdx, rm)
    ELSE FnBuildDCERemap(reach, n, pos + 1, nextIdx + 1,
                          [rm EXCEPT ![pos] = nextIdx])

RECURSIVE FnDCECompact(_, _, _, _, _, _, _, _)
FnDCECompact(ops, as, reach, drm, n, pos, newOps, newArgs) ==
    IF pos = n THEN <<newOps, newArgs>>
    ELSE IF ~reach[pos]
    THEN FnDCECompact(ops, as, reach, drm, n, pos + 1, newOps, newArgs)
    ELSE
        LET wp == drm[pos]
            a == as[pos][1]
            b == as[pos][2]
            ra == IF a # UNUSED THEN drm[a] ELSE UNUSED
            rb == IF b # UNUSED THEN drm[b] ELSE UNUSED
        IN FnDCECompact(ops, as, reach, drm, n, pos + 1,
                         [newOps EXCEPT ![wp] = ops[pos]],
                         [newArgs EXCEPT ![wp] = <<ra, rb>>])

FnDCE(ops, as, n, out) ==
    LET initReach == [i \in FullDomain |-> i < NumInputs]
        reach == FnDCEMark(as, <<out>>, initReach)
        remapResult == FnBuildDCERemap(reach, n, 0, 0,
                           [i \in FullDomain |-> 0])
        drm == remapResult[1]
        newN == remapResult[2]
        compactResult == FnDCECompact(ops, as, reach, drm, n, 0,
                             [i \in FullDomain |-> "Const"],
                             [i \in FullDomain |-> <<UNUSED, UNUSED>>])
    IN <<compactResult[1], compactResult[2], newN, drm[out]>>

---------------------------------------------------------------------------
(* Functional Optimize (CSE then DCE) *)
---------------------------------------------------------------------------

FnOptimize(ops, as, n, out) ==
    LET cse == FnCSE(ops, as, n, out)
    IN FnDCE(cse[1], cse[2], cse[3], cse[4])

---------------------------------------------------------------------------
(* Variables -- build phase only *)
---------------------------------------------------------------------------

VARIABLES opcodes, args, numEntries, outputIdx, phase

vars == <<opcodes, args, numEntries, outputIdx, phase>>

---------------------------------------------------------------------------
(* Build phase (same nondeterministic construction as TapeOptimizer) *)
---------------------------------------------------------------------------

Init ==
    /\ numEntries = NumInputs
    /\ opcodes = [i \in FullDomain |->
                    IF i < NumInputs THEN "Input" ELSE "Const"]
    /\ args = [i \in FullDomain |-> <<UNUSED, UNUSED>>]
    /\ outputIdx = 0
    /\ phase = "build"

BuildStep ==
    /\ phase = "build"
    /\ numEntries < MaxTapeLen
    /\ \E entry \in ValidBuildEntries(numEntries) :
        /\ opcodes' = [opcodes EXCEPT ![numEntries] = entry[1]]
        /\ args' = [args EXCEPT ![numEntries] = <<entry[2], entry[3]>>]
        /\ numEntries' = numEntries + 1
    /\ UNCHANGED <<outputIdx, phase>>

BuildDone ==
    /\ phase = "build"
    /\ \E out \in 0 .. (numEntries - 1) :
        /\ outputIdx' = out
        /\ phase' = "done"
    /\ UNCHANGED <<opcodes, args, numEntries>>

Next == BuildStep \/ BuildDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

---------------------------------------------------------------------------
(* Invariants *)
---------------------------------------------------------------------------

(*
 * IDEMPOTENCY: Optimizing an already-optimized tape produces the same tape.
 *
 * This is the core property: the optimizer is a fixed-point transformation.
 * If this fails, repeated optimization would keep changing the tape,
 * indicating the optimizer missed something on the first pass or
 * introduced new optimization opportunities.
 *)
IdempotencyProperty ==
    phase = "done" =>
        LET tape1 == FnOptimize(opcodes, args, numEntries, outputIdx)
            tape2 == FnOptimize(tape1[1], tape1[2], tape1[3], tape1[4])
        IN tape1 = tape2

---------------------------------------------------------------------------
(* Temporal properties *)
---------------------------------------------------------------------------

Termination == <>(phase = "done")

==========================================================================
