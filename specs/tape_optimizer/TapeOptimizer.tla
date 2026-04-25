------------------------------ MODULE TapeOptimizer ------------------------------
(*
 * Formal specification of the bytecode tape optimizer (CSE + DCE)
 * used in echidna.
 *
 * Models the full optimization pipeline as a stepwise state machine:
 *   Phase 0 (Build):       Nondeterministic construction of a valid tape
 *   Phase 1 (CSE Scan):    Forward scan deduplicating identical operations
 *   Phase 2 (CSE Remap):   Apply final CSE remap to all arg references
 *   Phase 3 (DCE Mark):    Backward reachability walk from output
 *   Phase 4 (DCE Compact): Forward compaction removing unreachable entries
 *   Phase 5 (Done):        Terminal state
 *
 * Opcode abstraction: 5 kinds instead of 44 real opcodes.
 *   Input      -- structural leaf, never removed by DCE, UNUSED args
 *   Const      -- structural leaf, can be removed by DCE, UNUSED args
 *   Unary      -- one operand (arg0), arg1 = UNUSED
 *   BinComm    -- two operands, commutative (CSE normalizes order)
 *   BinNonComm -- two operands, non-commutative
 *
 * This is the minimal partition capturing every structurally distinct code
 * path in the optimizer. The optimizer is opcode-aware only for commutative
 * normalization (captured by BinComm/BinNonComm split); CSE keys on opcode
 * identity and DCE is purely structural.
 *
 * The 5-kind abstraction is a safe overapproximation: the spec's CSE is
 * MORE aggressive than the real CSE (it merges all Unary ops with the same
 * arg, even if they represent different operations like Sin vs Cos). If
 * structural invariants hold under this more aggressive CSE, they hold for
 * the real less-aggressive CSE too.
 *
 * Excluded: Powi (exponent-as-u32) and Custom (callback index, side table)
 * -- encoding tricks well-covered by Rust debug assertions and unit tests.
 * Values abstracted away (structural properties only).
 * Single-output only (multi-output deferred).
 *
 * Rust correspondence: optimize() -> cse() -> dce_compact()
 *   in src/bytecode_tape/optimize.rs
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    MaxTapeLen,   \* Maximum tape length (must be > NumInputs)
    NumInputs     \* Number of input variables (>= 1)

ASSUME MaxTapeLen >= NumInputs + 1
ASSUME NumInputs >= 1

---------------------------------------------------------------------------
(* Opcode kinds and sentinel *)
---------------------------------------------------------------------------

OpKind == {"Input", "Const", "Unary", "BinComm", "BinNonComm"}

\* Sentinel for unused argument slots. Guaranteed out of range for any
\* valid tape index (0..MaxTapeLen-1).
UNUSED == MaxTapeLen + 100

\* Fixed-size domain for function variables. Entries beyond numEntries
\* are don't-care but must exist for TLC.
FullDomain == 0 .. (MaxTapeLen - 1)

---------------------------------------------------------------------------
(* Variables *)
---------------------------------------------------------------------------

VARIABLES
    opcodes,      \* [FullDomain -> OpKind]
    args,         \* [FullDomain -> <<arg0, arg1>>]
    numEntries,   \* Current number of live tape entries
    outputIdx,    \* Primary output index (single-output)
    phase,        \* Current pipeline phase
    \* CSE working state
    remap,        \* [FullDomain -> Nat] -- CSE remap table (index -> canonical)
    seen,         \* Set of <<key, index>> -- CSE deduplication table
    scanPos,      \* Forward scan position (reused across stepwise phases)
    \* DCE working state
    reachable,    \* [FullDomain -> BOOLEAN] -- reachability flags
    dceStack,     \* Sequence of Nat -- reachability worklist
    writePos,     \* Compact write cursor
    dceRemap      \* [FullDomain -> Nat] -- compaction index remap

vars == <<opcodes, args, numEntries, outputIdx, phase,
          remap, seen, scanPos, reachable, dceStack,
          writePos, dceRemap>>

---------------------------------------------------------------------------
(* Helper operators *)
---------------------------------------------------------------------------

IsLeaf(op) == op \in {"Input", "Const"}

Min2(a, b) == IF a <= b THEN a ELSE b
Max2(a, b) == IF a >= b THEN a ELSE b

(*
 * Canonical CSE key for an operation with remapped args.
 * Commutative binary ops normalize argument order (min, max).
 *
 * Mirrors the key construction in cse() at optimize.rs:177-186.
 *)
CSEKey(op, a, b) ==
    IF b = UNUSED
    THEN <<op, a, UNUSED>>
    ELSE IF op = "BinComm"
         THEN <<op, Min2(a, b), Max2(a, b)>>
         ELSE <<op, a, b>>

(*
 * Lookup in the seen table (set of <<key, index>> pairs).
 * Returns the canonical index for the key, or UNUSED if not seen.
 *)
SeenLookup(key, seenSet) ==
    LET matches == { pair \in seenSet : pair[1] = key }
    IN IF matches = {}
       THEN UNUSED
       ELSE (CHOOSE pair \in matches : TRUE)[2]

(*
 * Valid entries at tape position i during the build phase.
 * Position < NumInputs: must be Input (structural prefix).
 * Position >= NumInputs: any non-Input op with DAG-order args.
 *)
ValidBuildEntries(i) ==
    IF i < NumInputs
    THEN { <<"Input", UNUSED, UNUSED>> }
    ELSE LET refs == 0 .. (i - 1)
         IN  { <<"Const", UNUSED, UNUSED>> }
             \union { <<"Unary", a, UNUSED>> : a \in refs }
             \union { <<"BinComm", a, b>> : a \in refs, b \in refs }
             \union { <<"BinNonComm", a, b>> : a \in refs, b \in refs }

---------------------------------------------------------------------------
(* Phase 0: Build -- nondeterministic tape construction *)
---------------------------------------------------------------------------
(*
 * The build phase constructs a valid tape entry by entry, with
 * nondeterministic choice of opcode and args at each position.
 * This avoids pre-computing the (potentially huge) set of all valid
 * tapes; TLC explores each branch naturally.
 *
 * Edge cases covered by nondeterminism:
 *   - Empty body (only inputs, no operations): BuildDone fires immediately
 *   - All-dead tape (output references only an input): output chosen freely
 *   - Self-referencing commutative: BinComm(i,i) is in ValidBuildEntries
 *)

Init ==
    /\ numEntries = NumInputs
    /\ opcodes = [i \in FullDomain |->
                    IF i < NumInputs THEN "Input" ELSE "Const"]
    /\ args = [i \in FullDomain |-> <<UNUSED, UNUSED>>]
    /\ outputIdx = 0
    /\ phase = "build"
    /\ remap = [i \in FullDomain |-> i]
    /\ seen = {}
    /\ scanPos = 0
    /\ reachable = [i \in FullDomain |-> FALSE]
    /\ dceStack = << >>
    /\ writePos = 0
    /\ dceRemap = [i \in FullDomain |-> 0]

\* Add one entry to the tape.
BuildStep ==
    /\ phase = "build"
    /\ numEntries < MaxTapeLen
    /\ \E entry \in ValidBuildEntries(numEntries) :
        /\ opcodes' = [opcodes EXCEPT ![numEntries] = entry[1]]
        /\ args' = [args EXCEPT ![numEntries] = <<entry[2], entry[3]>>]
        /\ numEntries' = numEntries + 1
    /\ UNCHANGED <<outputIdx, phase, remap, seen, scanPos,
                   reachable, dceStack, writePos, dceRemap>>

\* Finish building: nondeterministically choose an output index and
\* begin the optimization pipeline.
BuildDone ==
    /\ phase = "build"
    /\ \E out \in 0 .. (numEntries - 1) :
        /\ outputIdx' = out
        /\ phase' = "cse_scan"
        /\ scanPos' = 0
    /\ UNCHANGED <<opcodes, args, numEntries, remap, seen,
                   reachable, dceStack, writePos, dceRemap>>

---------------------------------------------------------------------------
(* Phase 1: CSE Scan -- one entry per step *)
---------------------------------------------------------------------------
(*
 * Forward scan building the CSE remap table. For each non-leaf entry:
 *   1. Remap args through the current remap table
 *   2. Build a canonical key (normalizing commutative arg order)
 *   3. If key seen before: redirect to the canonical entry
 *   4. If key new: record this entry as canonical
 *
 * Mirrors cse() lines 154-192 of optimize.rs.
 *)

CSEScanStep ==
    /\ phase = "cse_scan"
    /\ scanPos < numEntries
    /\ LET op == opcodes[scanPos]
       IN
       IF IsLeaf(op)
       THEN
           \* Skip Input/Const
           /\ scanPos' = scanPos + 1
           /\ UNCHANGED <<opcodes, args, numEntries, outputIdx, phase,
                          remap, seen, reachable, dceStack, writePos, dceRemap>>
       ELSE
           LET a0 == args[scanPos][1]
               b0 == args[scanPos][2]
               \* Apply current remap to args
               a == remap[a0]
               b == IF b0 # UNUSED THEN remap[b0] ELSE UNUSED
               \* Canonical key
               key == CSEKey(op, a, b)
               existing == SeenLookup(key, seen)
           IN
           /\ args' = [args EXCEPT ![scanPos] = <<a, b>>]
           /\ IF existing # UNUSED
              THEN \* Duplicate found: redirect to canonical
                   /\ remap' = [remap EXCEPT ![scanPos] = existing]
                   /\ seen' = seen
              ELSE \* First occurrence: record as canonical
                   /\ remap' = remap
                   /\ seen' = seen \union { <<key, scanPos>> }
           /\ scanPos' = scanPos + 1
           /\ UNCHANGED <<opcodes, numEntries, outputIdx, phase,
                          reachable, dceStack, writePos, dceRemap>>

CSEScanDone ==
    /\ phase = "cse_scan"
    /\ scanPos = numEntries
    /\ phase' = "cse_remap"
    /\ scanPos' = 0
    /\ UNCHANGED <<opcodes, args, numEntries, outputIdx,
                   remap, seen, reachable, dceStack, writePos, dceRemap>>

---------------------------------------------------------------------------
(* Phase 2: CSE Remap -- apply final remap to all args *)
---------------------------------------------------------------------------
(*
 * Forward pass applying the complete remap table to every entry's
 * arg_indices. In practice this pass is idempotent on args (the scan
 * already applied the remap), but it also remaps the output index
 * when the scan completes. Modelled for structural correspondence
 * with the Rust code.
 *
 * Mirrors cse() lines 194-226 of optimize.rs.
 *)

CSERemapStep ==
    /\ phase = "cse_remap"
    /\ scanPos < numEntries
    /\ LET op == opcodes[scanPos]
       IN
       IF IsLeaf(op)
       THEN
           /\ scanPos' = scanPos + 1
           /\ UNCHANGED <<opcodes, args, numEntries, outputIdx, phase,
                          remap, seen, reachable, dceStack, writePos, dceRemap>>
       ELSE
           LET a == args[scanPos][1]
               b == args[scanPos][2]
               ra == IF a # UNUSED THEN remap[a] ELSE UNUSED
               rb == IF b # UNUSED THEN remap[b] ELSE UNUSED
           IN
           /\ args' = [args EXCEPT ![scanPos] = <<ra, rb>>]
           /\ scanPos' = scanPos + 1
           /\ UNCHANGED <<opcodes, numEntries, outputIdx, phase,
                          remap, seen, reachable, dceStack, writePos, dceRemap>>

CSERemapDone ==
    /\ phase = "cse_remap"
    /\ scanPos = numEntries
    /\ LET newOutput == remap[outputIdx]
       IN
       /\ outputIdx' = newOutput
       /\ phase' = "dce_mark"
       \* Initialize DCE: all inputs pre-marked reachable, output seeded.
       \* Rust: reachable[..num_inputs] = true; stack.push(output_index);
       /\ reachable' = [i \in FullDomain |->
                            IF i < NumInputs THEN TRUE ELSE FALSE]
       /\ dceStack' = << newOutput >>
       /\ scanPos' = 0
    /\ UNCHANGED <<opcodes, args, numEntries,
                   remap, seen, writePos, dceRemap>>

---------------------------------------------------------------------------
(* Phase 3: DCE Mark -- worklist-based reachability *)
---------------------------------------------------------------------------
(*
 * Pop an index from the stack, mark it reachable, push its unreached
 * operands. Inputs are pre-marked reachable. Terminates when the
 * stack is empty.
 *
 * Mirrors dce_compact() lines 17-46 of optimize.rs.
 *)

DCEMarkStep ==
    /\ phase = "dce_mark"
    /\ dceStack # << >>
    /\ LET idx == Head(dceStack)
           rest == Tail(dceStack)
       IN
       IF reachable[idx]
       THEN
           \* Already reachable: just pop
           /\ dceStack' = rest
           /\ UNCHANGED <<opcodes, args, numEntries, outputIdx, phase,
                          remap, seen, scanPos, reachable, writePos, dceRemap>>
       ELSE
           LET a == args[idx][1]
               b == args[idx][2]
               \* Push unreached operands onto the stack
               pushA == IF a # UNUSED /\ ~reachable[a] THEN <<a>> ELSE << >>
               pushB == IF b # UNUSED /\ ~reachable[b] THEN <<b>> ELSE << >>
           IN
           /\ reachable' = [reachable EXCEPT ![idx] = TRUE]
           /\ dceStack' = pushA \o pushB \o rest
           /\ UNCHANGED <<opcodes, args, numEntries, outputIdx, phase,
                          remap, seen, scanPos, writePos, dceRemap>>

DCEMarkDone ==
    /\ phase = "dce_mark"
    /\ dceStack = << >>
    /\ phase' = "dce_compact"
    /\ scanPos' = 0
    /\ writePos' = 0
    /\ dceRemap' = [i \in FullDomain |-> 0]
    /\ UNCHANGED <<opcodes, args, numEntries, outputIdx,
                   remap, seen, reachable, dceStack>>

---------------------------------------------------------------------------
(* Phase 4: DCE Compact -- forward compaction *)
---------------------------------------------------------------------------
(*
 * Forward pass through the tape. For each reachable entry:
 *   - Record its new index in dceRemap
 *   - Copy it to the write position with remapped arg references
 * Unreachable entries are skipped.
 *
 * DAG order guarantees that referenced entries are always before the
 * current entry, so their dceRemap values are already computed when
 * needed.
 *
 * Mirrors dce_compact() lines 48-99 of optimize.rs.
 *)

DCECompactStep ==
    /\ phase = "dce_compact"
    /\ scanPos < numEntries
    /\ IF reachable[scanPos]
       THEN
           LET op == opcodes[scanPos]
               a == args[scanPos][1]
               b == args[scanPos][2]
               \* Remap arg references through compaction remap
               ra == IF a # UNUSED THEN dceRemap[a] ELSE UNUSED
               rb == IF b # UNUSED THEN dceRemap[b] ELSE UNUSED
           IN
           /\ dceRemap' = [dceRemap EXCEPT ![scanPos] = writePos]
           /\ opcodes' = [opcodes EXCEPT ![writePos] = op]
           /\ args' = [args EXCEPT ![writePos] = <<ra, rb>>]
           /\ writePos' = writePos + 1
       ELSE
           /\ UNCHANGED <<opcodes, args, writePos, dceRemap>>
    /\ scanPos' = scanPos + 1
    /\ UNCHANGED <<numEntries, outputIdx, phase,
                   remap, seen, reachable, dceStack>>

DCECompactDone ==
    /\ phase = "dce_compact"
    /\ scanPos = numEntries
    /\ numEntries' = writePos
    /\ outputIdx' = dceRemap[outputIdx]
    /\ phase' = "done"
    /\ UNCHANGED <<opcodes, args, remap, seen, scanPos,
                   reachable, dceStack, writePos, dceRemap>>

---------------------------------------------------------------------------
(* Next-state relation *)
---------------------------------------------------------------------------

Next ==
    \/ BuildStep
    \/ BuildDone
    \/ CSEScanStep
    \/ CSEScanDone
    \/ CSERemapStep
    \/ CSERemapDone
    \/ DCEMarkStep
    \/ DCEMarkDone
    \/ DCECompactStep
    \/ DCECompactDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

---------------------------------------------------------------------------
(* Invariants *)
---------------------------------------------------------------------------

(* Phases where the tape is fully consistent (not mid-compaction). *)
ConsistentPhase ==
    phase \in {"build", "cse_scan", "cse_remap", "dce_mark", "done"}

(*
 * Input prefix: first NumInputs entries are Input with UNUSED args.
 * Inputs are never modified or relocated.
 *
 * Rust: inputs are pushed first, never touched by CSE/DCE.
 *)
InputPrefixInvariant ==
    ConsistentPhase =>
        \A i \in 0 .. (NumInputs - 1) :
            /\ opcodes[i] = "Input"
            /\ args[i] = <<UNUSED, UNUSED>>

(*
 * DAG order: non-leaf entry i has arg references strictly before i.
 *
 * Rust: enforced by recording order (push_op) and preserved by optimize.
 *)
DAGOrderInvariant ==
    ConsistentPhase =>
        \A i \in 0 .. (numEntries - 1) :
            ~IsLeaf(opcodes[i]) =>
                /\ args[i][1] < i
                /\ (args[i][2] # UNUSED => args[i][2] < i)

(*
 * Valid refs: all arg references are within tape bounds.
 *)
ValidRefsInvariant ==
    ConsistentPhase =>
        \A i \in 0 .. (numEntries - 1) :
            ~IsLeaf(opcodes[i]) =>
                /\ args[i][1] < numEntries
                /\ (args[i][2] # UNUSED => args[i][2] < numEntries)

(*
 * Output index is always within tape bounds.
 *)
OutputValidInvariant ==
    phase # "build" => outputIdx < numEntries

(*
 * Input count is preserved across all optimization phases.
 *
 * Rust: inputs are always reachable (pre-marked in DCE) and never
 * created or destroyed by CSE.
 *)
InputsPreserved ==
    ConsistentPhase =>
        Cardinality({i \in 0 .. (numEntries - 1) :
                        opcodes[i] = "Input"}) = NumInputs

(*
 * CSE remap is monotone: entries only redirect to earlier/equal indices.
 * remap[i] <= i for all active entries.
 *
 * This holds because CSE only deduplicates against earlier entries.
 *)
CSERemapMonotone ==
    phase # "build" =>
        \A i \in 0 .. (numEntries - 1) : remap[i] <= i

(*
 * CSE remap is idempotent: canonical indices are fixed points.
 * remap[remap[i]] = remap[i] for all active entries.
 *
 * This ensures the remap chain has depth 1 (no transitive chains).
 * Checked in all post-build phases because remap is never modified
 * after CSE completes.
 *)
CSERemapIdempotent ==
    phase # "build" =>
        \A i \in 0 .. (numEntries - 1) : remap[remap[i]] = remap[i]

(*
 * DCE always marks all inputs as reachable (pre-marked at init).
 *
 * Rust: reachable[..num_inputs] = true in dce_compact().
 *)
DCEInputsReachable ==
    phase \in {"dce_mark", "dce_compact"} =>
        \A i \in 0 .. (NumInputs - 1) : reachable[i]

(*
 * DCE always marks the output as reachable.
 * Checked during compact (before outputIdx is remapped to new indices).
 * At "done", outputIdx has been remapped via dceRemap, so reachable[]
 * (which uses old indices) no longer corresponds.
 *)
DCEOutputReachable ==
    phase = "dce_compact" => reachable[outputIdx]

(*
 * DCE compact write cursor never exceeds the read cursor.
 * writePos <= scanPos because we can't write more entries than we've read.
 *)
DCECompactProgress ==
    phase = "dce_compact" => writePos <= scanPos

(*
 * Comprehensive post-optimization validity check.
 * Verifies all structural properties on the compacted tape.
 *
 * Maps to debug assertions at optimize.rs:235-299.
 *)
PostOptValid ==
    phase = "done" =>
        \* Input prefix preserved
        /\ \A i \in 0 .. (NumInputs - 1) :
               /\ opcodes[i] = "Input"
               /\ args[i] = <<UNUSED, UNUSED>>
        \* Leaf args are UNUSED
        /\ \A i \in 0 .. (numEntries - 1) :
               IsLeaf(opcodes[i]) =>
                   args[i] = <<UNUSED, UNUSED>>
        \* DAG order
        /\ \A i \in 0 .. (numEntries - 1) :
               ~IsLeaf(opcodes[i]) =>
                   /\ args[i][1] < i
                   /\ (args[i][2] # UNUSED => args[i][2] < i)
        \* Valid refs
        /\ \A i \in 0 .. (numEntries - 1) :
               ~IsLeaf(opcodes[i]) =>
                   /\ args[i][1] < numEntries
                   /\ (args[i][2] # UNUSED => args[i][2] < numEntries)
        \* Output index valid
        /\ outputIdx < numEntries
        \* Input count preserved
        /\ Cardinality({i \in 0 .. (numEntries - 1) :
                            opcodes[i] = "Input"}) = NumInputs
        \* No CSE duplicates among non-leaf entries.
        \* NOTE: This property is specific to the 5-kind abstraction.
        \* The real code (44 opcodes) can have entries like Sin(x) and
        \* Cos(x) that share a key under the abstraction but are distinct
        \* operations. If the opcode set is refined, this check needs
        \* adjustment.
        /\ \A i, j \in 0 .. (numEntries - 1) :
               /\ i # j
               /\ ~IsLeaf(opcodes[i])
               /\ ~IsLeaf(opcodes[j])
               => CSEKey(opcodes[i], args[i][1], args[i][2])
                  # CSEKey(opcodes[j], args[j][1], args[j][2])

---------------------------------------------------------------------------
(* Temporal properties *)
---------------------------------------------------------------------------

(*
 * LIVENESS: The optimization pipeline always terminates.
 * Each phase makes bounded progress (scanPos/writePos advance,
 * stack drains), so the system always reaches "done".
 *)
Termination == <>(phase = "done")

==========================================================================
