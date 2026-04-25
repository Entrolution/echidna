#!/usr/bin/env bash
# Verify that every TLA+ invariant listed in specs/README.md has a matching
# `// SPEC: <InvariantName>` anchor in the declared source file.
#
# Parses the Invariant Cross-Reference tables, extracts (invariant, file) pairs,
# and greps for `SPEC: <invariant>` in the declared file.
#
# Exits non-zero if any anchor is missing. Run from the repository root.

set -euo pipefail

README="specs/README.md"

if [[ ! -f "$README" ]]; then
  echo "error: $README not found — run from the repository root." >&2
  exit 2
fi

# Extract rows of the form:
#   | `InvariantName` | `src/path/to/file.rs` | description |
# from the cross-reference tables. The first column is the invariant, the
# second is the source file carrying the anchor.
rows=$(awk '
  /^### .* \(anchors in / { in_table = 1; next }
  /^## / { in_table = 0 }
  in_table && /^\| *`[A-Za-z_][A-Za-z0-9_]*` *\| *`src\// {
    line = $0
    # Invariant is the first backticked token.
    match(line, /`[A-Za-z_][A-Za-z0-9_]*`/)
    inv = substr(line, RSTART + 1, RLENGTH - 2)
    # Source path: first backticked token starting with src/.
    rest = substr(line, RSTART + RLENGTH)
    match(rest, /`src\/[^`]+`/)
    path = substr(rest, RSTART + 1, RLENGTH - 2)
    print inv "\t" path
  }
' "$README")

if [[ -z "$rows" ]]; then
  echo "error: no invariant rows parsed from $README" >&2
  echo "hint: table rows must match | \`Invariant\` | \`src/...\` | ... |" >&2
  exit 2
fi

missing=0
checked=0

while IFS=$'\t' read -r invariant path; do
  [[ -z "$invariant" ]] && continue
  checked=$((checked + 1))
  if [[ ! -f "$path" ]]; then
    echo "MISSING: $invariant — declared source file $path does not exist" >&2
    missing=$((missing + 1))
    continue
  fi
  # Match `// SPEC: <Invariant>` followed by end-of-line, space, or non-alphanumeric —
  # prevents `BudgetInvariant` from matching an anchor named `Budget`.
  if ! grep -Eq "// SPEC: ${invariant}([^A-Za-z0-9_]|$)" "$path"; then
    echo "MISSING: $invariant — no \`// SPEC: ${invariant}\` anchor in $path" >&2
    missing=$((missing + 1))
  fi
done <<< "$rows"

if [[ $missing -gt 0 ]]; then
  echo ""
  echo "anchor verification failed: $missing missing of $checked declared"
  echo "fix: add \`// SPEC: <InvariantName>\` next to the relevant code,"
  echo "     or update specs/README.md if the invariant moved to a different file."
  exit 1
fi

echo "anchor verification OK: $checked invariants mapped and found."
