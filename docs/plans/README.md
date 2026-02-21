# echidna Implementation Plans

This directory contains detailed implementation plans for building the echidna library incrementally. Each plan corresponds to a phase from the [book breakdown](../book-breakdown.md).

## Plan Index

| # | Plan | Chapters | Status |
|---|------|----------|--------|
| 1 | Core Foundation | 2–3 | Not started |
| 2 | Production Reverse Mode | 4, 7, 12 | Not started |
| 3 | Second-Order Derivatives | 5, 8 | Not started |
| 4 | Sparse Computation | 9, 11 | Not started |
| 5 | Advanced Jacobian Accumulation | 10 | Not started |
| 6 | Higher-Order Derivatives | 13 | Not started |
| 7 | Nonsmooth Extensions | 14 | Not started |
| 8 | Implicit and Iterative Differentiation | 15 | Not started |

## Ordering

Plans should be executed roughly in order — each phase builds on the previous. However, phases 4 and 5 are somewhat independent and could be reordered. Phases 6–8 are also relatively independent of each other (but all depend on phases 1–3).
