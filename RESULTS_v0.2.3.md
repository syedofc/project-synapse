# Results Summary

Release: `v0.2.3`
Date: `2026-06-04`

## Benchmark Note

`v0.2.3` does not introduce new benchmark claims. It is a runtime-correctness release.

The main public metrics remain:

1. Public digits quick-start evaluation accuracy: `96.94%`
2. Resource-aware digits demo best validation accuracy: `96.11%`

## Additional Verification

This release additionally verified:

1. `11/11` unit tests passed, including new tests for per-sample head execution, explicit multimodal routing, cache-key ordering, CPU-safe ledger storage, and deterministic semantic placeholder behavior.
2. The public digits evaluation path still reproduces `96.94%` accuracy from the existing checkpoint.
3. The resource-aware adaptation script still reports nonzero generated-weight drift while preserving prediction agreement on the sampled batches.
4. The public toy training path still runs end-to-end successfully after the routing changes.
