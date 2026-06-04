# Release Notes

Release: `v0.3.0`
Date: `2026-06-04`

## Summary

`v0.3.0` is the first release of Project Synapse that includes **comparative research evidence** rather than only a runnable prototype.

This release adds a controlled public benchmark against a fixed-head baseline on the public digits task, using the same data split and epoch budget.

## Highlights

1. Added `scripts/benchmark_public_digits.py`.
2. Added a seeded end-to-end benchmark comparing Synapse against a fixed-head MLP baseline.
3. Documented comparative results for accuracy, latency, parameter count, and marginal task growth.

## Main Benchmark Result

From the seeded public digits benchmark:

1. Synapse resource-aware model: `98.61%` validation accuracy
2. Fixed-head baseline: `95.56%` validation accuracy
3. Synapse CPU evaluation latency: `1.39 ms/batch`
4. Fixed-head CPU evaluation latency: `0.02 ms/batch`
5. Synapse marginal task-specific parameters: `8`
6. Fixed-head marginal task-specific parameters: `2410`
7. Estimated break-even task count under this setup: `102`

## Interpretation

This benchmark sharpens the story:

1. Synapse is no longer just an architectural idea; it now has one reproducible comparison.
2. On tiny single-task workloads, Synapse still carries much higher shared overhead and slower runtime.
3. Its clearest efficiency advantage remains task scaling and context-conditioned behavior, not small-model simplicity.

## Known Limitations

1. This is still only one baseline and one public benchmark task.
2. Semantic context remains placeholder logic.
3. Adapter and LoRA baselines are still future work.
