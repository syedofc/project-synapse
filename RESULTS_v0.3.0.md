# Results Summary

Release: `v0.3.0`
Date: `2026-06-04`

## Public Digits Fixed-Head Benchmark

Configuration:

- Synapse config: `configs/public_digits_resource_exp.yaml`
- Baseline: fixed-head MLP with the same `64 -> 32 -> 10` task-head scale
- Training budget: `8` epochs
- Device: CPU

## Seeded End-to-End Result

### Synapse

- Validation accuracy: `98.61%`
- Validation loss: `0.0648`
- CPU eval latency: `1.39 ms/batch`
- Trainable parameters: `244338`
- Marginal task-specific parameters: `8`
- Shared trainable parameters: `244330`
- State dict size: `986992` bytes
- Context-response weight delta L2: `1.5401`
- Prediction agreement across two sampled contexts: `1.0`

### Fixed Head Baseline

- Validation accuracy: `95.56%`
- Validation loss: `0.1175`
- CPU eval latency: `0.02 ms/batch`
- Trainable parameters: `2410`
- Marginal task-specific parameters: `2410`
- Shared trainable parameters: `0`
- State dict size: `9640` bytes
- Context-response weight delta L2: `0.0`

## Comparison

- Accuracy delta, Synapse minus fixed head: `+3.06` percentage points
- Latency delta, Synapse minus fixed head: `+1.37 ms/batch`
- Marginal task parameter delta, Synapse minus fixed head: `-2402`
- Estimated break-even task count under this setup: `102`

## Honest Reading

This benchmark does **not** show that Synapse is a better default model for small single-task settings.

It does show:

1. Synapse can be competitive or better in accuracy on the public digits path under the tested budget.
2. Synapse remains substantially slower and heavier in shared parameters on CPU.
3. Synapse has a clear systems advantage only when you care about marginal task growth and context-conditioned parameter change.

## Repository Artifact

The exact seeded result used for this note is stored in:

- [benchmarks/public_digits_fixed_head_vs_synapse_seed42.json](benchmarks/public_digits_fixed_head_vs_synapse_seed42.json)
