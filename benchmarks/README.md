# Benchmarks

This directory stores small, curated benchmark artifacts that are meant to be easy to inspect and cite from the public repository.

## Current Artifact

### `public_digits_fixed_head_vs_synapse_seed42.json`

This file records the seeded end-to-end result of the first controlled public benchmark in Project Synapse:

- task: public digits classification
- Synapse config: `configs/public_digits_resource_exp.yaml`
- baseline: fixed-head MLP
- training budget: `8` epochs
- device: CPU

## Reproduce

```bash
python scripts/benchmark_public_digits.py --config configs/public_digits_resource_exp.yaml
```

The benchmark script writes a fresh JSON summary to `training_logs/public_digits_benchmark.json`.

## How To Read It

- `synapse`: metrics for the resource-conditioned Synapse model
- `fixed_head`: metrics for the static baseline
- `comparison`: deltas and the simple break-even task-count estimate
- `interpretation`: a short plain-language summary of the tradeoff
