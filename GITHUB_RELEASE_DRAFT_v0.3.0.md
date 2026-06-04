# Project Synapse v0.3.0

Project Synapse is a public research prototype for **resource-conditioned hypernetwork adaptation**.

This `v0.3.0` release is the first one backed by a reproducible comparative benchmark, not just architecture and tooling.

## Highlights

- Added `scripts/benchmark_public_digits.py`
- Added the first controlled Synapse vs fixed-head benchmark on the public digits task
- Published comparative metrics for accuracy, CPU latency, parameter count, and marginal task growth

## Benchmark Snapshot

Seeded end-to-end result on the public digits benchmark:

- Synapse validation accuracy: `98.61%`
- Fixed-head validation accuracy: `95.56%`
- Synapse CPU eval latency: `1.39 ms/batch`
- Fixed-head CPU eval latency: `0.02 ms/batch`
- Synapse marginal task-specific parameters: `8`
- Fixed-head marginal task-specific parameters: `2410`
- Estimated break-even task count: `102`

## Honest Interpretation

This does not mean Synapse is the better default for small single-task models.

It does mean the project now has one reproducible piece of comparative evidence showing:

- meaningful context-conditioned behavior,
- dramatically lower marginal task growth,
- significant shared-overhead tradeoffs that future baselines should test more broadly.

## What Still Remains

- adapter baseline
- LoRA baseline
- stronger multitask benchmark
- richer edge-hardware measurements
