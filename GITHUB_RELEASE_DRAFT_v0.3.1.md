# Project Synapse v0.3.1

Project Synapse is a public research prototype for **resource-conditioned hypernetwork adaptation**.

This `v0.3.1` patch release polishes the first benchmark-backed release by making the evidence easier to discover and reuse.

## Highlights

- Added a curated `benchmarks/` directory
- Added a machine-readable seeded benchmark artifact to the repo
- Tightened the README and public-facing repository copy

## What Changed

This release does not change the underlying benchmark result from `v0.3.0`.

It does make the project easier for others to:

- understand quickly,
- inspect without searching through logs,
- reuse as a starting point for follow-up baselines.

## Benchmark Snapshot

- Synapse validation accuracy: `98.61%`
- Fixed-head validation accuracy: `95.56%`
- Synapse CPU eval latency: `1.39 ms/batch`
- Fixed-head CPU eval latency: `0.02 ms/batch`
- Synapse marginal task-specific parameters: `8`
- Fixed-head marginal task-specific parameters: `2410`
