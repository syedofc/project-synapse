# Project Synapse v0.2.3

Project Synapse is a public research prototype for **resource-conditioned hypernetwork adaptation**.

This `v0.2.3` release focuses on **runtime correctness and research integrity**. It does not introduce a new benchmark claim, but it makes the public artifact more faithful to the adaptive behavior it is meant to study.

## Highlights

- Correct per-sample execution when the weaver generates distinct head weights across a batch
- Explicit `input_key_map` support for processor-to-input routing
- Cache keys that preserve processor order
- CPU-safe ledger storage for cached weights
- Deterministic semantic placeholder behavior
- Stronger targeted unit tests for the reviewed failure modes

## Why This Matters

The earlier releases were already runnable and useful, but `v0.2.3` closes several gaps between the repository's research framing and its actual runtime semantics.

That makes this release more suitable for:

- advisor review,
- open-source research reuse,
- thesis or lab artifact citation,
- future baseline benchmarking work.

## Verification

- `11/11` unit tests passed
- public digits evaluation path still reproduces `96.94%` accuracy
- resource-aware adaptation analysis still runs successfully
- public toy training path still runs end-to-end

## Scope

- No new benchmark superiority claim
- No new baseline suite yet
- Stronger runtime semantics and cleaner public experiment naming
