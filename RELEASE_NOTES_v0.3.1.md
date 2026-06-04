# Release Notes

Release: `v0.3.1`
Date: `2026-06-04`

## Summary

`v0.3.1` is a polish release on top of the first benchmark-backed `v0.3.0` release.

It does not change the benchmark outcome. Instead, it makes the public repository easier to understand and reuse by packaging the benchmark artifact directly inside the repo and tightening the public-facing copy.

## Highlights

1. Added a curated `benchmarks/` directory.
2. Added the seeded public digits benchmark JSON artifact to the repository.
3. Improved README discoverability for benchmark evidence and reuse.
4. Refreshed the repository description pack and pinned issue draft to match the benchmark-backed state.

## Benchmark Status

The benchmark results from `v0.3.0` remain unchanged:

1. Synapse: `98.61%` validation accuracy
2. Fixed head baseline: `95.56%` validation accuracy
3. Synapse CPU evaluation latency: `1.39 ms/batch`
4. Fixed head CPU evaluation latency: `0.02 ms/batch`

## Why This Release Matters

This release makes it easier for:

1. researchers to inspect the exact benchmark artifact,
2. advisors or reviewers to understand the point of the repository quickly,
3. new users to see what the project does without reading through logs or code first.
