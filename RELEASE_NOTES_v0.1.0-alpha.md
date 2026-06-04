# Release Notes

Release: `v0.1.0-alpha`  
Date: `2026-06-04`

## Summary

`v0.1.0-alpha` is the first public-facing documentation release for Project Synapse. It packages the repository as an **experimental research prototype** focused on resource-conditioned hypernetwork-based task adaptation.

This release is meant to make the project easier to share with advisors, collaborators, and early open-source users without overstating maturity.

## Highlights

1. Added a professional public `README.md`.
2. Added a versioned research-positioning brief.
3. Added a changelog and explicit release notes.
4. Added a root `VERSION` file with the current release marker.
5. Fixed `03_requirements.txt` so it can be used directly with `pip install -r`.
6. Added substrate support for `IdentityProcessor` to better match the feature-based experiment path.
7. Moved the former YAML stub out of `scripts/extract_imagenet_features.py` into `configs/config_for_feature_extraction.yaml` and replaced the script with a valid placeholder entrypoint.
8. Added a self-contained public toy experiment and a working evaluation script.

## Scope Of This Release

What this release is:

1. A structured research prototype
2. A basis for benchmarking and refinement
3. A credible alpha-stage open-source artifact

What this release is not:

1. A production-ready framework
2. A fully reproducible benchmark suite
3. A complete edge-deployment system

## Main Research Direction

The project is now explicitly framed around this question:

> Can a resource-conditioned hypernetwork generate lightweight task heads or adapters that improve task switching, memory use, or on-device adaptation latency over fixed heads, adapters, or LoRA baselines?

## Known Limitations

1. Device context is still simulated.
2. Semantic context is still placeholder logic.
3. The adaptation entrypoint is still incomplete.
4. Some prototype configs remain exploratory.
5. The feature extraction helper is still only a placeholder.

## Recommended Next Release Targets

### `v0.2.0`

1. One reproducible end-to-end training path
2. One working baseline
3. Passing smoke tests
4. Clean setup instructions validated on a fresh environment

### `v0.3.0`

1. Real resource telemetry
2. Resource-aware ablations
3. Warm-cache versus cold-cache analysis

### `v0.4.0`

1. Fixed-head baseline
2. Adapter baseline
3. LoRA baseline
4. Hypernetwork ablations without context and without ledger

## Suggested Release Label

Recommended public label:

`Project Synapse v0.1.0-alpha: experimental research prototype for resource-conditioned hypernetwork-based task adaptation`
