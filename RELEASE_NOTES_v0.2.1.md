# Release Notes

Release: `v0.2.1`  
Date: `2026-06-04`

## Summary

`v0.2.1` closes several of the most visible remaining caveats from the public `v0.2.0` release:

1. the resource monitor now supports real telemetry via `psutil`,
2. the adaptation script is now functional,
3. the feature-extraction helper is now a working utility rather than a placeholder.

This release keeps the stable digits quick-start intact while adding stronger resource-aware tooling around it.

## Highlights

1. Added telemetry-aware context support with deterministic fallbacks in `src/resource_monitor.py`.
2. Added a verified resource-aware digits config in `configs/public_digits_resource_exp.yaml`.
3. Implemented `scripts/test_adaptation.py` for context-sensitivity analysis.
4. Implemented `scripts/extract_imagenet_features.py` as a working feature extraction pipeline.
5. Added `psutil` to environment and pip requirements.
6. Added resource monitor unit tests.

## Verified Improvements

1. Resource-aware digits training completed successfully.
2. Adaptation analysis ran successfully on a resource-aware checkpoint.
3. Feature extraction completed successfully on a miniature ImageFolder-style verification dataset.

## Known Limitations

1. Semantic context is still placeholder logic.
2. Legacy multimodal configs remain experimental.
3. The project still lacks a formal baseline suite against adapters and LoRA.
4. Resource-aware experiments are functional, but not yet presented as a benchmark study.
