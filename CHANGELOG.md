# Changelog

All notable changes to this project will be documented in this file.

The format is based on semantic versioning with project-stage labels for research releases.

## [0.2.1] - 2026-06-04

### Added

1. Real telemetry-aware context support via `psutil`.
2. Functional adaptation analysis script.
3. Functional feature extraction utility.
4. Resource-aware digits demo config in `configs/public_digits_resource_exp.yaml`.
5. `RESULTS_v0.2.1.md` and `RELEASE_NOTES_v0.2.1.md`.
6. Resource monitor unit tests.

### Changed

1. Replaced the simulated-only resource monitor with telemetry-plus-fallback behavior.
2. Promoted adaptation analysis from placeholder to working research utility.
3. Promoted feature extraction from placeholder to working utility.
4. Updated public docs to reflect the stronger `v0.2.1` state.

### Known Issues

1. Semantic context is still placeholder logic.
2. Legacy multimodal configs remain experimental.
3. The project still lacks a formal baseline suite against adapters and LoRA.

## [0.2.0] - 2026-06-04

### Added

1. Reproducible real-dataset experiment in `configs/public_digits_exp.yaml`.
2. `data/digits_loader.py` based on `sklearn.datasets.load_digits`.
3. `.gitignore`, `CITATION.cff`, and `CONTRIBUTING.md`.
4. Release notes for `v0.2.0`.
5. Verified public results summary in `RESULTS_v0.2.0.md`.

### Changed

1. Promoted the recommended public quick-start from toy-only to a real dataset path.
2. Updated `README.md` to reflect a public release candidate.
3. Kept the toy experiment as a smoke-test path rather than the primary public example.
4. Advanced the release version marker to `0.2.0`.

### Known Issues

1. Runtime resource conditioning still uses simulated context rather than real telemetry.
2. `scripts/test_adaptation.py` remains incomplete.
3. Legacy multimodal configs remain experimental.
4. Feature extraction remains a placeholder workflow.

## [0.1.0-alpha] - 2026-06-04

### Added

1. Public-facing root `README.md` for repository landing-page use.
2. Versioned research framing in `04_RESEARCH_POSITIONING_v1.0.md`.
3. Release notes for `v0.1.0-alpha`.
4. Root `VERSION` file.
5. Self-contained public toy experiment config and synthetic dataset loader.

### Changed

1. Reframed the repository as an experimental research prototype for resource-conditioned hypernetwork-based task adaptation.
2. Clarified the alpha-stage limitations, scope, and roadmap.
3. Converted `03_requirements.txt` into installable pip requirements format.
4. Added `IdentityProcessor` wiring support in the substrate component.
5. Split the former YAML stub out of `scripts/extract_imagenet_features.py` into `configs/config_for_feature_extraction.yaml`.
6. Replaced the old numbered README content with a legacy pointer to the public README.
7. Replaced the old training script with a cleaner public release entrypoint.
8. Implemented the evaluation script.
9. Updated tests to match the current config-driven architecture.
10. Hardened dashboard and logger behavior for public use.

### Known Issues

1. `scripts/test_adaptation.py` is still a skeleton script.
2. Runtime resource conditioning still uses simulated random context rather than real telemetry.
3. Some experiment configurations are incomplete or dataset-path dependent.
4. The feature extraction workflow is still a placeholder rather than a completed pipeline.
