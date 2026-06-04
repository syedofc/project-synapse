# Release Notes

Release: `v0.2.0`  
Date: `2026-06-04`

## Summary

`v0.2.0` promotes Project Synapse from an architecture-focused alpha into a **public-ready reproducible research artifact** with a verified single-task experiment path.

The key change in this release is the addition of a real-dataset quick-start that does not depend on private data, along with stronger top-level release metadata and a cleaner contributor-facing structure.

## Highlights

1. Added a reproducible `scikit-learn` digits experiment in `configs/public_digits_exp.yaml`.
2. Added a digits data loader with no private dataset dependency.
3. Verified training and evaluation entrypoints for both toy and digits public configs.
4. Added `CITATION.cff`, `.gitignore`, and `CONTRIBUTING.md`.
5. Updated the README to reflect a public release candidate rather than an alpha-only state.
6. Added a small verified-results artifact for the public digits path.

## Verified Public Paths

The following paths are intended for public use in this release:

1. `configs/public_toy_exp.yaml`
   Fully self-contained smoke test
2. `configs/public_digits_exp.yaml`
   Recommended public quick-start on a real dataset
3. `scripts/train.py`
   Verified training entrypoint
4. `scripts/eval.py`
   Verified evaluation entrypoint

## Known Limitations

1. Runtime resource context is still simulated.
2. Semantic context is still placeholder logic.
3. `scripts/test_adaptation.py` is still incomplete.
4. Legacy multimodal configs remain experimental.
5. Feature extraction remains a placeholder workflow.

## Why `v0.2.0`

This version aligns with the roadmap milestone for a **reproducible single-task artifact**:

1. Public documentation is in place.
2. Tests are aligned with the current architecture.
3. There is at least one real-dataset experiment path without private data requirements.
4. Training and evaluation are verified end-to-end.
