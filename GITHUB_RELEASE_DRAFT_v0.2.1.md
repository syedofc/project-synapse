# Project Synapse v0.2.1

Project Synapse is a research codebase for studying **resource-conditioned hypernetworks for modular task adaptation**.

In practical terms, this repository asks whether a compact hypernetwork can generate lightweight task parameters from task identity and runtime device context, while a ledger-like memory helps recall previously useful parameter states.

This `v0.2.1` release is the first version I would consider a credible **public research release candidate**. It includes:

1. a reproducible real-dataset public quick-start,
2. a resource-aware demo path,
3. a working adaptation analysis script,
4. a working feature extraction utility,
5. verified tests and documented public results.

## Highlights

- Public quick-start on `sklearn` digits with no private dataset dependency
- Resource-aware digits demo using telemetry-aware context handling
- Adaptation analysis reporting accuracy, prediction agreement, and generated-weight drift
- Feature extraction utility for ImageFolder-style datasets
- Synced release docs, citation metadata, and contributor guidance

## Who This Release Is For

- Researchers studying hypernetworks, PEFT, continual learning, or edge adaptation
- Students who want a runnable base for resource-aware adaptation experiments
- Engineers exploring whether context-conditioned parameter generation is worth benchmarking further

## What This Release Contributes

- A public and inspectable systems prototype
- A verified quick-start path on a real dataset
- A resource-aware demo that exposes context-conditioned parameter changes
- A clean base for future comparisons against fixed heads, adapters, and LoRA

## Recommended Quick Start

```bash
python -m unittest discover -s tests -v
python scripts/train.py --config configs/public_digits_exp.yaml
python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint <checkpoint_path>
```

## Verified Public Results

- Public digits quick-start best validation accuracy: `96.94%`
- Resource-aware digits demo best validation accuracy: `96.11%`

## Important Limitations

- Semantic context is still placeholder logic
- Legacy multimodal configs remain experimental
- Baseline comparisons against adapters and LoRA are not yet included

## Recommended Interpretation

This release should be read as a meaningful **research artifact** and **benchmark starting point**, not as a claim that the method is already benchmark-superior or production-ready.

## Research Framing

The core question behind this repository is:

> Can a resource-conditioned hypernetwork generate lightweight task heads or adapters that improve task switching, memory use, or on-device adaptation latency over fixed heads, adapters, or LoRA baselines?

This release is intended as a solid public artifact for that direction, not as a final benchmark paper claim.
