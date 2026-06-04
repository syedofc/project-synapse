# Project Synapse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20544669.svg)](https://doi.org/10.5281/zenodo.20544669)

Version: `v0.3.2`
Release Date: `2026-06-04`
Status: `Polished evidence-backed research release candidate`

Project Synapse is a research codebase for studying **resource-conditioned hypernetworks for modular task adaptation**. The core idea is to use a compact hypernetwork to generate lightweight task-specific parameters on demand, conditioned on task identity and runtime device context, while a cache-like memory module reuses previously successful parameter configurations.

This repository is being released as a **public research artifact** with a verified quick-start path. It is still research software, not a production-ready framework.

## Quick Links

1. Latest public release: `v0.3.2`
2. Zenodo concept DOI: `10.5281/zenodo.20544669`
3. Zenodo version DOI for `v0.3.2`: `10.5281/zenodo.20544670`
4. Public benchmark note: [RESULTS_v0.3.0.md](RESULTS_v0.3.0.md)
5. Seeded benchmark artifact: [benchmarks/public_digits_fixed_head_vs_synapse_seed42.json](benchmarks/public_digits_fixed_head_vs_synapse_seed42.json)
6. Citation metadata: [CITATION.cff](CITATION.cff)

## Essence

In one sentence:

> Generate small task-specific parameters from task identity and device context, then reuse successful parameter states through a cache-like memory.

## Why This Exists

Most parameter-efficient adaptation workflows are still largely static. Project Synapse explores a different systems question:

> Can adaptation become **context-aware at runtime**, so a model changes how it parameterizes itself depending on task demands and device constraints?

That makes this repository most useful as a platform for studying:

1. edge or on-device adaptation,
2. task switching under changing resource conditions,
3. lightweight hypernetwork-generated heads or adapters,
4. cache or recall mechanisms for previously successful parameter states.

## Who This Is For

This release is primarily for:

1. researchers studying hypernetworks, PEFT, continual learning, or edge adaptation,
2. students who want a runnable starting point for resource-aware adaptation experiments,
3. practitioners exploring ideas for modular on-device adaptation before investing in a larger benchmark effort.

If you need a production PEFT stack today, standard LoRA or adapter pipelines are usually the better default.

## Research Focus

The main research question for this release is:

> Can a resource-conditioned hypernetwork generate lightweight task heads or adapters that improve task switching, memory use, or on-device adaptation latency over fixed heads, adapters, or LoRA baselines?

The code currently explores that question through three architectural components:

1. **Neural Substrate**
   A modular execution layer that selects input processors and task heads.
2. **Task Weaver**
   A hypernetwork that generates lightweight head parameters conditioned on task and context.
3. **Synaptic Ledger**
   A cache that stores successful context-to-parameter mappings for fast recall.

## Release Scope

`v0.3.2` is intended to provide a public-ready, reproducible research artifact on top of the broader research codebase. It is suitable for:

1. Advisor review
2. Early open-source release
3. Research planning
4. Incremental benchmarking work

It is not yet a complete benchmark suite, but it now includes verified public experiment paths.

## Current Benchmark Evidence

The first controlled public benchmark in this repository compares Synapse against a fixed-head baseline on the public digits task under the same `8`-epoch training budget.

Headline result from the seeded end-to-end run:

1. Synapse resource-aware model: `98.61%` validation accuracy
2. Fixed-head baseline: `95.56%` validation accuracy
3. Synapse latency remains much higher on CPU: `1.39 ms/batch` vs `0.02 ms/batch`
4. Synapse marginal task-specific parameters remain much smaller: `8` vs `2410`

Curated artifact:

1. benchmark summary note: [RESULTS_v0.3.0.md](RESULTS_v0.3.0.md)
2. machine-readable result: [benchmarks/public_digits_fixed_head_vs_synapse_seed42.json](benchmarks/public_digits_fixed_head_vs_synapse_seed42.json)

Interpretation:

1. Synapse is no longer only a conceptual artifact; it now has one reproducible comparative result.
2. The system still carries a large shared overhead on small single-task problems.
3. Its most defensible efficiency advantage remains **marginal task growth**, not single-task simplicity.

## What A Viewer Should Take Away

The point of this repository is not to claim a new fundamental learning primitive. The point is to provide a meaningful, inspectable public system for testing whether:

1. runtime resource context can shape lightweight parameter generation,
2. modular task-head generation can support efficient switching,
3. a memory-like ledger can help recall useful parameter states.

That makes Synapse most valuable as:

1. a research prototype,
2. a benchmark starting point,
3. a thesis or lab artifact that can be extended with stronger baselines.

## What Is Implemented

The current codebase includes:

1. A modular `SynapseModel` orchestrator in [src/synapse_model.py](src/synapse_model.py)
2. A task-conditioned hypernetwork in [src/fabric_weaver.py](src/fabric_weaver.py)
3. A modular substrate in [src/fabric_substrate.py](src/fabric_substrate.py)
4. A ledger-based cache in [src/fabric_ledger.py](src/fabric_ledger.py)
5. Dataset loaders and experiment configs for several prototype scenarios
6. A training entrypoint in [scripts/train.py](scripts/train.py)
7. Exploratory notebooks and historical training logs
8. A fully self-contained public toy experiment in [configs/public_toy_exp.yaml](configs/public_toy_exp.yaml)
9. A reproducible real-dataset experiment in [configs/public_digits_exp.yaml](configs/public_digits_exp.yaml)
10. A resource-aware public demo in [configs/public_digits_resource_exp.yaml](configs/public_digits_resource_exp.yaml)
11. A working adaptation analysis tool in [scripts/test_adaptation.py](scripts/test_adaptation.py)
12. A working feature extraction utility in [scripts/extract_imagenet_features.py](scripts/extract_imagenet_features.py)
13. A public fixed-head vs Synapse benchmark script in [scripts/benchmark_public_digits.py](scripts/benchmark_public_digits.py)

## Current Limitations

This release still has important limitations:

1. The semantic context extractor is still a placeholder in [src/synapse_model.py](src/synapse_model.py).
2. Several legacy configs are exploratory and may require local path fixes, dataset preparation, or additional code work.
3. Some planned multimodal and detection paths remain partial or aspirational.
4. The repo now supports resource-aware experiments, but it does not yet include the broader baseline suite promised in the longer-term roadmap.

## Recent Integrity Fixes

The `v0.2.3` release tightens several semantics that matter for research correctness:

1. per-sample generated weights are now handled explicitly instead of being silently collapsed to the first sample,
2. processor-to-input routing can now be declared explicitly through `input_key_map`,
3. ledger cache keys now preserve processor order and store cached weights on CPU.

## Repository Layout

The repository uses a numbered root layout for internal organization, but the active source directories are standard:

1. `configs/` experiment YAML files
2. `data/` dataset loaders
3. `src/` core model code
4. `training/` trainer and dashboard
5. `utils/` logging and metrics helpers
6. `scripts/` training, evaluation, extraction, and adaptation entrypoints
7. `benchmarks/` curated public benchmark artifacts
8. `tests/` prototype tests
9. `notebooks/` analysis and debugging notebooks
10. `training_logs/` archived run logs

## Installation

### Conda

```bash
conda env create -f 02_environment.yml
conda activate synapse
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 03_requirements.txt
```

## Example Entrypoints

### Train

```bash
python scripts/train.py --config configs/public_digits_exp.yaml
```

Notes:

1. `configs/public_digits_exp.yaml` is the recommended public quick-start because it uses a real dataset bundled via `scikit-learn` with no private data requirement.
2. `configs/public_toy_exp.yaml` remains available as a fully self-contained smoke-test path.
3. `configs/public_digits_resource_exp.yaml` is the recommended resource-aware demo path when you want to exercise telemetry-conditioned parameter generation.
4. Other configs are still experimental and often expect local datasets that are not bundled in the repo.
5. Some legacy configs require preparatory steps such as feature extraction.

### Evaluate

```bash
python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint <checkpoint_path>
```

The training script saves checkpoints to the directory configured by `checkpoint_dir`.

### Adaptation Analysis

```bash
python scripts/test_adaptation.py --config configs/public_digits_resource_exp.yaml --checkpoint <checkpoint_path> --max-batches 2
```

This script compares generated task-head weights and predictions under two contexts and reports:

1. sampled-batch accuracy under each context,
2. prediction agreement,
3. drift in generated head weights.

### Benchmark

```bash
python scripts/benchmark_public_digits.py --config configs/public_digits_resource_exp.yaml
```

This benchmark trains a fixed-head baseline and compares it against Synapse on the public digits path using the same epoch budget.

Curated seeded result artifact:

```bash
cat benchmarks/public_digits_fixed_head_vs_synapse_seed42.json
```

### Feature Extraction

```bash
python scripts/extract_imagenet_features.py --config configs/config_for_feature_extraction.yaml --output-dir data/imagenet_features
```

This utility extracts backbone features from an ImageFolder-style dataset and writes feature and label batches for later use with `get_imagenet_feature_loaders`.

### Reproducibility

The following commands have been verified in this repository:

```bash
python -m unittest discover -s tests -v
python scripts/train.py --config configs/public_toy_exp.yaml
python scripts/train.py --config configs/public_digits_exp.yaml
python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint <checkpoint_path>
python scripts/benchmark_public_digits.py --config configs/public_digits_resource_exp.yaml
```

### What You Can Use Today

Today, this repository can already be used to:

1. train and evaluate a reproducible public classification experiment,
2. run a resource-aware demo that changes generated head weights across contexts,
3. inspect whether context changes alter generated parameters without destroying predictions,
4. run a fixed-head versus Synapse benchmark on the public digits task,
5. extract backbone features from an ImageFolder-style dataset for further experiments.

### What You Should Not Assume Yet

This release should not yet be interpreted as:

1. a production framework,
2. a benchmark-proven replacement for adapters or LoRA,
3. a claim that resource-conditioned hypernetworks are broadly superior.

### Notebooks

The notebooks in `notebooks/` are useful for exploratory debugging and architecture inspection, but they should be treated as research workpads rather than stable reproducibility artifacts.

## Release Documents

This release includes:

1. Research positioning: [04_RESEARCH_POSITIONING_v1.0.md](04_RESEARCH_POSITIONING_v1.0.md)
2. Changelog: [CHANGELOG.md](CHANGELOG.md)
3. Release notes: [RELEASE_NOTES_v0.3.2.md](RELEASE_NOTES_v0.3.2.md)
4. Previous release notes: [RELEASE_NOTES_v0.3.1.md](RELEASE_NOTES_v0.3.1.md)
5. Previous release notes: [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md)
6. Previous release notes: [RELEASE_NOTES_v0.2.3.md](RELEASE_NOTES_v0.2.3.md)
7. Previous release notes: [RELEASE_NOTES_v0.2.2.md](RELEASE_NOTES_v0.2.2.md)
8. Previous release notes: [RELEASE_NOTES_v0.2.1.md](RELEASE_NOTES_v0.2.1.md)
9. Previous release notes: [RELEASE_NOTES_v0.2.0.md](RELEASE_NOTES_v0.2.0.md)
10. Previous alpha notes: [RELEASE_NOTES_v0.1.0-alpha.md](RELEASE_NOTES_v0.1.0-alpha.md)
11. Verified public results: [RESULTS_v0.3.0.md](RESULTS_v0.3.0.md)
12. Curated benchmark artifacts: [benchmarks/README.md](benchmarks/README.md)
13. Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
14. Authors: [AUTHORS.md](AUTHORS.md)
15. Citation metadata: [CITATION.cff](CITATION.cff)
16. Zenodo DOI setup: [ZENODO_DOI_SETUP.md](ZENODO_DOI_SETUP.md)
17. Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
18. Security policy: [SECURITY.md](SECURITY.md)
19. Version marker: [VERSION](VERSION)

## Recommended Interpretation

The strongest way to read this repository is:

1. As a **research prototype** for resource-aware parameter generation
2. As a **platform for benchmarking hypernetwork-based task adaptation**
3. As an **evidence-backed public release candidate** that still needs broader baseline work, richer systems evaluation, and real telemetry support

## Near-Term Roadmap

The current roadmap is:

1. `v0.1.0-alpha`
   Architecture release and positioning package
2. `v0.2.0`
   Reproducible single-task public artifact
3. `v0.3.0`
   First comparative fixed-head benchmark on the public digits path
4. `v0.4.0`
   Broader baseline suite including adapters and LoRA
5. `v0.5.0`
   Multitask edge benchmark study
6. `v1.0.0`
   Research-quality open-source release

## License

The project is distributed under the MIT license in [01_LICENSE](01_LICENSE).

## Authorship And Citation

The current public release is maintained by Syed Amir Hamza. Authorship details are listed in [AUTHORS.md](AUTHORS.md), and software citation metadata is provided in [CITATION.cff](CITATION.cff).

Preferred DOI usage:

1. cite `10.5281/zenodo.20544670` when you want the exact `v0.3.2` release,
2. cite `10.5281/zenodo.20544669` when you want the evolving project across versions.

## Zenodo And DOI

This repository now has a Zenodo record. The current concept DOI is `10.5281/zenodo.20544669`, and the current version DOI for `v0.3.2` is `10.5281/zenodo.20544670`. The archival workflow and future-version notes are documented in [ZENODO_DOI_SETUP.md](ZENODO_DOI_SETUP.md).
