# Contributing

Project Synapse is currently maintained as a research codebase. Contributions are welcome, especially in the following areas:

1. Reproducibility and experiment cleanup
2. Resource-aware conditioning with real telemetry
3. Adapter and LoRA baselines
4. Evaluation scripts and benchmark automation
5. Test coverage for additional experiment paths

## Local Setup

```bash
conda env create -f 02_environment.yml
conda activate synapse
```

Or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 03_requirements.txt
```

## Quick Verification

```bash
python -m unittest discover -s tests -v
python scripts/train.py --config configs/public_toy_exp.yaml
python scripts/train.py --config configs/public_digits_exp.yaml
python scripts/train.py --config configs/public_digits_resource_exp.yaml
```

## Contribution Style

1. Keep changes focused and well-scoped.
2. Prefer adding or updating tests when behavior changes.
3. Document limitations honestly.
4. Preserve the repository's research-oriented tone and modular structure.
