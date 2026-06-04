# Results

Release: `v0.2.0`  
Date: `2026-06-04`

## Verified Public Results

### Smoke Test

Config: `configs/public_toy_exp.yaml`

Observed outcome:

1. Training and evaluation completed successfully.
2. The synthetic task reached perfect validation accuracy as expected for a separable toy benchmark.

### Public Real-Dataset Quick-Start

Config: `configs/public_digits_exp.yaml`

Dataset:

1. `sklearn.datasets.load_digits`
2. 10-way handwritten digit classification
3. No private or manually downloaded dataset dependency

Verified command sequence:

```bash
python scripts/train.py --config configs/public_digits_exp.yaml
python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint checkpoints/public_digits/Synapse_Public_Digits_v0_2_0_best_model_epoch_7_val_metric_0.9694.pth
```

Observed best validation result:

1. Best validation accuracy: `96.94%`
2. Best checkpoint: `epoch 7`
3. Evaluation loss at best checkpoint: `0.0897`

## Interpretation

These results do not establish state-of-the-art performance. Their purpose is to show that:

1. the public release is runnable,
2. the training and evaluation path is verified,
3. the current architecture can learn a real dataset end-to-end without private data.
