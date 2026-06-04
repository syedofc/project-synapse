# Results

Release: `v0.2.1`  
Date: `2026-06-04`

## Verified Public Results

### Stable Public Quick-Start

Config: `configs/public_digits_exp.yaml`

Verified command sequence:

```bash
python scripts/train.py --config configs/public_digits_exp.yaml
python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint checkpoints/public_digits/Synapse_Public_Digits_v0_2_0_best_model_epoch_7_val_metric_0.9694.pth
```

Observed best validation result:

1. Best validation accuracy: `96.94%`
2. Best checkpoint: `epoch 7`
3. Evaluation loss at best checkpoint: `0.0897`

### Resource-Aware Public Demo

Config: `configs/public_digits_resource_exp.yaml`

Verified command sequence:

```bash
python scripts/train.py --config configs/public_digits_resource_exp.yaml
python scripts/test_adaptation.py --config configs/public_digits_resource_exp.yaml --checkpoint checkpoints/public_digits_resource/Synapse_Public_Digits_Resource_v0_2_1_best_model_epoch_5_val_metric_0.9611.pth --max-batches 2
```

Observed training result:

1. Best validation accuracy: `96.11%`
2. Best checkpoint: `epoch 5`

Observed adaptation-analysis result:

1. Sampled-batch accuracy under context A: `93.75%`
2. Sampled-batch accuracy under context B: `93.75%`
3. Prediction agreement across the two contexts: `100%`
4. Total L2 drift in generated head weights: `1.2438`
5. Largest layer drift: `layers.0.weight` with L2 `1.1599`

### Feature Extraction Utility Check

Script: `scripts/extract_imagenet_features.py`

Verified outcome:

1. The script successfully extracted features from a miniature ImageFolder-style dataset.
2. It wrote train and validation feature batches plus metadata to disk.

## Interpretation

These results do not establish a benchmark claim. They show that:

1. the public release is runnable end-to-end,
2. the real-dataset path is reproducible without private data,
3. the resource-aware pathway is functional and analyzable,
4. the feature-extraction tooling now works as an actual utility.
