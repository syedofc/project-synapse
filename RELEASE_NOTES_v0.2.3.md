# Release Notes

Release: `v0.2.3`
Date: `2026-06-04`

## Summary

`v0.2.3` is a research-integrity release. It focuses on tightening the runtime semantics behind Project Synapse's adaptation story rather than adding new headline features.

This release makes the public artifact more trustworthy by fixing several issues that could otherwise blur what the system is actually doing during context-conditioned parameter generation.

## Highlights

1. Added correct per-sample execution when distinct head weights are generated across a batch.
2. Replaced positional multimodal routing with explicit processor-to-input binding support through `input_key_map`.
3. Preserved processor order in ledger cache keys.
4. Moved cached weights to CPU storage to avoid silent accelerator-memory growth.
5. Made the semantic context placeholder deterministic.
6. Corrected training and evaluation loss averaging for skipped-batch scenarios.

## Why This Release Matters

The previous public releases were already runnable, but `v0.2.3` closes several gaps between the repository's research framing and its actual runtime behavior.

In practical terms, this means:

1. batch-conditioned execution is now more honest,
2. multimodal wiring is safer and more explicit,
3. cache behavior better reflects true blueprint differences,
4. the public artifact is more defensible for advisor, reviewer, or lab inspection.

## Verification

This release was checked with:

1. `python -m unittest discover -s tests -v`
2. `python -m py_compile ...` on updated runtime files
3. `python scripts/eval.py --config configs/public_digits_exp.yaml --checkpoint checkpoints/public_digits/Synapse_Public_Digits_v0_2_0_best_model_epoch_7_val_metric_0.9694.pth`
4. `python scripts/test_adaptation.py --config configs/public_digits_resource_exp.yaml --checkpoint checkpoints/public_digits_resource/Synapse_Public_Digits_Resource_v0_2_1_best_model_epoch_5_val_metric_0.9611.pth --max-batches 2`
5. `python scripts/train.py --config configs/public_toy_exp.yaml`

## Known Limitations

1. Semantic context is still placeholder logic rather than a real learned extractor.
2. Several legacy multimodal and detection configs remain experimental.
3. The repository still does not include a formal comparison suite against fixed heads, adapters, or LoRA.
