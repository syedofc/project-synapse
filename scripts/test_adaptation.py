import argparse
import json
import os
import pprint
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import synapse_model
from training import trainer as trainer_module
from utils import experiment as experiment_utils
from utils import logger as logger_module
from utils import metrics as metrics_utils


def _parse_context_arg(raw_value, expected_dim):
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if len(values) != expected_dim:
        raise ValueError(
            f"Context override has {len(values)} values, but expected {expected_dim}."
        )
    return torch.tensor(values, dtype=torch.float32)


def _move_batch_to_device(batch_inputs, batch_labels, device):
    inputs_on_device = {}
    for key, value in batch_inputs.items():
        inputs_on_device[key] = value.to(device) if torch.is_tensor(value) else value
    labels_on_device = batch_labels.to(device) if torch.is_tensor(batch_labels) else batch_labels
    return inputs_on_device, labels_on_device


def _clone_head_weights(weights_dict):
    return {key: value.detach().cpu().clone() for key, value in weights_dict.items()}


def _weight_delta_summary(weights_a, weights_b):
    total_sq = 0.0
    max_layer = ("", 0.0)
    for key in weights_a.keys():
        diff = (weights_a[key] - weights_b[key]).float()
        norm = diff.norm().item()
        total_sq += norm ** 2
        if norm > max_layer[1]:
            max_layer = (key, norm)
    return {
        "total_l2": total_sq ** 0.5,
        "largest_layer": max_layer[0],
        "largest_layer_l2": max_layer[1],
    }


def _prediction_agreement(logits_a, logits_b):
    preds_a = logits_a.argmax(dim=1)
    preds_b = logits_b.argmax(dim=1)
    return float((preds_a == preds_b).float().mean().item())


def _run_context_pass(model, loader, task_id, device, context_override=None, max_batches=1):
    if context_override is not None:
        model.monitor.set_context_override(context_override)
    else:
        model.monitor.clear_context_override()

    logits_list = []
    labels_list = []
    weight_snapshot = None
    context_source = None

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):
            batch_inputs, batch_labels = _move_batch_to_device(batch_inputs, batch_labels, device)
            outputs = model(batch_inputs, task_id)
            if weight_snapshot is None and model.last_weights_generated is not None:
                weight_snapshot = _clone_head_weights(model.last_weights_generated)
                context_source = model.monitor.last_source
            logits_list.append(outputs.detach().cpu())
            labels_list.append(batch_labels.detach().cpu())
            if batch_idx + 1 >= max_batches:
                break

    return {
        "logits": torch.cat(logits_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "weights": weight_snapshot,
        "context_source": context_source or model.monitor.last_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptation analysis script for Project Synapse.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved checkpoint.")
    parser.add_argument("--task", type=str, default=None, help="Task name to analyze. Defaults to the first available task.")
    parser.add_argument("--max-batches", type=int, default=1, help="Number of evaluation batches to compare.")
    parser.add_argument("--context-a", type=str, default=None, help="Comma-separated override context for pass A.")
    parser.add_argument("--context-b", type=str, default=None, help="Comma-separated override context for pass B.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the analysis as JSON.")
    args = parser.parse_args()

    config = experiment_utils.load_config(args.config)

    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['project_name']}_adaptation.log")
    logger = logger_module.setup_logger(name="synapse_adaptation_logger", log_file=log_file_path)

    logger.info("=" * 50)
    logger.info("STARTING ADAPTATION ANALYSIS: %s", config["project_name"])
    logger.info("=" * 50)
    logger.info("Loaded configuration:\n%s", pprint.pformat(config))

    experiment_utils.set_random_seed(config["seed"])
    train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, logger)

    model = synapse_model.SynapseModel(config)
    state_dict = torch.load(args.checkpoint, map_location=config["training"]["device"])
    model.load_state_dict(state_dict)
    device = torch.device(config["training"]["device"])

    trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)
    _ = trainer  # Keeps parity with other scripts and ensures initialization path is valid.

    task_name = args.task or next(iter(test_loaders.keys()))
    if task_name not in test_loaders:
        raise ValueError(f"Task '{task_name}' is not available in evaluation loaders.")
    task_id = config["tasks"][task_name]["id"]
    loader = test_loaders[task_name]

    context_dim = config["model"]["weaver"]["device_context_dim"]
    if context_dim == 0:
        logger.info("This configuration does not use device context (device_context_dim=0).")
        baseline = _run_context_pass(model, loader, task_id, device, context_override=None, max_batches=args.max_batches)
        accuracy = metrics_utils.calculate_accuracy(baseline["logits"], baseline["labels"])
        summary = {
            "task": task_name,
            "context_dim": 0,
            "message": "No resource-conditioned adaptation is possible for this configuration.",
            "baseline_accuracy": accuracy,
        }
        logger.info("Baseline sampled-batch accuracy: %.2f%%", accuracy * 100.0)
        if args.output_json:
            with open(args.output_json, "w") as handle:
                json.dump(summary, handle, indent=2)
        return

    if args.context_a:
        context_a = _parse_context_arg(args.context_a, context_dim)
    else:
        context_a = model.monitor.get_context()

    if args.context_b:
        context_b = _parse_context_arg(args.context_b, context_dim)
    else:
        context_b = torch.clamp(1.0 - context_a, 0.0, 1.0)

    pass_a = _run_context_pass(model, loader, task_id, device, context_override=context_a, max_batches=args.max_batches)
    pass_b = _run_context_pass(model, loader, task_id, device, context_override=context_b, max_batches=args.max_batches)
    model.monitor.clear_context_override()

    acc_a = metrics_utils.calculate_accuracy(pass_a["logits"], pass_a["labels"])
    acc_b = metrics_utils.calculate_accuracy(pass_b["logits"], pass_b["labels"])
    agreement = _prediction_agreement(pass_a["logits"], pass_b["logits"])
    weight_summary = _weight_delta_summary(pass_a["weights"], pass_b["weights"])

    result = {
        "task": task_name,
        "context_dim": context_dim,
        "context_a": [round(value, 4) for value in context_a.tolist()],
        "context_b": [round(value, 4) for value in context_b.tolist()],
        "context_source_a": pass_a["context_source"],
        "context_source_b": pass_b["context_source"],
        "sampled_batch_accuracy_a": acc_a,
        "sampled_batch_accuracy_b": acc_b,
        "prediction_agreement": agreement,
        "weight_delta": weight_summary,
    }

    logger.info("Adaptation summary:\n%s", pprint.pformat(result))
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
