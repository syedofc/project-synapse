import argparse
import copy
import glob
import json
import math
import os
import pprint
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import synapse_model
from training import trainer as trainer_module
from utils import experiment as experiment_utils
from utils import logger as logger_module
from utils import metrics as metrics_utils


class FixedHeadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _state_dict_size_bytes(state_dict):
    total_bytes = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()
    return total_bytes


def _find_best_checkpoint(checkpoint_dir, project_name):
    pattern = os.path.join(checkpoint_dir, f"{project_name}_best_model_epoch_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern!r} was found after training."
        )
    return matches[-1]


def _prepare_single_task_loaders(config, logger):
    train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, logger)
    task_name = next(iter(train_loaders.keys()))
    return task_name, train_loaders[task_name], test_loaders[task_name]


def _move_fixed_batch_to_device(batch, device):
    inputs, labels = batch
    features = inputs["features"].to(device)
    labels = labels.to(device)
    return features, labels


def _evaluate_fixed_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            features, labels = _move_fixed_batch_to_device(batch, device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def _train_fixed_baseline(config, train_loader, val_loader, device, logger):
    task_name = next(iter(config["tasks"].keys()))
    task_def = config["tasks"][task_name]
    input_dim = task_def.get("input_dim_original") or task_def.get("input_features_dim") or 64
    hidden_dim = config["model"]["task_heads_lib"][0].get("head_hidden_dim", 32)
    num_classes = task_def["output_dim"]

    model = FixedHeadMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            features, labels = _move_fixed_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = _evaluate_fixed_model(model, val_loader, device)
        logger.info(
            "Fixed baseline epoch %d/%d | val_loss=%.4f | val_accuracy=%.4f",
            epoch + 1,
            config["training"]["epochs"],
            val_loss,
            val_accuracy,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_val_loss, final_val_accuracy = _evaluate_fixed_model(model, val_loader, device)
    return model, {
        "val_loss": final_val_loss,
        "val_accuracy": final_val_accuracy,
    }


def _measure_fixed_latency(model, loader, device, warmup_batches=2, measure_batches=20, repeat_forwards=50):
    model.eval()
    latencies = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features, _ = _move_fixed_batch_to_device(batch, device)
            start = time.perf_counter()
            for _ in range(repeat_forwards):
                _ = model(features)
            end = time.perf_counter()
            if batch_idx >= warmup_batches:
                latencies.append((end - start) / repeat_forwards)
            if len(latencies) >= measure_batches:
                break

    if not latencies:
        return 0.0
    return (sum(latencies) / len(latencies)) * 1000.0


def _measure_synapse_latency(
    model,
    loader,
    task_id,
    device,
    context_override,
    warmup_batches=2,
    measure_batches=20,
    repeat_forwards=50,
):
    model.eval()
    latencies = []
    if context_override is not None:
        model.monitor.set_context_override(context_override)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs_on_device = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            start = time.perf_counter()
            for _ in range(repeat_forwards):
                _ = model(inputs_on_device, task_id)
            end = time.perf_counter()
            if batch_idx >= warmup_batches:
                latencies.append((end - start) / repeat_forwards)
            if len(latencies) >= measure_batches:
                break

    model.monitor.clear_context_override()
    if not latencies:
        return 0.0
    return (sum(latencies) / len(latencies)) * 1000.0


def _run_synapse_context_probe(model, loader, task_id, device):
    context_dim = model.config["model"]["weaver"]["device_context_dim"]
    if context_dim == 0:
        return {
            "context_dim": 0,
            "weight_delta_l2": 0.0,
            "prediction_agreement": 1.0,
            "note": "No device context enabled.",
        }

    context_a = torch.linspace(0.1, 0.4, steps=context_dim)
    context_b = torch.linspace(0.9, 0.6, steps=context_dim)

    def run_pass(context_override):
        model.monitor.set_context_override(context_override)
        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(loader))
            inputs_on_device = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            logits = model(inputs_on_device, task_id)
            weights = {
                key: value.detach().cpu().clone()
                for key, value in model.last_weights_generated.items()
            }
        return logits.detach().cpu(), weights

    logits_a, weights_a = run_pass(context_a)
    logits_b, weights_b = run_pass(context_b)
    model.monitor.clear_context_override()

    total_sq = 0.0
    for key in weights_a.keys():
        total_sq += (weights_a[key] - weights_b[key]).float().norm().item() ** 2

    agreement = float(
        (
            logits_a.argmax(dim=1) == logits_b.argmax(dim=1)
        ).float().mean().item()
    )
    return {
        "context_dim": context_dim,
        "weight_delta_l2": total_sq ** 0.5,
        "prediction_agreement": agreement,
    }


def _benchmark_synapse(config, device, logger, checkpoint_override=None):
    config = copy.deepcopy(config)
    experiment_utils.set_random_seed(config["seed"])
    train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, logger)
    model = synapse_model.SynapseModel(config)

    if checkpoint_override:
        logger.info("Loading Synapse checkpoint from %s", checkpoint_override)
        state_dict = torch.load(checkpoint_override, map_location=device)
        model.load_state_dict(state_dict)
    else:
        trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)
        trainer.train()
        checkpoint_path = _find_best_checkpoint(config["checkpoint_dir"], config["project_name"])
        logger.info("Reloading best Synapse checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)
    val_loss, val_metrics, _ = trainer.evaluate()

    task_name = next(iter(test_loaders.keys()))
    task_id = config["tasks"][task_name]["id"]
    context_dim = config["model"]["weaver"]["device_context_dim"]
    context_override = (
        torch.linspace(0.2, 0.8, steps=context_dim)
        if context_dim > 0
        else None
    )
    latency_ms = _measure_synapse_latency(
        model,
        test_loaders[task_name],
        task_id,
        device,
        context_override,
    )
    total_params, trainable_params = _count_parameters(model)
    task_embedding_dim = config["model"]["weaver"]["task_embedding_dim"]
    shared_trainable = max(trainable_params - task_embedding_dim, 0)

    return {
        "model": model,
        "metrics": {
            "val_loss": val_loss,
            "val_accuracy": val_metrics.get("accuracy", 0.0),
            "eval_latency_ms_per_batch": latency_ms,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "marginal_task_parameters": task_embedding_dim,
            "shared_trainable_parameters": shared_trainable,
            "state_dict_bytes": _state_dict_size_bytes(model.state_dict()),
            "context_response": _run_synapse_context_probe(
                model,
                test_loaders[task_name],
                task_id,
                device,
            ),
        },
    }


def _benchmark_fixed_head(config, device, logger):
    config = copy.deepcopy(config)
    experiment_utils.set_random_seed(config["seed"])
    task_name, train_loader, val_loader = _prepare_single_task_loaders(config, logger)
    model, metrics = _train_fixed_baseline(config, train_loader, val_loader, device, logger)
    total_params, trainable_params = _count_parameters(model)
    metrics.update(
        {
            "eval_latency_ms_per_batch": _measure_fixed_latency(model, val_loader, device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "marginal_task_parameters": trainable_params,
            "shared_trainable_parameters": 0,
            "state_dict_bytes": _state_dict_size_bytes(model.state_dict()),
            "context_response": {
                "context_dim": 0,
                "weight_delta_l2": 0.0,
                "prediction_agreement": 1.0,
                "note": "Static baseline; no context-conditioned parameter change.",
            },
        }
    )
    return {"model": model, "metrics": metrics}


def _estimate_break_even_tasks(fixed_task_params, synapse_shared_params, synapse_task_params):
    denominator = fixed_task_params - synapse_task_params
    if denominator <= 0:
        return None
    return int(math.floor(synapse_shared_params / denominator) + 1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Synapse against a fixed-head baseline on public digits.")
    parser.add_argument("--config", type=str, default="configs/public_digits_resource_exp.yaml")
    parser.add_argument("--synapse-checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="training_logs/public_digits_benchmark.json")
    args = parser.parse_args()

    config = experiment_utils.load_config(args.config)
    experiment_utils.set_random_seed(config["seed"])

    os.makedirs("training_logs", exist_ok=True)
    logger = logger_module.setup_logger(
        name="synapse_benchmark_logger",
        log_file=os.path.join("training_logs", "public_digits_benchmark.log"),
    )
    logger.info("=" * 50)
    logger.info("STARTING PUBLIC DIGITS BENCHMARK")
    logger.info("=" * 50)
    logger.info("Loaded benchmark config:\n%s", pprint.pformat(config))

    device = torch.device(config["training"]["device"])

    synapse_result = _benchmark_synapse(config, device, logger, checkpoint_override=args.synapse_checkpoint)
    fixed_result = _benchmark_fixed_head(config, device, logger)

    break_even_tasks = _estimate_break_even_tasks(
        fixed_task_params=fixed_result["metrics"]["marginal_task_parameters"],
        synapse_shared_params=synapse_result["metrics"]["shared_trainable_parameters"],
        synapse_task_params=synapse_result["metrics"]["marginal_task_parameters"],
    )

    result = {
        "benchmark_name": "public_digits_fixed_head_vs_synapse",
        "date": "2026-06-04",
        "config": args.config,
        "synapse": synapse_result["metrics"],
        "fixed_head": fixed_result["metrics"],
        "comparison": {
            "accuracy_delta_synapse_minus_fixed": (
                synapse_result["metrics"]["val_accuracy"] - fixed_result["metrics"]["val_accuracy"]
            ),
            "latency_delta_ms_synapse_minus_fixed": (
                synapse_result["metrics"]["eval_latency_ms_per_batch"]
                - fixed_result["metrics"]["eval_latency_ms_per_batch"]
            ),
            "marginal_task_parameter_delta_synapse_minus_fixed": (
                synapse_result["metrics"]["marginal_task_parameters"]
                - fixed_result["metrics"]["marginal_task_parameters"]
            ),
            "break_even_task_count_estimate": break_even_tasks,
        },
        "interpretation": {
            "summary": (
                "Fixed heads are the conservative single-task baseline. "
                "Synapse carries higher shared overhead but lower marginal task-specific growth."
            )
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    logger.info("Benchmark summary:\n%s", pprint.pformat(result))


if __name__ == "__main__":
    main()
