import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import get_imagenet_loaders
from src import input_processors
from utils import experiment as experiment_utils


def _find_processor_config(config, processor_name):
    for proc_conf in config["model"]["input_processors_lib"]:
        if proc_conf["name"] == processor_name:
            return proc_conf
    raise ValueError(f"Processor '{processor_name}' was not found in config['model']['input_processors_lib'].")


def _build_extractor(proc_conf, device):
    if proc_conf["type"] != "ImageProcessor":
        raise ValueError("Feature extraction currently supports only ImageProcessor entries.")
    extractor = input_processors.ImageProcessor(
        backbone_name=proc_conf["backbone_name"],
        pretrained=proc_conf.get("pretrained", True),
        freeze_backbone=proc_conf.get("freeze_backbone", True),
        output_flattened_features=proc_conf.get("output_flattened_features", True),
        input_channels=proc_conf.get("input_channels", 3),
    ).to(device)
    extractor.eval()
    return extractor


def _save_split_features(loader, extractor, device, output_dir, max_batches=None):
    os.makedirs(output_dir, exist_ok=True)
    saved_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs_raw, labels = batch
            if isinstance(inputs_raw, dict):
                first_tensor = next(iter(inputs_raw.values()))
                images = first_tensor
            else:
                images = inputs_raw

            images = images.to(device)
            features = extractor(images).cpu()
            labels = labels.cpu()

            torch.save(features, os.path.join(output_dir, f"batch_{batch_idx:05d}_features.pt"))
            torch.save(labels, os.path.join(output_dir, f"batch_{batch_idx:05d}_labels.pt"))

            saved_batches += 1
            total_samples += labels.size(0)

    return {"saved_batches": saved_batches, "total_samples": total_samples}


def main():
    parser = argparse.ArgumentParser(description="Extract ImageNet features for Project Synapse.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_for_feature_extraction.yaml",
        help="Path to the feature extraction config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/imagenet_features",
        help="Directory where extracted features will be written.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit for the number of training batches to extract.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional limit for the number of validation batches to extract.",
    )
    args = parser.parse_args()

    config = experiment_utils.load_config(args.config)
    experiment_utils.set_random_seed(config["seed"])

    task_name = "imagenet_classification"
    if task_name not in config["tasks"]:
        raise ValueError("Feature extraction config must define the 'imagenet_classification' task.")

    task_def = config["tasks"][task_name]
    dataset_params = task_def.get("dataset_specific_params", {})
    data_path = config["data"][dataset_params["data_path_key"]]
    img_size = config["data"][dataset_params["img_size_key"]]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    subset_size = dataset_params.get("subset_size")

    processor_name = task_def["input_processor_names"][0]
    proc_conf = _find_processor_config(config, processor_name)
    device = torch.device(config["training"]["device"])
    extractor = _build_extractor(proc_conf, device)

    train_loader, val_loader, input_dim, num_classes = get_imagenet_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        subset_size=subset_size,
    )

    train_stats = _save_split_features(
        train_loader,
        extractor,
        device,
        os.path.join(args.output_dir, "train"),
        max_batches=args.max_train_batches,
    )
    val_stats = _save_split_features(
        val_loader,
        extractor,
        device,
        os.path.join(args.output_dir, "val"),
        max_batches=args.max_val_batches,
    )

    metadata = {
        "project_name": config["project_name"],
        "backbone_name": proc_conf["backbone_name"],
        "output_dir": args.output_dir,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "train": train_stats,
        "val": val_stats,
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("--- Feature Extraction Complete ---")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
