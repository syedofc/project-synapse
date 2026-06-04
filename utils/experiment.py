import copy
import os
import pprint
import traceback

import numpy as np
import torch
import yaml

import data as data_module


def load_config(config_path):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_loader_args(config, task_name, task_def, logger):
    loader_func_name = task_def.get("data_loader_func")
    dataset_params = task_def.get("dataset_specific_params", {})

    current_loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
    }
    current_loader_args.update(dataset_params.get("loader_kwargs", {}))

    if loader_func_name == "get_imagenet_loaders":
        data_path_key = dataset_params.get("data_path_key")
        img_size_key = dataset_params.get("img_size_key")
        if not data_path_key or data_path_key not in config["data"] or \
           not img_size_key or img_size_key not in config["data"]:
            raise ValueError(
                f"Task '{task_name}' is missing ImageNet data path or img size keys."
            )
        current_loader_args["data_path"] = config["data"][data_path_key]
        current_loader_args["img_size"] = config["data"][img_size_key]

    elif loader_func_name == "get_a2d2_loaders":
        label_root_key = dataset_params.get("label_root_path_key")
        depth_root_key = dataset_params.get("depth_root_path_key")
        img_size_key = dataset_params.get("img_size_a2d2_output_mask_key")
        color_map_path = dataset_params.get("color_map_json_path")

        if not all(
            [
                label_root_key,
                depth_root_key,
                img_size_key,
                label_root_key in config["data"],
                depth_root_key in config["data"],
                img_size_key in config["data"],
            ]
        ):
            raise ValueError(
                f"A2D2 task '{task_name}' is missing one or more required path/size keys."
            )

        current_loader_args["task_config"] = {
            "label_root": config["data"][label_root_key],
            "depth_root": config["data"][depth_root_key],
            "img_size_a2d2_output_mask": config["data"][img_size_key],
            "num_classes": dataset_params.get("num_classes"),
            "color_map_json_path": color_map_path,
            "seed": config["seed"],
            "camera_view": dataset_params.get("camera_view", "cam_front_center"),
        }

        data_path_key = dataset_params.get("data_path_key")
        if data_path_key and data_path_key in config["data"]:
            current_loader_args["data_path"] = config["data"][data_path_key]

    elif loader_func_name == "get_coco_loaders":
        data_path_key = dataset_params.get("data_path_key")
        img_size_key = dataset_params.get("img_size_key")
        if not data_path_key or data_path_key not in config["data"]:
            raise ValueError(f"COCO task '{task_name}' is missing a valid data_path_key.")

        current_loader_args["data_path"] = config["data"][data_path_key]
        current_loader_args["ann_file_name_train"] = dataset_params["ann_file_train_name"]
        current_loader_args["ann_file_name_val"] = dataset_params["ann_file_val_name"]
        if img_size_key and img_size_key in config["data"]:
            current_loader_args["img_size"] = config["data"][img_size_key]

    else:
        generic_data_path_key = dataset_params.get("data_path_key")
        generic_img_size_key = dataset_params.get("img_size_key")

        if generic_data_path_key and generic_data_path_key in config["data"]:
            current_loader_args["data_path"] = config["data"][generic_data_path_key]
        if generic_img_size_key and generic_img_size_key in config["data"]:
            current_loader_args["img_size"] = config["data"][generic_img_size_key]

    logger.debug(
        "Arguments for %s on task '%s': %s",
        loader_func_name,
        task_name,
        pprint.pformat(current_loader_args),
    )
    return current_loader_args


def prepare_dataloaders(config, logger):
    train_loaders = {}
    test_loaders = {}

    logger.info("Checking 'tasks' in configuration...")
    if not config.get("tasks") or not isinstance(config["tasks"], dict) or not config["tasks"]:
        raise ValueError("Configuration file must contain a non-empty 'tasks' dictionary.")

    logger.info("Found %d task(s) defined in config.", len(config["tasks"]))

    for task_name, task_def in config["tasks"].items():
        logger.info(
            "--- Preparing data for task: '%s' (ID: %s) ---",
            task_name,
            task_def.get("id", "N/A"),
        )

        loader_func_name = task_def.get("data_loader_func")
        if not loader_func_name:
            logger.error("Task '%s' is missing 'data_loader_func'. Skipping.", task_name)
            continue

        if not hasattr(data_module, loader_func_name):
            logger.error(
                "Data loader function '%s' was not found in data package. Skipping task '%s'.",
                loader_func_name,
                task_name,
            )
            continue

        loader_function = getattr(data_module, loader_func_name)

        try:
            current_loader_args = _resolve_loader_args(config, task_name, task_def, logger)
            loader_results = loader_function(**current_loader_args)

            if loader_results is None or not isinstance(loader_results, tuple) or len(loader_results) != 4:
                raise ValueError(
                    "Loader did not return the expected "
                    "(train_loader, val_loader, input_dim_info, num_classes) tuple."
                )

            train_loader, val_loader, input_dim_info, num_classes = loader_results
            if train_loader is None:
                raise ValueError("Train loader is None.")

            train_loaders[task_name] = train_loader
            test_loaders[task_name] = val_loader

            config["tasks"][task_name]["output_dim"] = num_classes
            if isinstance(input_dim_info, tuple):
                config["tasks"][task_name]["input_dim_tuple_original"] = input_dim_info
            elif isinstance(input_dim_info, dict):
                config["tasks"][task_name]["input_dims_dict_original"] = input_dim_info
            else:
                config["tasks"][task_name]["input_dim_original"] = input_dim_info

            logger.info(
                "Data for task '%s' prepared successfully. Train batches: %d, Val batches: %d. Output dim: %s",
                task_name,
                len(train_loader),
                len(val_loader) if val_loader is not None else 0,
                num_classes,
            )

        except Exception as exc:
            logger.error("CRITICAL ERROR loading data for task '%s': %s", task_name, exc)
            logger.error(traceback.format_exc())
            logger.error("Skipping task '%s' due to loader failure.", task_name)
            continue

    if not train_loaders:
        raise ValueError(
            "No data loaders were successfully created after checking all tasks."
        )

    active_task_ids = [
        task_def["id"]
        for task_name, task_def in config["tasks"].items()
        if task_name in train_loaders and "id" in task_def
    ]
    config["model"]["weaver"]["num_unique_tasks"] = (max(active_task_ids) + 1) if active_task_ids else 0

    logger.info("Successfully prepared data loaders for tasks: %s", list(train_loaders.keys()))
    return train_loaders, test_loaders


def prepare_config(config_path):
    config = load_config(config_path)
    return copy.deepcopy(config)
