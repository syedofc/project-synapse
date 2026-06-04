import os
import sys
import argparse
import pprint

import torch.multiprocessing

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import synapse_model
from training import trainer as trainer_module
from utils import experiment as experiment_utils
from utils import logger as logger_module


def main():
    parser = argparse.ArgumentParser(description="Main training script for Project Synapse.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML file.")
    args = parser.parse_args()

    config = experiment_utils.load_config(args.config)

    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['project_name']}.log")
    logger = logger_module.setup_logger(log_file=log_file_path)

    logger.info("=" * 50)
    logger.info("STARTING EXPERIMENT: %s", config["project_name"])
    logger.info("=" * 50)
    logger.info("Loaded configuration:\n%s", pprint.pformat(config))

    experiment_utils.set_random_seed(config["seed"])
    logger.info("Random seed set to %s", config["seed"])

    train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, logger)

    logger.info("Initializing SynapseModel...")
    model = synapse_model.SynapseModel(config)

    logger.info("Initializing Trainer...")
    trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)

    logger.info("Starting training process...")
    trainer.train()

    logger.info("=" * 50)
    logger.info("EXPERIMENT %s FINISHED", config["project_name"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
