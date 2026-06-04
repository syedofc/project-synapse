import argparse
import os
import pprint
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src import synapse_model
from training import trainer as trainer_module
from utils import experiment as experiment_utils
from utils import logger as logger_module


def main():
    parser = argparse.ArgumentParser(description="Evaluation script for Project Synapse.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved checkpoint.")
    args = parser.parse_args()

    config = experiment_utils.load_config(args.config)

    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['project_name']}_eval.log")
    logger = logger_module.setup_logger(name="synapse_eval_logger", log_file=log_file_path)

    logger.info("=" * 50)
    logger.info("STARTING EVALUATION: %s", config["project_name"])
    logger.info("=" * 50)
    logger.info("Loaded configuration:\n%s", pprint.pformat(config))

    experiment_utils.set_random_seed(config["seed"])
    train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, logger)

    model = synapse_model.SynapseModel(config)
    state_dict = torch.load(args.checkpoint, map_location=config["training"]["device"])
    model.load_state_dict(state_dict)

    trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)
    trainer.evaluate()


if __name__ == "__main__":
    main()
