import os
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import synapse_model
from training import trainer as trainer_module
from utils import experiment as experiment_utils


class TestPublicToyPipeline(unittest.TestCase):
    def test_public_toy_config_can_run_one_eval_pass(self):
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "public_toy_exp.yaml")
        )
        config = experiment_utils.load_config(config_path)
        config["training"]["epochs"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            config["checkpoint_dir"] = tmp_dir
            experiment_utils.set_random_seed(config["seed"])

            class _Logger:
                def info(self, *args, **kwargs):
                    return None
                def error(self, *args, **kwargs):
                    return None
                def debug(self, *args, **kwargs):
                    return None

            train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, _Logger())
            model = synapse_model.SynapseModel(config)
            trainer = trainer_module.Trainer(model, config, train_loaders, test_loaders)
            val_loss, metrics_agg, metrics_by_task = trainer.evaluate(epoch=0)

        self.assertGreaterEqual(val_loss, 0.0)
        self.assertIn("accuracy", metrics_agg)
        self.assertIn("toy_classification", metrics_by_task)


if __name__ == "__main__":
    unittest.main()
