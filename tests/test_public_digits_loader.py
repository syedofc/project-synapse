import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import experiment as experiment_utils


class TestPublicDigitsLoader(unittest.TestCase):
    def test_public_digits_config_prepares_batches(self):
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "public_digits_exp.yaml")
        )
        config = experiment_utils.load_config(config_path)

        class _Logger:
            def info(self, *args, **kwargs):
                return None
            def error(self, *args, **kwargs):
                return None
            def debug(self, *args, **kwargs):
                return None

        train_loaders, test_loaders = experiment_utils.prepare_dataloaders(config, _Logger())
        batch_inputs, batch_labels = next(iter(train_loaders["digits_classification"]))

        self.assertEqual(batch_inputs["features"].dim(), 2)
        self.assertEqual(batch_inputs["features"].shape[1], 64)
        self.assertEqual(batch_labels.dim(), 1)
        self.assertEqual(config["tasks"]["digits_classification"]["output_dim"], 10)
        self.assertIn("digits_classification", test_loaders)


if __name__ == "__main__":
    unittest.main()
