import os
import sys
import unittest
from collections import OrderedDict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import fabric_substrate


def make_test_config():
    return {
        "model": {
            "input_processors_lib": [
                {
                    "name": "Toy_Feature_Identity",
                    "type": "IdentityProcessor",
                    "output_features_dim": 8,
                }
            ],
            "task_heads_lib": [
                {
                    "name": "Toy_Classifier_Head",
                    "type": "ClassificationHead",
                    "input_features_dim": 8,
                    "num_classes": 3,
                    "num_hidden_layers_in_head": 1,
                    "head_hidden_dim": 6,
                }
            ],
        }
    }


class TestNeuralSubstrate(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.substrate = fabric_substrate.NeuralSubstrate(
            config=make_test_config(),
            device=torch.device("cpu"),
        )

    def test_forward_pass(self):
        mock_weights = OrderedDict()
        for name, shape in self.substrate.get_head_parameter_shapes("Toy_Classifier_Head").items():
            mock_weights[name] = torch.rand(shape)

        blueprint = {
            "input_processor_names": ["Toy_Feature_Identity"],
            "head_name": "Toy_Classifier_Head",
            "fusion_strategy": "none",
        }
        dummy_inputs = {"features": torch.rand(self.batch_size, 8)}

        output = self.substrate(dummy_inputs, blueprint, mock_weights)
        self.assertEqual(output.shape, (self.batch_size, 3))


if __name__ == "__main__":
    unittest.main()
