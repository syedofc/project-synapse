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


def make_multimodal_test_config():
    return {
        "model": {
            "input_processors_lib": [
                {
                    "name": "Vision_Identity",
                    "type": "IdentityProcessor",
                    "output_features_dim": 2,
                },
                {
                    "name": "Audio_Identity",
                    "type": "IdentityProcessor",
                    "output_features_dim": 3,
                },
            ],
            "task_heads_lib": [
                {
                    "name": "Fusion_Classifier_Head",
                    "type": "ClassificationHead",
                    "input_features_dim": 5,
                    "num_classes": 2,
                    "num_hidden_layers_in_head": 0,
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
            "input_key_map": {"Toy_Feature_Identity": "features"},
        }
        dummy_inputs = {"features": torch.rand(self.batch_size, 8)}

        output = self.substrate(dummy_inputs, blueprint, mock_weights)
        self.assertEqual(output.shape, (self.batch_size, 3))

    def test_per_sample_weights_match_individual_runs(self):
        target_shapes = self.substrate.get_head_parameter_shapes("Toy_Classifier_Head")
        sample_weights = []
        for seed in (0, 1):
            generator = torch.Generator().manual_seed(seed)
            sample_weights.append(
                OrderedDict(
                    (name, torch.rand(shape, generator=generator))
                    for name, shape in target_shapes.items()
                )
            )

        blueprint = {
            "input_processor_names": ["Toy_Feature_Identity"],
            "head_name": "Toy_Classifier_Head",
            "fusion_strategy": "none",
            "input_key_map": {"Toy_Feature_Identity": "features"},
        }
        dummy_inputs = {"features": torch.rand(2, 8)}

        batched_output = self.substrate(dummy_inputs, blueprint, sample_weights)
        expected_outputs = []
        for idx, weights in enumerate(sample_weights):
            single_inputs = {"features": dummy_inputs["features"][idx : idx + 1]}
            expected_outputs.append(self.substrate(single_inputs, blueprint, weights))
        expected_output = torch.cat(expected_outputs, dim=0)

        self.assertTrue(torch.allclose(batched_output, expected_output))

    def test_multimodal_inputs_require_explicit_mapping(self):
        substrate = fabric_substrate.NeuralSubstrate(
            config=make_multimodal_test_config(),
            device=torch.device("cpu"),
        )
        mock_weights = OrderedDict()
        for name, shape in substrate.get_head_parameter_shapes("Fusion_Classifier_Head").items():
            mock_weights[name] = torch.rand(shape)

        dummy_inputs = {
            "vision": torch.rand(2, 2),
            "audio": torch.rand(2, 3),
        }
        blueprint_without_mapping = {
            "input_processor_names": ["Vision_Identity", "Audio_Identity"],
            "head_name": "Fusion_Classifier_Head",
            "fusion_strategy": "concat",
        }
        with self.assertRaises(ValueError):
            substrate(dummy_inputs, blueprint_without_mapping, mock_weights)

        blueprint_with_mapping = {
            "input_processor_names": ["Vision_Identity", "Audio_Identity"],
            "head_name": "Fusion_Classifier_Head",
            "fusion_strategy": "concat",
            "input_key_map": {
                "Vision_Identity": "vision",
                "Audio_Identity": "audio",
            },
        }
        output = substrate(dummy_inputs, blueprint_with_mapping, mock_weights)
        self.assertEqual(output.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
