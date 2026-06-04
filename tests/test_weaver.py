import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import fabric_substrate, fabric_weaver


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


class TestTaskWeaver(unittest.TestCase):
    def setUp(self):
        self.substrate = fabric_substrate.NeuralSubstrate(
            config=make_test_config(),
            device=torch.device("cpu"),
        )
        self.target_shapes = self.substrate.get_head_parameter_shapes("Toy_Classifier_Head")
        max_head_parameters = sum(torch.Size(shape).numel() for shape in self.target_shapes.values())
        self.weaver = fabric_weaver.TaskWeaver(
            num_unique_tasks=2,
            task_embedding_dim=4,
            device_context_dim=0,
            semantic_context_dim=0,
            hidden_dim=16,
            num_hidden_layers=1,
            max_head_parameters=max_head_parameters,
        )

    def test_output_structure_and_shapes(self):
        task_id = torch.tensor([0])
        context = torch.empty(1, 0)

        weights = self.weaver(task_id, context, None, self.target_shapes)

        self.assertEqual(list(weights.keys()), list(self.target_shapes.keys()))
        for name, shape in self.target_shapes.items():
            self.assertEqual(weights[name].shape, shape)


if __name__ == "__main__":
    unittest.main()
