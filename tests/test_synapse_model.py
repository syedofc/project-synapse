import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.synapse_model import SemanticContextExtractor


class TestSemanticContextExtractor(unittest.TestCase):
    def test_placeholder_is_deterministic_for_same_input(self):
        extractor = SemanticContextExtractor({"output_dim": 4}, torch.device("cpu"))
        sample = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])

        first = extractor(sample)
        second = extractor(sample)

        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(first.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
