import os
import sys
import unittest
from collections import OrderedDict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fabric_ledger import SynapticLedger


class TestSynapticLedger(unittest.TestCase):
    def test_processor_order_is_preserved_in_cache_key(self):
        ledger = SynapticLedger(capacity=2)
        weights = OrderedDict({"output.weight": torch.ones(2, 2)})
        blueprint_a = {
            "input_processor_names": ["vision", "audio"],
            "head_name": "fusion_head",
        }
        blueprint_b = {
            "input_processor_names": ["audio", "vision"],
            "head_name": "fusion_head",
        }

        ledger.store(
            task_id_scalar=0,
            device_context_tensor=torch.tensor([0.1, 0.2]),
            semantic_context_tensor=None,
            blueprint_signals_dict=blueprint_a,
            weights_dict=weights,
        )

        hit = ledger.retrieve(0, torch.tensor([0.1, 0.2]), None, blueprint_a)
        miss = ledger.retrieve(0, torch.tensor([0.1, 0.2]), None, blueprint_b)

        self.assertIsNotNone(hit)
        self.assertIsNone(miss)

    def test_cached_weights_are_stored_on_cpu(self):
        ledger = SynapticLedger(capacity=1)
        weights = OrderedDict({"output.weight": torch.randn(2, 2)})

        ledger.store(
            task_id_scalar=1,
            device_context_tensor=torch.tensor([0.3, 0.4]),
            semantic_context_tensor=None,
            blueprint_signals_dict={"input_processor_names": ["features"], "head_name": "classifier"},
            weights_dict=weights,
        )

        cached_weights = next(iter(ledger.cache.values()))
        self.assertEqual(cached_weights["output.weight"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
