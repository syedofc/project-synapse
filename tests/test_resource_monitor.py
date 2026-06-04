import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.resource_monitor import ResourceMonitor


class TestResourceMonitor(unittest.TestCase):
    def test_override_context_is_returned_exactly(self):
        monitor = ResourceMonitor(context_dim=4, mode="zeros")
        override = torch.tensor([0.1, 0.2, 0.3, 0.4])
        monitor.set_context_override(override)
        returned = monitor.get_context()
        self.assertTrue(torch.allclose(returned, override))
        self.assertEqual(monitor.last_source, "override")

    def test_zero_dim_context_is_supported(self):
        monitor = ResourceMonitor(context_dim=0)
        context = monitor.get_context()
        self.assertEqual(context.numel(), 0)
        self.assertEqual(monitor.last_source, "zero-dim")


if __name__ == "__main__":
    unittest.main()
