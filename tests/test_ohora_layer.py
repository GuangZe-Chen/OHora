from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ohora_layer import OHoRALinear, compute_rhigh


class TestOHoRALinear(unittest.TestCase):
    def test_minimal_square_linear(self) -> None:
        torch.manual_seed(0)
        base_linear = nn.Linear(16, 16, bias=True)

        layer = OHoRALinear.from_linear(base_linear)

        self.assertEqual(layer.rhigh, compute_rhigh(16, 16))
        self.assertEqual(tuple(layer.Ahigh.shape), (4, 4))
        self.assertEqual(tuple(layer.Bhigh.shape), (4, 4))
        self.assertEqual(tuple(layer.wres.shape), tuple(base_linear.weight.shape))
        self.assertEqual(tuple(layer.delta_weight().shape), tuple(base_linear.weight.shape))

        trainable_params = {name for name, param in layer.named_parameters() if param.requires_grad}
        self.assertEqual(trainable_params, {"Ahigh", "Bhigh"})

        x = torch.randn(2, 3, 16)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 16))


if __name__ == "__main__":
    unittest.main()