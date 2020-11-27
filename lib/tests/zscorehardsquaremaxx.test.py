import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import zScoreHardSquareMax


class ZScoreHardSquareMaxTestCase(unittest.TestCase):
    def setUp(self):
        self.zScoreHardSquareMax = zScoreHardSquareMax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_zScoreHardSquareMax(self):
        logits = [
            torch.tensor([[-60, 0, 10, 250]], dtype=torch.float16, device=self.device),
            torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float16, device=self.device),
        ]
        expected_results = [
            [[0.13, 0.20, 0.21, 0.46]],
            [[0.10, 0.33, 0.57]],
        ]

        for idx, z in enumerate(logits):
            result = self.zScoreHardSquareMax(z)
            np.testing.assert_almost_equal(expected_results[idx], result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
