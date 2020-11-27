import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import zScoreHardMax


class ZScoreHardMaxTestCase(unittest.TestCase):
    def setUp(self):
        self.zScoreHardMax = zScoreHardMax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_zScoreHardMax(self):
        logits = [
            torch.tensor([[-60, 0, 10, 250]], dtype=torch.float16, device=self.device),
        ]
        expected_results = [
            [[0.18, 0.22, 0.23, 0.37]],
        ]

        for idx, z in enumerate(logits):
            result = self.zScoreHardMax(z)
            np.testing.assert_almost_equal(expected_results[idx], result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
