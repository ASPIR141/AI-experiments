import torch
import unittest
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import z_score_hardsquaremax


class ZScoreHardSquareMaxTestCase(unittest.TestCase):
    def setUp(self):
        self.z_score_hardsquaremax = z_score_hardsquaremax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_z_score_hardsquaremax(self):
        logits = [
            torch.tensor([[-60., 0., 10., 250.]], device=self.device)
        ]
        expected_results = [
            [[0.134, 0.197, 0.208, 0.461]]
        ]

        for idx, z in enumerate(logits):
            result = self.z_score_hardsquaremax(z)
            np.testing.assert_almost_equal(
                expected_results[idx], result.tolist(), decimal=3)


if __name__ == '__main__':
    unittest.main()
