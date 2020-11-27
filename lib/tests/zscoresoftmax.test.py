import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import zScoreSoftMax


class ZScoreSoftMaxTestCase(unittest.TestCase):
    def setUp(self):
        self.zScoreSoftMax = zScoreSoftMax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_zScoreSoftMax(self):
        logits = [
            torch.tensor([[-60, 0, 10, 250]], dtype=torch.float16, device=self.device),
        ]
        expected_results = [
            [[0.05514, 0.09148, 0.0952, 0.7539]],
        ]

        for idx, z in enumerate(logits):
            result = self.zScoreSoftMax(z)
            np.testing.assert_almost_equal(expected_results[idx], result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
