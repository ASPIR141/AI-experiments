import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import z_score_softmax


class z_score_softmaxTestCase(unittest.TestCase):
    def setUp(self):
        self.z_score_softmax = z_score_softmax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_z_score_softmax(self):
        logits = [
            torch.tensor([[-60, 0, 10, 250]], dtype=torch.float16, device=self.device),
        ]
        expected_results = [
            [[0.06, 0.09, 0.10, 0.75]],
        ]

        for idx, z in enumerate(logits):
            result = self.z_score_softmax(z)
            np.testing.assert_almost_equal(expected_results[idx], result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
