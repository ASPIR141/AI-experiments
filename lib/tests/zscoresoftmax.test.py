import torch
import unittest
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import z_score_softmax


class ZScoreSoftMaxTestCase(unittest.TestCase):
    def setUp(self):
        self.z_score_softmax = z_score_softmax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_z_score_softmax(self):
        logits = [
            torch.tensor([[-60., 0., 10., 250.]], device=self.device),
        ]
        expected_results = [
            [[0.06, 0.09, 0.10, 0.75]],
        ]

        for idx, z in enumerate(logits):
            result = self.z_score_softmax(z)
            np.testing.assert_almost_equal(
                expected_results[idx], result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
