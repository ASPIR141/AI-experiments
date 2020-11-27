import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import zScore, zScoreHardSquareMax


class ZScoreHardSquareMaxTestCase(unittest.TestCase):
    def setUp(self):
        # self.zScore = zScore
        self.zScoreHardSquareMax = zScoreHardSquareMax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # def test_zScore(self):
    #     pass

    def test_zScoreHardSquareMax(self):
        logits = torch.tensor([-60, 0, 10, 250], dtype=torch.float16, device=device)
        expected_results = [[0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.1666, 0.3333, 0.5   ],
                            [0.1666, 0.4167, 0.4167],
                            [0.286 , 0.2144, 0.5005]]

        result = self.hardmax(input)
        np.testing.assert_almost_equal(expected_results, result.tolist(), decimal=2)


if __name__ == '__main__':
    unittest.main()
