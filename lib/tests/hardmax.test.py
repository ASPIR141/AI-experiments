import unittest

import torch
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import hardmax


class HardmaxTestCase(unittest.TestCase):
    def setUp(self):
        self.hardmax = hardmax

    def test_hardmax(self):
        input = torch.tensor([
                            [0, 0, 0],
                            [1, 1, 1],
                            [-2, -2, -2],
                            [-1, 0, 1],
                            [-2, 1, 1],
                            [-10, -15, 5]], dtype=torch.float16)
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
