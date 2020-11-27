import unittest

import torch

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.layers import confusion_layer


class LayerTestCase(unittest.TestCase):
    def setUp(self):
        self.layer = confusion_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_probabilities(self):
        probabilities = [
            torch.tensor([[0, 1], [0.8, 0.2], [0.75, 0.25], [0.6, 0.4], [
                         0.5, 0.5], [0.4, 0.6], [0.25, 0.75], [0.2, 0.8], [1, 0]], device=self.device),
        ]
        labels = ['0', '1']
        expected_results = [[1.0, 0.48, 0.38, 0.48, 0.5, 0.48, 0.38, 0.48, 1.0]]

        for idx, p in enumerate(probabilities):
            result, _ = self.layer(p, labels, 2)
            self.assertEqual(expected_results[idx], result)


if __name__ == '__main__':
    unittest.main()
