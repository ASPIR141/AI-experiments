import unittest

import torch

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import hardmax
from lib.modules.layers import dummy_confusion_layer


class LayerTestCase(unittest.TestCase):
    def setUp(self):
        self.hardmax = hardmax
        self.layer = dummy_confusion_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_labels(self):
        X_in = torch.tensor([
                            [0, 0, 0],
                            [1, 1, 1],
                            [-2, -2, -2],
                            [-1, 0, 1],
                            [-2, 1, 1],
                            [-10, -15, 5]], dtype=torch.float16, device=self.device)
        probabilities = self.hardmax(X_in)
        labels = ['0', '1', '2', '2', '1', '2']
        result = self.layer(probabilities, labels, 3)
        print(result)

        self.assertFalse()


    # def test_probabilities(self):
    #     probabilities = [
    #         torch.tensor([[0, 1], [0.8, 0.2], [0.75, 0.25], [0.6, 0.4], [
    #                      0.5, 0.5], [0.4, 0.6], [0.25, 0.75], [0.2, 0.8], [1, 0]], device=self.device),
    #     ]
    #     expected_results = [[1.0, 0.48, 0.38, 0.48, 0.5, 0.48, 0.38, 0.48, 1.0]]

    #     for idx, p in enumerate(probabilities):
    #         result, _ = self.layer(p, labels, 2)
    #         self.assertEqual(expected_results[idx], result)


if __name__ == '__main__':
    unittest.main()
