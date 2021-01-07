import torch
import unittest

import sys
[sys.path.append(i) for i in ['.', '..']]
from lib.modules.activation import hardmax
from lib.modules.layers import dummy_confusion_layer


class DummyLayerTestCase(unittest.TestCase):
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

        self.assertFalse()


if __name__ == '__main__':
    unittest.main()
