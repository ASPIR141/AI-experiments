import torch
from itertools import product
from torch.tensor import Tensor
from typing import Iterable, List, Tuple


def confusion_layer(p: Tensor, l: Tensor, k: int) -> Tuple[List[float], List[str]]:
    """
    Arguments:
        p: Tensor, probabilities from softmax layer.
        l: Tensor, list of labels.
        k: Integer, a number of classes.

    Returns:
        probabilities, labels: Tuple[List[Float], List[Str]]
    """

    def combinations(iterable: Iterable, r: int):
        pool = tuple(iterable)
        n = len(pool)
        for indices in product(range(n), repeat=r):
            yield tuple(pool[i] for i in indices)

    def calculate_probabilities(array: List[Tuple[int]], v: int, tax: int) -> Tensor:
        """
        Arguments:
            array: [(1, 1), (1, 0), (0, 1), (0, 0)]
            v: Integer
            tax: Integer

        Returns:
            A list of Tensors: tensor([1., 0., 0., 0.])
        """

        probabilities = []
        for value in array:
            i, j = value
            if i == j:
                probabilities.append(i * v)
            else:
                probabilities.append(torch.tensor(0., device='cuda') if i == 1 else ((i * j) / (1 - i)) * tax)
        return torch.stack(probabilities)

    v = torch.sqrt((k / (k - 1)) * torch.sum((p - 1 / k) ** 2, dim=1))
    # print(f'V: {v}')
    tax = 1 - v
    # print(f'TAX: {tax}')

    # create all possible combinations
    c = [list(combinations(x, 2)) for x in p]
    labels = ([f'{x}_{y}' for x, y in product(l, l)])

    # calculate probabilities for each combination
    probabilities = [calculate_probabilities(
        t, v[idx], tax[idx]) for idx, t in enumerate(c)]

    # get indexes of the max probabilities, f.e. [tensor([0.1200, 0.0800, 0.3200, 0.4800]), tensor([0.1200, 0.4800, 0.3200, 0.0800])]
    max_probabilities_indexes = [torch.argmax(x).item() for x in probabilities]

    labels = [labels[i] for i in max_probabilities_indexes]
    probabilities = [round(torch.max(x).item(), 2) for x in probabilities]
    return probabilities, labels
