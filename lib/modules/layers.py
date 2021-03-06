import torch
from torch import FloatTensor
from torch.tensor import Tensor
from typing import List, Tuple, Iterable
from itertools import product
from typing import Iterable


def combinations(iterable: Iterable, r: int):
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=r):
        yield tuple(pool[i] for i in indices)

def calculate_probabilities(array: FloatTensor, v: FloatTensor, tax: FloatTensor) -> Tensor:
    """
    Arguments:
        array: tensor([[[0., 0.],[0., 1.]], [[1., 0.],[1., 1.]]])
        v: Integer
        tax: Integer

    Returns:
        A list of Tensors: tensor([0., 0., 0., 1.])
    """

    probabilities = []
    for idx, a in enumerate(array):
        for i, value in enumerate(a):
            if idx == i:
                i, _ = value
                probabilities.append(i * v)
            else:
                i, j = value
                probabilities.append(torch.tensor(0., device='cuda' if torch.cuda.is_available() else 'cpu') if i == 1 else ((i * j) / (1 - i)) * tax)
    return torch.stack(probabilities)

def confusion_layer(p: Tensor, l: Tensor, k: int) -> Tuple[List[float], List[str]]:
    """
    Arguments:
        p: Tensor, probabilities from softmax layer.
        l: Tensor, list of labels.
        k: Integer, a number of classes.

    Returns:
        probabilities, labels: Tuple[List[Float], List[Str]]
    """

    v = torch.sqrt((k / (k - 1)) * torch.sum((p - 1 / k) ** 2, dim=1))
    tax = 1 - v

    # create all possible combinations and represent each combination as a tensor to find the diagonal elements
    # tensor([0., 1.]) -> tensor([[[0., 0.],[0., 1.]], [[1., 0.],[1., 1.]]])
    c = [torch.tensor(list(combinations(x, 2)), dtype=torch.float).view(k, -1, 2) for x in p]

    labels = ([f'{x}_{y}' for x, y in product(l, l)])

    # calculate probabilities for each combination
    probabilities = [calculate_probabilities(c, v[idx], tax[idx]) for idx, c in enumerate(c)]

    # get indexes of the max probabilities, f.e. [tensor([0.1200, 0.0800, 0.3200, 0.4800]), tensor([0.1200, 0.4800, 0.3200, 0.0800])]
    max_probabilities_indexes = [torch.argmax(x).item() for x in probabilities]
    labels = [labels[i] for i in max_probabilities_indexes]
    probabilities = [round(torch.max(x).item(), 2) for x in probabilities]
    return probabilities, labels
