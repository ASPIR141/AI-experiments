import torch
from torch import FloatTensor
from torch.tensor import Tensor
from typing import List, Tuple, Iterable


def naive_confusion_layer(p: Tensor, l: List, k: int) -> Tuple[List[float], List[str]]:
    """
    Arguments:
        p: Tensor, probabilities from softmax layer.
        l: List, list of labels.
        k: Integer, a number of classes.

    Returns:
        probabilities, labels: Tuple[List[Float], List[Str]]
    """
    device = p.device

    v = (p.std(dim=1, unbiased=True, keepdim=True)**2)*(k)
    tax = 1 - v

    v, t = v.to(device), tax.to(device)
    # probabilities = torch.einsum('bi,bj->bij', (p, p)) # batch outer product TODO
    mask = torch.eye(k, dtype=torch.bool).to(device)
    probabilities = (p*v).reshape(p.shape[0], 1, p.shape[1])*mask
    for i, vec in enumerate(p):
        outer = torch.where(vec == 1, torch.zeros(1).to(device), torch.outer(vec, vec)*tax[i]/(1-vec))
        probabilities[i] = torch.where(mask, probabilities[i], outer)

    probabilities, labels = zip(*(torch.topk(x.flatten(), 1) for x in probabilities))
    labels = list(map(lambda i: f'{l[i.item()//k]}_{l[i.item()%k]}', labels))
    probabilities = list(map(lambda x: round(x.item(), 2), probabilities))
    return probabilities, labels

def confusion_layer(p: Tensor, l: List, k: int) -> Tuple[List[float], List[str]]:
    """
    Arguments:
        p: Tensor, probabilities from softmax layer.
        l: List, list of labels.
        k: Integer, a number of classes.

    Returns:
        probabilities, labels: Tuple[List[Float], List[Str]]
    """
    device = p.device

    v = (p.std(dim=1, unbiased=True, keepdim=True))*(k**0.5)
    tax = 1 - v

    v, t = v.to(device), tax.to(device)
    # probabilities = torch.einsum('bi,bj->bij', (p, p)) # batch outer product TODOs
    mask = torch.eye(k, dtype=torch.bool).to(device)
    probabilities = (p*v).reshape(p.shape[0], 1, p.shape[1])*mask
    for i, vec in enumerate(p):
        outer = torch.where(vec == 1, torch.zeros(1).to(device), torch.outer(vec, vec)*tax[i]/(1-vec))
        probabilities[i] = torch.where(mask, probabilities[i], outer)

    probabilities, labels = zip(*(torch.topk(x.flatten(), 1) for x in probabilities))
    labels = list(map(lambda i: f'{l[i.item()//k]}_{l[i.item()%k]}', labels))
    probabilities = list(map(lambda x: round(x.item(), 2), probabilities))
    return probabilities, labels


def dummy_confusion_layer(probabilities: Tensor, labels: List, k: int) -> List[str]:
    new_labels = []
    lbls_to_idx = list(map(int, labels))
    max_probabilities_indexes = [torch.argmax(sample).item() for sample in probabilities]

    for row, (i, j) in enumerate(zip(lbls_to_idx, max_probabilities_indexes)):
        if probabilities[row][i] != probabilities[row][j]:
            new_labels.append(f'{i}_{j}')
        else:
            row_p = probabilities[row].clone()
            row_p[i] = -row_p[i]
            j = torch.argmax(row_p).item()
            if probabilities[row][i] <= 2*probabilities[row][j]:
                new_labels.append(f'{i}_{j}')
            else:
                new_labels.append(f'{i}_{i}')
    return new_labels