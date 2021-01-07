import torch
from torch.tensor import Tensor


def hardmax(X_in: Tensor) -> Tensor:
    r"""hardmax(X_in) -> Tensor

    Applies hardmax function element-wise.
    Elements will be probability distributions that sum to 1
    and in such a way, that makes it possible to avoid unnecessary
    non-linearity effect typical to the Softmax procedure.

    Args:
        X_in (Tensor): the input tensor.

    Returns:
        A tensor of the same shape as input.
    """
    assert(len(X_in.shape) >= 2)

    m = torch.sum(torch.abs(X_in), dim=1, keepdim=True)
    for idx, sum_i in enumerate(m):
        if sum_i == 0:
            m[idx] = 1
            X_in[idx][X_in[idx] == 0] = 1
    s = X_in.add(m) / (2 * m)
    output = s / torch.sum(s, dim=1, keepdim=True)
    return output


def hardsquaremax(X_in: Tensor) -> Tensor:
    assert(len(X_in.shape) >= 2)
    m = torch.sqrt(torch.sum((torch.pow(X_in, 2)), dim=1, keepdim=True))
    for idx, sum_i in enumerate(m):
        if sum_i == 0:
            m[idx] = 1
            X_in[idx][X_in[idx] == 0] = 1
    s = X_in.add(m) / (2 * m)
    output = s / torch.sum(s, dim=1, keepdim=True)
    return output


def z_score(logits: Tensor) -> Tensor:
    '''
    Normalize logits
    '''
    assert(len(logits.shape) >= 2)

    mean = torch.mean(logits, dim=1, keepdim=True)
    deviation = torch.sqrt(
        torch.sum(torch.pow(logits - mean, 2), dim=1, keepdim=True)/logits.size()[1])
    z_norm = (logits - mean)/deviation
    return z_norm


def z_score_hardsquaremax(logits: Tensor) -> Tensor:
    '''
    logits: Tensor, k-dimensional output from last layer.
        Each value is  a score defined on the interval (-inf, +inf)

    Sum of probalities = 1.
    Makes it possible to compromise between SoftMax and HardMax.
    '''
    z_norm = z_score(logits)
    probabilities = hardsquaremax(z_norm)
    return probabilities


def z_score_hardmax(logits: Tensor) -> Tensor:
    '''
    logits: Tensor, k-dimensional output from last layer.
        Each value is  a score defined on the interval (-inf, +inf)

    Sum of probalities = 1.
    Makes it possible to compromise between SoftMax and HardMax.
    '''
    z_norm = z_score(logits)
    probabilities = hardmax(z_norm)
    return probabilities


def z_score_softmax(logits: Tensor) -> Tensor:
    '''
    logits: Tensor, k-dimensional output from last layer.
        Each value is  a score defined on the interval (-inf, +inf)

    Sum of probalities = 1.
    Makes it possible to soften the effect "winners takes all" of SoftMax.
    '''
    z_norm = z_score(logits)
    probabilities = torch.nn.functional.softmax(z_norm, dim=1)
    return probabilities
