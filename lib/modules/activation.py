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

    x = X_in.clone()
    m = torch.sum(torch.abs(x), dim=1, keepdim=True)
    for idx, sum_i in enumerate(m):
        if sum_i == 0:
            m[idx] = 1
            x[idx][x[idx] == 0] = 1
    s = x.add(m) / (2 * m)
    output = s / torch.sum(s, dim=1, keepdim=True)
    return output


def hardsquaremax(X_in: Tensor) -> Tensor:
    x = X_in.clone()
    m = torch.sqrt(torch.sum((torch.pow(x, 2)), dim=1, keepdim=True))
    for idx, sum_i in enumerate(m):
        if sum_i == 0:
            m[idx] = 1
            x[idx][x[idx] == 0] = 1
    s = x.add(m) / (2 * m)
    output = s / torch.sum(s, dim=1, keepdim=True)
<<<<<<< HEAD
    return output


def zScore(logits: Tensor) -> Tensor:
    '''
    Normalize logits
    '''
    mean1 = torch.mean(logits)
    k = len(logits)
    mean2 = torch.sum([z/k for z in logits])
    print(mean1, mean2)
    deviation = torch.sqrt(torch.sum([torch.pow(z-mean1, 2) for z in logits])/k)
    z_norm = [(z-mean1)/deviation for z in logits]
    return z_norm


def zScoreHardSquareMax(logits: Tensor) -> Tensor:
    '''
    logits: Tensor, k-dimensional output from last layer. 
        Each value is  a score defined on the interval (-inf, +inf)
    
    Sum of probalities = 1.
    Makes it possible to compromise between SoftMax and HardMax.
    '''
    # stage 1 z-score normalization
    z_norm = zScore(logits)
    # stage 2 hardsquaremax
    probabilities = hardsquaremax(z_norm)
    return probabilities

=======
    return output
>>>>>>> refactor
