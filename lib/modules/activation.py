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
  return output