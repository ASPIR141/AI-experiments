import torch
from torch.tensor import Tensor


def hardmax(input: Tensor) -> Tensor:
  r"""hardmax(input) -> Tensor

  Applies hardmax function element-wise.
  Elements will be probability distributions that sum to 1
  and in such a way, that makes it possible to avoid unnecessary
  non-linearity effect typical to the Softmax procedure.

  Args:
    input (Tensor): the input tensor.

  Returns:
    A tensor of the same shape as input.
  """

  x = input.clone()
  m = torch.sum(torch.abs(input), dim=1, keepdim=True)
  for idx, sum_i in enumerate(m):
    if sum_i == 0:
      m[idx] = 1
      x[idx][x == 0] = 1
  s = x.add(m) / (2 * m)
  output = s / torch.sum(s, dim=1, keepdim=True)
  return output


def hardsquaremax(input: Tensor) -> Tensor:
  x = input.clone()
  m = torch.sum(torch.sqrt(input), dim=1, keepdim=True)
  for idx, sum_i in enumerate(m):
    if sum_i == 0:
      m[idx] = 1
      x[idx][x == 0] = 1
  s = x.add(m) / (2 * m)
  output = s / torch.sum(s, dim=1, keepdim=True)
  return output