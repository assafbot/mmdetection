import torch
from torch.nn import Module


class ScaleLayer(Module):
    def __init__(self, scale=True, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor([1])) if scale else None
        self.bias = torch.nn.Parameter(torch.FloatTensor([0])) if bias else None

    def forward(self, x):
        if self.weight is not None:
            x *= self.weight
        if self.bias is not None:
            x += self.bias

        return x
