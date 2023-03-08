import torch
from mmengine.model import bias_init_with_prob
from torch.nn import Module


class ScaleLayer(Module):
    def __init__(self, scale=True, bias=True, use_input=False, in_channels=None):
        super().__init__()

        self.use_input = use_input
        if self.use_input:
            self.scale = torch.nn.Linear(in_channels, 1, bias=True)
            self.shift = torch.nn.Linear(in_channels, 1, bias=True)

            torch.nn.init.normal_(self.scale.weight, std=0.01)
            torch.nn.init.constant_(self.scale.bias, 0)

            torch.nn.init.normal_(self.shift.weight, std=0.01)
            torch.nn.init.constant_(self.shift.bias, bias_init_with_prob(0.01))
        else:
            self.weight = torch.nn.Parameter(torch.FloatTensor([1])) if scale else None
            self.bias = torch.nn.Parameter(torch.FloatTensor([0])) if bias else None

    def forward(self, x, y):
        if self.use_input:
            scale = self.scale(y)
            scale = torch.nn.functional.elu(scale) + 1
            shift = self.shift(y)
            x = x * scale + shift
        else:
            if self.weight is not None:
                x *= self.weight
            if self.bias is not None:
                x += self.bias

        return x
