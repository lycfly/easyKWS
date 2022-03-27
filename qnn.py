import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch import Tensor
import torch.nn as nn
from typing import Union, Tuple, Type, Optional

def torchFixpoint(tensor, qlist):
    s, w, f = qlist
    base = torch.tensor(2**f)
    mul = torch.multiply(tensor, base)
    mul_round = torch.round(mul)
    clip = torch.clip(mul_round, -2**(w-s), 2**(w-s)-1)
    quant = torch.div(clip, base)
    return tensor + (quant - tensor).detach()

class QuantConv(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        out_channels:int,
        kernel_size:Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation:Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_type: str = 'standard',
        qlist: Tuple[int,int,int] = [1,8,6],
    ):
        super(QuantConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_type = padding_type
        self.qlist = qlist
        self.conv_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
            )
    def forward(self, x):
        weight = self.conv_module.weight
        self.quant_weight = torchFixpoint(weight, self.qlist)
        qconv = F.conv2d(
            input=x,
            weight=self.quant_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
            )
        return qconv

if __name__ == '__main__':
    x = torch.randn(1,1,4,4)
    quantconv = QuantConv(1, 1, 3,1)
    a = quantconv(x).sum().backward()

    quantconv.zero_grad()
    quantconv.qlist = [1,4,3]
    a = quantconv(x).sum().backward()
    print(quantconv.quant_weight)