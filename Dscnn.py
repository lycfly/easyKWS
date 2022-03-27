import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.pooling import AvgPool2d

from qnn import QuantConv, torchFixpoint
from torch.utils.tensorboard import SummaryWriter

class mySubSpectralNorm(nn.Module):
    def __init__(self, C, S, BW, eps=1e-5):
        super(mySubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.BW = BW
        self.bn = nn.BatchNorm2d(C*S)
    
    def forward(self, x):
        N, C, F, T = x.size()
        padnum = self.BW * self.S -F
        if(padnum < 0):
            raise('SubSpectral padding < 0!')
        x = x.view(N,C,T,F)
        x = torch.nn.functional.pad(x, pad=(0,padnum), mode='constant', value=0)
        x = x.view(N, C, F + padnum, T)
        x = x.view(N, C * self.S, self.BW, T)

        x = self.bn(x)
        x = x.view(N, C, self.BW * self.S, T)
        return x[:N,:C,:F,:T]
                 

class DS_Block(nn.Module):
    def __init__(
        self,
        planes: int,
        ssn_band : int,
        stride = 1,
        temp_pad = (0,0),
    ) -> None:
        super(DS_Block, self).__init__()
        self.conv = nn.Conv2d(planes, planes, kernel_size=(3,3), groups=planes, stride=stride, padding=temp_pad)
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ssn = SubSpectralNorm(planes, ssn_band)

    def forward(self, x:Tensor) -> Tensor:
        conv_ds = self.conv(x)
        conv_ps = self.conv1x1(conv_ds)
        bnout = self.ssn(conv_ps)
        out = self.relu(bnout)
        return out
    
class DSCNN(torch.nn.Module):
    def __init__(self, label_num=12):
        super(DSCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5,10), stride=(2,2), padding=(0,0))
        planes = 64
        self.bn_block_1 = DS_Block(planes=planes,stride=1,temp_pad=(0,0), ssn_band=4)
        self.bn_block_2 = DS_Block(planes=planes,stride=1,temp_pad=(0,0), ssn_band=7)
        self.bn_block_3 = DS_Block(planes=planes,stride=1,temp_pad=(0,0), ssn_band=6)
        self.bn_block_4 = DS_Block(planes=planes,stride=1,temp_pad=(0,0), ssn_band=5)
        self.convout = nn.Conv2d(planes, label_num, 1, stride=1, padding=(0,0))
    def forward(self, x):
        ifprint=False
        
        print_shape("Input layer", x.shape, ifprint)
        out = self.conv1(x)

        print_shape("BLOCK1", out.shape, ifprint)
        out = self.bn_block_1(x)

        print_shape("BLOCK2", out.shape, ifprint)
        out = self.bn_block_2(x)

        print_shape("BLOCK3", out.shape, ifprint)
        out = self.bn_block_3(x)

        print_shape("BLOCK4", out.shape, ifprint)
        out = self.bn_block_4(x)

        print_shape("Mean", out.shape, ifprint)
        out = out.mean(-1, keepdim=True)
        out = out.mean(-2, keepdim=True)

        print_shape("CONV_OUT", out.shape, ifprint)
        out = self.convout(out)
        out = out.view(out.size(0), out.size(1))
        print_shape("OUT", out.shape, ifprint)
        return out 


class qDS_Block(nn.Module):
    def __init__(
        self,
        planes: int,
        ssn_band : int,
        ssn_bw : int,
        weightq = None,
        actq = None,
        stride = 1,
        temp_pad = (0,0),
    ) -> None:
        super(qDS_Block, self).__init__()
        self.actq = actq
        self.conv = QuantConv(planes, planes, kernel_size=(3,3), groups=planes, stride=stride, padding=temp_pad, bias=False, qlist = weightq)
        self.conv1x1 = QuantConv(planes, planes, kernel_size=(1,1), bias=False, qlist = weightq)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ssn = mySubSpectralNorm(planes, ssn_band, ssn_bw)

    def forward(self, x:Tensor) -> Tensor:
        conv_ds = self.conv(x)
        conv_ps = self.conv1x1(conv_ds)
        out = self.ssn(conv_ps)
        out = torchFixpoint(out, self.actq)
        out = self.relu(out)
        return out
    
class qDSCNN(torch.nn.Module):
    def __init__(self, label_num=12):
        super(qDSCNN, self).__init__()
        planes = 32
        self.weightq = [1,8,6]
        self.actq = [1,8, 6]
        self.firstbn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, planes, (5,10), stride=(2,2), padding=(0,0), bias=False, qlist=self.weightq)
        self.bn_block_1 = qDS_Block(planes=planes,stride=1,temp_pad=(0,0), weightq=self.weightq, actq=self.actq, ssn_band=4, ssn_bw = 2)
        self.bn_block_2 = qDS_Block(planes=planes,stride=1,temp_pad=(0,0), weightq=self.weightq, actq=self.actq, ssn_band=8, ssn_bw = 2)
        self.bn_block_3 = qDS_Block(planes=planes,stride=1,temp_pad=(0,0), weightq=self.weightq, actq=self.actq, ssn_band=7, ssn_bw = 2)
        self.convout = QuantConvd(planes, label_num, 1, stride=1, padding=(0,0),bias=False, qlist=self.weightq)
        self.writer = None
        self.step = 0
        self.ifwriter = False

    def forward(self, x):
        ifprint=False
        x = x / 2**6
        print_shape("Input layer", x.shape, ifprint)
        out = self.conv1(x)

        print_shape("BLOCK1", out.shape, ifprint)
        out = self.bn_block_1(x)

        print_shape("BLOCK2", out.shape, ifprint)
        out = self.bn_block_2(x)

        print_shape("BLOCK3", out.shape, ifprint)
        out = self.bn_block_3(x)

        print_shape("Mean", out.shape, ifprint)
        out = out.mean(-1, keepdim=True)
        out = out.mean(-2, keepdim=True)

        print_shape("CONV_OUT", out.shape, ifprint)
        out = self.convout(out)
        out = out.view(out.size(0), out.size(1))
        print_shape("OUT", out.shape, ifprint)
        return out 
        