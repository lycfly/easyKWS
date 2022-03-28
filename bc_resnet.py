import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.pooling import AvgPool2d
from myBatchNorm import *

from qnn import QuantConv, torchFixpoint
from torch.utils.tensorboard import SummaryWriter

def print_shape(block_name, shape, ifprint):
    if ifprint:
        print('{} INPUT SHAPE: {}'.format(block_name, shape))

class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)

class mySubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(mySubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = MyBatchnorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class myQSubSpectralNorm(nn.Module):
    def __init__(self, C, S, qlistm=None,qlistb=None):
        super(myQSubSpectralNorm, self).__init__()
        self.S = S
        eps=1e-5
        self.eps = eps
        self.bn = MyQBatchnorm2d(C*S, qlistm, qlistb)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)

class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.0) # 0.5
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.0) #0.5
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(self, label_num=12, finetune = False):
        super(BCResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32, label_num, 1, bias=False)

    def forward(self, x):
        ifprint = False
        print_shape('INPUT SHAPE:', x.shape, ifprint)
        out = self.conv1(x)

        print_shape('BLOCK1 INPUT SHAPE:', out.shape, ifprint)
        out = self.block1_1(out)
        out = self.block1_2(out)

        print_shape('BLOCK2 INPUT SHAPE:', out.shape, ifprint)
        out = self.block2_1(out)
        out = self.block2_2(out)

        print_shape('BLOCK3 INPUT SHAPE:', out.shape, ifprint)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        print_shape('BLOCK4 INPUT SHAPE:', out.shape, ifprint)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        print_shape('Conv2 INPUT SHAPE:', out.shape, ifprint)
        out = self.conv2(out)

        print_shape('Conv3 INPUT SHAPE:', out.shape, ifprint)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        print_shape('Conv4 INPUT SHAPE:', out.shape, ifprint)
        out = self.conv4(out)
        out = out.view(out.size(0), out.size(1))
        print_shape('OUTPUT SHAPE:', out.shape, ifprint)
        return out

'''
    my BC-Resnet
'''
class myBroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            sub : int,
            dilation=1,
            stride=1,
            temp_pad=(0, 0),
    ) -> None:
        super(myBroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, sub)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.0) # 0.5
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class myTransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            sub : int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(myTransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, sub)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.swish = nn.SiLU()
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.freq_dw_conv(x)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out

class myBCResNet(torch.nn.Module):
    def __init__(self, label_num=12, finetune = False):
        super(myBCResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))

        self.block1_1 = myBroadcastedBlock(planes=16, sub=5, temp_pad=(0, 1))
        self.block1_2 = myBroadcastedBlock(planes=16, sub=5, temp_pad=(0, 1))

        self.pooling1 = myTransitionBlock( planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2) ,stride=(2, 1))
        self.block2_1 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2))

        self.pooling2 = myTransitionBlock( planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4) ,stride=(2, 1))
        self.block3_2 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = myBroadcastedBlock(planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 8))

        self.conv4 = nn.Conv2d(16, label_num, (5,1), bias=False)

    def forward(self, x):
        ifprint = False
        print_shape('INPUT SHAPE:', x.shape, ifprint)
        out = self.conv1(x)

        print_shape('BLOCK1 INPUT SHAPE:', out.shape, ifprint)
        out = self.block1_1(out)
        out = self.block1_2(out)

        print_shape('BLOCK2 INPUT SHAPE:', out.shape, ifprint)
        out = self.pooling1(out)
        out = self.block2_1(out)
        out = self.block2_2(out)

        print_shape('BLOCK3 INPUT SHAPE:', out.shape, ifprint)
        out = self.pooling2(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        print_shape('BLOCK4 INPUT SHAPE:', out.shape, ifprint)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = out.mean(-1, keepdim=True)

        print_shape('Conv4 INPUT SHAPE:', out.shape, ifprint)
        out = self.conv4(out)
        out = out.view(out.size(0), out.size(1))
        print_shape('OUTPUT SHAPE:', out.shape, ifprint)
        return out


'''
    my quant BC-Resnet
'''
class myQBroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            sub : int,
            dilation=1,
            stride=1,
            temp_pad=(0, 0),
            weightq = None,
            actq = None,
            actq_sram = None,
            bnmq = None,
            bnbq = None,

    ) -> None:
        super(myQBroadcastedBlock, self).__init__()
        self.actq = actq
        self.actq_sram = actq_sram
        self.freq_dw_conv = QuantConv(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=1, stride=stride, bias=False, qlist = weightq)
        #self.ssn1 = SubSpectralNorm(planes, sub)
        self.ssn1 = myQSubSpectralNorm(planes, sub, bnmq, bnbq)
        self.temp_dw_conv = QuantConv(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=1, bias=False, qlist = weightq)
        #self.bn = nn.BatchNorm2d(planes)
        self.bn = MyQBatchnorm2d(planes, bnmq, bnbq)
        self.relu = nn.ReLU(inplace=True)
        #self.swish = nn.SiLU()
        self.swish = nn.ReLU()
        self.conv1x1 = QuantConv(planes, planes, kernel_size=(1, 1), bias=False, qlist = weightq)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = torchFixpoint(out, self.actq)
        out = self.ssn1(out)
        out = torchFixpoint(out, self.actq)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling
        out = torchFixpoint(out, self.actq)
        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = torchFixpoint(out, self.actq)
        out = self.bn(out)
        out = torchFixpoint(out, self.actq)
        out = self.swish(out)
        out = torchFixpoint(out, self.actq)
        out = self.conv1x1(out)
        out = torchFixpoint(out, self.actq)
        ############################
        out = out + auxilary
        out = torchFixpoint(out, self.actq_sram)
        out = out + identity
        out = torchFixpoint(out, self.actq)
        out = self.relu(out)

        return out


class myQTransitionBlock(nn.Module):

    def __init__(
            self,
            planes: int,
            sub : int,
            dilation=(1,1),
            stride=(1,1),
            temp_pad=(0, 0),
            weightq = None,
            actq = None,
            actq_sram = None,
            bnmq = None,
            bnbq = None,
    ) -> None:
        super(myQTransitionBlock, self).__init__()
        self.actq = actq
        self.actq_sram = actq_sram
        self.freq_dw_conv = QuantConv(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride, dilation=dilation, bias=False, qlist = weightq)
        #self.ssn = SubSpectralNorm(planes, sub)
        self.ssn = myQSubSpectralNorm(planes, sub, bnmq, bnbq)
        self.temp_dw_conv = QuantConv(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=1, bias=False, qlist = weightq)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = MyQBatchnorm2d(planes, bnmq, bnbq)

        self.relu = nn.ReLU(inplace=True)
        #self.swish = nn.SiLU()
        self.swish = nn.ReLU()
        self.conv1x1_2 = QuantConv(planes, planes, kernel_size=(1, 1), bias=False, qlist = weightq)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.freq_dw_conv(x)
        out = torchFixpoint(out,self.actq)
        out = self.ssn(out)
        out = torchFixpoint(out,self.actq)
        #############################
        auxilary = torchFixpoint(out,self.actq_sram)
        out = out.mean(2, keepdim=True)  # frequency average pooling
        out = torchFixpoint(out,self.actq)
        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = torchFixpoint(out,self.actq)
        out = self.bn2(out)
        out = torchFixpoint(out,self.actq)
        out = self.swish(out)
        out = torchFixpoint(out,self.actq)
        out = self.conv1x1_2(out)
        out = torchFixpoint(out,self.actq)
        #############################

        out = auxilary + out
        out = torchFixpoint(out,self.actq)
        out = self.relu(out)

        return out

class myQBCResNet(torch.nn.Module):
    def __init__(self, label_num=12, finetune = False):
        super(myQBCResNet, self).__init__()
        self.weightq = [1,8,6]
        self.actq_sram = [1,8,4]
        self.actq = [1,16,9]
        if finetune:
            self.bnmq = [1,8,4]
        else:
            self.bnmq = [1,16,9]
        self.bnbq = [1,16,9]
        self.conv1 = QuantConv(1, 16, 5, stride=(2, 1), padding=(2, 2), qlist = self.weightq)

        self.block1_1 = myQBroadcastedBlock(planes=16, sub=5, temp_pad=(0, 1), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block1_2 = myQBroadcastedBlock(planes=16, sub=5, temp_pad=(0, 1), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)

        self.block2_1 = myQTransitionBlock (planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2), stride=(2, 1), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block2_2 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block2_3 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 2), temp_pad=(0, 2), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)

        self.block3_1 = myQTransitionBlock( planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4) ,stride=(2, 1), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block3_2 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block3_3 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block3_4 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 4), temp_pad=(0, 4), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)

        self.block4_1 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 8), temp_pad=(0, 8), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block4_2 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 8), temp_pad=(0, 8), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)
        self.block4_3 = myQBroadcastedBlock(planes=16, sub=5, dilation=(1, 8), temp_pad=(0, 8), weightq=self.weightq, actq=self.actq, actq_sram=self.actq_sram, bnmq=self.bnmq, bnbq=self.bnbq)

        self.conv4 = QuantConv(16, label_num, (5,1), bias=False, qlist=self.weightq)
        self.writer = None
        self.step = 0
        self.ifwriter = False
    def forward(self, x):
        ifprint = False
        x = x / (2**6)
        print_shape('INPUT SHAPE:', x.shape, ifprint)
        out = self.conv1(x)
        out = torchFixpoint(out, self.actq)

        print_shape('BLOCK1 INPUT SHAPE:', out.shape, ifprint)
        out = self.block1_1(out)
        out = self.block1_2(out)

        print_shape('BLOCK2 INPUT SHAPE:', out.shape, ifprint)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block2_3(out)

        print_shape('BLOCK3 INPUT SHAPE:', out.shape, ifprint)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        print_shape('BLOCK4 INPUT SHAPE:', out.shape, ifprint)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)

        out = out.mean(-1, keepdim=True)

        print_shape('Conv4 INPUT SHAPE:', out.shape, ifprint)
        out = self.conv4(out)
        out = out.view(out.size(0), out.size(1))
        print_shape('OUTPUT SHAPE:', out.shape, ifprint)
        return out

