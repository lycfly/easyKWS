import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.pooling import AvgPool2d

from qnn import QuantConv, torchFixpoint
from torch.utils.tensorboard import SummaryWriter

def print_shape(block_name, shape, ifprint):
    if ifprint:
        print('{} INPUT SHAPE: {}'.format(block_name, shape))

class GRU_Model(torch.nn.Module):
    def __init__(self, label_num=12):
        super(GRU_Model, self).__init__()
        n_maps = 64
        self.bn0 = nn.BatchNorm2d(1)
        self.gru_layer1 = nn.GRU(input_size=40, hidden_size=n_maps, batch_first=True)
        self.gru_layer2 = nn.GRU(input_size=n_maps, hidden_size=n_maps, batch_first=True)
        self.bn1 = nn.BatchNorm1d(n_maps)
        self.fc = nn.Liner(in_features=n_maps, out_features=label_num)

    def forward(self, x):
        x = self.bn0(x)
        x = x.view(x.size(0),x.size(3),x.size(2))
        x, hidden = self.gru_layer1(x)
        x, hidden = self.gru_layer2(x)
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        x = x[:,:,-1]
        x = self.bn1(x)
        x = self.fc(x)
        return x

class ResModel(torch.nn.Module):
    def __init__(self):
        super(ResModel,self).__init__()
        n_labels = 12
        n_maps = 128
        self.conv0 = torch.nn.Conv2d((1, n_maps, (3,3)), padding=(1,1), bias=False)
        self.n_layers = n_layers = 9
        self.convs = torch.nn.ModuleList([torch.nn.conv2d(n_maps, n_maps,(3,3), padding=1, dilation=1,
                                        bias=False) for _ in range(n_layers)])
        self.pool = MaxPool2d(2, return_indices=True)
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i+1), torch.nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i+1), conv)
        self.output = torch.nn.Linear(n_maps, n_labels)

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
            pooling = False
            if pooling:
                x_pool, pool_indices = self.pool(x)
                x = self.unpool(x_pool, pool_indices, output_size=x.size())
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = self.output(x)
        return x

