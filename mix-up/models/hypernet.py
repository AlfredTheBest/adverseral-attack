import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.resnet import ResNet18
import numpy as np


class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()
        self.net = ResNet18()
        # self.main_percent = torch.tensor([[0.8]]).cuda()


    def forward(self, x, train=False):
        hyper = F.sigmoid(self.hyper)
        batch_size = x.size()[0]
        # hyper = torch.cat([self.main_percent, hyper], 0)
        lam = np.random.beta(1, 1)
        if train:
            out = x.reshape(batch_size, -1)

            if batch_size != 128:
                hyper = hyper[:batch_size, ]

            noise = out * hyper
            out = out*lam + noise
            out = out.reshape(batch_size, 3, 32, 32)
            out = self.net(out)
            return out, lam, hyper
        else:
            out = self.net(x)
            return out

