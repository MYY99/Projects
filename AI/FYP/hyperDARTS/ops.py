""" Ops """

from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn

import math
import torch


class HyperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_hyper,
                 stride=1, padding=0, groups=1, dilation=1, bias=False):
        """ Initialize a class StnConv2d.
        :param in_channels: int
        :param out_channels: int
        :param kernel_size: int or (int, int)
        :param num_hyper: int
        :param feature_map_wh: int
        :param stride: int or (int, int)
        :param padding: int or (int, int)
        :param groups: int
        :param dilation: int or (int, int)
        :param bias: bool
        """
        super(HyperConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if bias:
            raise ValueError("bias is true")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_hyper = num_hyper
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.hypernet_out_len = self.in_channels // self.groups * self.out_channels * self.kernel_size ** 2 \
                                + bias * self.out_channels

        # self.hypernet = nn.Linear(self.num_hyper, self.hypernet_out_len, bias=True)
        self.hypernet = nn.Sequential(
            nn.Linear(self.num_hyper, self.num_hyper // 2, bias=True),
            nn.Sigmoid(),
            nn.Linear(self.num_hyper // 2, self.hypernet_out_len, bias=True)
        )

    def forward(self, inputs, hparams):
        """ Returns a forward pass.
        :param inputs: Tensor of size 'batch_size x in_channels x height x width'
        :param h_net: Tensor of size 'batch_size x num_hyper'
        :return: Tensor of size 'batch_size x out_channels x height x width'
        """

        linear_hyper = self.hypernet(hparams)

        elem_weight = torch.reshape(linear_hyper, (self.out_channels, self.in_channels // self.groups,
                                                   self.kernel_size, self.kernel_size))

        output = F.conv2d(inputs, elem_weight, padding=self.padding, stride=self.stride,
                          groups=self.groups, dilation=self.dilation)

        return output


class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features, num_hyper, bias=True):
        """ Initialize a class StnLinear.
        :param in_features: int
        :param out_features: int
        :param num_hyper: int
        :param bias: bool
        """
        super(HyperLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_hyper = num_hyper

        self.hypernet_out_len = self.in_features * self.out_features + bias * self.out_features

        self.hypernet = nn.Linear(self.num_hyper, self.hypernet_out_len, bias=True)

    def forward(self, inputs, hparams):
        """ Returns a forward pass.
        :param inputs: Tensor of size 'batch_size x in_features'
        :param h_net: Tensor of size 'batch_size x num_hyper'
        :return: Tensor of size 'batch_size x out_features'
        """
        linear_hyper = self.hypernet(hparams)
        elem_weight = linear_hyper[:self.in_features * self.out_features] \
            .reshape((self.out_features, self.in_features))
        elem_bias = linear_hyper[self.in_features * self.out_features:]
        output = F.linear(inputs, elem_weight, elem_bias)  # Welem x + belem

        return output


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_hyper, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.hyperconv = HyperConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     num_hyper=num_hyper, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x, h_net):
        out = self.relu(x)
        out = self.hyperconv(out, h_net)
        out = self.bn(out)
        return out


class StdConvNoHyper(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, in_channels, out_channels, num_hyper, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.hyperconv1 = HyperConv2d(in_channels, out_channels // 2, kernel_size=1, num_hyper=num_hyper,
                                      stride=2, padding=0, bias=False)
        self.hyperconv2 = HyperConv2d(in_channels, out_channels // 2, kernel_size=1, num_hyper=num_hyper,
                                      stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x, h_net):
        out = self.relu(x)
        # oxoxox
        # xoxoxo
        out = torch.cat([self.hyperconv1(out, h_net), self.hyperconv2(out[:, :, 1:, 1:], h_net)], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduceNoHyper(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, num_hyper, affine=True):
        super().__init__()

        self.relu = nn.ReLU()
        self.hyperconv1 = HyperConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                      num_hyper=num_hyper, stride=stride, padding=padding, dilation=dilation,
                                      groups=in_channels, bias=False)
        self.hyperconv2 = HyperConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                      num_hyper=num_hyper, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x, h_net):
        out = self.relu(x)
        out = self.hyperconv1(out, h_net)
        out = self.hyperconv2(out, h_net)
        out = self.bn(out)
        return out


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_hyper, affine=True):
        super().__init__()
        self.dilconv1 = DilConv(in_channels, in_channels, kernel_size, stride, padding, dilation=1,
                                num_hyper=num_hyper, affine=affine)
        self.dilconv2 = DilConv(in_channels, out_channels, kernel_size, 1, padding, dilation=1,
                                num_hyper=num_hyper, affine=affine)

    def forward(self, x, h_net):
        out = self.dilconv1(x, h_net)
        out = self.dilconv2(out, h_net)
        return out


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x, h_net):
        out = self.pool(x)
        out = self.bn(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, h_net):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x, h_net):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.

