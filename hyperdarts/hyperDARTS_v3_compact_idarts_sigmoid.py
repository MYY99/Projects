#!/usr/bin/env python
# coding: utf-8

# ## Utils

# In[1]:


import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def save_checkpoint(epoch, model, phi_optimizer, hparam_optimizer, hparams, phi_lr_scheduler, seed, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'phi_optimizer': phi_optimizer.state_dict(),
        'hparam_optimizer': hparam_optimizer.state_dict(),
        'hparams': hparams,
        'phi_lr_scheduler': phi_lr_scheduler.state_dict(),
        'seed': seed
        }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
        

def print_hparams(logger):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
        org_formatters.append(handler.formatter)
        handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info("####### HPARAMS #######")
    logger.info("# Hparams - normal")
    logger.info(hparams[:4])
    for hp in hparams[:4]:
        logger.info(F.softmax(hp.flatten(), dim=-1).reshape(hp.shape))

    logger.info("\n# Hparams - reduce")
    logger.info(hparams[4:])
    for hp in hparams[4:]:
        logger.info(F.softmax(hp.flatten(), dim=-1).reshape(hp.shape))
    logger.info("#####################")
    
    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
        handler.setFormatter(formatter)


# ## Data Loader

# In[2]:


""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
# import preproc


def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        num_class = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        num_class = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        num_class = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, num_class, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform


# ## Genotype

# In[4]:


from collections import namedtuple
import sys
# from graphviz import Digraph

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',  # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]

def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

def create_genotype(hparams_normal, hparams_reduce, num_int_nodes, k=2):
    gene_normal = parse(hparams_normal, k=k)
    gene_reduce = parse(hparams_reduce, k=k)
    concat = range(2, 2 + num_int_nodes)  # concat all intermediate nodes

    return Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)

# def plot(genotype, file_path, caption=None):
#     """ make DAG plot and save to file_path as .png """
#     edge_attr = {
#         'fontsize': '20',
#         'fontname': 'times'
#     }
#     node_attr = {
#         'style': 'filled',
#         'shape': 'rect',
#         'align': 'center',
#         'fontsize': '20',
#         'height': '0.5',
#         'width': '0.5',
#         'penwidth': '2',
#         'fontname': 'times'
#     }
#     g = Digraph(
#         format='png',
#         edge_attr=edge_attr,
#         node_attr=node_attr,
#         engine='dot')
#     g.body.extend(['rankdir=LR'])

#     # input nodes
#     g.node("c_{k-2}", fillcolor='darkseagreen2')
#     g.node("c_{k-1}", fillcolor='darkseagreen2')

#     # intermediate nodes
#     n_nodes = len(genotype)
#     for i in range(n_nodes):
#         g.node(str(i), fillcolor='lightblue')

#     for i, edges in enumerate(genotype):
#         for op, j in edges:
#             if j == 0:
#                 u = "c_{k-2}"
#             elif j == 1:
#                 u = "c_{k-1}"
#             else:
#                 u = str(j-2)

#             v = str(i)
#             g.edge(u, v, label=op, fillcolor="gray")

#     # output node
#     g.node("c_{k}", fillcolor='palegoldenrod')
#     for i in range(n_nodes):
#         g.edge(str(i), "c_{k}", fillcolor="gray")

#     # add image caption
#     if caption:
#         g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

#     g.render(file_path, view=False)


# ## Config

# In[5]:


import argparse
import os
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text
    
class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', default='cifar10_experiment74')

        # data
        parser.add_argument('--dataset', type=str, default='CIFAR10')
        parser.add_argument("--percent_valid", type=float, default=0.5)
        parser.add_argument("--percent_train", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=64)

        # environment setting
        parser.add_argument("--seed", type=int, default=2, help='random seed')

        # dataloader setting
        parser.add_argument('--num_workers', type=int, default=4, help='# of workers')

        # training details
        parser.add_argument("--total_epochs", type=int, default=50)

        # phi
        parser.add_argument('--phi_lr', type=float, default=0.05, help='lr for phi')
        parser.add_argument('--phi_lr_min', type=float, default=0.001, help='minimum lr for phi')
        parser.add_argument('--phi_momentum', type=float, default=0.9, help='momentum for phi')
        parser.add_argument('--phi_weight_decay', type=float, default=3e-4, help='weight decay for phi')
        parser.add_argument('--phi_grad_clip', type=float, default=5., help='gradient clipping for phi')

        # hparam
        parser.add_argument('--hparam_lr', type=float, default=6e-4, help='lr for hparam')
        parser.add_argument('--hparam_beta_1', type=float, default=0.5)
        parser.add_argument('--hparam_beta_2', type=float, default=0.999)
        parser.add_argument('--hparam_weight_decay', type=float, default=1e-3, help='weight decay for hparam')

        # network details
        parser.add_argument("--num_cells", type=int, default=8)
        parser.add_argument("--num_int_nodes", type=int, default=4)
        parser.add_argument("--init_channels", type=int, default=16)
        parser.add_argument("--num_ops", type=int, default=8)

        # logger
        parser.add_argument('--print_freq', type=int, default=100, help='print frequency')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args, unknown = parser.parse_known_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        # self.plot_path = os.path.join(self.path, 'plots')


# ## Ops

# In[6]:


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

        elem_weight = torch.reshape(linear_hyper, (self.out_channels, self.in_channels//self.groups, 
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


# ## Network Architecture

# In[7]:

class Cell(nn.Module):
    def __init__(self, c_pp, c_p, c_int_node, num_hyper, num_int_nodes=4, reduction_p=False, reduction=False):
        super(Cell, self).__init__()
        
        self.c_pp = c_pp
        self.c_p = c_p
        self.c_int_node = c_int_node
        self.num_hyper = num_hyper
        self.num_int_nodes = num_int_nodes
        self.reduction_p = reduction_p
        self.reduction = reduction
        
        # cell preproc
        self.preproc0 = self.__cell_preproc(self.c_pp, self.c_int_node, reduction_p)
        self.preproc1 = self.__cell_preproc(self.c_p, self.c_int_node)
        
        # cell intermediate nodes
        self.nodes = nn.ModuleList()
        for int_node_th in range(self.num_int_nodes):
            input_edges = nn.ModuleList()
            for parent_node_th in range(int_node_th+2):
                stride = 2 if self.reduction and parent_node_th < 2 else 1
                input_edges.append(self.__ops(channels=self.c_int_node, stride=stride))
            
            self.nodes.append(input_edges)
            
    def forward(self, s0, s1, hps_flatten, hp_normal):
        
        # cell preproc
#         print(hps_flatten)
#         print(hp_normal)
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        
        # cell intermediate nodes
        # output from each intermediate node
        out_nodes = []
        for int_node_th in range(self.num_int_nodes):
            # input after preprocessing
#             print(int_node_th)
            out_node = sum(hp * op(s0, hps_flatten) 
                           for hp, op in zip(hp_normal[int_node_th][0], self.nodes[int_node_th][0]))
            out_node += sum(hp * op(s1, hps_flatten) 
                           for hp, op in zip(hp_normal[int_node_th][1], self.nodes[int_node_th][1]))
            
            # intermediate parent nodes
            for ipn_th in range(int_node_th):
                out_node += sum(hp * op(out_nodes[ipn_th], hps_flatten) 
                                for hp, op in zip(hp_normal[int_node_th][ipn_th+2], self.nodes[int_node_th][ipn_th+2]))
                
            out_nodes.append(out_node)

        # cell output
        # concat output from every intermediate node before classifier
        out_add = torch.cat(out_nodes, dim=1)
        
        return out_add
        
    def __cell_preproc(self, in_channels, out_channels, reduction_p=False):
        if reduction_p:
            op = FactorizedReduceNoHyper(in_channels, out_channels, affine=False)
        else:
            op = StdConvNoHyper(in_channels, out_channels, kernel_size=1, stride=1, padding=0, affine=False)
        return op
                
    
    def __ops(self, channels, stride):
        ops = nn.ModuleList()
        
        # maxpool 3x3
        ops.append(PoolBN('max', channels, 3, stride=stride, padding=1, affine=False))
        # avgpool 3x3
        ops.append(PoolBN('avg', channels, 3, stride=stride, padding=1, affine=False))
        # skip connect
        if stride == 2:
            ops.append(FactorizedReduce(channels, channels, num_hyper=self.num_hyper, affine=False))
        else:
            ops.append(Identity())
        # sepconv 3x3
        ops.append(SepConv(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, 
                           padding=1, num_hyper=self.num_hyper, affine=False))
        # sepconv 5x5
        ops.append(SepConv(in_channels=channels, out_channels=channels, kernel_size=5, stride=stride, 
                           padding=2, num_hyper=self.num_hyper, affine=False))
        # dilconv 3x3
        ops.append(DilConv(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, 
                           padding=2, dilation=2, num_hyper=self.num_hyper, affine=False))
        # dilconv 5x5
        ops.append(DilConv(in_channels=channels, out_channels=channels, kernel_size=5, stride=stride, 
                           padding=4, dilation=2, num_hyper=self.num_hyper, affine=False))
        # zero
        ops.append(Zero(stride))
        
        return ops

    
class demo_net(nn.Module):
    def __init__(self, in_channels, init_channels, num_hyper, num_int_nodes=4, num_cells=2, stem_multiplier=3,
                 num_class=10):
        super(demo_net, self).__init__()
        
        self.in_channels = in_channels
        self.init_channels = init_channels
        self.num_hyper = num_hyper
        self.num_int_nodes = num_int_nodes
        self.num_cells = num_cells
        self.stem_multiplier = stem_multiplier
        self.num_class = num_class
        
        # preproc
        c_stem = self.init_channels*self.stem_multiplier
        self.stem = self.__stem_preproc(self.in_channels, c_stem)
        reduction_p = False
            
        # cells
        c_pp, c_p, c_cur = c_stem, c_stem, self.init_channels
        self.cells = nn.ModuleList()
        for cell_th in range(self.num_cells):
            if cell_th in [self.num_cells // 3, 2 * self.num_cells // 3]:
                c_cur *= 2
                reduction = True
            else:
                reduction = False
                
            self.cells.append(Cell(c_pp, c_p, c_cur, 
                                   num_hyper=self.num_hyper, num_int_nodes=self.num_int_nodes, 
                                   reduction_p=reduction_p, reduction=reduction))
            reduction_p = reduction
            c_cur_out = c_cur * self.num_int_nodes
            c_pp, c_p = c_p, c_cur_out
        
        # classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 2 reduction cells therefore *4
        self.linear1 = nn.Linear(self.num_int_nodes*self.init_channels*4, self.num_class)

    def forward(self, x, hps_normal, hps_reduce):
        
        # hps flatten to input hypernetworks
        hps_normal_flatten = torch.cat(hps_normal, dim=0).flatten()
        hps_reduce_flatten = torch.cat(hps_reduce, dim=0).flatten()
        
        # softmax (normalize) hparams to weight (times) each op
        hps_normal_softmax = [F.softmax(hp.flatten(), dim=-1).reshape(hp.shape) for hp in hps_normal]
        hps_reduce_softmax = [F.softmax(hp.flatten(), dim=-1).reshape(hp.shape) for hp in hps_reduce]

        # input after preprocessing
        out_stem = self.stem[1](self.stem[0](x))
        
        # cells
        s0, s1 = out_stem, out_stem
        for cell_th in range(self.num_cells):
            if self.cells[cell_th].reduction:
                hps_flatten = hps_reduce_flatten
                hps_softmax = hps_reduce_softmax
            else:
                hps_flatten = hps_normal_flatten
                hps_softmax = hps_normal_softmax
                
            out_add = self.cells[cell_th](s0, s1, hps_flatten, hps_softmax)
            s0, s1 = s1, out_add

        # classifier output
        out = self.gap(out_add)
        out = self.linear1(out.view(out.shape[0], -1))
        
        return out
    
    def __stem_preproc(self, in_channels, out_channels):
        ops = nn.ModuleList()
        ops.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        ops.append(nn.BatchNorm2d(out_channels, affine=True))
        return ops
        


# ## Main Program

# In[8]:


import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import argparse
import sys
import random
import torch
# import wandb
import os
import time
# from tensorboardX import SummaryWriter

# config
config = SearchConfig()
# config, unknown = parser.parse_known_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1 if 'mnist' in config.dataset.lower() else 3

# tensorboard
# writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))  # config.path = "searchs" folder
# writer.add_text('config', config.as_markdown(), 0)  # tensorboard text
logger = get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)
print('914')
# cudnn speedup & deterministic
cudnn.benchmark = True
# cudnn.deterministic = True

# set seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# stringify argument call
info = vars(config)

# load data
# train_loader, valid_loader, test_loader = mnist_loader(info, num_workers=config.num_workers)
# get data with meta info
input_size, in_channels, num_class, train_data = get_data(
    config.dataset, config.data_path, cutout_length=0, validation=False)    # search no cutout
# split data to train/validation
n_train = len(train_data)
split_train = int(n_train * config.percent_train)  # 25000:25000:10000
split_valid = int(n_train * config.percent_valid)
indices = list(range(n_train))
np.random.shuffle(indices)
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_train])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_train:split_train+split_valid])
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config.batch_size,
                                           sampler=train_sampler,
                                           num_workers=config.num_workers,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config.batch_size,
                                           sampler=valid_sampler,
                                           num_workers=config.num_workers,
                                           pin_memory=True)

# configure hyperparameters
hparams = []  #nn.ParameterList()
# hparams_normal = nn.ParameterList()
# hparams_reduce = nn.ParameterList()
for i in range(2):
    for int_node_th in range(config.num_int_nodes):
        hparams.append(nn.Parameter(1e-3 * torch.randn(int_node_th+2, config.num_ops).to(device), 
                                    requires_grad=True))



# In[9]:


model = demo_net(in_channels=in_channels, init_channels=config.init_channels, 
                 num_hyper=config.num_ops * sum(range(2, config.num_int_nodes+2)),
                 num_cells=config.num_cells)
model = model.to(device)

total_params = sum(param.numel() for param in model.parameters())
logger.info("Args: {}".format(str(config)))
logger.info("Model total parameters: {}".format(total_params))

# In[11]:


phi_optimizer = torch.optim.SGD(model.parameters(), config.phi_lr, momentum=config.phi_momentum,
                                weight_decay=config.phi_weight_decay)
# hparam_optimizer = torch.optim.SGD(hparams, config.hparam_lr)
hparam_optimizer = torch.optim.Adam(hparams, config.hparam_lr, betas=(config.hparam_beta_1, config.hparam_beta_2),
                                    weight_decay=config.hparam_weight_decay)

phi_loss_criterion = nn.CrossEntropyLoss().to(device)
hparam_loss_criterion = nn.CrossEntropyLoss().to(device)

# for phi
phi_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(phi_optimizer, config.total_epochs, 
                                                          eta_min=config.phi_lr_min)


# In[12]:


logger.info("Logger is set - training start")

start_time = time.time()
total_time = 0

best_top1 = 0.
for epoch in range(config.total_epochs):
   
    print_hparams(logger)

    lr = phi_lr_scheduler.get_last_lr()[0]
        
    train_top_1 = AverageMeter()
    train_top_5 = AverageMeter()
    val_top_1 = AverageMeter()
    val_top_5 = AverageMeter()

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    
    cur_step = epoch*len(train_loader)
    # writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    # grad_debug = np.zeros(shape=(50, 2, 8))
    # hps_debug = np.zeros(shape=(50, 2, 8))
    for step, ((train_x, train_y), (val_x, val_y)) in enumerate(zip(train_loader, valid_loader)):
               
        train_x, train_y = train_x.to(device, non_blocking=True), train_y.to(device, non_blocking=True)
        val_x, val_y = val_x.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = train_x.size(0)

        # phi gradient descent (train set)
        phi_optimizer.zero_grad()
        train_pred = model(train_x, hparams[:4], hparams[4:])
        phi_loss = phi_loss_criterion(train_pred, train_y)
        phi_loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.phi_grad_clip)
        phi_optimizer.step()

        # hparam gradient descent (val set)
        hparam_optimizer.zero_grad()
        val_pred = model(val_x, hparams[:4], hparams[4:])
        hparam_loss = hparam_loss_criterion(val_pred, val_y)
        hparam_loss.backward()
        hparam_optimizer.step()

        train_prec_1, train_prec_5 = accuracy(train_pred, train_y, topk=(1, 5))
        train_losses.update(phi_loss.item(), N)
        train_top_1.update(train_prec_1.item(), N)
        train_top_5.update(train_prec_5.item(), N)

        val_prec_1, val_prec_5 = accuracy(val_pred, val_y, topk=(1, 5))
        val_losses.update(hparam_loss.item(), N)
        val_top_1.update(val_prec_1.item(), N)
        val_top_5.update(val_prec_5.item(), N)
        
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.total_epochs, step, len(train_loader)-1, losses=train_losses,
                    top1=train_top_1, top5=train_top_5))
            
        if step % config.print_freq == 0 or step == len(valid_loader)-1:
            logger.info(
                "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.total_epochs, step, len(valid_loader)-1, losses=val_losses,
                    top1=val_top_1, top5=val_top_5))
            
        # writer.add_scalar('train/loss', phi_loss.item(), cur_step)
        # writer.add_scalar('train/top1', train_prec_1.item(), cur_step)
        # writer.add_scalar('train/top5', train_prec_5.item(), cur_step)
        
        # writer.add_scalar('val/loss', hparam_loss.item(), cur_step)
        # writer.add_scalar('val/top1', val_prec_1.item(), cur_step)
        # writer.add_scalar('val/top5', val_prec_5.item(), cur_step)
    
        cur_step += 1
        
    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.total_epochs, train_top_1.avg))
    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.total_epochs, val_top_1.avg))

    val_epoch_top1 = AverageMeter()
    val_epoch_top5 = AverageMeter()
    val_epoch_losses = AverageMeter()

    cur_step = (epoch + 1) * len(train_loader)

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X, hparams[:4], hparams[4:])
            loss = hparam_loss_criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            val_epoch_losses.update(loss.item(), N)
            val_epoch_top1.update(prec1.item(), N)
            val_epoch_top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.total_epochs, step, len(valid_loader) - 1, losses=val_epoch_losses,
                        top1=val_epoch_top1, top5=val_epoch_top5))

    # writer.add_scalar('val_epoch/loss', val_epoch_losses.avg, cur_step)
    # writer.add_scalar('val_epoch/top1', val_epoch_top1.avg, cur_step)
    # writer.add_scalar('val_epoch/top5', val_epoch_top5.avg, cur_step)

    logger.info("Valid (Epoch) : [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.total_epochs, val_epoch_top1.avg))
            
    phi_lr_scheduler.step()

    # genotype
    genotype = create_genotype(hparams[:4], hparams[4:], num_int_nodes=config.num_int_nodes)
    logger.info("genotype = {}".format(genotype))

    # genotype as a image
    # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
    # caption = "Epoch {}".format(epoch+1)
    # plot(genotype.normal, plot_path + "-normal", caption)
    # plot(genotype.reduce, plot_path + "-reduce", caption)
    
    if best_top1 < val_epoch_top1.avg:
        best_top1 = val_epoch_top1.avg
        best_genotype = genotype
        is_best = True
    else:
        is_best = False
    save_checkpoint(epoch, model, phi_optimizer, hparam_optimizer, hparams, phi_lr_scheduler, config.seed, config.path, is_best)
    print("")

logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
logger.info("Best Genotype = {}".format(best_genotype))


# In[ ]:




