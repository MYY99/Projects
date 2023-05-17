""" Network Architecture"""
import torch
import torch.nn as nn
from ops import *

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
            for parent_node_th in range(int_node_th + 2):
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
                                for hp, op in
                                zip(hp_normal[int_node_th][ipn_th + 2], self.nodes[int_node_th][ipn_th + 2]))

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
        c_stem = self.init_channels * self.stem_multiplier
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
        self.linear1 = nn.Linear(self.num_int_nodes * self.init_channels * 4, self.num_class)

    def forward(self, x, hps_normal, hps_reduce):

        # hps flatten to input hypernetworks
        hps_normal_flatten = torch.cat(hps_normal, dim=0).flatten()
        hps_reduce_flatten = torch.cat(hps_reduce, dim=0).flatten()

        # softmax (normalize) hparams to weight (times) each op
        hps_normal_softmax = [F.softmax(hp, dim=-1) for hp in hps_normal]
        hps_reduce_softmax = [F.softmax(hp, dim=-1) for hp in hps_reduce]

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


