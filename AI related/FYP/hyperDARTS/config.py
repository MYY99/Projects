""" Config """


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
        parser.add_argument('--name', default='cifar10_experiment_c1')

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
        parser.add_argument('--phi_lr', type=float, default=0.025, help='lr for phi')
        parser.add_argument('--phi_lr_min', type=float, default=0.001, help='minimum lr for phi')
        parser.add_argument('--phi_momentum', type=float, default=0.9, help='momentum for phi')
        parser.add_argument('--phi_weight_decay', type=float, default=3e-4, help='weight decay for phi')
        parser.add_argument('--phi_grad_clip', type=float, default=5., help='gradient clipping for phi')

        # hparam
        parser.add_argument('--hparam_lr', type=float, default=3e-4, help='lr for hparam')
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
        self.plot_path = os.path.join(self.path, 'plots')
        
