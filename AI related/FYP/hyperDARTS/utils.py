""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import torch.nn.functional as F

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


def save_checkpoint(epoch, model, phi_optimizer, hparam_optimizer, hparams, phi_lr_scheduler, seed, ckpt_dir,
                    is_best=False):
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


def print_hparams(logger, hparams):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
        org_formatters.append(handler.formatter)
        handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info("####### HPARAMS #######")
    logger.info("# Hparams - normal")
    logger.info(hparams[:4])
    for hp in hparams[:4]:
        logger.info(F.softmax(hp, dim=-1))

    logger.info("\n# Hparams - reduce")
    logger.info(hparams[4:])
    for hp in hparams[4:]:
        logger.info(F.softmax(hp, dim=-1))
    logger.info("#####################")

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
        handler.setFormatter(formatter)
