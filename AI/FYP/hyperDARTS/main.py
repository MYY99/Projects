""" Main Program """

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
import random

import time
from archi import *
from config import *
from data_loader import *
from genotypes import *
from utils import *

from tensorboardX import SummaryWriter

# config
config = SearchConfig()
# config, unknown = parser.parse_known_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1 if 'mnist' in config.dataset.lower() else 3

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))  # config.path = "searchs" folder
writer.add_text('config', config.as_markdown(), 0)  # tensorboard text
logger = get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

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
    config.dataset, config.data_path, cutout_length=0, validation=False)  # search no cutout
# split data to train/validation
n_train = len(train_data)
split_train = int(n_train * config.percent_train)  # 25000:25000:10000
split_valid = int(n_train * config.percent_valid)
indices = list(range(n_train))
np.random.shuffle(indices)
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_train])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_train:split_train + split_valid])
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
hparams = []  # nn.ParameterList()
# hparams_normal = nn.ParameterList()
# hparams_reduce = nn.ParameterList()
for i in range(2):
    for int_node_th in range(config.num_int_nodes):
        hparams.append(nn.Parameter(1e-3 * torch.randn(int_node_th + 2, config.num_ops).to(device),
                                    requires_grad=True))

# In[9]:


model = demo_net(in_channels=in_channels, init_channels=config.init_channels,
                 num_hyper=config.num_ops * sum(range(2, config.num_int_nodes + 2)),
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

    print_hparams(logger, hparams)

    lr = phi_lr_scheduler.get_last_lr()[0]

    train_top_1 = AverageMeter()
    train_top_5 = AverageMeter()
    val_top_1 = AverageMeter()
    val_top_5 = AverageMeter()

    train_losses = AverageMeter()
    val_losses = AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

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

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.total_epochs, step, len(train_loader) - 1, losses=train_losses,
                    top1=train_top_1, top5=train_top_5))

        if step % config.print_freq == 0 or step == len(valid_loader) - 1:
            logger.info(
                "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.total_epochs, step, len(valid_loader) - 1, losses=val_losses,
                    top1=val_top_1, top5=val_top_5))

        writer.add_scalar('train/loss', phi_loss.item(), cur_step)
        writer.add_scalar('train/top1', train_prec_1.item(), cur_step)
        writer.add_scalar('train/top5', train_prec_5.item(), cur_step)

        writer.add_scalar('val/loss', hparam_loss.item(), cur_step)
        writer.add_scalar('val/top1', val_prec_1.item(), cur_step)
        writer.add_scalar('val/top5', val_prec_5.item(), cur_step)

        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.total_epochs, train_top_1.avg))
    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.total_epochs, val_top_1.avg))

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

    writer.add_scalar('val_epoch/loss', val_epoch_losses.avg, cur_step)
    writer.add_scalar('val_epoch/top1', val_epoch_top1.avg, cur_step)
    writer.add_scalar('val_epoch/top5', val_epoch_top5.avg, cur_step)

    logger.info(
        "Valid (Epoch) : [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.total_epochs, val_epoch_top1.avg))

    phi_lr_scheduler.step()

    # genotype
    genotype = create_genotype(hparams[:4], hparams[4:], num_int_nodes=config.num_int_nodes)
    logger.info("genotype = {}".format(genotype))

    # genotype as a image
    plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
    caption = "Epoch {}".format(epoch+1)
    plot(genotype.normal, plot_path + "-normal", caption)
    plot(genotype.reduce, plot_path + "-reduce", caption)

    if best_top1 < val_epoch_top1.avg:
        best_top1 = val_epoch_top1.avg
        best_genotype = genotype
        is_best = True
    else:
        is_best = False
    save_checkpoint(epoch, model, phi_optimizer, hparam_optimizer, hparams, phi_lr_scheduler, config.seed, config.path,
                    is_best)
    print("")

logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
logger.info("Best Genotype = {}".format(best_genotype))




