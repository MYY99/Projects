import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets

# show_reconstruction
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# train
from time import time
import csv

# load_mnist
from torch.utils.data import random_split

from capsulelayers import DenseCapsule, PrimaryCapsule
from utils import combine_images, plot_log

ROOT = "/content/gdrive/My Drive/ECE4179_COLAB/Project2"

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)           # The Frobenius norm is the Euclidian norm of a matrix. The L2 (or L^2) norm is the Euclidian norm of a vector. 
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).to(device))
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)

class CapsuleNet56(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet56, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 128, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=9, stride=2, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)           # The Frobenius norm is the Euclidian norm of a matrix. The L2 (or L^2) norm is the Euclidian norm of a vector. 
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).to(device))
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)

def caps_loss(y_true, y_pred, x=None, x_recon=None, lam_recon=None):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean() 

    if x is None:
      output = L_margin
    else:
      L_recon = nn.MSELoss()(x_recon, x)
      output = L_margin + lam_recon * L_recon

    return output


def show_reconstruction(model, test_loader, n_images, args):

    model.eval()
    with torch.no_grad():
      for x, _ in test_loader:
          x = Variable(x[:min(n_images, x.size(0))].to(device))
          _, x_recon = model(x)
          data = np.concatenate([x.data.cpu(), x_recon.data.cpu()])
          img = combine_images(np.transpose(data, [0, 2, 3, 1]))
          image = img * 255
          Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
          print()
          print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
          print('-' * 70)
          plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
          plt.show()
          break


def test(model, test_loader, args, recon=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for x, y in test_loader:
          y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
          x, y = Variable(x.to(device)), Variable(y.to(device))
          
          if recon:
              y_pred, x_recon = model(x)
              loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
          else:
              y_pred = model(x)
              loss = caps_loss(y, y_pred)

          test_loss += loss.data.item() * x.size(0)  # sum up batch loss
          y_pred = y_pred.data.max(1)[1]
          y_true = y.data.max(1)[1]
          correct += y_pred.eq(y_true).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args, recon=True):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        ti = time()
        training_loss = 0
        correct = 0
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.to(device)), Variable(y.to(device))  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            
            if recon:
                y_pred, x_recon = model(x, y)  # forward
                loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            else:
                y_pred = model(x)  # forward
                loss = caps_loss(y, y_pred)

            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.data.item() * x.size(0)  # record the batch loss
            y_pred = y_pred.data.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()
            optimizer.step()  # update the trainable parameters with computed gradients

        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args, recon=recon)
        logwriter.writerow(dict(epoch=epoch, train_loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, train_acc=correct / len(train_loader.dataset), val_acc=val_acc))
        print("==> Epoch %02d: train_loss=%.5f, val_loss=%.5f, train_acc=%.4f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset), val_loss, 
                 correct / len(train_loader.dataset), val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    
    if next(iter(train_loader))[0].shape[2] == 56:
        torch.save(model.state_dict(), args.save_dir + '/trained_model56.pkl')
        print('Trained model saved to \'%s/trained_model56.pkl\'' % args.save_dir)
    else:
        torch.save(model.state_dict(), args.save_dir + '/trained_model28.pkl')
        print('Trained model saved to \'%s/trained_model28.pkl\'' % args.save_dir)
    
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model


def load_mnist(path=ROOT+'/data', download=False, batch_size=100, shift_pixels=2, input56=False, train_600=False, multi_test=False, aff_test=False):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path         : file path of the dataset
    :param download     : whether to download the original data
    :param batch_size   : batch size
    :param shift_pixels : maximum number of pixels to shift in each direction
    :param input56      : True if want input images to zero-pad to size 56x56 from size 28x28 (6000 input images)
    :param train_600    : True if train only 600 input images
    :param multi_test   : True if test double digits in 1 image of size 56x56 (only for trained using input56)
    :param aff_test     : True if test affine-transformed input images of size 56x56 (only for trained using input56)
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}

    if multi_test:
        test_data = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.Compose([transforms.ToTensor()]))
        # part_test = random_split(test_data, [int(len(test_data)*0.1), int(len(test_data)*0.9)])[0]

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
        train_loader = None
    elif aff_test:
        test_data = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.Compose([transforms.Pad(14),
                                                                 transforms.RandomAffine(degrees=25, translate=(0.3, 0.3), scale=(0.7, 1.5), shear=0.4),
                                                                 transforms.ToTensor()]))
        # part_test = random_split(test_data, [int(len(test_data)*0.1), int(len(test_data)*0.9)])[0]

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
        train_loader = None
    elif input56:
        train_data =  datasets.MNIST(path, train=True, download=download,
                                     transform=transforms.Compose([transforms.Pad(14),
                                                                  transforms.RandomCrop(size=56, padding=14),
                                                                  transforms.ToTensor()]))
        part_train = torch.utils.data.random_split(train_data, [6000, 54000])[0]
        train_loader = torch.utils.data.DataLoader(part_train, batch_size=batch_size, shuffle=True, **kwargs)
        
        test_data = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.Compose([transforms.Pad(14),
                                                                 transforms.ToTensor()]))
        # part_test = random_split(test_data, [int(len(test_data)*0.1), int(len(test_data)*0.9)])[0]

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    elif train_600:
        train_data =  datasets.MNIST(path, train=True, download=download,
                                     transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                        transforms.ToTensor()]))
        part_train = torch.utils.data.random_split(train_data, [600, 59400])[0]
        train_loader = torch.utils.data.DataLoader(part_train, batch_size=batch_size, shuffle=True, **kwargs)
        
        test_data = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.ToTensor())
        part_test = random_split(test_data, [int(len(test_data)*0.1), int(len(test_data)*0.9)])[0]

        test_loader = torch.utils.data.DataLoader(part_test, batch_size=batch_size, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=download,
                          transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                        transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True, **kwargs)
            
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=download,
                          transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

