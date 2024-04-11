import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from IPython.display import clear_output

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1)        # in [1,28,28]   -> out [256,24,24]
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1)      # in [256,24,24] -> out [256,20,20]
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)      # in [256,20,20] -> out [128,16,16]
        self.fc1 = nn.Linear(128*16*16, 328)
        self.fc2 = nn.Linear(328, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(-1, 128*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return (F.log_softmax(x, dim=1))
		
class ConvNet56(nn.Module):
    def __init__(self):
        super(ConvNet56, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1)        # in [1,56,56]   -> out [128,52,52]
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2)      # in [128,52,52] -> out [256,24,24]
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1)      # in [256,24,24] -> out [256,20,20]
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)      # in [256,20,20] -> out [128,16,16]
        self.fc1 = nn.Linear(128*16*16, 328)
        self.fc2 = nn.Linear(328, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = x.view(-1, 128*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return (F.log_softmax(x, dim=1))
        
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
    
#This function should perform a single training epoch using our training data
def train_cnn(net, device, loader, optimizer, Loss_fun, acc_logger, loss_logger):
    
    #initialise counters
    epoch_loss = 0
    epoch_acc = 0
    
    #Set Network in train mode
    net.train()
    
    for i, (x, y) in enumerate(loader):
        
        #load images and labels to device
        x = x.to(device) # x is the image
        y = y.to(device) # y is the corresponding label
                
        #Forward pass of image through network and get output
        fx = net(x)
        
        #Calculate loss using loss function
        loss = Loss_fun(fx, y)
        
        #calculate the accuracy
        acc = calculate_accuracy(fx, y)

        #Zero Gradents
        optimizer.zero_grad()
        #Backpropagate Gradents
        loss.backward()
        #Do a single optimization step
        optimizer.step()
        
        #create the cumulative sum of the loss and acc
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        #log the loss for plotting
        loss_logger.append(loss.item())
        acc_logger.append(acc.item())

        #clear_output is a handy function from the IPython.display module
        #it simply clears the output of the running cell
        
        clear_output(True)
        print("TRAINING: | Itteration [%d/%d] | Loss %.2f |" %(i+1 ,len(loader) , loss.item()))
        
    #return the avaerage loss and acc from the epoch as well as the logger array       
    return epoch_loss / len(loader), epoch_acc / len(loader), acc_logger, loss_logger
    
#This function should perform a single evaluation epoch and will be passed our validation or evaluation/test data
#it WILL NOT be used to train out model
def evaluate_cnn(net, device, loader, Loss_fun, acc_logger, loss_logger=None):
    
    epoch_loss = 0
    epoch_acc = 0
    
    #Set network in evaluation mode
    #Layers like Dropout will be disabled
    #Layers like Batchnorm will stop calculating running mean and standard deviation
    #and use current stored values
    net.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            
            #load images and labels to device
            x = x.to(device)
            y = y.to(device)
            
            #Forward pass of image through network
            fx = net(x)
            
            #Calculate loss using loss function
            loss = Loss_fun(fx, y)
            
            #calculate the accuracy
            acc = calculate_accuracy(fx, y)
            
            #log the cumulative sum of the loss and acc
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            #log the loss for plotting if we passed a logger to the function
            if not (loss_logger is None):
              loss_logger.append(loss.item())
            acc_logger.append(acc.item())

            clear_output(True)
            print("EVALUATION: | Itteration [%d/%d] | Loss %.2f | Accuracy %.2f%% |" %(i+1 ,len(loader), loss.item(), 100*(epoch_acc/ len(loader))))
    
    #return the avaerage loss and acc from the epoch as well as the logger array       
    return epoch_loss / len(loader), epoch_acc / len(loader), acc_logger, loss_logger