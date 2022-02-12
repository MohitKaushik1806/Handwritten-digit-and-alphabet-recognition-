import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# this function opens csv filws, reads the image pixels and labels separately in numpy arrays, and returns those arrays
def read_data(path):
    data = []
    labels = []
    
    for row in open(path):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        
        image = image.reshape((28, 28))
        
        data.append(image)
        labels.append(label)
        
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    
    return data, labels

alphabets, labels_alphabets = read_data("../inputs/A_Z Handwritten Data.csv")
labels_alphabets += 10

digits, labels_digits = read_data("../inputs/mnist_train.csv")

# to combine alphabets and digits images
data = np.vstack([alphabets, digits])
labels = np.hstack([labels_alphabets, labels_digits])
data /= 255.0

# creating dataset class for the images, as required by Pytorch DataLoader (see Pytorch documentation for more details)
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return (image, label)
    
# converting numpy array to tensors, and normalizing
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))]
)
dataset = MyDataset(data, labels, transform=transform)
# train-val split
train_data, val_data = torch.utils.data.random_split(dataset, [400000, 32451])

# creating separate data loaders for training data and validation data
# training data will be passed in batches of 64 images
# for validation, 1000 images will be tested at once
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1000, shuffle=True)

# defining the model architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(4608, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 36)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 4608)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)

# initializing the model
net = Net()

# defining the optimizer to Adam
optimizer = optim.Adam(net.parameters())

# setting device to CUDA for training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

# function to get validation accuracy
def test():
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            val_loss += F.nll_loss(outputs, labels, size_average=False).item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
        val_loss /= len(val_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

# function to train the model on the training data
def train(epoch):
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:    
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(inputs), len(train_loader.dataset), 100. * i / len(train_loader), loss.item()))

# setting number of epochs to 2
n_epochs = 2
for epoch in range(1, n_epochs + 1):
    train(epoch)
    print()
    test()

# saving the model state; requires the model class definition to be availale when loading the model
torch.save(net.state_dict(), '../models/model.pt')
print("Model saved successfully!")
    
