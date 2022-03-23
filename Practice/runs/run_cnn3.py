

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from models.cnn import CNN
from dataset.MNIST_LOADER import make_loader
import argparse
import yaml

def train(epoch, model, loss_func, train_loader, optimizer):
    model.train()
    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch+1} | Batch Status: {batch_index*len(x)}/{len(train_loader.dataset)} \
            ({100. * batch_index * batch_size / len(train_loader.dataset):.0f}% | Loss: {loss.item():.6f}')

def test(model, loss_func, test_loader):
    model.eval()
    test_loss = 0
    correct_count = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        test_loss += loss_func(y_pred, y).item()
        pred = y_pred.data.max(1, keepdim=True)[1]
        # torch.eq : Computes element-wise equality. return counts value
        correct_count += pred.eq(y.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print(f'=======================\n Test set: Average loss: {test_loss:.4f}, Accuracy: {correct_count/len(test_loader.dataset):.3}')

# parser 정의
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--config_path', type=str, default='configs/cnn.yaml', help = "Score of english")

args = parser.parse_args()

train_loader, vaild_loader, test_loader, shape = make_loader(16)
C = shape[0]
W = shape[1]
H = shape[2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

cnn = CNN(C=C, W=W, H=H, K=3, S=2) 
cnn = cnn.to(device)
ce_loss = nn.CrossEntropyLoss()

# with 구문으로 파일을 불러옵니다.

with open(args.config_path) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
    print(type(config))

# Hyperparameters
batch_size = config['batch_size']
learning_rate = config['learning_rate']
epochs = config['epochs']
kernel_size = config['kernel_size']
stride = config['stride']

optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, optimizer)

test(cnn, ce_loss, test_loader)
