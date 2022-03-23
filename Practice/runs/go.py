
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from models.cnn import CNN
from dataset.MNIST_LOADER import make_loader
import argparse
import yaml

parser = argparse.ArgumentParser(description = 'quiz')
parser.add_argument('-c','--config_path', type=str, default='configs/',help='config path')
parser.add_argument('--save_path', type=str, default='weights/', help='save path')
parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
parser.add_argument('--model_name', type=str, default="CNN", help='model name')

## 1) 구현
# 1. args 출력하기
# 2. args 들 중 config_path를 통해 yaml 파일을 config변수에 할당.
# 3. config출력하기

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


with open(args.config_path) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

print(config)

pre_trained = args.pretrain
save_path = args.save_path
model_name = args.model_name
model = cnn

if pre_trained:
    model_dict = torch.load(save_path+model_name)
    model.load_state_dict(model_dict)

writer = SummaryWriter('runs/cnn/')


def train(epoch, model, loss_func, train_loader, optimizer):
    model.train()
    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss, epoch*batch_size + batch_index)
        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch+1} | Batch Status: {batch_index*len(x)}/{len(train_loader.dataset)} \
            ({100. * batch_index * batch_size / len(train_loader.dataset):.0f}% | Loss: {loss.item():.6f}')
    torch.save(model.state_dict(), save_path + model_name+str(epoch))
    print(model.state_dict())
            

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


optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train(epoch, cnn, ce_loss, train_loader, optimizer)


test(cnn, ce_loss, test_loader)
