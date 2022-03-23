import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset 
from torchvision import datasets, transforms


def make_loader(batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True, transform=transforms.ToTensor())
    valid_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    vaild_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    shape = train_dataset[0][0].shape
    channel =shape[0]
    width = shape[1]
    height = shape[2]
    print(f"channel: {channel}, width: {width}, height: {height}")
    
    return train_loader, vaild_loader, test_loader, shape