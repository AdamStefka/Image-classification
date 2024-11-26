import torch
import torch.nn as nn
import torch.utils.data
from dataset import ImagesDataset
from torch.utils.data import DataLoader
import numpy as np
import random

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(32 * 40 * 40, 256)
        self.fc2 = nn.Linear(256, 20)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation_function = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        x = input_images
        x = self.max_pool(self.batch_norm1(self.activation_function(self.conv1(x))))
        x = self.max_pool(self.batch_norm2(self.activation_function(self.conv2(x))))
        x = self.max_pool(self.batch_norm3(self.activation_function(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.activation_function(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MyCNN()
