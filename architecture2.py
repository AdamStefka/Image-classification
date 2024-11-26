import torch
import torch.nn as nn
import torch.utils.data
from dataset import ImagesDataset
from torch.utils.data import DataLoader
import numpy as np
import random

class MyCNN2(nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()

        self.architect = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),

            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(),

            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(),

            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(),

            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(),

            nn.Flatten(),

            nn.LazyLinear(2048),
            nn.ReLU(),

            nn.LazyLinear(1024),
            nn.ReLU(),

            nn.LazyLinear(512),
            nn.ReLU(),

            nn.LazyLinear(20)
        )


    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        return self.architect(input_images)

model = MyCNN2()
