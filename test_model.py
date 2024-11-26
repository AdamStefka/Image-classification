from architecture2 import MyCNN2
from architecture import MyCNN
import torch
from functions import test_model
from dataset import ImagesDataset
from functions import get_loader
import numpy as np
import random

seed = 123
torch.initial_seed()
torch.random.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dataset = ImagesDataset("training_data")
train_loader, valid_loader, test_loader = get_loader(img_dataset, batch_size=32)


loaded_model = MyCNN()
loaded_model.load_state_dict(torch.load('model.pth'))
test_model(loaded_model, test_loader, device, class_names)