import torch.utils.data
import numpy as np
import random
from torchvision import transforms
from os.path import join, dirname, abspath
import shutil


# import from other files
from functions import train_model
from functions import get_loader_for_training
from architecture import MyCNN
from architecture2 import MyCNN2
from functions import plot_summary
from functions import load_dataset
from functions import test_model

seed = 123
torch.initial_seed()
torch.random.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 32
CLASS_NAMES = ['book', 'bootle', 'car', 'cat', 'chair', 'computermouse', 'cup', 'dog', 'flower', 'fork', 'glass', 'glasses', 'headphones', 'knife', 'laptop', 'pen', 'plate', 'shoes', 'spoon', 'tree']
WORKING_DIRECTORY = abspath(join(dirname(__file__), "../"))
DATASET_PATH = "training_data"
DATASET_DUMP_PATH = "saved_data/dataset.pkl"


train_set, valid_set, test_set = load_dataset(DATASET_PATH, DATASET_DUMP_PATH)

choice = input("TRAIN OR TEST? (y/n): ")
if choice == "y":
    train_loader, valid_loader = get_loader_for_training(train_set, valid_set, batch_size=BATCH_SIZE)
    model = MyCNN2()
    model.to(DEVICE)
    train_losses, train_acc, eval_losses, eval_acc = train_model(model, train_loader, valid_loader, num_epochs=NUM_EPOCHS, device=DEVICE)
    print("Training complete")
    for i in range(NUM_EPOCHS):
        print(f"Epoch: {i}, Train Loss: {train_losses[i]}, Train Accuracy: {train_acc[i]} --- Eval Loss: {eval_losses[i]}, Eval Accuracy: {eval_acc[i]}")
    plot_summary(train_losses, train_acc, eval_losses, eval_acc)
if choice == "n":
    model = MyCNN2()
    model.to(DEVICE)
    model.load_state_dict(torch.load("model2-74.13-72.28.pth"))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    test_model(model, test_loader, DEVICE, CLASS_NAMES)