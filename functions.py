import pickle
import random
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transformed_dataset import TransformedImagesDataset
from dataset import ImagesDataset

def load_dataset(dataset_path: str, dataset_dump_path: str | None) -> tuple[TransformedImagesDataset, TransformedImagesDataset, TransformedImagesDataset]:
    try:
        if dataset_dump_path is None:
            raise ValueError()

        print("Trying to restore saved dataset")
        with open(dataset_dump_path, "rb") as f:
            train_set, validation_set, test_set = pickle.load(f)
            print("\tDone")

            return train_set, validation_set, test_set

    except:
        print("\tFail")
        pass

    print("Creating dataset")
    image_dataset = ImagesDataset(dataset_path)
    torch.random.manual_seed(123)
    train_set, valid_set, test_set = torch.utils.data.random_split(image_dataset, [0.85, 0.10, 0.05])

    train_set = TransformedImagesDataset(train_set)
    validation_set = valid_set
    test_set = test_set

    if dataset_dump_path is not None:
        print("\tSaving to a file")
        with open(dataset_dump_path, "wb") as f:
            pickle.dump((train_set,  validation_set, test_set), f)
        print("\tDone")

    return train_set, validation_set, test_set

def get_loader_for_training(train_set, valid_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def evaluate(model, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0
    accuracy = 0
    counter = 1
    with torch.no_grad():
        for mini_batch in eval_loader:
            print(f"Batch: {counter}")
            counter += 1
            images, labels = mini_batch[0].to(device), mini_batch[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * labels.size(0)
            accuracy += (outputs.max(1).indices == labels).sum().item() / labels.size(0)
    return eval_loss/len(eval_loader), accuracy * 100 / len(eval_loader)

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, criterion: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer):
    model.train()
    epoch_loss = 0
    accuracy = 0
    counter = 1
    for mini_batch in train_loader:
        print(f"Batch: {counter}")
        counter+=1
        images, labels = mini_batch[0].to(device), mini_batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * labels.size(0)
        accuracy += (outputs.max(1).indices == labels).sum().item() / labels.size(0)
    return epoch_loss / len(train_loader), accuracy * 100 / len(train_loader)

def test_model(model, test_loader, device, class_names):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            if predicted[0] == labels[0]:
                correct += 1
            total += 1
            print(f'Prediction: {class_names[predicted[0]]} - True: {class_names[labels[0]]}')
    print(f'Accuracy: {correct/total}')

def train_model(model, train_loader, valid_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_accuracy = train_epoch(model, train_loader, criterion, device, optimizer)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        eval_loss, eval_accuracy = evaluate(model, valid_loader, criterion, device)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)
        torch.save(model.state_dict(), f'model2.pth')
        torch.save(model.state_dict(), f'model2-{epoch_train_accuracy:.2f}-{eval_accuracy:.2f}.pth')
        print(f'Epochs: {epoch}/{num_epochs}')
    return train_losses, train_accuracies, eval_losses, eval_accuracies

def plot_summary(
    train_losses: list, train_accuracies: list, eval_losses: list, eval_accuracies: list
):
    _, (losses, accuracies) = plt.subplots(2)

    x = range(1, len(train_losses) + 1)
    losses.set_xlabel("Epoch")
    losses.set_ylabel("Loss")
    losses.plot(x, train_losses, label="Training loss")
    losses.plot(x, eval_losses, label="Evaluation loss")
    losses.legend()

    accuracies.set_xlabel("Epoch")
    accuracies.set_ylabel("Accuracy %")
    accuracies.plot(x, train_accuracies, label="Training accuracy")
    accuracies.plot(x, eval_accuracies, label="Evaluation accuracy")
    accuracies.legend()

    plt.show()