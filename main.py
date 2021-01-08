import numpy as np
import torch
from torch.optim import lr_scheduler
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dataset import data_loaders, dataset_sizes, class_labels
from models import create_model
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        print(f'Batch {i + 1}/{len(data_loader)}')
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, data_loaders, dataset_sizes, device, n_epochs=5):
    learning_params = list(model.fc.parameters())
    learning_params += list(model.fc2.parameters())
    learning_params += list(model.fc3.parameters())
    optimizer = torch.optim.SGD(learning_params, lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            data_loaders['test'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            dataset_sizes['test']
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            model,
            data_loaders['val'],
            loss_fn,
            device,
            dataset_sizes['val']
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    print(f'Best val accuracy: {best_accuracy}')
    model.load_state_dict(torch.load('best_model_state.bin'))
    return model, history


base_model = create_model(len(class_labels))

base_model, history = train_model(base_model, data_loaders, dataset_sizes, device)



def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='validation loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(history['train_acc'], label='train accuracy')
    ax2.plot(history['val_acc'], label='validation accuracy')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    fig.suptitle('Training history')

plot_training_history(history)
